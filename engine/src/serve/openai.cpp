#include "serve/openai.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>

namespace ingine::serve {
namespace {

struct Bound {
  double value;
  const char* text;  // Python's str() of the bound, as printed in the error message
};

const Json* get(const Json& body, const char* key) {
  auto it = body.find(key);
  return it == body.end() || it->is_null() ? nullptr : &*it;
}

std::string range_message(const char* key, const std::optional<Bound>& lo, const std::optional<Bound>& hi) {
  return std::string("'") + key + "' must be in [" + (lo ? lo->text : "None") + ", " + (hi ? hi->text : "None") +
         "]";
}

int64_t saturate(double v) {
  if (v >= 9.2233720368547758e18) return std::numeric_limits<int64_t>::max();
  if (v <= -9.2233720368547758e18) return std::numeric_limits<int64_t>::min();
  return static_cast<int64_t>(v);
}

// server._num(body, key, int, lo, hi)
std::optional<int64_t> num_int(const Json& body, const char* key, std::optional<Bound> lo = std::nullopt,
                               std::optional<Bound> hi = std::nullopt) {
  const Json* v = get(body, key);
  if (!v) return std::nullopt;
  if (v->is_boolean() || !v->is_number() ||
      (v->is_number_float() && std::trunc(v->get<double>()) != v->get<double>())) {
    throw BadRequest(std::string("'") + key + "' must be an integer", key);
  }
  int64_t value;
  if (v->is_number_unsigned()) {
    value = int64_t(std::min<uint64_t>(v->get<uint64_t>(), uint64_t(std::numeric_limits<int64_t>::max())));
  } else if (v->is_number_integer()) {
    value = v->get<int64_t>();
  } else {
    value = saturate(v->get<double>());
  }
  if ((lo && double(value) < lo->value) || (hi && double(value) > hi->value)) {
    throw BadRequest(range_message(key, lo, hi), key);
  }
  return value;
}

// server._num(body, key, float, lo, hi)
std::optional<double> num_float(const Json& body, const char* key, std::optional<Bound> lo, std::optional<Bound> hi) {
  const Json* v = get(body, key);
  if (!v) return std::nullopt;
  if (v->is_boolean() || !v->is_number()) throw BadRequest(std::string("'") + key + "' must be a number", key);
  const double value = v->get<double>();
  if ((lo && value < lo->value) || (hi && value > hi->value)) throw BadRequest(range_message(key, lo, hi), key);
  return value;
}

int clamp_int(int64_t v) { return int(std::clamp<int64_t>(v, INT_MIN, INT_MAX)); }

// An image content part: its place in the request (for messages) and its data, a data: URI or bare base64.
struct ImagePart {
  std::string where;         // "messages[i].content[j]"
  const std::string* uri = nullptr;
  const std::string* base64 = nullptr;
};

const std::string* string_at(const Json& object, const char* key) {
  const auto it = object.find(key);
  return it != object.end() && it->is_string() ? &it->get_ref<const std::string&>() : nullptr;
}

// The image parts of the request in the order the chat template renders them: messages in order, the parts
// of a message in order. Shapes the template would reject anyway (non-array messages, ...) are skipped here.
std::vector<ImagePart> collect_images(const Json& messages) {
  std::vector<ImagePart> parts;
  if (!messages.is_array()) return parts;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const Json& message = messages[i];
    if (!message.is_object()) continue;
    const auto content = message.find("content");
    if (content == message.end() || !content->is_array()) continue;
    for (std::size_t j = 0; j < content->size(); ++j) {
      const Json& part = (*content)[j];
      if (!part.is_object()) continue;
      const std::string* type = string_at(part, "type");
      if (!type || (*type != "image_url" && *type != "input_image" && *type != "image")) continue;
      ImagePart found;
      found.where = "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "]";
      if (const auto url = part.find("image_url"); url != part.end() && url->is_object()) {
        found.uri = string_at(*url, "url");
      } else {
        found.uri = string_at(part, "image_url");
      }
      if (!found.uri) found.uri = string_at(part, "image");
      if (!found.uri) found.uri = string_at(part, "url");
      if (!found.uri) {
        if (const auto source = part.find("source"); source != part.end() && source->is_object()) {
          const std::string* kind = string_at(*source, "type");
          if (kind && *kind == "base64") found.base64 = string_at(*source, "data");
          if (kind && *kind == "url") found.uri = string_at(*source, "url");
        }
      }
      if (!found.uri && !found.base64) {
        throw BadRequest(found.where + ": an image part needs a base64 data: URI (image_url.url)", "messages");
      }
      parts.push_back(std::move(found));
    }
  }
  return parts;
}

// Decodes the request's images and replaces the template's placeholder tokens by the images' tokens.
void expand_images(const std::vector<ImagePart>& parts, const ImageOptions& options, std::vector<int32_t>& ids,
                   std::vector<model::PromptImage>& images) {
  if (int64_t(parts.size()) > options.max_images) {
    throw BadRequest("the request has " + std::to_string(parts.size()) + " images; at most " +
                         std::to_string(options.max_images) + " are accepted",
                     "messages");
  }
  // The template renders exactly one placeholder per image part, in order. More placeholders than images
  // means message text spells the placeholder token, and no pairing of the two is right.
  const auto placeholders = std::count(ids.begin(), ids.end(), model::IMAGE_PLACEHOLDER_ID);
  if (placeholders != std::ptrdiff_t(parts.size())) {
    throw BadRequest("message text must not contain the image placeholder token <|unused_200054|> in a request "
                     "with images",
                     "messages");
  }
  std::vector<image::Image> decoded;
  decoded.reserve(parts.size());
  int64_t total = 0;
  for (const ImagePart& part : parts) {
    try {
      if (part.uri) {
        decoded.push_back(image::decode_data_uri(*part.uri, options.limits));
      } else {
        const std::string bytes = image::decode_base64(*part.base64);
        decoded.push_back(image::decode(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()),
            options.limits));
      }
    } catch (const image::Error& e) {
      throw BadRequest(part.where + ": " + e.what(), "messages");
    }
    total += decoded.back().tokens();
  }
  if (total > options.max_total_tokens) {
    throw BadRequest("the images of the request take " + std::to_string(total) + " tokens; at most " +
                         std::to_string(options.max_total_tokens) + " are accepted",
                     "messages");
  }
  std::vector<int32_t> expanded;
  expanded.reserve(ids.size() + std::size_t(total));
  std::size_t next = 0;
  for (const int32_t id : ids) {
    if (id != model::IMAGE_PLACEHOLDER_ID) {
      expanded.push_back(id);
      continue;
    }
    image::Image& img = decoded[next++];
    model::PromptImage entry;
    entry.pos = int(expanded.size());
    entry.height = img.height;
    entry.width = img.width;
    const int tokens = img.tokens();
    for (int k = 0; k < tokens; ++k) expanded.push_back(image::token_id(img.hash, k));
    entry.rgb = std::make_shared<const std::vector<std::uint8_t>>(std::move(img.rgb));
    images.push_back(std::move(entry));
  }
  ids = std::move(expanded);
}

}  // namespace

bool truthy(const Json& value) {
  switch (value.type()) {
    case Json::value_t::null: return false;
    case Json::value_t::boolean: return value.get<bool>();
    case Json::value_t::number_integer: return value.get<int64_t>() != 0;
    case Json::value_t::number_unsigned: return value.get<uint64_t>() != 0;
    case Json::value_t::number_float: return value.get<double>() != 0.0;
    case Json::value_t::string: return !value.get_ref<const std::string&>().empty();
    case Json::value_t::array:
    case Json::value_t::object: return !value.empty();
    default: return true;
  }
}

ChatRequest parse_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                          const model::Sampling& defaults, const ImageOptions& image_options) {
  if (num_int(body, "n", Bound{1, "1"}, Bound{1, "1"}).value_or(1) != 1) {
    throw BadRequest("only n=1 is supported", "n");
  }
  Json kwargs = Json::object();
  if (auto it = body.find("chat_template_kwargs"); it != body.end() && truthy(*it)) {
    if (!it->is_object()) {
      throw BadRequest("'chat_template_kwargs' must be an object", "chat_template_kwargs");
    }
    kwargs = *it;
  }
  // body.get("reasoning_effort", kwargs.pop("reasoning_effort", None))
  Json effort = kwargs.contains("reasoning_effort") ? kwargs["reasoning_effort"] : Json();
  kwargs.erase("reasoning_effort");
  if (auto it = body.find("reasoning_effort"); it != body.end()) effort = *it;
  // kwargs.pop("enable_thinking", body.get("enable_thinking", True))
  Json enable_thinking = true;
  if (auto it = body.find("enable_thinking"); it != body.end()) enable_thinking = *it;
  if (kwargs.contains("enable_thinking")) {
    enable_thinking = kwargs["enable_thinking"];
    kwargs.erase("enable_thinking");
  }
  if (!enable_thinking.is_boolean()) throw BadRequest("'enable_thinking' must be a boolean", "enable_thinking");

  text::ChatOptions options;
  options.reasoning_effort = std::move(effort);
  options.enable_thinking = enable_thinking.get<bool>();
  options.template_kwargs = std::move(kwargs);
  const auto messages = body.find("messages");
  const auto tools = body.find("tools");
  static const Json kAbsent;  // null: no messages/tools key (passed by reference, not a deep copy of the body)
  const std::vector<ImagePart> image_parts = collect_images(messages == body.end() ? kAbsent : *messages);
  if (!image_parts.empty() && !image_options.enabled) {
    throw BadRequest("this server does not take images (it runs without the vision tower)", "messages");
  }
  text::RenderedPrompt prompt = fmt.render(messages == body.end() ? kAbsent : *messages,
                                           tools == body.end() ? kAbsent : *tools, options);
  ChatRequest req;
  if (!image_parts.empty()) expand_images(image_parts, image_options, prompt.ids, req.images);

  const int64_t room = int64_t(max_model_len) - int64_t(prompt.ids.size());
  if (room <= 0) {
    throw BadRequest("prompt has " + std::to_string(prompt.ids.size()) + " tokens; the model context is " +
                         std::to_string(max_model_len),
                     "messages");
  }
  std::optional<int64_t> max_tokens = num_int(body, "max_completion_tokens", Bound{1, "1"});
  if (!max_tokens) max_tokens = num_int(body, "max_tokens", Bound{1, "1"});

  req.sampling.temperature =
      float(num_float(body, "temperature", Bound{0.0, "0.0"}, Bound{2.0, "2.0"}).value_or(defaults.temperature));
  req.sampling.top_p = float(num_float(body, "top_p", Bound{0.0, "0.0"}, Bound{1.0, "1.0"}).value_or(defaults.top_p));
  req.sampling.top_k =
      clamp_int(std::max<int64_t>(0, num_int(body, "top_k", Bound{-1, "-1"}).value_or(defaults.top_k)));
  if (auto seed = num_int(body, "seed")) req.sampling.seed = static_cast<uint64_t>(*seed);
  if (req.sampling.top_p <= 0.0f) throw BadRequest("'top_p' must be in (0, 1]", "top_p");

  if (const Json* stop = get(body, "stop")) {
    if (stop->is_string()) {
      req.stop.push_back(stop->get<std::string>());
    } else if (stop->is_array() &&
               std::all_of(stop->begin(), stop->end(), [](const Json& s) { return s.is_string(); })) {
      for (const Json& s : *stop) req.stop.push_back(s.get<std::string>());
    } else {
      throw BadRequest("'stop' must be a string or a list of strings", "stop");
    }
  }
  if (auto it = body.find("stream"); it != body.end()) {
    if (!it->is_boolean()) throw BadRequest("'stream' must be a boolean", "stream");
    req.stream = it->get<bool>();
  }
  if (auto it = body.find("stream_options"); it != body.end() && truthy(*it)) {
    if (!it->is_object()) throw BadRequest("'stream_options' must be an object", "stream_options");
    if (auto u = it->find("include_usage"); u != it->end()) req.include_usage = truthy(*u);
  }
  req.max_new_tokens = int(std::min(max_tokens.value_or(room), room));
  req.prompt_ids = std::move(prompt.ids);
  req.prefill_ids = std::move(prompt.prefill_ids);
  return req;
}

Json error_body(int status, std::string_view message, const std::optional<std::string>& param) {
  const char* kind = status == 400 ? "invalid_request_error"
                     : status == 404 ? "not_found_error"
                     : status == 503 ? "server_busy"
                                     : "server_error";
  return Json{{"error",
               {{"message", message}, {"type", kind}, {"param", param ? Json(*param) : Json()}, {"code", status}}}};
}

Json delta_json(const text::Delta& d) {
  if (!d.reasoning_content.empty()) return Json{{"reasoning_content", d.reasoning_content}};
  if (!d.content.empty()) return Json{{"content", d.content}};
  const text::ToolCallDelta& tc = *d.tool_call;
  if (tc.id) {
    return Json{{"tool_calls",
                 Json::array({Json{{"index", tc.index},
                                   {"id", *tc.id},
                                   {"type", "function"},
                                   {"function", {{"name", tc.name ? Json(*tc.name) : Json()},
                                                 {"arguments", tc.arguments}}}}})}};
  }
  return Json{{"tool_calls", Json::array({Json{{"index", tc.index}, {"function", {{"arguments", tc.arguments}}}}})}};
}

std::string dump(const Json& value) { return value.dump(-1, ' ', false, Json::error_handler_t::replace); }

}  // namespace ingine::serve
