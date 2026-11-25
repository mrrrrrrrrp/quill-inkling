# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
# Repository-pinned third-party libraries. No configure-time downloads, no system packages.

# nlohmann json (header only): #include <nlohmann/json.hpp>
add_library(ingine_nlohmann_json INTERFACE)
target_include_directories(ingine_nlohmann_json SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(nlohmann_json::nlohmann_json ALIAS ingine_nlohmann_json)

# cpp-httplib (header only): #include <httplib.h>
add_library(ingine_httplib INTERFACE)
target_include_directories(ingine_httplib SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp-httplib)
target_link_libraries(ingine_httplib INTERFACE Threads::Threads)
add_library(httplib::httplib ALIAS ingine_httplib)

# utf8proc (compiled, static): #include <utf8proc/utf8proc.h>
add_library(ingine_utf8proc STATIC utf8proc/utf8proc.c)
target_include_directories(ingine_utf8proc SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ingine_utf8proc PUBLIC UTF8PROC_STATIC)

# spdlog with bundled fmt: spdlog::spdlog
add_subdirectory(spdlog)

# llama.cpp's Jinja runtime: ingine_jinja (#include <jinja/runtime.h>)
add_subdirectory(llama-jinja)

# stb_image (header only, compiled once by src/image/stb_image_impl.cpp): #include <stb/stb_image.h>
add_library(ingine_stb INTERFACE)
target_include_directories(ingine_stb SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# libwebp 1.6.0, decoder only (compiled, static): src/dec, src/demux and the decoding parts of src/dsp and
# src/utils, generic C and x86 SIMD (run-time dispatch; no threads). #include <webp/decode.h>, <webp/demux.h>
add_library(ingine_webp STATIC
  libwebp/src/dec/alpha_dec.c
  libwebp/src/dec/buffer_dec.c
  libwebp/src/dec/frame_dec.c
  libwebp/src/dec/idec_dec.c
  libwebp/src/dec/io_dec.c
  libwebp/src/dec/quant_dec.c
  libwebp/src/dec/tree_dec.c
  libwebp/src/dec/vp8_dec.c
  libwebp/src/dec/vp8l_dec.c
  libwebp/src/dec/webp_dec.c
  libwebp/src/demux/anim_decode.c
  libwebp/src/demux/demux.c
  libwebp/src/dsp/alpha_processing.c
  libwebp/src/dsp/alpha_processing_sse2.c
  libwebp/src/dsp/alpha_processing_sse41.c
  libwebp/src/dsp/cpu.c
  libwebp/src/dsp/dec.c
  libwebp/src/dsp/dec_clip_tables.c
  libwebp/src/dsp/dec_sse2.c
  libwebp/src/dsp/dec_sse41.c
  libwebp/src/dsp/filters.c
  libwebp/src/dsp/filters_sse2.c
  libwebp/src/dsp/lossless_avx2.c
  libwebp/src/dsp/lossless.c
  libwebp/src/dsp/lossless_sse2.c
  libwebp/src/dsp/lossless_sse41.c
  libwebp/src/dsp/rescaler.c
  libwebp/src/dsp/rescaler_sse2.c
  libwebp/src/dsp/upsampling.c
  libwebp/src/dsp/upsampling_sse2.c
  libwebp/src/dsp/upsampling_sse41.c
  libwebp/src/dsp/yuv.c
  libwebp/src/dsp/yuv_sse2.c
  libwebp/src/dsp/yuv_sse41.c
  libwebp/src/utils/bit_reader_utils.c
  libwebp/src/utils/color_cache_utils.c
  libwebp/src/utils/filters_utils.c
  libwebp/src/utils/huffman_utils.c
  libwebp/src/utils/palette.c
  libwebp/src/utils/quant_levels_dec_utils.c
  libwebp/src/utils/random_utils.c
  libwebp/src/utils/rescaler_utils.c
  libwebp/src/utils/thread_utils.c
  libwebp/src/utils/utils.c
)
target_include_directories(ingine_webp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libwebp)
target_include_directories(ingine_webp SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libwebp/src)
target_compile_options(ingine_webp PRIVATE -w)
set_target_properties(ingine_webp PROPERTIES C_VISIBILITY_PRESET hidden)
