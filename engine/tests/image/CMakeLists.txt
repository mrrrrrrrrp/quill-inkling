# Image decoding and preprocessing (CPU): base64 and data: URIs, limits, compositing and reduction, the
# processor's pixels (goldens of tools/vision_ref.py), content hashes and prompt ids.
ingine_add_test(NAME image SOURCES test_image.cpp LIBS ingine_image ingine_core)
