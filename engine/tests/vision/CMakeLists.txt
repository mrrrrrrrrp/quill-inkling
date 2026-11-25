# Vision tower: kernels against FP64 oracles and the reference processor (GPU, small).
ingine_add_test(NAME vision_kernels SOURCES test_vision_kernels.cpp LIBS ingine_kernels_vision
                LABELS vision GPU ORACLE TIMEOUT 600)
# The whole tower with the checkpoint's vision weights (636 MB of the model file, about 1.5 GB of GPU memory)
# against the transformers goldens; skipped without the model directory.
if(TARGET ingine_model AND TARGET ingine_image)
  ingine_add_test(NAME vision_model SOURCES test_vision_model.cpp LIBS ingine_model ingine_image
                  LABELS vision GPU TIMEOUT 900)
endif()
