# ingine_image: image decoding and preprocessing for the vision tower (CUDA-free).
add_library(ingine_image STATIC
  image.cpp
  resize.cpp
  stb_image_impl.cpp
)
ingine_internal_includes(ingine_image)
target_include_directories(ingine_image PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_image PRIVATE ingine_stb ingine_webp)
