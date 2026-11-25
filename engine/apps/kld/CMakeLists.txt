# quill-kld: teacher-forced KL divergence of a configuration against a saved reference.
add_executable(quill-kld main.cpp)
ingine_internal_includes(quill-kld)
target_link_libraries(quill-kld PRIVATE ingine_model ingine_serve ingine_text CUDA::cudart)
