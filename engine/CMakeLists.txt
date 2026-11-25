# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
cmake_minimum_required(VERSION 3.28)

# quill targets Ada (sm_89, RTX 4070 SUPER class). Set the architecture before CUDA compiler detection.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build")
endif()

# Prefer the installer's private toolkit, then /opt/cuda, when nvcc is not chosen explicitly.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  foreach(_nvcc ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/nvidia/cu13/bin/nvcc
                ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/bin/nvcc /opt/cuda/bin/nvcc)
    if(EXISTS ${_nvcc})
      set(CMAKE_CUDA_COMPILER ${_nvcc} CACHE FILEPATH "CUDA compiler")
      break()
    endif()
  endforeach()
endif()

project(ingine LANGUAGES C CXX CUDA)

if(MSVC)
  # UTF-8 sources, standard preprocessor, large objects, no min/max macros, M_PI, quiet CRT deprecations.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8> $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>
                      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
                      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS _SILENCE_ALL_CXX20_DEPRECATION_WARNINGS)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(INGINE_BUILD_TESTS "Build ingine tests" OFF)
option(INGINE_BUILD_APPS "Build the ingine executables (serve, bench, validate)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) # no C++20 modules; skip the dependency scan
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ingine_link=1)
  set(CMAKE_JOB_POOL_LINK ingine_link)
endif()

include(cmake/Dependencies.cmake)

add_subdirectory(third_party)
add_subdirectory(src/core)

# Every other component directory owns its CMakeLists.txt and defines the target named here; a
# directory is added only once it has one, so the tree builds while components are in progress.
#   src/text              -> ingine_text
#   src/image             -> ingine_image
#   src/kernels/<area>    -> ingine_kernels_<area>   (aggregated by ingine_kernels)
#   src/model             -> ingine_model
#   src/engine            -> ingine_engine
#   src/serve             -> ingine_serve
#   apps/serve|bench|validate -> ingine-serve | ingine-bench | ingine-validate
function(ingine_add_component dir)
  if(EXISTS ${PROJECT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endfunction()

ingine_add_component(src/text)
ingine_add_component(src/image)

set(INGINE_KERNEL_AREAS common dense moe attn head prefill draft vision)
foreach(area ${INGINE_KERNEL_AREAS})
  ingine_add_component(src/kernels/${area})
endforeach()
add_library(ingine_kernels INTERFACE)
foreach(area ${INGINE_KERNEL_AREAS})
  if(TARGET ingine_kernels_${area})
    target_link_libraries(ingine_kernels INTERFACE ingine_kernels_${area})
  endif()
endforeach()

ingine_add_component(src/model)
ingine_add_component(src/engine)
ingine_add_component(src/serve)

if(INGINE_BUILD_APPS)
  foreach(app serve bench validate kld)
    ingine_add_component(apps/${app})
  endforeach()
endif()

if(INGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
