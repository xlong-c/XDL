cmake_minimum_required(VERSION 3.25)

# WSL + CMake Tools can accidentally select a Windows kit for a Linux
# configure. Fail early with a clear message instead of surfacing a later
# FindThreads/CUDAToolkit error.
function(xdl_require_linux_host_compiler compiler_var)
  if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    return()
  endif()

  if(NOT DEFINED ${compiler_var})
    return()
  endif()

  set(compiler_path "${${compiler_var}}")
  if(compiler_path MATCHES "\\.exe$")
    message(FATAL_ERROR
      "Detected a Windows compiler in a Linux/WSL configure: "
      "${compiler_var}='${compiler_path}'.\n"
      "Use a Linux host compiler such as /usr/bin/clang++ or /usr/bin/g++, "
      "then reconfigure with a fresh build directory.")
  endif()
endfunction()

xdl_require_linux_host_compiler(CMAKE_CXX_COMPILER)

project(xdl_cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(XDL_CUDA_ARCHITECTURE "90a" CACHE STRING "CUDA SM architecture, for example 80, 90, or 90a")

set(XDL_CUTLASS_INCLUDE
    "${CMAKE_SOURCE_DIR}/third_party/cutlass/include"
    CACHE PATH "CUTLASS/CuTe include directory")

if(DEFINED ENV{CONDA_PREFIX})
  set(XDL_CONDA_PREFIX "$ENV{CONDA_PREFIX}")
elseif(EXISTS "/root/miniconda3/envs/xdl")
  set(XDL_CONDA_PREFIX "/root/miniconda3/envs/xdl")
else()
  set(XDL_CONDA_PREFIX "")
endif()

set(XDL_CUDA_INCLUDE_DIRS "")

function(xdl_append_include_if_exists include_dir)
  if(include_dir AND EXISTS "${include_dir}")
    list(APPEND XDL_CUDA_INCLUDE_DIRS "${include_dir}")
    set(XDL_CUDA_INCLUDE_DIRS "${XDL_CUDA_INCLUDE_DIRS}" PARENT_SCOPE)
  endif()
endfunction()

xdl_append_include_if_exists("${CMAKE_SOURCE_DIR}")
xdl_append_include_if_exists("${XDL_CUTLASS_INCLUDE}")
xdl_append_include_if_exists("${XDL_CONDA_PREFIX}/targets/x86_64-linux/include")
xdl_append_include_if_exists("${XDL_CONDA_PREFIX}/targets/x86_64-linux/include/cccl")

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  xdl_append_include_if_exists("${CUDAToolkit_INCLUDE_DIRS}")
endif()

add_library(xdl_cuda_options INTERFACE)
target_include_directories(xdl_cuda_options INTERFACE ${XDL_CUDA_INCLUDE_DIRS})
target_compile_options(xdl_cuda_options INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${XDL_CUDA_ARCHITECTURE}>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

function(xdl_add_cuda_object target_name source_file)
  add_library(${target_name} OBJECT "${source_file}")
  target_link_libraries(${target_name} PRIVATE xdl_cuda_options)
  set_target_properties(${target_name} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
  )
endfunction()

function(xdl_add_cuda_demo target_name source_file)
  add_executable(${target_name} "${source_file}")
  target_link_libraries(${target_name} PRIVATE xdl_cuda_options)
  if(CUDAToolkit_FOUND)
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
  endif()
endfunction()

# Object targets do not require a main() function. They are useful for
# kernel-only files and for producing accurate compile_commands.json entries.
xdl_add_cuda_object(xdl_hgemm_mma_stage_tn_cute
  learn/cuda/kernels/hgemm/hgemm_mma_stage_tn_cute.cu)

xdl_add_cuda_object(xdl_hgemm_wgmma_stages
  learn/cuda/kernels/hgemm/hgemm_wgmma_stages.cu)

# Examples with main() can be enabled here as needed.
# xdl_add_cuda_demo(xdl_sgemm_cute learn/cuda/kernels/gemm/sgemm_cute.cu)
