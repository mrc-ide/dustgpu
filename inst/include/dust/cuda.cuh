#ifndef DUST_CUDA_CUH
#define DUST_CUDA_CUH

#ifdef __NVCC__
#include "cuda.cuh"
#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__
#define KERNEL __global__

// This is necessary due to templates which are __host__ __device__
#define __nv_exec_check_disable__ _Pragma("nv_exec_check_disable")

#include <stdio.h>

// Standard cuda library functions
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

// cub functions (included with CUDA>=11; needs downloading otherwise)
#include <cub/device/device_select.cuh>

// R/cpp11 includes
#include <cpp11/protect.hpp>

const int warp_size = 32;

static void HandleCUDAError(const char *file, int line,
                            cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
  cudaDeviceSynchronize();
#endif

  if (status != CUDA_SUCCESS || (status = cudaGetLastError()) != CUDA_SUCCESS) {
    if (status == cudaErrorUnknown) {
      cudaProfilerStop();
      cpp11::stop("%s(%i) An Unknown CUDA Error Occurred :(\n",
                  file, line);
    }
    cudaProfilerStop();
    cpp11::stop("%s(%i) CUDA Error Occurred;\n%s\n",
                file, line, cudaGetErrorString(status));
  }
}

#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))

#else
#define DEVICE
#define HOST
#define HOSTDEVICE
#define KERNEL
#define __nv_exec_check_disable__
#endif

#endif