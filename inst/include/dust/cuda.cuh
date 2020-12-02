#ifndef DUST_CUDA_CUH
#define DUST_CUDA_CUH

#ifdef __NVCC__
#include "cuda.cuh"
#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__
#define KERNEL __global__

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>
#include <cpp11/protect.hpp>

#include <cub/device/device_select.cuh>

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
#endif

#endif