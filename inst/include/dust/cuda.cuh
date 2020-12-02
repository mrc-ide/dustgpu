#ifndef DUST_CUDA_CUH
#define DUST_CUDA_CUH

#ifdef __NVCC__
#include "cuda.cuh"
#define D __device__
#define H __host__
#define HD __host__ __device
#define KERNEL __global__

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cpp11/protect.hpp>

static void HandleCUDAError(const char *file, int line,
                            cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
  cudaDeviceSynchronize();
#endif

  if (status != CUDA_SUCCESS || (status = cudaGetLastError()) != CUDA_SUCCESS) {
    if (status == cudaErrorUnknown) {
      cpp11::stop("%s(%i) An Unknown CUDA Error Occurred :(\n",
                  file, line);
    }
    cpp11::stop("%s(%i) CUDA Error Occurred;\n%s\n",
                file, line, cudaGetErrorString(status));
  }
}

#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))

#else
#define D
#define H
#define HD
#define KERNEL
#endif

#endif