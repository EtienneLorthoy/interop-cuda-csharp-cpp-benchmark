
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "stdlib.h"
#include <chrono>
#include <windows.h>
#include <cinttypes>

#include "Timer.cpp"

__global__ void some_calculations(float* a)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	a[idx] = a[idx] * a[idx] * 0.1F - a[idx] - 10.0F;
}


// Cuda wrapper function
extern "C" int __declspec(dllexport) __stdcall SomeCalculationsGPU(
	float* a_h,                               // pointer to input array
	const unsigned long N,                     // input array size
	const int cuBlockSize = 512              // kernel block size (max 512)
)
{
	Timer sw;
	printf("	Start unmanaged CUDA %+" PRId64 " ms\n", sw.get_elapsed_time() / CLOCKS_PER_SEC / 1000);
	float* a_d;
	size_t size = N * sizeof(float);
	int cuerr = 0;
	unsigned int timer = 0;

	cudaMalloc((void**)&a_d, size);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	int n_blocks = N / cuBlockSize + (N % cuBlockSize == 0 ? 0 : 1);
	printf("	Allocating GPU DRAM %+" PRId64 " ms\n", sw.get_elapsed_time() / CLOCKS_PER_SEC / 1000);

	Timer sw2;
	printf("	Start Kernel CUDA %+" PRId64 " ms\n", sw2.get_elapsed_time() / CLOCKS_PER_SEC / 1000);
	some_calculations<<<n_blocks, cuBlockSize>>> (a_d);
	cudaDeviceSynchronize();
	printf("	Kernel CUDA stopped %+" PRId64 " ms\n", sw2.get_elapsed_time() / CLOCKS_PER_SEC / 1000);

	Timer sw3;
	cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
	cudaFree(a_d);
	printf("	Releasing GPU DRAM %+" PRId64 " ms\n", sw3.get_elapsed_time() / CLOCKS_PER_SEC / 1000);

	printf("	Unmanaged CUDA stopped %+" PRId64 " ms\n", sw.get_elapsed_time() / CLOCKS_PER_SEC / 1000);
	return cuerr;
}
