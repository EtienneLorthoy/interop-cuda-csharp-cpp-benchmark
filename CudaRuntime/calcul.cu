
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "stdlib.h"
#include <chrono>
#include <windows.h>
#include <cinttypes>

#include "Timer.cpp"

__global__ void some_calculations(double* a)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	a[idx] = a[idx] * a[idx] * 0.1 - a[idx] - 10;
}


// Cuda wrapper function
extern "C" int __declspec(dllexport) __stdcall SomeCalculationsGPU(
	double* a_h,                               // pointer to input array
	const unsigned int N,                     // input array size
	const unsigned int M,                     // kernel M parameter
	const int cuBlockSize = 512,              // kernel block size (max 512)
	const int showErrors = 1                  // show CUDA errors in console window
)
{
	Timer sw;
	printf("	Starto GPU %d ms\n", sw.get_elapsed_time() / CLOCKS_PER_SEC / CLOCKS_PER_SEC);
	double* a_d;
	size_t size = N * sizeof(double);
	int cuerr = 0;
	unsigned int timer = 0;

	cudaMalloc((void**)&a_d, size);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	int n_blocks = N / cuBlockSize + (N % cuBlockSize == 0 ? 0 : 1);

	some_calculations<<<n_blocks, cuBlockSize>>> (a_d);
	cudaDeviceSynchronize();

	cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);

	cudaFree(a_d);

	printf("	Finito GPU %d ms\n", sw.get_elapsed_time() / CLOCKS_PER_SEC / CLOCKS_PER_SEC);
	return cuerr;
}
