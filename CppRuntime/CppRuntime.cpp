#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <cinttypes>

#include "Timer.cpp"

extern "C" int __declspec(dllexport) __stdcall SomeCalculationsCPU
(
	double* a_h,
	const unsigned int N,
	const unsigned int M
)
{
	Timer sw;
	printf("	Start unmanaged C %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);
	for (unsigned int i = 0; i < N; i++)
		*(a_h + i) = *(a_h + i) * *(a_h + i) * 0.1 - *(a_h + i) - 10;
	printf("	Unmanaged C stopped %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);
	return 0;
}

// Switch project type to .exe Application to run directly from a native C runtime
int main(void)
{
	clock_t start, stop;
	double* a_h;
	const unsigned int N = 2048 * 2048 * 4;
	const unsigned int M = 10;
	const int cuBlockSize = 512;

	while (true)
	{
		Timer sw;
		size_t size = N * sizeof(double);
		a_h = (double*)malloc(size);
		for (unsigned i = 0; i < N; i++) *(a_h + i) = (double)i;

		SomeCalculationsCPU(a_h, N, M);
		stop = clock();

		printf("Finito C CPU %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);
	}
}