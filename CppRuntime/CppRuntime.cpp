#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <cinttypes>

#include "Timer.cpp"

extern "C" void __declspec(dllexport) __stdcall SomeCalculationsCPU
(
	float* a_h,
	const unsigned int N,
	const unsigned int M
)
{
	Timer sw;
	printf("	Start unmanaged C %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);
	for (unsigned int i = 0; i < N; i++)
		*(a_h + i) = *(a_h + i) * *(a_h + i) * 0.1F - *(a_h + i) - 10.0F;
	printf("	Unmanaged C stopped %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);
}

void ResetCursor() {
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	COORD pos = { 0, 0 };
	SetConsoleCursorPosition(hConsole, pos);
}

// Switch project type to .exe Application to run directly from a native C runtime
int main(void)
{
	clock_t start, stop;
	float* a_h;
	const unsigned int N = 2048 * 2048 * 8 * 4;
	const unsigned int M = 10;
	const int cuBlockSize = 512;

	while (true)
	{
		Timer sw;
		printf("Start C CPU\n");

		size_t size = N * sizeof(float);
		a_h = (float*)malloc(size);
		for (unsigned i = 0; i < N; i++) *(a_h + i) = (float)i;
		printf(" Allocating %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);

		printf(" Start computing %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);
		SomeCalculationsCPU(a_h, N, M);
		printf(" Computing finish %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);
		free(a_h);
		printf(" Releasing %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);

		printf("Total C CPU %+" PRId64 " ms\n", sw.get_elapsed_time().count() / CLOCKS_PER_SEC / 1000);
		ResetCursor();
	}
}