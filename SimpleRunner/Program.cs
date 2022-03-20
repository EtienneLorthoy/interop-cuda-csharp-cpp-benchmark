using System;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace SimpleRunner
{
    internal class Program
    {
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
        internal static extern IntPtr GetProcesAddress(IntPtr hModule, string procName);

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
        internal static extern IntPtr LoadLibrary(string lpszLib);

        [DllImport("CudaRuntime.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SomeCalculationsGPU(float[] a_h, uint N, int cuBlockSize);

        [DllImport("CppRuntime.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SomeCalculationsCPU(float[] a_h, uint N);

        private const int SET_SIZE = 2048 * 2048 * 8 * 8;

        static void Main(string[] args)
        {
            while (true)
            {
                float[] a_test = new float[SET_SIZE];
                const int N = SET_SIZE;
                var sizeOfCurrent = sizeof(float);
                var stp = new Stopwatch();

                // Dummy data and second data set for validation
                for (int i = 0; i < N; i++) a_test[i] = (float)i;
                CSharpRuntime.CSharpRuntime.SomeCalculationsCsharpCPU(a_test, N);

                var itemsNumberFormat = SET_SIZE.ToString("##,#");
                Console.WriteLine("Array of " + itemsNumberFormat + " elements of size " + sizeOfCurrent + "bytes");
                Console.WriteLine("");

                Console.WriteLine("Start managed C# ");
                {
                    float[] a_h = new float[SET_SIZE];
                    for (int i = 0; i < N; i++) a_h[i] = (float)i;

                    stp.Restart();
                    CSharpRuntime.CSharpRuntime.SomeCalculationsCsharpCPU(a_h, N);
                    stp.Stop();

                    Console.WriteLine("Total compute time for C# (ms): " + stp.Elapsed.TotalMilliseconds);
                    Console.Write("Data 32 set:" + a_h[32]);
                    Console.Write(" test:" + a_test[32]);
                    Console.WriteLine(" validated:" + VerifyEquality(a_h, a_test));
                    Console.WriteLine("");
                    GC.Collect(3, GCCollectionMode.Forced, true);
                }


                Console.WriteLine("Start CUDA GPU ");
                {
                    float[] a_h = new float[SET_SIZE];
                    for (int i = 0; i < N; i++) a_h[i] = (float)i;

                    stp.Restart();
                    int cublocks = 256;
                    SomeCalculationsGPU(a_h, N, cublocks);
                    stp.Stop();

                    Console.WriteLine("Total compute time for CUDA (ms): " + stp.Elapsed.TotalMilliseconds);
                    Console.Write("Data 32 set:" + a_h[32]);
                    Console.Write(" test:" + a_test[32]);
                    Console.WriteLine(" validated:" + VerifyEquality(a_h, a_test));
                    Console.WriteLine("");
                    GC.Collect(3, GCCollectionMode.Forced, true);
                }


                Console.WriteLine("Start C CPU ");
                {
                    float[] a_h = new float[SET_SIZE];
                    for (int i = 0; i < N; i++) a_h[i] = (float)i;

                    stp.Start();
                    SomeCalculationsCPU(a_h, N);
                    stp.Stop();

                    Console.WriteLine("Total compute time for C (ms): " + stp.Elapsed.TotalMilliseconds);
                    Console.Write("Data 32 set:" + a_h[32]);
                    Console.Write(" test:" + a_test[32]);
                    Console.WriteLine(" validated:" + VerifyEquality(a_h, a_test));
                    Console.WriteLine("");
                    GC.Collect(3, GCCollectionMode.Forced, true);
                }

                GC.Collect(5, GCCollectionMode.Forced, true);
                Console.WriteLine("Ctrl + C to exit...");
                Console.SetCursorPosition(0, 0);
            }
        }

        private static bool VerifyEquality(float[] f_a, float[] f_b)
        {
            if (f_a.Length != f_b.Length) return false;
            for (int i = 0; i < f_a.Length; i++)
            {
                if (f_a[i] != f_b[i]) return false;
            }
            return true;
        }
    }
}