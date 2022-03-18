using System;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace Runner
{
    internal class Program
    {
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
        internal static extern IntPtr GetProcesAddress(IntPtr hModule, string procName);

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
        internal static extern IntPtr LoadLibrary(string lpszLib);

        [DllImport("CudaRuntime.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SomeCalculationsGPU(double[] a_h, uint N, uint M, int cuBlockSize);

        [DllImport("CppRuntime.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SomeCalculationsCPU(double[] a_h, uint N, uint M);

        static double ReadsExecutionTime()
        {
            Console.WriteLine("Hello World!");

            IntPtr hdl = LoadLibrary("CudaRuntime.dll");
            if (hdl != IntPtr.Zero)
            {
                IntPtr addr = GetProcesAddress(hdl, "sExecutionTime");
                if (addr != IntPtr.Zero)
                {
                    double[] managedArray = new double[1];
                    Marshal.Copy(addr, managedArray, 0, 1);
                    return managedArray[0];
                }
            }
            return 0;
        }

        private const int SET_SIZE = 2048 * 2048 * 8 * 4;

        static void Main(string[] args)
        {
            while (true)
            {
                double[] a_test = new double[SET_SIZE];
                const int N = SET_SIZE;
                const int M = 8;
                var stp = new Stopwatch();

                // Dummy data and second data set for validation
                for (int i = 0; i < N; i++) a_test[i] = (double)i;
                SomeCalculationsCsharpCPU(a_test, N, M);

                Console.WriteLine("Start managed C# ");
                {
                    double[] a_h = new double[SET_SIZE];
                    for (int i = 0; i < N; i++) a_h[i] = (double)i;

                    stp.Restart();
                    SomeCalculationsCsharpCPU(a_h, N, M);
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
                    double[] a_h = new double[SET_SIZE];
                    for (int i = 0; i < N; i++) a_h[i] = (double)i;

                    stp.Restart();
                    int cublocks = 256;
                    SomeCalculationsGPU(a_h, N, M, cublocks);
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
                    double[] a_h = new double[SET_SIZE];
                    for (int i = 0; i < N; i++) a_h[i] = (double)i;

                    stp.Start();
                    SomeCalculationsCPU(a_h, N, M);
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

        private static bool VerifyEquality(double[] f_a, double[] f_b)
        {
            if (f_a.Length != f_b.Length) return false;
            for (int i = 0; i < f_a.Length; i++)
            {
                // We lose precision during the default marshaling process
                if (f_a[i] - f_b[i] > 1) return false;
            }
            return true;
        }

        private static void SomeCalculationsCsharpCPU(double[] farr3, uint N, uint M)
        {
            for (uint i = 0; i < N; i++)
                farr3[i] = farr3[i] * farr3[i] * 0.1f - farr3[i] - 10f;
        }
    }
}