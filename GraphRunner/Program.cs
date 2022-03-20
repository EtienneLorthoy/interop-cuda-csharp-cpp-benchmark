using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace GraphRunner
{
    internal class Program
    {
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
        internal static extern IntPtr GetProcesAddress(IntPtr hModule, string procName);

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
        internal static extern IntPtr LoadLibrary(string lpszLib);

        [DllImport("CudaRuntime.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SomeCalculationsGPU(float[] a_h, ulong N, int cuBlockSize);

        [DllImport("CppRuntime.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SomeCalculationsCPU(float[] a_h, ulong N);

        static float ReadsExecutionTime()
        {
            IntPtr hdl = LoadLibrary("CudaRuntime.dll");
            if (hdl != IntPtr.Zero)
            {
                IntPtr addr = GetProcesAddress(hdl, "sExecutionTime");
                if (addr != IntPtr.Zero)
                {
                    float[] managedArray = new float[1];
                    Marshal.Copy(addr, managedArray, 0, 1);
                    return managedArray[0];
                }
            }
            return 0;
        }

        private const ulong BASE_SET_SIZE = 2048 * 2048;
        private const int PASSES = 12;

        private static ulong[] SetSizes = new ulong[PASSES];
        private static double[] SetCSharpTimings = new double[PASSES];
        private static double[] SetCppTimings = new double[PASSES];
        private static double[] SetCUDATimings = new double[PASSES];
        
        static void Main(string[] args)
        {
            // Warmup 
            float[] a_warm = new float[100];
            CSharpRuntime.CSharpRuntime.SomeCalculationsCsharpCPU(a_warm, 100);
            SomeCalculationsCPU(a_warm, 100);
            SomeCalculationsGPU(a_warm, 100, 512);
            Console.Clear();

            for (uint p=0; p<PASSES; p++)
            {
                ulong N = BASE_SET_SIZE * p * p;
                float[] a_test = new float[N];
                SetSizes[p] = N;
                var stp = new Stopwatch();

                // Dummy data and second data set for validation
                for (ulong i = 0; i < N; i++) a_test[i] = (float)i;
                CSharpRuntime.CSharpRuntime.SomeCalculationsCsharpCPU(a_test, N);

                var itemsNumbersFormat = a_test.Length.ToString("##,#");
                Console.WriteLine("Pass #" + p);
                Console.WriteLine("Array of " + itemsNumbersFormat + " elements of size");
                Console.WriteLine("");

                Console.WriteLine("Start managed C# ");
                {
                    float[] a_h = new float[N];
                    for (ulong i = 0; i < N; i++) a_h[i] = (float)i;

                    stp.Restart();
                    CSharpRuntime.CSharpRuntime.SomeCalculationsCsharpCPU(a_h, N);
                    stp.Stop();

                    var elapsed = stp.Elapsed.TotalMilliseconds;
                    SetCSharpTimings[p] = elapsed;
                    Console.WriteLine("");
                    GC.Collect(2, GCCollectionMode.Forced, true);
                }


                Console.WriteLine("Start CUDA GPU ");
                {
                    float[] a_h = new float[N];
                    for (ulong i = 0; i < N; i++) a_h[i] = (float)i;

                    stp.Restart();
                    int cublocks = 256;
                    SomeCalculationsGPU(a_h, N, cublocks);
                    stp.Stop();

                    var elapsed = stp.Elapsed.TotalMilliseconds;
                    SetCUDATimings[p] = elapsed;
                    Console.WriteLine("");
                    GC.Collect(2, GCCollectionMode.Forced, true);
                }


                Console.WriteLine("Start C CPU ");
                {
                    float[] a_h = new float[N];
                    for (ulong i = 0; i < N; i++) a_h[i] = (float)i;

                    stp.Start();
                    SomeCalculationsCPU(a_h, N);
                    stp.Stop();

                    var elapsed = stp.Elapsed.TotalMilliseconds;
                    SetCppTimings[p] = elapsed;
                    Console.WriteLine("");
                    GC.Collect(2, GCCollectionMode.Forced, true);
                }

                GC.Collect(5, GCCollectionMode.Forced, true);
                Console.WriteLine("Ctrl + C to exit...");
                Console.SetCursorPosition(0, 0);
            }

            Plot();
            Console.WriteLine("Plotting saved in timings.png");
            Console.WriteLine("Press any key to exit...");

            Console.ReadKey();
        }

        private static void Plot()
        {
            var setSizeDivided = SetSizes.Select(s => (double)(s)).ToArray();

            var plt = new ScottPlot.Plot(900, 600);
            var sp = plt.AddScatter(setSizeDivided, SetCSharpTimings);
            sp.LineWidth = 2;
            sp.Label = "CSharp Timings";
            sp = plt.AddScatter(setSizeDivided, SetCUDATimings);
            sp.LineWidth = 2;
            sp.Label = "CUDA Timings";
            sp = plt.AddScatter(setSizeDivided, SetCppTimings);
            sp.LineWidth = 2;
            sp.Label = "Cpp Timings";

            plt.Legend(true, ScottPlot.Alignment.MiddleRight);
            plt.XAxis.Label("Float Set Size");
            plt.YAxis.Label("Compute Time (ms)");
            plt.SaveFig("timings.png");

            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                Arguments = "timings.png",
                FileName = "explorer.exe"
            };

            Process.Start(startInfo);
        }
    }
}