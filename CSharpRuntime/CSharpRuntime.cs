using System;

namespace CSharpRuntime
{
    public class CSharpRuntime
    {
        public static void SomeCalculationsCsharpCPU(float[] farr3, ulong N)
        {
            for (ulong i = 0; i < N; i++)
                farr3[i] = farr3[i] * farr3[i] * 0.1f - farr3[i] - 10f;
        }
    }
}
