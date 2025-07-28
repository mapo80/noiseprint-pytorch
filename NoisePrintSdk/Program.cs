using System;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: NoisePrintSdk <image> [--provider CPU|CUDA]");
            return;
        }

        string image = args[0];
        string provider = "CPU";
        for (int i = 1; i < args.Length - 1; i++)
        {
            if (args[i] == "--provider")
                provider = args[i + 1];
        }

        var result = OnnxRunner.Run(image, provider);
        Console.WriteLine($"QF={result.Qf} shape=({string.Join(',', result.Shape)}) time={result.Time:F3}s");
    }
}
