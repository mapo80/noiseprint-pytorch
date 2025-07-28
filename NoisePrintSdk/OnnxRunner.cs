using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

/// <summary>
/// Helper for running ONNX Noiseprint models on a single image.
/// </summary>
public static class OnnxRunner
{
    /// <summary>
    /// Run the appropriate ONNX model on <paramref name="imagePath"/>.
    /// </summary>
    /// <param name="imagePath">Input image.</param>
    /// <param name="provider">Execution provider name (CPU or CUDA).</param>
    /// <returns>Tuple with estimated quality factor, output shape and elapsed seconds.</returns>
    public static (int Qf, int[] Shape, double Time) Run(string imagePath, string provider = "CPU")
    {
        var estimator = new JpegQualityEstimator();
        int? qf = estimator.EstimateQuality(imagePath);
        if (!qf.HasValue)
            qf = Path.GetExtension(imagePath).ToLowerInvariant() == ".png" ? 101 : 100;

        string baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../"));
        string modelPath = Path.Combine(baseDir, "onnx_models", $"model_qf{qf}.onnx");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException(modelPath);

        using Image<L8> img = Image.Load<L8>(imagePath);
        int w = Math.Min(64, img.Width);
        int h = Math.Min(64, img.Height);
        img.Mutate(x => x.Crop(new Rectangle(0, 0, w, h)));

        float[] data = new float[w * h];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
                data[y * w + x] = img[x, y].PackedValue / 256f;
        }

        var tensor = new DenseTensor<float>(data, new[] { 1, 1, h, w });

        using var options = CreateOptions(provider);
        using var session = new InferenceSession(modelPath, options);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", tensor) };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
        sw.Stop();
        var output = results.First().AsTensor<float>();
        return (qf.Value, output.Dimensions.ToArray(), sw.Elapsed.TotalSeconds);
    }

    private static SessionOptions CreateOptions(string provider)
    {
        var opt = new SessionOptions();
        if (provider.Equals("cuda", StringComparison.OrdinalIgnoreCase))
            opt.AppendExecutionProvider_CUDA();
        else
            opt.AppendExecutionProvider_CPU();
        return opt;
    }
}
