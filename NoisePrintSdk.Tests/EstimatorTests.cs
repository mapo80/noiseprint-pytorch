using System;
using System.IO;
using Xunit;

public class EstimatorTests
{
    private static string Map(string relative)
        => Path.Combine(Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../")), relative);

    [Theory]
    [InlineData("examples/NC2016_2564.jpeg")]
    [InlineData("examples/faceswap.jpeg")]
    public void Jpeg_ReturnsValue(string path)
    {
        var est = new JpegQualityEstimator();
        int? qf = est.EstimateQuality(Map(path));
        Assert.True(qf.HasValue && qf.Value >= 1 && qf.Value <= 100);
    }

    [Theory]
    [InlineData("examples/inpainting.png")]
    [InlineData("examples/seamcarving.png")]
    public void Png_ReturnsNull(string path)
    {
        var est = new JpegQualityEstimator();
        int? qf = est.EstimateQuality(Map(path));
        Assert.Null(qf);
    }
}
