using System;
using System.IO;
using Xunit;

public class RunnerTests
{
    [Fact]
    public void Run_ReturnsOutput()
    {
        string img = Path.Combine(Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../")), "examples/faceswap.jpeg");
        var (qf, shape, time) = OnnxRunner.Run(img);
        Assert.InRange(qf, 1, 101);
        Assert.Equal(4, shape.Length);
        Assert.True(time > 0);
    }

    [Fact]
    public void Run_JpegKnownQuality()
    {
        string img = Path.Combine(Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../")), "examples/faceswap.jpeg");
        var (qf, shape, _) = OnnxRunner.Run(img);
        Assert.Equal(95, qf);
        Assert.Equal(4, shape.Length);
    }

    [Fact]
    public void Run_PngReturns101()
    {
        string img = Path.Combine(Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../")), "examples/splicing.png");
        var (qf, shape, _) = OnnxRunner.Run(img);
        Assert.Equal(101, qf);
        Assert.Equal(4, shape.Length);
    }
}
