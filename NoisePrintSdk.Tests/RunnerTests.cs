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
}
