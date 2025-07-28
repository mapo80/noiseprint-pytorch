# NoisePrint PyTorch

This repository contains a PyTorch implementation of **NoisePrint**, an image forensics technique used to extract the sensor noise fingerprint of a camera model. Pretrained weights are provided and also exported to the ONNX format so that they can be executed in other environments such as .NET.

## ONNX models

The ONNX versions of the pretrained models are stored in the `onnx_models` directory. If you update the weights you can regenerate the models with:

```bash
python convert_to_onnx.py
```

The script `test_onnx.py` evaluates all the ONNX models on the sample images found in `examples`. It automatically estimates the image Quality Factor (QF) and chooses the appropriate model. A typical run on CPU looks as follows:

```
NC2016_2564.jpeg: QF=99 output=(1, 1, 64, 64) time=0.022s
faceswap.jpeg: QF=95 output=(1, 1, 64, 64) time=0.018s
inpainting.png: QF=101 output=(1, 1, 64, 64) time=0.017s
seamcarving.png: QF=101 output=(1, 1, 64, 64) time=0.021s
splicing.png: QF=101 output=(1, 1, 64, 64) time=0.018s
```

The average execution time is about **0.019 s** per image.

## .NET SDK

The `NoisePrintSdk` folder contains a .NET 8 SDK that allows running the ONNX models directly from C#. The SDK exposes both a programmatic API and a simple command-line application.

### Build and test

```bash
dotnet build NoisePrintSdk/NoisePrintSdk.csproj -c Release
dotnet test NoisePrintSdk.Tests/NoisePrintSdk.Tests.csproj -c Release
```

### Command line usage

After building, run the application by specifying an image path:

```bash
dotnet NoisePrintSdk/bin/Release/net8.0/NoisePrintSdk.dll examples/faceswap.jpeg
```

Example output on the files in `examples`:

```
QF=99 shape=(1,1,64,64) time=0.052s
QF=95 shape=(1,1,64,64) time=0.044s
QF=101 shape=(1,1,64,64) time=0.040s
QF=101 shape=(1,1,64,64) time=0.039s
QF=101 shape=(1,1,64,64) time=0.048s
```

The average processing time is approximately **0.045 s** per image.

### Library usage

```csharp
var (qf, shape, secs) = OnnxRunner.Run("my.jpg");
Console.WriteLine($"QF={qf} shape=({string.Join(',', shape)}) time={secs:F3}s");
```

The helper `JpegQualityEstimator` relies on the `MetadataExtractor` package to read the JPEG quantization tables and compute the Quality Factor (PNG images are assigned QF 101).

## Dependencies

Install the Python dependencies with:

```bash
pip install torch torchvision torchaudio onnx scipy onnxruntime pillow numpy
```

## Citation

```text
@article{Cozzolino2019_Noiseprint,
  title={Noiseprint: A CNN-Based Camera Model Fingerprint},
  author={D. Cozzolino and L. Verdoliva},
  journal={IEEE Transactions on Information Forensics and Security},
  doi={10.1109/TIFS.2019.2916364},
  pages={144-159},
  year={2020},
  volume={15}
}
```
