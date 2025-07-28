# Repo Guidelines

This repository contains a PyTorch implementation of Noiseprint. The codebase is
maintained for Python 3.12 and PyTorch 2.7 or newer.

## Running
- Install dependencies using `pip install torch torchvision torchaudio onnx scipy`.
- Execute `python convert_to_onnx.py` if you need to regenerate ONNX weights.
- Example usage is provided in `playground.ipynb` and the `Noiseprint` module.

## Contributing
- Keep style consistent with existing code (PEP8).
- When adding or updating weights also update the ONNX versions in
  `onnx_models/` by running `convert_to_onnx.py`.
