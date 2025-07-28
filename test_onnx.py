import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from utilityRead import jpeg_qtableinv

EXAMPLES_DIR = 'examples'
MODELS_DIR = 'onnx_models'


def estimate_qf(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png':
        return 101
    try:
        return int(round(jpeg_qtableinv(path)))
    except Exception:
        return 100


def load_image(path: str):
    img = Image.open(path).convert('L')
    img = img.crop((0, 0, min(64, img.width), min(64, img.height)))
    arr = np.asarray(img, dtype=np.float32) / 256.0
    return arr[None, None]


def run_model(image_path: str):
    qf = estimate_qf(image_path)
    model_path = os.path.join(MODELS_DIR, f"model_qf{qf}.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    img = load_image(image_path)
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    start = time.perf_counter()
    result = sess.run(None, {'input': img})[0]
    elapsed = time.perf_counter() - start
    return qf, result.shape, elapsed


def main():
    for fname in sorted(os.listdir(EXAMPLES_DIR)):
        path = os.path.join(EXAMPLES_DIR, fname)
        qf, shape, secs = run_model(path)
        print(f"{fname}: QF={qf} output={shape} time={secs:.3f}s")


if __name__ == '__main__':
    main()
