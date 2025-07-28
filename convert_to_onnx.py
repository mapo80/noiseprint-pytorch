import os
import torch
from Noiseprint import FullConvNet

WEIGHTS_DIR = 'pretrained_weights'
ONNX_DIR = 'onnx_models'
os.makedirs(ONNX_DIR, exist_ok=True)

for fname in os.listdir(WEIGHTS_DIR):
    if not fname.endswith('.pth'):
        continue
    weight_path = os.path.join(WEIGHTS_DIR, fname)
    onnx_path = os.path.join(ONNX_DIR, fname.replace('.pth', '.onnx'))
    if os.path.exists(onnx_path):
        print(f'Skip {onnx_path}, already exists')
        continue
    print('Converting', weight_path)
    net = FullConvNet(0.9, torch.tensor(False), num_levels=17)
    state = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(state)
    net.eval()
    dummy = torch.randn(1,1,64,64)
    torch.onnx.export(
        net, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {2: 'h', 3: 'w'}, 'output': {2: 'h', 3: 'w'}},
        opset_version=12
    )
print('Done')

