import os
VAIP = os.environ.get("VAIP_CONFIG", r"C:\Program Files\RyzenAI.8.0oe-4.0-win_amd64aip_config.json")
import numpy as np, onnxruntime as ort, time
VAIP = rVAIP
pv = np.load('bench_pv.npz')['pv']
s = ort.InferenceSession('export/d1-3B_vision_tile512.onnx', providers=['VitisAIExecutionProvider'], provider_options=[{'config_file': VAIP}])
s.run(None, {'pixel_values': pv})
print('NPU BUSY WINDOW START (watch Task Manager now)')
for _ in range(20): s.run(None, {'pixel_values': pv})
print('NPU BUSY WINDOW END')
