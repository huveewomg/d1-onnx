"""Keep the NPU busy for ~8 s (20 vision tiles) so Task Manager's NPU meter becomes visible.

The per-decision DPU burst is ~0.4 s — invisible to task-manager sampling. This loop
sustains the load. Run with the ryzen-ai-1.8.0 python.
"""
import os
import sys

import numpy as np
import onnxruntime as ort

VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")
MODEL = os.environ.get("D1_VISION_GRAPH",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                    "models", "onnx", "d1-3B_vision_tile512.onnx"))
# patchified input: re-create deterministically, or load bench_pv.npz if present
if os.path.exists('bench_pv.npz'):
    pv = np.load('bench_pv.npz')['pv']
else:
    pv = np.random.default_rng(0).standard_normal((1, 1024, 768)).astype(np.float32)

s = ort.InferenceSession(os.path.abspath(MODEL),
                         providers=['VitisAIExecutionProvider'],
                         provider_options=[{'config_file': VAIP}])
s.run(None, {'pixel_values': pv})
print('NPU BUSY WINDOW START (watch Task Manager now)')
for _ in range(20):
    s.run(None, {'pixel_values': pv})
print('NPU BUSY WINDOW END')