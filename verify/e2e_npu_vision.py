"""E2E step B (ryzen env): image tiles through the NPU vision graph.

Run:  <ryzen-ai-1.8.0 python> e2e_npu_vision.py
Reads e2e_inputs.npz (from e2e_prep.py); writes e2e_img_embeds_npu.npz.
VAIP config/env: set VAIP_CONFIG to the SDK's vaip_config.json path.
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")
MODEL = os.environ.get("D1_VISION_GRAPH",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                    "models", "onnx", "d1-3B_vision_tile512.onnx"))

d = np.load('e2e_inputs.npz')
pv, pam, ss = d['pixel_values'], d['pixel_attention_mask'], d['spatial_shapes']
K = pv.shape[0]

sess = ort.InferenceSession(os.path.abspath(MODEL),
                            providers=['VitisAIExecutionProvider'],
                            provider_options=[{'config_file': VAIP}])
t0 = time.time()
embeds = []
for k in range(K):
    n = int(pam[k].sum())  # valid patches (full tile = 1024 for our fixed geometry)
    o = sess.run(None, {'pixel_values': pv[k:k+1].astype(np.float32)})[0]
    print(f'tile {k}: patches={n} -> embeds {o.shape}')
    embeds.append(o)
out = np.concatenate(embeds, axis=0)  # (K*256, 2048) row-major — matches the host-merge order
np.savez('e2e_img_embeds_npu.npz', embeds=out)
print(f'NPU vision total (compile excluded): {time.time()-t0:.1f}s; {out.shape}')