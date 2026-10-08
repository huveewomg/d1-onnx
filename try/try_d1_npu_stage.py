"""NPU vision stage for try_d1.py — runs in the ryzen-ai-1.8.0 env.

Input npz: pixel_values (K, 1024, 768), pixel_attention_mask (K, 1024) int32, spatial_shapes (K, 2)
Output npz: embeds (K*256, 2048)
"""
import os
import os
import sys

import numpy as np
import onnxruntime as ort

VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")

src, dst, model = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(src)
pv, mask = d["pv"], d["mask"]
sess = ort.InferenceSession(model, providers=["VitisAIExecutionProvider"],
                            provider_options=[{"config_file": VAIP}])
# single-tile contract: (1024, 768) full-valid = exact graph shape
out = sess.run(None, {"pixel_values": pv[0:1].astype(np.float32)})[0]
np.savez(dst, embeds=out.reshape(-1, 2048))
# bench mode: python try_d1_npu_stage.py <src> <dst> <model> bench -> loops 20 inferences (keeps NPU busy)
if len(sys.argv) > 4 and sys.argv[4] == "bench":
    for _ in range(20):
        sess.run(None, {"pixel_values": pv[0:1].astype(np.float32)})
    print("20 vision tiles done")
