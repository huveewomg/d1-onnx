"""Compile + benchmark the d1-3B VISION encoder on the XDNA2 NPU (VitisAI EP).

Parity is checked against a CPU EP reference computed here; then warm latency over
one 512x512 tile (~1024 patches -> 256 image tokens). Writes npu_vision_out.npz.
"""
import glob
import os
import time

import numpy as np
import onnxruntime as ort

VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")
MODEL = os.environ.get("D1_VISION_GRAPH",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                    "models", "onnx", "d1-3B_vision_tile512.onnx"))

rng = np.random.default_rng(0)
pv = rng.standard_normal((1, 1024, 768)).astype(np.float32)
feeds = {"pixel_values": pv}

s_cpu = ort.InferenceSession(os.path.abspath(MODEL), providers=["CPUExecutionProvider"])
ref = s_cpu.run(None, feeds)[0]
print(f"CPU reference embeds: {ref.shape}")

t0 = time.time()
s = ort.InferenceSession(os.path.abspath(MODEL),
                         providers=["VitisAIExecutionProvider"],
                         provider_options=[{"config_file": VAIP}])
print(f"NPU session created in {time.time() - t0:.0f}s; providers={s.get_providers()}")

out = s.run(None, feeds)[0]
np.savez("npu_vision_out.npz", embeds=out)
d = float(np.abs(out.astype(np.float64) - ref.astype(np.float64)).max())
print(f"maxdiff NPU vs CPU: {d:.3e}")

for _ in range(3):
    s.run(None, feeds)
ts = []
for _ in range(20):
    a = time.perf_counter()
    s.run(None, feeds)
    ts.append((time.perf_counter() - a) * 1000)
print(f"NPU tile latency: p50={np.percentile(ts, 50):.0f}ms mean={np.mean(ts):.0f}ms p95={np.percentile(ts, 95):.0f}ms")

caches = sorted(glob.glob(os.path.join(os.environ.get("TEMP", r"C:\temp"), "RT_*", "vaip", ".cache", "*")),
                key=os.path.getmtime)
if caches:
    summary = os.path.join(caches[-1], "preliminary-vaiml-pass-summary.txt")
    if os.path.exists(summary):
        print("--- latest partition summary ---")
        print("".join(open(summary).readlines()[:12]))