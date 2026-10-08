import os
VAIP = os.environ.get("VAIP_CONFIG", r"C:\Program Files\RyzenAI.8.0oe-4.0-win_amd64aip_config.json")
import onnxruntime as ort, numpy as np, time
log = open('npu_vision3.log', 'w', buffering=1)
def P(*a): print(*a); print(*a, file=log)
rng = np.random.default_rng(0)
pv = rng.standard_normal((1,1024,768)).astype(np.float32)
feeds = {'pixel_values': pv}
t0=time.time()
s = ort.InferenceSession('export/d1-3B_vision_tile512.onnx', providers=['VitisAIExecutionProvider'], provider_options=[{'config_file': rVAIP}])
P('NPU session created:', round(time.time()-t0), 's')
o = s.run(None, feeds)[0]
P('run ok:', o.shape, o.dtype)
ref = ort.InferenceSession('export/d1-3B_vision_tile512.onnx', providers=['CPUExecutionProvider']).run(None, feeds)[0]
P('maxdiff NPU vs CPU:', f"{float(np.abs(o.astype(np.float64)-ref.astype(np.float64)).max()):.3e}")
for _ in range(3): s.run(None, feeds)
ts=[]
for _ in range(20):
    a=time.perf_counter(); s.run(None, feeds); ts.append((time.perf_counter()-a)*1000)
P(f'warm p50 {np.percentile(ts,50):.0f}ms mean {np.mean(ts):.0f}ms p95 {np.percentile(ts,95):.0f}ms')
np.savez('npu_vision_out.npz', embeds=o)
P('saved npu_vision_out.npz')
