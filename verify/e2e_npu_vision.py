import os
VAIP = os.environ.get("VAIP_CONFIG", r"C:\Program Files\RyzenAI.8.0oe-4.0-win_amd64aip_config.json")
"""E2E step B (ryzen env): d1 cats image tiles through the NPU vision graph."""
import numpy as np, onnxruntime as ort, time
d = np.load('e2e_inputs.npz')
pv, pam, ss = d['pixel_values'], d['pixel_attention_mask'], d['spatial_shapes']
K = pv.shape[0]
sess = ort.InferenceSession('export/d1-3B_vision_tile512.onnx',
    providers=['VitisAIExecutionProvider'],
    provider_options=[{'config_file': rVAIP}])
t0 = time.time()
embeds = []
for k in range(K):
    n = int(pam[k].sum())  # valid patches (full tile = 1024)
    o = sess.run(None, {'pixel_values': pv[k:k+1].astype(np.float32)})[0]
    print(f'tile {k}: patches={n} -> embeds {o.shape}')
    embeds.append(o)
out = np.concatenate(embeds, axis=0)  # (K*256, 2048) row-major
np.savez('e2e_img_embeds_npu.npz', embeds=out)
print(f'NPU vision total (compile excluded): {time.time()-t0:.1f}s; {out.shape}')
