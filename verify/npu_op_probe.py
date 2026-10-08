import os
VAIP = os.environ.get("VAIP_CONFIG", r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")
import sys, time, io, contextlib
import onnxruntime as ort
import numpy as np
model = sys.argv[1]
res = open('op_probe_result.txt','w',buffering=1)
try:
    t0=time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        s = ort.InferenceSession(model, providers=['VitisAIExecutionProvider'], provider_options=[{'config_file': VAIP}])
    x = np.random.default_rng(0).standard_normal((1,1024,768)).astype(np.float32)
    o = s.run(None, {'X': x})[0]
    res.write(f'OK {time.time()-t0:.0f}s out={o.shape} {o.dtype}\n')
except Exception as e:
    res.write(f'EXC {str(e)[-200:]}\n')
