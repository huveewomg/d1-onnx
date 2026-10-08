"""Ticket-1 repro: does the VAIML cache RELOAD in a fresh process? Run twice.

1st run: compiles + writes cache (timed); 2nd run: loads from cache (timed) -
a healthy reload is fast; a fatal error or silent recompile = the bug.
Usage: <ryzen python> cache_reload_test.py run1 <graph.onnx>
"""
import glob
import io
import time

import contextlib
import os

import onnxruntime as ort

VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")
run_id = sys_arg = None
try:
    run_id, model = os.sys.argv[1], os.sys.argv[2]
except Exception:
    run_id, model = "run?", "probe_LayerNormalization.onnx"

model = os.path.abspath(model)
HERE = os.path.dirname(os.path.abspath(__file__))
res = open(os.path.join(HERE, "cache_reload_result.txt"), "a", encoding="utf-8")
res.write(f"=== {run_id} {time.strftime('%H:%M:%S')} model={os.path.basename(model)} ===\n")

buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        s = ort.InferenceSession(model, providers=["VitisAIExecutionProvider"],
                                 provider_options=[{"config_file": VAIP}])
        np = __import__("numpy")
        o = s.run(None, {"X": np.random.default_rng(0).standard_normal((1, 1024, 768)).astype(np.float32)})[0]
        res.write(f"session OK in {time.time()-t0:.1f}s, out {o.shape} {o.dtype}\n")
    except Exception as e:
        res.write(f"FATAL after {time.time()-t0:.1f}s: {str(e)[:400]}\n")

caches = sorted(glob.glob(r"C:\temp\RT_*w*\vaip\.cache\*"), key=os.path.getmtime)
if caches:
    newest = caches[-1]
    rai = [f for f in os.listdir(newest) if f.endswith(".rai")]
    res.write(f"newest cache: {os.path.basename(newest)} rai files: {rai}\n")
res.close()
print(open(os.path.join(HERE, "cache_reload_result.txt"), encoding="utf-8").read()[-400:])