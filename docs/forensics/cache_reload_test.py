"""Ticket-1 repro: does the VAIML cache RELOAD in a fresh process? Run twice.

1st run: compiles + writes cache (timed); 2nd run: loads from cache (timed) -
a healthy reload is fast; a fatal error or silent recompile = the bug.
Feeds are derived from each graph's DECLARED inputs (name/shape/dtype), so the
same script demos first-inference on any graph: vision-shaped or text-shaped.
Usage: <ryzen python> cache_reload_test.py run1 <graph.onnx>
"""
import glob
import io
import sys
import tempfile
import time
import contextlib
import os

import numpy as np
import onnxruntime as ort

VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")

DTYPE = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
         "tensor(int64)": np.int64, "tensor(int32)": np.int32,
         "tensor(int8)": np.int8, "tensor(uint8)": np.uint8,
         "tensor(bool)": np.bool_}

def _feed_for(inp):
    """Valid feed matching a declared graph input: name/shape/dtype.

    attention_mask = all-ones, position_ids = sequential; dtype is taken from
    the declaration. Matching shapes and dtypes alone does not ensure valid
    model inputs, so semantically typed inputs get real semantics.
    """
    dims = [d if isinstance(d, int) else 1 for d in inp.shape]
    if inp.type not in DTYPE:
        raise RuntimeError(f"unsupported input type {inp.type!r} for {inp.name}")
    dt = DTYPE[inp.type]
    n = int(np.prod(dims)) if dims else 1
    if inp.name == "attention_mask":
        return inp.name, np.ones(dims, dtype=dt)
    if inp.name == "position_ids":
        return inp.name, np.arange(n, dtype=dt).reshape(dims)
    if np.issubdtype(dt, np.floating):
        return inp.name, np.random.default_rng(0).standard_normal(dims).astype(dt)
    # token ids (and other integer inputs): small valid ids
    return inp.name, np.random.default_rng(0).integers(1, 127, dims).astype(dt)

run_id, model = sys.argv[1:3] if len(sys.argv) >= 3 else ("run?", "probe_LayerNormalization.onnx")
model = os.path.abspath(model)
HERE = os.path.dirname(os.path.abspath(__file__))
res = open(os.path.join(HERE, "cache_reload_result.txt"), "a", encoding="utf-8")
res.write(f"=== {run_id} {time.strftime('%Y-%m-%d %H:%M:%S')} model={os.path.basename(model)} ===\n")

# VAIP drops runtime side-effect files (e.g. original-model-signature.txt,
# md5 of the last-loaded model) into the process cwd. Sandbox the cwd so a
# test run can never clobber recorded evidence files.
sandbox = os.path.join(tempfile.gettempdir(), "RT_vaip_sidefx_cwd")
os.makedirs(sandbox, exist_ok=True)
cwd0 = os.getcwd()
try:
    os.chdir(sandbox)

    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            s = ort.InferenceSession(model, providers=["VitisAIExecutionProvider"],
                                     provider_options=[{"config_file": VAIP}])
            declared = [(i.name, i.shape, i.type) for i in s.get_inputs()]
            res.write(f"declared inputs: {declared}\n")
            feeds = dict(_feed_for(i) for i in s.get_inputs())
            o = s.run(None, feeds)[0]
            res.write(f"session OK in {time.time()-t0:.1f}s, out {o.shape} {o.dtype}\n")
        except Exception as e:
            res.write(f"FATAL after {time.time()-t0:.1f}s: {str(e)[:400]}\n")

    caches = sorted(glob.glob(r"C:\temp\RT_*w*\vaip\.cache\*"), key=os.path.getmtime)
    if caches:
        newest = caches[-1]
        rai = [f for f in os.listdir(newest) if f.endswith(".rai")]
        res.write(f"newest cache: {os.path.basename(newest)} rai files: {rai}\n")
finally:
    os.chdir(cwd0)
    res.close()
print(open(os.path.join(HERE, "cache_reload_result.txt"), encoding="utf-8").read()[-400:])