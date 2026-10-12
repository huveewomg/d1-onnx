"""Ticket-2 repro: a minimal ONNX graph with ONE MatMulInteger: what does VAIML say?

Builds: MatMulInteger(x_uint8, W_int8, zps) -> Y (int32), loads it on the NPU EP,
runs the real op, and asserts FULL output equality against the spec semantics
(x_uint8 - x_zp) @ (w_int8 - w_zp): this catches the stale-expected-value class
of bug that the old append-mode log hid.
Result file: minimal_matmulint_result.txt (one dated block per run; delete the
file for a fully clean log).
"""
import glob
import os
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

E = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(E, "probe_matmulinteger.onnx")
RESULT = os.path.join(E, "minimal_matmulint_result.txt")

rng = np.random.default_rng(0)
K, N = 64, 32
X = rng.standard_normal((1, K)).astype(np.float32)
scale = np.float32(0.02)

W_int8 = rng.integers(-64, 64, size=(K, N)).astype(np.int8)

# graph: MatMulInteger(Xq, Wq, Xzp=128, Wzp=0) -> Y int32
Xqp = np.clip(X / scale + 128, 0, 255).astype(np.uint8)
t_x = helper.make_tensor("Xq", TensorProto.UINT8, X.shape, Xqp.flatten().tolist())
t_w = helper.make_tensor("Wq", TensorProto.INT8, (K, N), W_int8.flatten().tolist())
t_xs = helper.make_tensor("Xs", TensorProto.FLOAT, [1], [float(scale)])
t_ws = helper.make_tensor("Ws", TensorProto.FLOAT, [1], [0.01])
x_zp = helper.make_tensor("Xzp", TensorProto.UINT8, [1], [128])
w_zp = helper.make_tensor("Wzp", TensorProto.INT8, [1], [0])

n2 = helper.make_node("MatMulInteger", ["Xq", "Wq", "Xzp", "Wzp"], ["Y"], name="mmi")
g_in = [helper.make_tensor_value_info("Xq", TensorProto.UINT8, X.shape)]
g_out = [helper.make_tensor_value_info("Y", TensorProto.INT32, [1, N])]
graph = helper.make_graph([n2], "mmi_probe", g_in, g_out,
                          [t_x, t_w, t_xs, t_ws, x_zp, w_zp])
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, PROBE)

# Expected per MatMulInteger spec: (x - x_zp) @ (w - w_zp), full output.
xq = Xqp.astype(np.int32) - int(numpy_helper.to_array(x_zp).reshape(-1)[0])
wq = numpy_helper.to_array(t_w).astype(np.int32).reshape(K, N)
expected = (xq @ wq).reshape(1, N)

def run_probe():
    """Load the probe graph on the NPU EP, run it, assert full equality.
    Returns the log block as one text blob."""
    lines = [f"=== repro_matmulinteger {time.strftime('%Y-%m-%d %H:%M:%S')} ==="]
    lines.append(f"expected (full, spec semantics) shape={expected.shape} "
                 f"Y[0,:4]={expected[0, :4].tolist()}")
    try:
        t0 = time.time()
        s = ort.InferenceSession(PROBE, providers=["VitisAIExecutionProvider"],
                                 provider_options=[{"config_file": VAIP}])
        lines.append(f"session created in {time.time() - t0:.1f}s")

        o = s.run(None, {t_x.name: Xqp})[0]
        # Full-output equality: the assert the old script never made.
        if o.shape != expected.shape or o.dtype != np.int32:
            lines.append(f"FAIL: output {o.shape} {o.dtype} != expected "
                         f"{expected.shape} int32")
        elif np.array_equal(o.astype(np.int64), expected.astype(np.int64)):
            lines.append("PASS: full output equals spec expected exactly")
        else:
            diff = np.argwhere(o != expected)
            lines.append(f"FAIL: {diff.size} mismatches; "
                         f"got Y[0,:4]={o[0, :4].tolist()} "
                         f"expected Y[0,:4]={expected[0, :4].tolist()}")

        # partition trace verdict: pick THIS probe's cache dir (mtime order
        # alone grabs a neighbor dir; match the summary's Model line)
        caches = sorted(glob.glob(r"C:\temp\RT_*w*\vaip\.cache\*"),
                        key=os.path.getmtime)
        cache_dir = next((c for c in caches
                          if os.path.exists(os.path.join(c, "preliminary-vaiml-pass-summary.txt"))
                          and "probe_matmulinteger" in open(
                              os.path.join(c, "preliminary-vaiml-pass-summary.txt"),
                              encoding="utf-8", errors="ignore").read()),
                         None)
        if cache_dir:
            trace = os.path.join(cache_dir, "graph_partition_trace.csv")
            if os.path.exists(trace):
                hits = [l.strip() for l in open(trace, encoding="utf-8", errors="ignore")
                        if "MatMulInteger" in l or "DequantizeLinear" in l]
                lines.append("partition trace (MatMulInteger/DQL):\n"
                             + "\n".join("  " + h for h in hits[:6]))
            summary = os.path.join(cache_dir, "preliminary-vaiml-pass-summary.txt")
            if os.path.exists(summary):
                lines.append(open(summary, encoding="utf-8").read()[:700])
    except Exception as e:
        lines.append(f"FATAL after {time.time() - t0:.1f}s: {str(e)[:300]}")
    return "\n".join(lines) + "\n"


VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")
# VAIP drops runtime side-effect files (original-model-signature.txt, md5 of
# the last-loaded model) into the process cwd: sandbox it (see LEARNED §3:
# process-level side effects eat evidence) and restore from a finally block.
sandbox = os.path.join(tempfile.gettempdir(), "RT_vaip_sidefx_cwd")
os.makedirs(sandbox, exist_ok=True)
cwd0 = os.getcwd()
try:
    os.chdir(sandbox)
    result_text = run_probe()
finally:
    os.chdir(cwd0)

with open(RESULT, "a", encoding="utf-8") as res:
    res.write(result_text)
print(result_text)