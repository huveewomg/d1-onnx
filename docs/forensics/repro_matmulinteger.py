"""Ticket-2 repro: a minimal ONNX graph with ONE MatMulInteger — what does VAIML say?

Builds: QuantizeLinear(fp32 X) -> MatMulInteger(x_uint8, W_int8, zps) -> DequantizeLinear -> Y.
Loads it on the NPU EP; prints the newest partition trace's verdict on MatMulInteger.
Result file: minimal_matmulint_result.txt
"""
import glob
import hashlib
import io
import contextlib
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

E = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(E, "probe_matmulinteger.onnx")

rng = np.random.default_rng(0)
K, N = 64, 32
X = rng.standard_normal((1, K)).astype(np.float32)
scale = np.float32(0.02)

W_int8 = rng.integers(-64, 64, size=(K, N)).astype(np.int8)
W_scale = np.array([0.01], dtype=np.float32)

# graph: DequantizeLinear(X-ish) -> MatMulInteger -> (int32 out) -> (int64 cast for display)
t_x = helper.make_tensor("Xq", TensorProto.UINT8, X.shape, X.view(np.uint8) if False else np.clip(X / scale + 128, 0, 255).astype(np.uint8).flatten().tolist())
t_w = helper.make_tensor("Wq", TensorProto.INT8, (K, N), W_int8.flatten().tolist())
t_xs = helper.make_tensor("Xs", TensorProto.FLOAT, [1], [float(scale)])
t_ws = helper.make_tensor("Ws", TensorProto.FLOAT, [1], [float(W_scale[0])])
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
print("built", PROBE)

VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")
res = open(os.path.join(E, "minimal_matmulint_result.txt"), "a", encoding="utf-8")
from onnx import numpy_helper
wq = numpy_helper.to_array(t_w).astype(np.int32).reshape(K, N)
xq = (numpy_helper.to_array(t_x).astype(np.int32) - int(numpy_helper.to_array(t_xzp).reshape(-1)[0])).reshape(1, K)  # zp-subtracted: MatMulInteger semantics
res.write(f"=== {time.strftime('%H:%M:%S')} ===\nexpected Y[0,:4] = {(xq @ wq)[0, :4].tolist()}\n")

buf = io.StringIO()
t0 = time.time()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        s = ort.InferenceSession(PROBE, providers=["VitisAIExecutionProvider"],
                                 provider_options=[{"config_file": VAIP}])
    res.write(f"session created in {time.time()-t0:.1f}s\n")
    import onnxruntime as ort
    o = s.run(None, {t_x.name: np.clip(X / scale + 128, 0, 255).astype(np.uint8)})[0]
    res.write(f"output: {o.shape} {o.dtype} Y[0,:4]={o[0,:4].tolist()}\n")
    # partition trace verdict on MatMulInteger
    caches = sorted(glob.glob(r"C:\temp\RT_*w*\vaip\.cache\*"), key=os.path.getmtime)
    if caches:
        trace = os.path.join(caches[-1], "graph_partition_trace.csv")
        if os.path.exists(trace):
            hits = [l.strip() for l in open(trace, encoding="utf-8", errors="ignore")
                    if "MatMulInteger" in l or "DequantizeLinear" in l]
            res.write("partition trace (MatMulInteger/DQL):\n" + "\n".join("  " + h for h in hits[:6]) + "\n")
        summary = os.path.join(caches[-1], "preliminary-vaiml-pass-summary.txt")
        if os.path.exists(summary):
            res.write(open(summary, encoding="utf-8").read()[:700] + "\n")
except Exception as e:
    res.write(f"FATAL after {time.time()-t0:.1f}s: {str(e)[:300]}\n")
res.close()
print(open(os.path.join(E, "minimal_matmulint_result.txt"), encoding="utf-8").read())