import os
VAIP = os.environ.get("VAIP_CONFIG", r"C:\Program Files\RyzenAI.8.0oe-4.0-win_amd64aip_config.json")
"""Run d1-3B single-pass ONNX on the AMD XDNA2 NPU via VitisAI EP, then verify + time.

Run in conda env ryzen-ai-1.8.0 (voe 1.8.0). Reference npz comes from make_reference.py
(d1-probe venv). Reports: load/compile time, active provider, NPU partition share
(from EP logs), parity vs CPU fp32 reference, and warm latency.

Usage: python npu_run.py [seq_len]
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VAIP_CONFIG = rVAIP
MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 128
REF = os.path.join(HERE, f"reference_seq{MAX_LEN}.npz")
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "export", f"d1-3B_text_pass_seq{MAX_LEN}.onnx")

import onnxruntime as ort  # noqa: E402


def main():
    import io
    import contextlib

    assert os.path.exists(REF), "run make_reference.py in tools/d1-probe/.venv first"
    ref = np.load(REF, allow_pickle=False)
    names = [str(n) for n in ref["names"]]

    print("creating NPU session (compile happens at load; first run can take minutes) ...")
    log = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        sess = ort.InferenceSession(
            MODEL,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": VAIP_CONFIG}],
        )
    load_s = time.time() - t0
    print(f"session created in {load_s:.0f}s; providers={sess.get_providers()}")

    # surface the EP's own partition/compile report (compute share lives there)
    print("--- EP log (partition/compile lines) ---")
    for line in log.getvalue().splitlines():
        low = line.lower()
        if any(k in low for k in ("npu", "cpu", "subgraph", "partition", "compile", "xclbin",
                                  "compute", "fusion", "quant")):
            print("  ", line.strip())
    print("---------------------------------------")

    n = len(names)
    feeds = [{"input_ids": ref["input_ids"][i:i + 1], "attention_mask": ref["attention_mask"][i:i + 1],
              "position_ids": ref["position_ids"][i:i + 1]} for i in range(n)]
    outs = []
    for f in feeds:
        outs.append(sess.run(None, f)[0][0, 0])  # (128000,) answer-slot logits
    np.savez(os.path.join(HERE, f"npu_{os.path.basename(os.path.splitext(os.path.basename(MODEL))[0])}_out.npz"),
             logits=np.concatenate(outs), names=np.array(names))
    print("dumped NPU logits to", f"npu_{os.path.basename(os.path.splitext(os.path.basename(MODEL))[0])}_out.npz")

    all_ok = True
    for i, name in enumerate(names):
        o = outs[i]
        r = ref["logits"][i]
        d = float(np.abs(o.astype(np.float64) - r).max())
        eq = bool(o.argmax(-1) == r.argmax(-1))
        all_ok &= eq
        print(f"[{name:6s}] maxdiff={d:.3e} argmax_eq={eq}")
    print("PARITY_ALL_OK" if all_ok else "PARITY_FAILED")

    print("timing warm runs (case 0) ...")
    for _ in range(3):
        sess.run(None, feeds[0])
    t0 = time.perf_counter()
    it = 20
    for _ in range(it):
        sess.run(None, feeds[0])
    print(f"NPU (VitisAI) single-pass latency seq={MAX_LEN}: {(time.perf_counter()-t0)/it*1000:.1f} ms")


if __name__ == "__main__":
    main()