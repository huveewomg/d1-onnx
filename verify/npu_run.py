"""Run d1-3B single-pass ONNX on the AMD XDNA2 NPU via VitisAI EP, then verify + time.

Run in the ryzen-ai-1.8.0 environment (voe 1.8.0). Reference tensors come from
make_reference.py (needs transformers 5.x; run it in the converter venv).

Usage: <ryzen python> npu_run.py <model.onnx> [seq_len]
Reports: load/compile time, provider, parity vs the fp32 npz reference, warm latency.
For NPU graphs the compile happens at session creation (one-time; cached by the SDK).
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

VAIP = os.environ.get("VAIP_CONFIG",
                      r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")

MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 128
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference", f"reference_seq{MAX_LEN}.npz")
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "onnx",
    f"d1-3B_text_pass_seq{MAX_LEN}.onnx")


def main():
    assert os.path.exists(REF), "run make_reference.py (in the converter venv) first"
    ref = np.load(REF, allow_pickle=False)
    names = [str(n) for n in ref["names"]]

    print("creating NPU session (compile happens at load; first run can take minutes) ...")
    t0 = time.time()
    sess = ort.InferenceSession(
        os.path.abspath(MODEL),
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": VAIP}],
    )
    load_s = time.time() - t0
    print(f"session created in {load_s:.0f}s; providers={sess.get_providers()}")

    n = len(names)
    feeds = [{"input_ids": ref["input_ids"][i:i + 1],
              "attention_mask": ref["attention_mask"][i:i + 1],
              "position_ids": ref["position_ids"][i:i + 1]} for i in range(n)]
    outs = []
    for f in feeds:
        outs.append(sess.run(None, f)[0])
    np.savez(os.path.join(HERE, f"npu_{os.path.basename(os.path.splitext(os.path.basename(MODEL))[0])}_out.npz"),
             logits=np.concatenate(outs), names=np.array(names))
    print("dumped NPU logits:", f"npu_{os.path.basename(os.path.splitext(os.path.basename(MODEL))[0])}_out.npz")

    all_ok = True
    for i, name in enumerate(names):
        o = outs[i][0, 0] if outs[i].ndim == 3 else outs[i][0]
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
    ts = [(time.perf_counter() - t0) / it * 1000]
    print(f"NPU (VitisAI) single-pass latency seq={MAX_LEN}: {ts[0]:.1f} ms")


if __name__ == "__main__":
    main()