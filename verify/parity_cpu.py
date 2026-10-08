"""Real-weights parity + latency: d1-3B ONNX single-pass graph vs native transformers.

Compares (a) last-position logits, (b) the actual d1 decision readout (softmax over
option tokens, using d1's own prompt.py), and (c) measures CPU EP latency.
Left-pad convention: the answer slot is the final position of the fixed-length window.
"""

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_DIR = os.path.join(ROOT, "models", "d1-3B")
ONNX_DIR = os.path.join(HERE, "export")
sys.path.insert(0, ROOT)  # repo-level d1 prompt modules live in the model dir
sys.path.insert(0, MODEL_DIR)

from transformers import AutoTokenizer, Lfm2VlForConditionalGeneration  # noqa: E402
import onnxruntime as ort  # noqa: E402
import prompt as d1_prompt  # noqa: E402  (models/d1-3B/prompt.py)

MAX_LEN = 128

PROMPTS = {
    "noul": ("I was charged twice this month, please refund one of them.",
             {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}),
    "choice": ("My app crashes whenever I open the settings page.",
               {"team": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"billing": "Charges, refunds, invoices",
                                      "technical": "App or site faults",
                                      "fraud": "Suspected unauthorised use"}}}),
    "long": ("Customer ticket: " + "The parcel was due Monday and still has not arrived. " * 8,
             {"late": {"type": "score", "instructions": "How urgent is this?",
                       "criteria": ["Can wait", "Today", "Blocking the customer now"]}}),
}

STATE_MODEL = "LiquidAI/LFM2.5-VL-3B"


def build_inputs(tok, state, questions):
    """Left-padded fixed-length inputs for one question over one state."""
    q = list(questions.values())[0]
    text = d1_prompt.render(tok, state, d1_prompt.as_question(q), lead="")
    ids = tok.encode(text, add_special_tokens=False)[-MAX_LEN:]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids_padded = [pad] * (MAX_LEN - len(ids)) + ids
    mask = [0] * (MAX_LEN - len(ids)) + [1] * len(ids)
    x = np.array([ids_padded], dtype=np.int64)
    m = np.array([mask], dtype=np.int64)
    pos = np.clip(np.cumsum(m, axis=1) - 1, 0, None).astype(np.int64)
    return x, m, pos, tok


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    onnx_path = os.path.join(ONNX_DIR, f"d1-3B_text_pass_seq{MAX_LEN}.onnx")

    print("loading torch reference ...")
    t0 = time.time()
    torch_m = Lfm2VlForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32).eval()
    print(f"  torch loaded in {time.time()-t0:.0f}s")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    all_ok = True
    for name, (state, questions) in PROMPTS.items():
        x, m, pos, tok = build_inputs(tok, state, questions)
        with torch.no_grad():
            ref_hidden = torch_m.model.language_model(
                input_ids=torch.from_numpy(x),
                attention_mask=torch.from_numpy(m),
                position_ids=torch.from_numpy(pos),
                use_cache=False).last_hidden_state
            ref_logits = torch_m.lm_head(ref_hidden[:, -1, :])[0].numpy()

        o = sess.run(None, {"input_ids": x, "attention_mask": m, "position_ids": pos})[0][0, 0]
        diff = float(np.abs(ref_logits - o).max())
        argmax_eq = bool(ref_logits.argmax(-1) == o.argmax(-1))

        # d1 decision readout over the option tokens (temperature: none for d1-3B)
        q = d1_prompt.as_question(list(questions.values())[0])
        pz_model = o - o.max()
        pz_torch = ref_logits - ref_logits.max()
        pr_onnx = d1_prompt.readout(tok, q, pz_model)
        pr_torch = d1_prompt.readout(tok, q, pz_torch)
        prob_diff = float(np.abs(np.array(pr_onnx) - np.array(pr_torch)).max())

        ok = diff < 1e-2 and prob_diff < 1e-3
        all_ok &= ok
        print(f"[{name:6s}] logits maxdiff={diff:.3e} argmax_eq={argmax_eq} "
              f"readout_probs(onnx)={[f'{p:.3f}' for p in pr_onnx]} "
              f"probs_diff={prob_diff:.2e} {'OK' if ok else 'FAIL'}")

    # latency (warm, question-sized input, full fixed window)
    state, questions = PROMPTS["noul"]
    x, m, pos, tok = build_inputs(tok, state, questions)
    feed = {"input_ids": x, "attention_mask": m, "position_ids": pos}
    for _ in range(3):
        sess.run(None, feed)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        sess.run(None, feed)
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"CPU EP single-pass latency (seq={MAX_LEN}, fp32): {dt:.1f} ms median-free mean of {n}")

    print("ALL_OK" if all_ok else "SOME_FAILED")


if __name__ == "__main__":
    main()