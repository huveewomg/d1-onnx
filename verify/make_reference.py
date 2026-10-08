"""Compute CPU reference for the three d1 test cases and save inputs+outputs to npz.

Run in tools/d1-probe/.venv (transformers 5.19): the ryzen-ai env has an old
transformers and cannot build the torch reference.
Output: reference_seq{N}.npz with input_ids/attention_mask/position_ids (3 cases)
and expected answer_logits (1,128000) per case + the per-case prompt name list.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_DIR = os.path.join(ROOT, "models", "d1-3B")
sys.path.insert(0, MODEL_DIR)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer, Lfm2VlForConditionalGeneration  # noqa: E402
import prompt as d1_prompt  # noqa: E402
from parity_cpu import PROMPTS, MAX_LEN  # noqa: E402


def main():
    out_path = os.path.join(HERE, f"reference_seq{MAX_LEN}.npz")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    torch_m = Lfm2VlForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32).eval()

    xs, ms, ps, refs, names = [], [], [], [], []
    for name, (state, questions) in PROMPTS.items():
        x, m, pos, tok = build_one(tok, state, questions)
        with torch.no_grad():
            hidden = torch_m.model.language_model(
                input_ids=torch.from_numpy(x), attention_mask=torch.from_numpy(m),
                position_ids=torch.from_numpy(pos), use_cache=False).last_hidden_state
            logits = torch_m.lm_head(hidden[:, -1, :])[0].numpy()
        xs.append(x); ms.append(m); ps.append(pos); refs.append(logits); names.append(name)
        print(f"case {name}: ok")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    np.savez(out_path, input_ids=np.concatenate(xs), attention_mask=np.concatenate(ms),
             position_ids=np.concatenate(ps), logits=np.stack(refs),
             names=np.array(names), pad_token=np.array([pad]))
    print("saved", out_path)


def build_one(tok, state, questions):
    import numpy as np
    q = list(questions.values())[0]
    text = d1_prompt.render(tok, state, d1_prompt.as_question(q), lead="")
    ids = tok.encode(text, add_special_tokens=False)[-MAX_LEN:]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids_p = [pad] * (MAX_LEN - len(ids)) + ids
    mask = [0] * (MAX_LEN - len(ids)) + [1] * len(ids)
    x = np.array([ids_p], dtype=np.int64)
    m = np.array([mask], dtype=np.int64)
    pos = np.clip(np.cumsum(m, axis=1) - 1, 0, None).astype(np.int64)
    return x, m, pos, tok


if __name__ == "__main__":
    main()