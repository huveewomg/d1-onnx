"""Export the d1-3B single-pass text graph taking PRE-ASSEMBLED input embeddings.

The host merges token embeddings (host numpy gather from lm.embed_tokens.weight) with
NPU vision output (image rows) — so this graph takes final inputs_embeds and contains
no embedding lookup and no masked_scatter — the cleanest graph for the NPU.

  inputs   inputs_embeds (1, 128, 2048) fp32, attention_mask (1, 128) int64,
           position_ids (1, 128) int64      [left-pad convention, answer slot = last position]
  output   answer_logits (1, 1, 128000) fp32

Run in tools/d1-probe/.venv. Sequential (export_d1_text.py family).
"""

import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "models", "d1-3B"))
OUT = os.path.join(HERE, "export", "d1-3B_embeds_pass_seq512.onnx")
MAX_LEN = 512
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import Lfm2VlForConditionalGeneration  # noqa: E402


def main():
    torch.manual_seed(0)
    m = Lfm2VlForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)

    class EmbedsPass(torch.nn.Module):
        def __init__(self, lm, head):
            super().__init__()
            self.lm, self.head = lm, head

        def forward(self, inputs_embeds, attention_mask, position_ids):
            out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                          position_ids=position_ids, use_cache=False)
            return self.head(out.last_hidden_state[:, -1:, :])

    w = EmbedsPass(m.model.language_model, m.lm_head).eval()
    dummy = (torch.randn(1, MAX_LEN, 2048), torch.ones(1, MAX_LEN, dtype=torch.int64),
             torch.arange(MAX_LEN, dtype=torch.int64).unsqueeze(0))
    t0 = time.time()
    torch.onnx.export(w, dummy, OUT, opset_version=18, dynamo=True,
                      input_names=["inputs_embeds", "attention_mask", "position_ids"],
                      output_names=["answer_logits"])
    print(f"export done in {time.time()-t0:.0f}s -> {OUT}")

    # CPU parity vs torch
    import onnxruntime as ort
    s = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = w(*dummy)[0, 0].numpy()
    o = s.run(None, {"inputs_embeds": dummy[0].numpy(),
                     "attention_mask": dummy[1].numpy(),
                     "position_ids": dummy[2].numpy()})[0][0, 0]
    print("maxdiff vs torch:", float(np.abs(o - ref).max()))


if __name__ == "__main__":
    main()