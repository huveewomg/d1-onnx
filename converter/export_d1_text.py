"""Export LiquidAI d1-3B text backbone as a fixed-length single-pass ONNX graph.

Graph contract:
  inputs  input_ids (1, MAX_LEN) int64, attention_mask (1, MAX_LEN) int64,
          position_ids (1, MAX_LEN) int64    [LEFT-pad convention: answer slot = last position]
  output  answer_logits (1, 1, vocab) fp32   [logits at the final position = the answer slot]

d1 semantics (prompt.py::readout): answer = softmax over option-code token ids at the
answer slot, so only the last position's logits leave the graph.
"""

import json
import os
import sys
import time

import numpy as np
import torch

MAX_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 128
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "d1-3B")
OUT_DIR = os.path.join(os.path.dirname(__file__), "export")

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoConfig, AutoTokenizer, Lfm2VlForConditionalGeneration  # noqa: E402


def build_model() -> Lfm2VlForConditionalGeneration:
    m = Lfm2VlForConditionalGeneration.from_pretrained(
        os.path.abspath(MODEL_DIR), torch_dtype=torch.float32
    ).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class SinglePassReadout(torch.nn.Module):
    """Text backbone + lm_head restricted to the LAST position (the answer slot)."""

    def __init__(self, lm, head):
        super().__init__()
        self.lm = lm
        self.head = head

    def forward(self, input_ids, attention_mask, position_ids):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        last = out[:, -1:, :]  # (1, 1, d)
        return self.head(last)  # (1, 1, vocab)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    m = build_model()
    lm, head = m.model.language_model, m.lm_head
    print(f"model loaded; params={sum(p.numel() for p in m.parameters()):,}")

    wrapper = SinglePassReadout(lm, head).eval()
    dummy = {
        "input_ids": torch.ones((1, MAX_LEN), dtype=torch.int64),
        "attention_mask": torch.ones((1, MAX_LEN), dtype=torch.int64),
        "position_ids": torch.arange(MAX_LEN, dtype=torch.int64).unsqueeze(0),
    }
    t0 = time.time()
    out_path = os.path.join(OUT_DIR, f"d1-3B_text_pass_seq{MAX_LEN}.onnx")
    torch.onnx.export(
        wrapper,
        tuple(dummy.values()),
        out_path,
        opset_version=17,
        dynamo=False,
        input_names=list(dummy.keys()),
        output_names=["answer_logits"],
        do_constant_folding=True,
        dynamic_axes=None,  # fixed shape — NPU compilation wants this
    )
    print(f"export done in {time.time() - t0:.1f}s -> {out_path}")

    # initializer dedupe check (tied head) + Gemm(transB=1) rewrite
    # torch.onnx.export duplicates the tied head as a transposed (d,) initializer;
    # share the embed initializer instead: Reshape -> Gemm(x, embed, transB=1) -> Reshape
    import onnx
    from onnx import helper, TensorProto

    g = onnx.load(out_path, load_external_data=False)
    inits = {t.name: t for t in g.graph.initializer}
    canon_name, canon = "lm.embed_tokens.weight", inits["lm.embed_tokens.weight"]
    c_dims = [int(d) for d in canon.dims]
    vocab = c_dims[0]

    dup_hits = [n for n in g.graph.node if n.op_type == "MatMul" and any(
        i in inits and int(np.prod(inits[i].dims)) == vocab * c_dims[1]
        for i in n.input)]
    assert len(dup_hits) >= 1, "tied-head MatMul not found"
    dup_names = {i for n in dup_hits for i in n.input
                 if i in inits and int(np.prod(inits[i].dims)) == vocab * c_dims[1]}
    for node in dup_hits:
        hidden_in = node.input[0] if node.input[1] in dup_names else node.input[1]
        out_name = node.output[0]
        idx = list(g.graph.node).index(node)
        sh2 = helper.make_tensor("d1_sh2", TensorProto.INT64, [2], [1, c_dims[1]])
        sh3 = helper.make_tensor("d1_sh3", TensorProto.INT64, [3], [1, 1, vocab])
        r1 = helper.make_node("Reshape", [hidden_in, "d1_sh2"], ["d1_hidden_2d"])
        gemm = helper.make_node("Gemm", ["d1_hidden_2d", canon_name], ["d1_gemm_out"],
                                name=node.name, alpha=1.0, beta=1.0, transB=1)
        r2 = helper.make_node("Reshape", ["d1_gemm_out", "d1_sh3"], [out_name])
        g.graph.node.remove(node)
        g.graph.node.insert(idx, r1)
        g.graph.node.insert(idx + 1, gemm)
        g.graph.node.insert(idx + 2, r2)
    if "d1_sh2" not in inits:
        g.graph.initializer.extend([sh2, sh3])
    for dn in dup_names - {canon_name}:
        for t in list(g.graph.initializer):
            if t.name == dn:
                g.graph.initializer.remove(t)
        ext = None
        for kv in inits[dn].external_data:
            if kv.key == "location":
                ext = os.path.join(os.path.dirname(out_path), kv.value)
        if ext and os.path.exists(ext):
            os.remove(ext)
    onnx.save(g, out_path)
    print(f"tied-head deduped: MatMul->Gemm(transB=1); removed {sorted(dup_names - {canon_name}) or 'nothing'}")
    total = sum(int(np.prod(t.dims)) if t.dims else 0 for t in g.graph.initializer)
    print(f"initializer elements total: {total/1e6:.0f}M")


if __name__ == "__main__":
    main()