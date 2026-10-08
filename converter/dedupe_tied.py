"""Dedupe the duplicated tied lm_head initializer in the exported d1-3B ONNX graph.

torch.onnx.export emits the tied embedding twice: `lm.embed_tokens.weight`
(128000,2048) for the Gather and `onnx::MatMul_NNNN` (2048,128000) — the same bytes
transposed — for the head MatMul. Replace that MatMul with Gemm(transB=1) reading the
canonical embed initializer, then delete the duplicate + its external file.
"""

import os
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto

MODEL_PATH = sys.argv[1]
DUP_NAME = sys.argv[2] if len(sys.argv) > 2 else "onnx::MatMul_4972"
CANON_NAME = "lm.embed_tokens.weight"


def external_file(t: onnx.TensorProto):
    for kv in t.external_data:
        if kv.key == "location":
            return os.path.join(os.path.dirname(MODEL_PATH), kv.value)
    return None


g = onnx.load(MODEL_PATH, load_external_data=False)
inits = {t.name: t for t in g.graph.initializer}
assert CANON_NAME in inits and DUP_NAME in inits, "expected initializers missing"

# safety: duplicated data is the transpose of the canonical tensor
c = np.memmap(external_file(inits[CANON_NAME]), dtype="<f4", mode="r").reshape(
    [int(d) for d in inits[CANON_NAME].dims]
)
d = np.memmap(external_file(inits[DUP_NAME]), dtype="<f4", mode="r").reshape(
    [int(d) for d in inits[DUP_NAME].dims]
)
err = float(np.abs(c - d.T).max())
ref_ok = err == 0.0
del c, d, inits  # release memmap handles before any file removal (Windows file locking)
print(f"transpose-equivalence max abs err: {err}")
assert ref_ok, "not a transposed duplicate — refusing"

# rewrite the head node: MatMul(x, dup^T) -> Gemm(x, canon, alpha=1, transB=1)
hits = [n for n in g.graph.node if DUP_NAME in n.input]
assert len(hits) == 1, f"expected exactly one consumer, got {len(hits)}"
node = hits[0]
hidden_in = node.input[0] if node.input[1] == DUP_NAME else node.input[1]
new_node = helper.make_node(
    "Gemm", [hidden_in, CANON_NAME], list(node.output), name=node.name,
    alpha=1.0, beta=1.0, transB=1,
)
idx = list(g.graph.node).index(node)
g.graph.node.remove(node)
g.graph.node.insert(idx, new_node)
print(f"replaced {node.op_type}({node.name}) with Gemm(transB=1)")

# drop the duplicate initializer + its external file
g.graph.initializer.remove(inits[DUP_NAME])
onnx.save(g, MODEL_PATH)
dup_file = external_file(inits[DUP_NAME])
os.remove(dup_file)
print("removed duplicate external file:", dup_file)
total = sum(int(np.prod(t.dims)) if t.dims else 0 for t in g.graph.initializer)
print(f"initializer elements now: {total/1e6:.1f}M")