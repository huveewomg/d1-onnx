"""Pre-quantization transform: make all weight-MatMuls rank-2 with Reshape wrappers.

onnxruntime's dynamic quantization cannot handle rank-3 MatMul(x, W) (LFM2 conv blocks
and FFN run everything as (1, S, D)); MatMulInteger is rank-2 only and the Q/DQ chain
corrupts shape inference downstream. This pass wraps each MatMul with a constant-B:

    Reshape(x, [-1, D]) -> MatMul -> Reshape -> (1, S, O)

using the concrete shapes from value_info. Run on the fp32 graph BEFORE quantize_dynamic.

Usage: python reshape_rank2.py <src.onnx> <dst.onnx>
"""

import os
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto

SRC, DST = sys.argv[1], sys.argv[2]
SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 128  # fixed sequence length of the exported graph

m_shape = onnx.load(SRC, load_external_data=False)  # metadata only: infer_shapes serializes the proto (>2GB rule)
inferred = onnx.shape_inference.infer_shapes(m_shape, strict_mode=False, data_prop=True)
g = inferred.graph
vi = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in g.value_info}

patched = 0
out_graph = onnx.GraphProto()
nodes_out = []
for node in g.node:
    if node.op_type == "MatMul" and len(node.input) == 2:
        a, b = node.input
        a_shape = vi.get(a)
        if a_shape is not None and len(a_shape) == 3 and a_shape[0] in (0, 1):
            out_shape = vi.get(node.output[0])
            if out_shape is None or len(out_shape) != 3:
                nodes_out.append(node)
                continue
            # 0 = symbolic; the graph is fixed-shape so recover the true value from a
            # concrete sibling, else fall back to the graph's fixed sequence length
            S = a_shape[1] or out_shape[1] or SEQ
            D = a_shape[2] or out_shape[2]
            O = out_shape[2] or a_shape[2]
            in_dims = [S, D]
            out_dims = [1, S, O]
            assert 0 not in in_dims and 0 not in out_dims, f"symbolic leak {in_dims} {out_dims}"
            idx = len(nodes_out)
            sh2 = helper.make_tensor(f"d1r2_in_{idx}", TensorProto.INT64, [2], in_dims)
            sh3 = helper.make_tensor(f"d1r3_in_{idx}", TensorProto.INT64, [3], out_dims)
            g.initializer.extend([sh2, sh3])
            r_in = helper.make_node("Reshape", [a, sh2.name], [f"d1r_in_{idx}"],
                                    name=f"d1r_a_{idx}")
            new_mm = helper.make_node("MatMul", [f"d1r_in_{idx}", b], [f"d1r_mm_{idx}"],
                                      name=f"{node.name}_2d")
            r_out = helper.make_node("Reshape", [f"d1r_mm_{idx}", sh3.name], list(node.output),
                                     name=f"d1r_b_{idx}")
            nodes_out.extend([r_in, new_mm, r_out])
            patched += 1
            continue
    nodes_out.append(node)

g.ClearField("node")
g.node.extend(nodes_out)
# patch graph-structure only: external weight refs are untouched, weights stay as-is
onnx.save(inferred, DST)
print(f"patched {patched} MatMuls to rank-2 -> {DST}")