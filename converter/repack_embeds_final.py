"""Dedupe embeds512's fp32 externals onto text_pass's per-tensor weight files.

Byte-matches every external initializer (sha256 of the file-slice bytes), rewrites
references to the text_pass file names, rewrites the duplicated tied head
(val_1810 == embed.T exactly) to Gemm(transB=1) on the shared embed file, and
externalizes the tiny leftovers into a small .data. Parity gate runs separately.
"""

import hashlib
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference
from onnx.external_data_helper import convert_model_to_external_data

E = 'export'
OUTP = os.path.join(E, 'd1-3B_embeds_pass_seq512_repacked.onnx')
LEFTOVER_DATA = 'd1-3B_embeds_pass_seq512_repacked.onnx.data'

# 1) hash text_pass per-tensor files
tf = {}
mt = onnx.load(os.path.join(E, 'd1-3B_text_pass_seq128.onnx'), load_external_data=False)
for t in mt.graph.initializer:
    if t.data_location == TensorProto.EXTERNAL:
        loc = next(kv.value for kv in t.external_data if kv.key == 'location')
        tf[hashlib.sha256(open(os.path.join(E, loc), 'rb').read()).hexdigest()] = loc
print('text files hashed:', len(tf))

# 2) map embeds externals by content
m = onnx.load(os.path.join(E, 'd1-3B_embeds_pass_seq512.onnx'), load_external_data=False)
mapping = {}
for t in m.graph.initializer:
    if t.data_location != TensorProto.EXTERNAL:
        continue
    off = int(next((kv.value for kv in t.external_data if kv.key == 'offset'), 0))
    ln = int(next(kv.value for kv in t.external_data if kv.key == 'length'))
    raw = open(os.path.join(E, next(kv.value for kv in t.external_data if kv.key == 'location')), 'rb').read()[off:off + ln]
    h = hashlib.sha256(raw).hexdigest()
    if h in tf:
        mapping[t.name] = tf[h]
leftovers = [t.name for t in m.graph.initializer
             if t.data_location == TensorProto.EXTERNAL and t.name not in mapping]
print('matched:', len(mapping), '-> leftover:', leftovers)

# 3) rewrite the duplicated tied head as Gemm(transB=1) on the shared embed file
big = next(t for t in m.graph.initializer if (int(np.prod(t.dims)) if t.dims else 0) > 200_000_000)
c = next(n for n in m.graph.node if big.name in n.input)
hidden_in = c.input[0] if c.input[1] == big.name else c.input[1]
out_name = c.output[0]
inf = shape_inference.infer_shapes(m, strict_mode=False)
vi = {v.name: [int(dd.dim_value) for dd in v.type.tensor_type.shape.dim] for v in inf.graph.value_info}
out_shp = [d if d else -1 for d in vi.get(out_name, [1, 1, 128000])]
idx = list(m.graph.node).index(c)
sh_in = helper.make_tensor('r2c_in_shape', TensorProto.INT64, [2], [1, 2048])
sh_out = helper.make_tensor('r3c_out_shape', TensorProto.INT64, [3], out_shp)
n1 = helper.make_node('Reshape', [hidden_in, 'r2c_in_shape'], ['r2c_hidden_2d'], name='repack_reshape_in')
n2 = helper.make_node('Gemm', ['r2c_hidden_2d', 'lm.embed_tokens.weight'], ['r2c_gemm_out'],
                      name='repack_head_gemm', alpha=1.0, beta=1.0, transB=1)
n3 = helper.make_node('Reshape', ['r2c_gemm_out', 'r3c_out_shape'], [out_name], name='repack_reshape_out')
m.graph.node.remove(c)
m.graph.node.insert(idx, n1)
m.graph.node.insert(idx + 1, n2)
m.graph.node.insert(idx + 2, n3)
m.graph.initializer.append(sh_in)
m.graph.initializer.append(sh_out)
m.graph.initializer.remove(big)
print('head rewritten; big duplicate dropped')

# 4) re-point externals, externalize leftovers small
for t in m.graph.initializer:
    if t.data_location == TensorProto.EXTERNAL and t.name in mapping:
        for kv in t.external_data:
            if kv.key == 'location':
                kv.value = mapping[t.name]
            if kv.key == 'offset':
                kv.value = '0'

if leftovers:
    convert_model_to_external_data(m, location=LEFTOVER_DATA)  # re-externalizes ALL inline+touched
    for t in m.graph.initializer:
        if t.name in mapping:
            for kv in t.external_data:
                if kv.key == 'location':
                    kv.value = mapping[t.name]
                if kv.key == 'offset':
                    kv.value = '0'

onnx.save(m, OUTP)
size = os.path.getsize(os.path.join(E, LEFTOVER_DATA)) if leftovers and os.path.exists(os.path.join(E, LEFTOVER_DATA)) else 0
print('saved', OUTP, '| leftover .data bytes:', size)