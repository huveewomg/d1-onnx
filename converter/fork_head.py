"""Fork the tied head initializer so the quantizer never shares the embed weight.

Generalized from the release-session inline steps (transcript L313/L315, 2026-10-08):
the head MatMul/Gemm must read its own FP32 copy of the embedding table while the
body keeps the shared `lm.embed_tokens.weight`. Quantize_dynamic is later run with
head excluded, so the readout stays fp32-exact.

Usage: python fork_head.py <graph.onnx> <head_node_name> [<head_weight_file>] [<embed_initializer_name>]
- graph modified IN PLACE (same convention as the release chain used)
- head_node_name: node whose input should point at the fork (text graph: /head/MatMul;
  embeds graph: repack_head_gemm reading the repack-shared initializer)
- head_weight_file: optional; default = copy <dir>/lm.embed_tokens.weight to <dir>/lm.head.weight
- embed_initializer_name: optional; default lm.embed_tokens.weight (embeds graph: repack_shared_embed)
"""
import os
import shutil
import sys

import onnx
from onnx import TensorProto

DEFAULT_EMBED = "lm.embed_tokens.weight"
HEAD_NAME = "lm.head.weight"


def main():
    graph = os.path.abspath(sys.argv[1])
    head_node = sys.argv[2]
    head_w = sys.argv[3] if len(sys.argv) > 3 else None
    embed_name = sys.argv[4] if len(sys.argv) > 4 else HEAD_NAME.replace("lm.head", "lm.embed_tokens")

    gdir = os.path.dirname(graph)
    embed_path = os.path.join(gdir, DEFAULT_EMBED)
    head_path = head_w or os.path.join(gdir, HEAD_NAME)
    if not os.path.exists(head_path):
        shutil.copyfile(embed_path, head_path)
        print(f"copied {embed_path} -> {head_path}")

    m = onnx.load(graph, load_external_data=False)
    # fork initializer: head gets its own fp32 weight file, embed stays Gather-only
    t = onnx.TensorProto()
    t.name = HEAD_NAME
    t.data_type = TensorProto.FLOAT
    t.dims.extend([128000, 2048])
    t.data_location = TensorProto.EXTERNAL
    for k, v in [("location", os.path.basename(head_path))]:
        kv = t.external_data.add()
        kv.key = k
        kv.value = v
    existing = {i.name for i in m.graph.initializer}
    if HEAD_NAME in existing:
        print(f"{HEAD_NAME} already in graph — no fork needed")
        return
    m.graph.initializer.append(t)
    fixed = 0
    for n in m.graph.node:
        if n.name == head_node:
            for i, nm in enumerate(n.input):
                if nm == embed_name:
                    n.input[i] = HEAD_NAME
                    fixed += 1
    print(f"head node {head_node!r}: {fixed} input(s) forked ({embed_name!r} -> {HEAD_NAME})")
    if fixed == 0:
        raise SystemExit(f"node {head_node!r} not found or does not read {embed_name!r}")
    onnx.save(m, graph)


if __name__ == "__main__":
    main()