"""Export d1-3B's vision stack (SigLIP2 NaFlex + connector) as a fixed-shape ONNX graph
with the positional-interpolation pipeline CONSTANT-FOLDED (fix for the VAIML segfault).

Graph contract (post-processor inputs — the HOST owns patchify/normalize):
  inputs   pixel_values (1, 1024, 768) fp32   [1024 patches of 3x16x16, already normalized]
           spatial_shapes (1, 2) int64        [patch grid (32, 32) for a 512x512 tile]
           pixel_attention_mask (1, 1024) int32
  output   image_embeds (256, 2048) fp32      [1024 patches -> PixelUnshuffle x2 -> 256 tokens]

Fold rationale: NaFlex embeds do bilinear F.interpolate + pad-scatter per call — that
emits Resize/ScatterND/ScatterElements/GatherND (ops absent from the text graph) which
segfault the aiecompiler. For the FIXED tile geometry the resized positional embedding
is a pure constant, so we compute it once and freeze it. Parity vs the canonical path
is the correctness gate; the exported graph must contain none of the exotic ops.
Run in tools/d1-probe/.venv.
"""

import os
import time
from types import MethodType

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "models", "d1-3B"))
OUT = os.path.join(HERE, "export", "d1-3B_vision_tile512.onnx")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import Lfm2VlForConditionalGeneration  # noqa: E402

POS_SIZE = 32      # patch grid of a 512x512 tile (patch 16)
MAX_PATCHES = 1024
FORBIDDEN_OPS = {"Resize", "ScatterND", "ScatterElements", "GatherND"}
FOLD_TOL = 1e-06
ORT_TOL = 1e-04


def main():
    torch.manual_seed(0)
    m = Lfm2VlForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    print("model loaded")

    lm_mod = m.model
    tower = lm_mod.vision_tower

    # STEP 1: canonical reference BEFORE any patching (unpatched embeddings path)
    patches = POS_SIZE * POS_SIZE
    pv = torch.randn(1, patches, 768)
    ss = torch.tensor([[POS_SIZE, POS_SIZE]], dtype=torch.int64)
    pam = torch.ones(1, patches, dtype=torch.int32)
    with torch.no_grad():
        canon = torch.cat(lm_mod.get_image_features(
            pixel_values=pv, spatial_shapes=ss, pixel_attention_mask=pam,
            return_dict=True).pooler_output, dim=0)

    # STEP 2: fold the positional pipeline
    emb = tower.embeddings
    with torch.no_grad():
        pos = emb.position_embedding.weight.reshape(emb.position_embedding_size,
                                                    emb.position_embedding_size, -1)
        folded_pos = emb.resize_positional_embeddings(
            pos, torch.tensor([[POS_SIZE, POS_SIZE]], dtype=torch.int64),
            max_length=MAX_PATCHES)
    assert folded_pos.shape == (1, MAX_PATCHES, pos.size(-1))
    emb.register_buffer("folded_pos", folded_pos, persistent=False)

    def folded_forward(self, pixel_values, spatial_shapes):
        """Same math as Siglip2VisionEmbeddings.forward, positional side frozen."""
        return (self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
                + self.folded_pos)

    emb.forward = MethodType(folded_forward, emb)

    class VisionTile(torch.nn.Module):
        """Fixed-geometry tower: folded positional pipeline + full-mask (no unpad/ReduceSum)."""

        def __init__(self):
            super().__init__()
            self.tower = tower
            self.projector = lm_mod.multi_modal_projector
            self.pos_size = POS_SIZE

        def forward(self, pixel_values):
            # mask + spatial shapes are CONSTANTS for the fixed full tile: bake them so
            # the whole mask-derivation (And/GatherND/Where chain) constant-folds away
            ss = torch.tensor([[self.pos_size, self.pos_size]], dtype=torch.int64)
            pam = torch.ones(1, MAX_PATCHES, dtype=torch.int32)
            out = self.tower(pixel_values=pixel_values, spatial_shapes=ss,
                             pixel_attention_mask=pam, return_dict=True)
            f = out.last_hidden_state                      # (1, 1024, 1152)
            f = f.reshape(1, self.pos_size, self.pos_size, -1)
            img = self.projector(f)                        # unshuffle + MLP
            return img.reshape(-1, img.size(-1))           # (256, 2048)

    vt = VisionTile().eval()

    # STEP 3: the genuine gate — folded wrapper vs the canonical computed pre-patch
    with torch.no_grad():
        ref = vt(pv)
    d = float((canon - ref).abs().max())
    print(f"canonical (unpatched) vs folded torch maxdiff: {d:.3e}  shapes {tuple(canon.shape)} vs {tuple(ref.shape)}")
    if not np.isfinite(d):
        raise SystemExit("FOLD GATE FAILED: non-finite difference (NaN/inf) — aborting")
    if d > FOLD_TOL:
        raise SystemExit(f"FOLD GATE FAILED: maxdiff {d:.3e} > tolerance {FOLD_TOL:.0e} — the constant-fold is not mathematically exact")
    print("FOLD GATE PASSED")

    t0 = time.time()
    torch.onnx.export(
        vt, (pv,), OUT, opset_version=18, dynamo=True,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
    )
    print(f"vision export done in {time.time()-t0:.0f}s -> {OUT}")

    # graph hygiene: exotic ops must be gone
    import onnx
    g = onnx.load(OUT, load_external_data=False)
    ops = {n.op_type for n in g.graph.node}
    bad = ops & FORBIDDEN_OPS
    print("graph ops check:", "CLEAN" if not bad else f"STILL PRESENT: {bad}")
    if bad:
        raise SystemExit(f"FORBIDDEN OPS PRESENT after folding: {bad} — the NPU compile will likely segfault")

    # CPU parity — single input now; this gate can also FAIL
    import onnxruntime as ort
    s = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    o = s.run(None, {"pixel_values": pv.numpy()})[0]
    d_ort = float(np.abs(o.astype(np.float64) - ref.numpy()).max())
    print("ORT maxdiff vs torch folded:", f"{d_ort:.3e}")
    if not np.isfinite(d_ort):
        raise SystemExit("ORT GATE FAILED: non-finite output diff (NaN/inf)")
    if d_ort > ORT_TOL:
        raise SystemExit(f"ORT GATE FAILED: maxdiff {d_ort:.3e} > tolerance {ORT_TOL:.0e} — ONNX graph does not match torch")
    print("ORT GATE PASSED")


if __name__ == "__main__":
    main()