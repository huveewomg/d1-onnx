import os
import sys
"""try_d1.py — interactive/single-shot interface for the ONNX-ported d1-3B (text + image).

Usage examples (run from npu-playground/tools/d1-probe with the d1-probe venv python):

  # text-only decision (int8 text path uses input_ids graph fallback... v1: fp32 embeds path)
  python try_d1.py --state "I was charged twice this month, please refund one of them." \\
      --type noul --instructions "Is the customer asking for a refund?"

  # image decision (the cats demo)
  python try_d1.py --image cats.jpg --type choice --instructions "How many cats are there?" \\
      --criteria "one=One,two=Two,more=Three or more"

  # score question (2-10 levels)
  python try_d1.py --state "Order was due Monday, still not delivered." --type score \\
      --instructions "How urgent is this?" --criteria "Can wait,Today,Blocking the customer now"

  # interactive loop: python try_d1.py (no args) -> keep typing states/questions

Options:
  --image PATH(.jpg/.png)     image input (single 512x512 tile used; larger images are downscaled)
  --state TEXT or @file.json  the state (text or JSON file); omit when the image IS the state
  --type noul|choice|score    question type (d1 decision schema)
  --instructions TEXT         the question itself
  --criteria A,B,C            choice labels or score levels (comma-separated);
                              choice may use "label=desc" pairs
The tool drives two stages: patchify here -> NPU vision (via the ryzen-ai env python,
CPU-vision fallback) -> host merge -> ONNX text pass -> readout. Prints probabilities.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))  # repo root
# public layout: download the graphs from HF once:
#   hf download huveewomg/d1-3B-ONNX --local-dir models
# (put the tokenizer/config from LiquidAI/d1-3B next to them, or let HF resolve it)
HF_MODEL = os.environ.get("D1_HF_MODEL", "LiquidAI/d1-3B")          # tokenizer/processor source
MODELS_DIR = os.environ.get("D1_MODELS_DIR", os.path.abspath(os.path.join(ROOT, "models")))
def _model_file(name):
    """Resolve a graph/weight file: <MODELS_DIR>/onnx/<name> (HF layout) or <MODELS_DIR>/<name>."""
    for cand in (os.path.join(MODELS_DIR, "onnx", name), os.path.join(MODELS_DIR, name)):
        if os.path.exists(cand):
            return cand
    return os.path.join(MODELS_DIR, "onnx", name)

EMBEDS_GRAPH = _model_file("d1-3B_embeds_pass_seq512.onnx")
EMBED_TABLE = _model_file("lm.embed_tokens.weight")
MODEL_DIR = os.environ.get("D1_LOCAL_MODEL_DIR", MODELS_DIR)         # local ckpt if downloaded
RYZEN_PY = os.environ.get("RYZEN_AI_PYTHON", r"C:\miniforge3\envs\ryzen-ai-1.8.0\python.exe")
if os.environ.get("D1_HUB_OFFLINE"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


import sys as _s
_s.path.insert(0, MODEL_DIR)

def tokenizer_source():
    """A local dir only counts if it actually has tokenizer files; else resolve from the Hub."""
    local_ok = os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "tokenizer.json"))
    return MODEL_DIR if local_ok else HF_MODEL


def load_env():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    from transformers import AutoProcessor, AutoTokenizer
    src = tokenizer_source()
    tok = AutoTokenizer.from_pretrained(src)
    proc = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    # make the checkpoint's remote prompt/readout modules importable (works for
    # both a local checkpoint dir and a Hub-cached one)
    probe = os.path.dirname(os.path.abspath(tok.__class__.__module__ and tok.name_or_path)) if hasattr(tok, 'name_or_path') else None
    _ensure_d1_modules(src)
    return tok, proc, None  # torch model loads LAZILY (CPU-vision fallback only)


def _ensure_d1_modules(src):
    """sys.path must contain the dir holding prompt.py/modeling_d1.py (local dir,
    or the trust_remote_code snapshot inside the HF cache)."""
    if _s.path and any(os.path.exists(os.path.join(p, "prompt.py")) for p in _s.path):
        return
    candidates = []
    if os.path.isdir(src) and os.path.exists(os.path.join(src, "prompt.py")):
        candidates.append(src)
    hub_modules = os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                               "modules", "transformers_modules")
    if os.path.isdir(hub_modules):
        import glob as _g
        # LiquidAI/d1-3B remote code lands here after any trust_remote_code load
        for pat in (os.path.join(hub_modules, "*d1*", "**", "prompt.py"),
                    os.path.join(hub_modules, "*", "*d1*", "**", "prompt.py")):
            for p_ in _g.glob(pat, recursive=True):
                candidates.append(os.path.dirname(p_))
    if not candidates:
        # fresh install: nothing cached and no local checkpoint — fetch prompt.py
        # explicitly from the Hub (it is self-contained: no relative imports)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_MODEL, filename="prompt.py")
        candidates = [os.path.dirname(path)]
    _s.path.insert(0, candidates[0])
    return


def load_torch_model():
    from transformers import Lfm2VlForConditionalGeneration
    m = Lfm2VlForConditionalGeneration.from_pretrained(
        tokenizer_source(), torch_dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return fold_vision(m)


def fold_vision(m):
    """Freeze the positional pipeline for the fixed 32x32 tile (same as export_vision.py)."""
    from types import MethodType
    tower = m.model.vision_tower
    emb = tower.embeddings
    with torch.no_grad():
        pos = emb.position_embedding.weight.reshape(
            emb.position_embedding_size, emb.position_embedding_size, -1)
        folded = emb.resize_positional_embeddings(
            pos, torch.tensor([[32, 32]], dtype=torch.int64), max_length=1024)
    emb.register_buffer("folded_pos", folded, persistent=False)

    def folded_forward(self, pixel_values, spatial_shapes):
        return self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype)) + self.folded_pos
    emb.forward = MethodType(folded_forward, emb)
    return m


def vision_embeds(m, im):
    """Image -> (256, 2048) embeds. NPU first (subprocess into ryzen env), CPU fallback."""
    with torch.no_grad():
        pv = m.model.get_image_features(pixel_values=im["pixel_values"],
                                        **{k: v for k, v in im.items() if k not in
                                           ("pixel_values", "input_ids", "input_embeds")},
                                        return_dict=True).pooler_output
    return torch.cat(pv, dim=0)


BENCH = False



def npu_stage_if_available(pv_np, mask_np, shapes_np):
    global BENCH
    BENCH = os.environ.get("TRY_D1_BENCH") == "1"
    """Try the ryzen-env NPU vision; return None to signal CPU fallback."""
    if not os.path.exists(RYZEN_PY):
        return None
    if not os.path.exists(_model_file("d1-3B_vision_tile512.onnx")):
        return None
    tmp = tempfile.mktemp(suffix=".npz")
    out = tempfile.mktemp(suffix=".npz")
    np.savez(tmp, pv=pv_np, mask=mask_np, shapes=shapes_np)
    script = os.path.join(HERE, "try_d1_npu_stage.py")
    r = subprocess.run(
        [RYZEN_PY, script, tmp, out, _model_file("d1-3B_vision_tile512.onnx")] + (["bench"] if BENCH else []),
        capture_output=True, text=True, timeout=2400)
    if os.path.exists(out): print(r.stdout.strip())
    if r.returncode != 0 or not os.path.exists(out):
        print("  (NPU stage failed — falling back to CPU vision)",
              (r.stderr or r.stdout)[-300:].strip())
        return None
    z = np.load(out)["embeds"]
    print(f"  [NPU vision: {len(pv_np)} tile(s)]")
    return z


def readout(logits, opt_groups):
    scores = [max(float(logits[i]) for i in g) for g in opt_groups]
    mx = max(scores)
    e = [math.exp(s - mx) for s in scores]
    return [x / sum(e) for x in e]


def decide(args, ctx):
    tok, proc, m = ctx
    import prompt as d1_prompt

    # question object
    if args.type == "choice":
        labels, criteria = [], {}
        for part in (args.criteria or "").split(","):
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                labels.append(k); criteria[k] = v
            else:
                labels.append(part); criteria[part] = None
        q = d1_prompt.Choice(args.instructions, criteria)
    elif args.type == "noul":
        q = d1_prompt.Noul(args.instructions)
    elif args.type == "score":
        q = d1_prompt.Score(args.instructions, (args.criteria or "").split(","))
    else:
        raise SystemExit(f"unknown type {args.type}")

    groups = d1_prompt.readout_ids(tok, q)
    labels = (list(criteria.keys()) if args.type == "choice"
              else (["yes", "no"] if args.type == "noul" else list(q.criteria)))

    # prompt construction (Liquid's markup path)
    markup = None
    if args.image:
        # single 512x512 input; the processor patchifies
        im = Image.open(args.image).convert("RGB").resize((512, 512))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "\x00"}]}]
        text_full = proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        head = "<|im_start|>user\n"
        markup = text_full[text_full.index(head) + len(head): text_full.index("\x00")]
    prefix = d1_prompt.prefix_text(tok, args.state if args.state else None, "",
                                   style="json_only", system="none", images=markup or "")
    suffix = d1_prompt.suffix_text(tok, q, "", "desc")

    if args.image:
        inputs = proc(text=[prefix + suffix], images=[[im]], return_tensors="pt", add_special_tokens=False)
        ids = inputs["input_ids"][0].numpy()
        pv = inputs["pixel_values"].numpy().astype(np.float32)
        pam = inputs["pixel_attention_mask"].numpy().astype(np.int32)
        ss = inputs["spatial_shapes"].numpy()
        img_pos = (ids == 124907).nonzero()[0]
        K = pv.shape[0]
        img_embeds = None
        if K == 1:
            # our fixed-shape graph handles exactly (1024, 768); full-valid tiles only
            if int(pam[0].sum()) == 1024 and tuple(pv.shape[1:]) == (1024, 768):
                img_embeds = npu_stage_if_available(pv, pam, ss)
        if img_embeds is None:
            if m is None:
                m = load_torch_model()  # lazy: 6 GB checkpoint, CPU-vision fallback only
            img_embeds = vision_embeds(m, torch_tensorize(m, pv, ss, pam)).reshape(-1, 2048).numpy()
            print(f"  [CPU vision, {K} tile(s)]")
        # host merge
        table = np.memmap(EMBED_TABLE, dtype="<f4", mode="r").reshape(128000, 2048)
        em = table[ids]
        em[img_pos] = img_embeds.reshape(-1, 2048)[: len(img_pos)]
    else:
        ids = None
        # text-only: token embeds via the table
        em_ids = tok.encode(prefix + suffix, add_special_tokens=False)
        table = np.memmap(EMBED_TABLE, dtype="<f4", mode="r").reshape(128000, 2048)
        em = table[np.array(em_ids)]

    # text pass (embeds graph, left-padded to 512)
    L = em.shape[0]
    assert L <= 512, f"prompt {L} tokens > 512 (too long for this build)"
    pad = 512 - L
    em5 = np.zeros((1, 512, 2048), dtype=np.float32)
    em5[0, pad:] = em
    mask = np.zeros((1, 512), dtype=np.int64); mask[0, pad:] = 1
    pos = np.clip(np.cumsum(mask, axis=1) - 1, 0, None).astype(np.int64)
    s = ort_session(EMBEDS_GRAPH)
    logits = s.run(None, {"inputs_embeds": em5, "attention_mask": mask, "position_ids": pos})[0][0, 0]

    probs = readout(logits, groups)
    answer = groups[int(np.argmax(probs))]
    # map option group -> label (groups align with option_codes order)
    top_label = labels[int(np.argmax(probs))]
    conf = max(probs)
    return top_label, conf, probs, L


def torch_tensorize(m, pv, ss, pam):
    return {"pixel_values": torch.from_numpy(pv),
            "spatial_shapes": torch.from_numpy(ss),
            "pixel_attention_mask": torch.from_numpy(pam)}


_ORT = {}
def ort_session(path):
    import onnxruntime as ort
    if path not in _ORT:
        _ORT[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _ORT[path]


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--image"); ap.add_argument("--state")
    ap.add_argument("--type", default="choice")
    ap.add_argument("--instructions")
    ap.add_argument("--criteria")
    if not sys.argv[1:]:
        interactive()
        return
    a = ap.parse_args()
    ctx = load_env()
    label, conf, probs, L = decide(a, ctx)
    print(f"\n=== d1 decision [{L} tokens read] ===")
    print(f"answer: {label}   confidence: {conf:.4f}")
    print(f"probabilities: {['%.4f' % p for p in probs]}")


def interactive():
    print("try_d1 interactive — for each round you'll be asked: type, instructions,")
    print("criteria (blank for noul), image path (blank = text mode), state text.")
    ctx = load_env()
    while True:
        t = input("type [choice/q/noul/score, q=quit] > ").strip()
        if t == "q":
            break
        inst = input("instructions/question > ").strip()
        crit = input("criteria (comma-separated; label=desc ok) > ") if t in ("choice", "score") else None
        img = input("image path (blank = no image) > ").strip() or None
        state = input("state text (blank if image-only) > ").strip() or None
        a = argparse.Namespace(image=img, state=state, type=t or "choice",
                               instructions=inst, criteria=crit)
        try:
            label, conf, probs, L = decide(a, ctx)
            print(f"\n=== d1 decision [{L} tokens read] ===")
            print(f"answer: {label}   confidence: {conf:.4f}")
            print(f"probabilities: {['%.4f' % p for p in probs]}\n")
        except Exception as e:
            print("error:", e)


if __name__ == "__main__":
    main()
