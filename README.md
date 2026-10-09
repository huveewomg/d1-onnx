---
license: other
license_name: lfm-1.0
base_model: LiquidAI/d1-3B
library_name: onnx
pipeline_tag: image-text-to-text
tags:
- onnx
- decision-model
- amd-npu
- xdna2
- ryzen-ai
- single-pass
- lfm2
---

# d1-3B — ONNX single-pass port (NPU-friendly)

**d1-3B image decisions with AMD XDNA2-accelerated vision — to our knowledge the first d1 execution on AMD NPU silicon (XDNA2) — verified against the official checkpoint.** The text leg runs on CPU (see the table). A parallel browser port exists: [onnx-community/d1-3B-ONNX](https://huggingface.co/onnx-community/d1-3B-ONNX) (Transformers.js/WebGPU, same readout approach, published the same week — credit to them for the browser port and the conversion methodology). On Ryzen AI NPU, the two ports behave the same way and neither reaches the DPU for the text decoder as-shipped: graphs carrying generic quantized-matmul ops (`MatMulInteger`/`MatMulNBits`) fall back to CPU, and the fp32 route hits either monolithic-graph inefficiency or, for NaFlex vision graphs, a compiler segfault — the exact issues dissected in the gating forensics below, this repo's novel contribution alongside the NPU-compiled vision graph (which resolves them by constant-folding the dynamic geometry).

**In one sentence: an AMD XDNA2 deployment study and ONNX implementation of d1-3B — fixed-geometry vision acceleration with measured silicon evidence, reference comparisons, and documented quantized-decoder fallback behavior.**

Liquid's d1 decision models answer typed questions in **one forward pass** (no output tokens):
yes/no (`noul`), pick-from-options (`choice`), ordered rubrics (`score`), with text, JSON or
images as the state. This repo contains that model as **decision-shaped ONNX graphs** —
cache-free, fixed-shape, verified 1:1 against the official checkpoint.

## What is verified (and how)

| Claim | Evidence |
|---|---|
| Text backbone matches Liquid's official code | ONNX logits vs torch-native: max abs diff **2.9e-05** (readout probabilities agree to 1e-06); torch-native vs Liquid's remote code: **0.0** (identical) — both on real weights |
| Vision pipeline fold is exact | canonical vs folded path: max abs diff **0.000e+00** |
| End-to-end image decision | COCO two-cats demo (as "two" choice): torch-native **0.9847** vs ONNX+NPU-vision **0.9849** — same answer, final-probability drift 2e-4 |
| Vision-stage speedup vs measured CPU baseline | one 512×512 tile: **378 ms NPU vs 7,242 ms CPU = 19.2×** (same graph; CPU 21–26% busy during NPU runs — real DPU compute; NPU meter at 100%). Whole-pipeline numbers: see "What runs where" |
| Final-probability agreement through the NPU (single example) | cats demo: drift 0.0002 — indicative, **not** a calibration study (dataset-scale robustness evaluation is future work) |
| Decisions match Liquid's own API | Liquid `system_one` → `two` @ conf 0.9859; our port → `two` @ 0.9847 (gap 0.001) |

*(Probabilities refer to the cats demo decision; exact reference tensors + parity scripts live in the GitHub repo's `verify/reference/`.)*

## What runs where

| Component | ONNX graph | Best measured | Notes |
|---|---|---|---|
| Vision encoder (SigLIP2-400M, 1 tile = 256 tokens) | `d1-3B_vision_tile512.onnx` | **378 ms / tile on NPU** (19.2× vs CPU) | single fixed-shape graph; exotic ops constant-folded away |
| Text decision, short states | `d1-3B_text_int8_seq128.onnx` (input_ids input) | **537 ms** (dynamic int8, CPU) | fp32 twin: 1750 ms; int8 keeps decisions + argmax, probs drift ~0.03-0.1 |
| Text decision with image embeds | `d1-3B_embeds_pass_seq512.onnx` (inputs_embeds input) | **4.7 s** (CPU fp32, 512-tok window) | NPU-vision output merged in host-side; int8 variant planned for v1.1 |
| Text decoder on NPU | — | CPU-speed | see "NPU gating" below |

## The NPU gating finding

*(Scope: tested on Ryzen AI Software 1.8.0, XDNA2/Krackan Point, our graph formats and the bare VitisAI EP path; other SDK versions or validated-zoo models may behave differently.)*

AMD's VitisAI EP ships fused quantized NPU kernels (`waic_target_qhw4`) that **only dispatch on
exact md5 signatures of models in AMD's own zoo** (`PHI_7B_4BIT`, `QWEN3_0.6B_8BIT`, ...; decoded
from `vaip_config.json`'s `mepTable`). Arbitrary models fall to the generic compiler flow where
`MatMulInteger`/`MatMulNBits` are "Not supported" → quantized matrix math silently runs on CPU,
even when the partition report claims 100% of GOPs placed. Requesting the zoo target explicitly
loads but binds no AIE (`XAie_SetIOBackend: Invalid backend request`).

In our testing (SDK 1.8.0, the tested quantized decoder graphs, bare-EP path), the 3.1B model's
0.2 ms theoretical NPU budget measures hundreds of milliseconds: the quantized matrix math does
not reach the DPU on the tested path. Encoder-style fp32 graphs in the same flow DO accelerate
(19.2× here; nomic control 33.5 ms NPU vs 79 ms CPU). Other SDK versions or zoo-validated
models may behave differently.

## Files

**This repo (github.com/huveewomg/d1-onnx):**

```
converter/  full export pipeline: export_d1_text / export_embeds_pass / export_vision,
            reshape_rank2, dedupe_tied, kaggle_d1_quark.ipynb (AMD Quark int4 recipe)
verify/     parity/benchmark/e2e harness + reference tensors (verify/reference/*.npz)
try/        try_d1.py — usable decision tool (text/JSON/image; NPU vision stage)
docs/       VAIML gating forensics: partition traces, GOP accounting, mepTable decode
assets/     cat-demo image used by the end-to-end check (COCO 39769)
```

**Model weights + graphs (the HF repo, onnx/):**

```
d1-3B_text_int8_seq128.onnx     input_ids (1,128) -> answer_logits (1,1,128000)   [int8, 537 ms]
d1-3B_text_pass_seq128.onnx     fp32 anchor of the same graph (shared canonical weight set)
d1-3B_embeds_pass_seq512.onnx   inputs_embeds (1,512,2048) -> answer_logits; shell that shares the fp32 weight files
d1-3B_vision_tile512.onnx       pixel_values (1,1024,768) -> image_embeds (256,2048)  [NPU-compiled, 19.2x]
```

All fp32 graphs share one weight set (`onnx/lm.*.weight` + `onnx/onnx__MatMul_*`), byte-matched per tensor.
`verify/reference/*.npz` = reference logits/embeds for offline parity checks.

## Setup

```bash
pip install -r requirements.txt
hf download huveewomg/d1-3B-ONNX --local-dir models      # graphs (17 GB; graphs only are much smaller)
# tokenizer/processor: either copy from LiquidAI/d1-3B or let the tools resolve it from the Hub

# CPU-only decision (no AMD SDK needed):
python try/try_d1.py --state "I was charged twice this month, please refund one of them." \
    --type noul --instructions "Is the customer asking for a refund?"

# NPU vision stage additionally needs: Ryzen AI Software 1.8.0 (VitisAI EP), XDNA2 silicon,
# and env vars RYZEN_AI_PYTHON + VAIP_CONFIG pointing at that installation (see try/try_d1.py).
```

## Reproduce

The conversion + verification tooling (export pipeline, parity/benchmark harness, `try_d1.py`, reference tensors) lives in the companion GitHub repo: **github.com/huveewomg/d1-onnx**.

Demo image: `assets/cats.jpg` (COCO 39769).

```bash
hf download huveewomg/d1-3B-ONNX --local-dir models

# text-only decision (CPU; ONNX graphs + Hub tokenizer suffice — no torch checkpoint)
python try/try_d1.py --state "I was charged twice this month, please refund one of them." \
    --type noul --instructions "Is the customer asking for a refund?"

# image decision through the NPU
python try/try_d1.py --image assets/cats.jpg --type choice \
    --instructions "How many cats are there?" --criteria "one=One,two=Two,more=Three or more"
```

Requirements: Python 3.12, `onnxruntime>=1.30`, `transformers>=5.19`; for the NPU path: Ryzen AI Software 1.8.0 (VitisAI EP) with an XDNA2 NPU (Ryzen AI 300 series or AI Max).

## Gotchas encoded in the converter (why a naive port fails)

1. **Tied lm_head** — torch tracer duplicates the tied embedding; quantizers must never share the
   initializer (fork `lm.head.weight`), or the embedding Gather silently collapses to vocab-sized
   activations and every RMSNorm explodes.
2. **Rank-3 MatMuls** — LFM2 conv blocks/FFN run `(1, S, D)`; ORT quantizers require rank 2
   (`reshape_rank2.py` wraps all 166 weight matmuls).
3. **NaFlex needs the dynamo exporter** (TorchScript chokes on its reshapes) and the
   positional-embedding interpolation must be **constant-folded for fixed tiles** or the NPU
   compiler segfaults on the `Resize`/`ScatterND`/`GatherND` cluster.
4. **VAIML model cache (SDK 1.8.0)** — with default cache settings the cache is wasted: every
   fresh process recompiles (~35-50 min for the int8 text graph). Set provider options
   `cache_dir` + `cache_key` + `enable_cache_file_io_in_mem="0"` and the behavior is what you
   expect: compile once, then fresh-process sessions load in seconds (verified: 2125s → 2.9s,
   zero cache regeneration). The fp32 (DPU-placed) graph still fails fatally on large-`.rai`
   cache reload — tracked upstream as [RyzenAI-SW#405](https://github.com/amd/RyzenAI-SW/issues/405).
5. **`use_cache=False`** — LFM2's cache layer rejects probe calls on attention-only layouts.

## Observed behavior (from hands-on use)

Same image (lobster + llama, from [ollama.ac.cn](https://ollama.ac.cn/), not redistributed):

| Question | Answer | Confidence |
|---|---|---|
| "How many creature are there?" | **two** | 0.95 |
| "How many **sea** creature..." | **one** | 0.67 |

Both right — the `sea` qualifier correctly excludes the llama from the count.
Built-in reproducible demo: `assets/cats.jpg`.

## License

Model weights and this conversion are released under Liquid AI's **LFM Open License v1.0**
(see `LICENSE` — carries over from the base model, Object form = conversions explicitly
covered). Base model: [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B).

## Credits

- [Liquid AI](https://huggingface.co/LiquidAI) — d1-3B, LFM2.5 architecture, open weights
- Verified against: native `transformers` (5.19) `Lfm2VlForConditionalGeneration` and Liquid's
  d1 remote code (`modeling_d1.py` / `runner.py`), both from the official checkpoint.
- Conversion/benchmark hardware: AMD Ryzen AI 7 Pro 350 (Krackan Point, XDNA2 50 TOPS).