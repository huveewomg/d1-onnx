# Disk-cache rerun result — the int8 "related observation" resolved (2026-10-09)

Companion to [RyzenAI-SW#405](https://github.com/amd/RyzenAI-SW/issues/405). #405's opening
bug is the fp32 large-`.rai` fresh-process reload abort and remains open. Its **related
observation** (int8: 3154 s session creation, partition artifacts regenerated mid-session,
no `.rai` afterwards, cache configuration unresolved) is hereby resolved — `run` with fully
explicit cache settings on the exact same int8 graph (HF revision `ced5f5a17d6f…`):

| run | provider options | session creation | cache mutation |
|---|---|---|---|
| 1 (compile) | `config_file`, `cache_dir`, `cache_key`, `enable_cache_file_io_in_mem="0"` | 2125.3 s | 147 files written (2.2 MB), `.rai`: **none** |
| 2 (fresh process, same options) | 〃 | **2.9 s** | **0 files regenerated** |

Forward passes on both runs: 0.5 s, logits `(1, 1, 128000)`. Script + results:
`npu_study/disk_cache_int8/` in the npu-playground workspace (evidence commit 45998a8).

## Verdicts

1. **One-compile-then-load works for the int8 graph.** The 3154 s mystery was the
   *unconfigured default path* (default `%TEMP%` cacheDir, default md5 key, in-memory cache
   IO) failing to reuse — config-by-default, not a defect in our graphs or a silent-recompile
   property of the SDK.
2. **No `.rai` for the int8 graph is correct behavior.** All quantized matmuls are dropped to
   CPU (see [#406](https://github.com/amd/RyzenAI-SW/issues/406) and
   `cache-5321ae398989eec5e1274ef98fa21d74` here), so nothing DPU-shaped gets compiled; the
   int8 cache is legitimately a partition/analysis metadata tree (context.json,
   subgraphs_cache.json, per-subgraph dirs). The `.rai` format belongs to DPU-placed graphs
   (fp32 = 3 GB of real aiecompiler output) — expecting one there was our wrong prior.
3. **#405 therefore stands on the fp32 reload abort alone**, isolated from cache-config
   confusion.

## Also found while re-downloading: int8 graph external-data fix (HF fa3184e)

The int8 graph's 3 Constant **node-attribute** tensors referenced a pre-pack filename
(`d1-3B_int8e_seq128.onnx.data`) that was never uploaded — a fresh download of
`onnx/d1-3B_text_int8_seq128.onnx` from either repo failed ONNX Runtime external-data
validation at session creation. Root cause: the pack/merge step rewrites
`graph.initializer` external-data locations but is blind to node-attribute tensors
(`converter/repack_embeds_final.py` walks initializers only). Fix shipped on HF as
`fa3184e` (3 location strings rewritten; bytes provably already inside the uploaded
merged file — tail offsets end at its size exactly; pinned revision `ced5f5a17d6f…`
untouched) with a card note at `a0d26c9`.

**Standing guard for future packs (quantized exports will create attribute tensors
again):** run `converter/validate_external_refs.py` before ANY upload — it walks both
surfaces (initializers + node-attribute tensors) and fails if any referenced external
file is missing. Also note: onnx's checker rejects multi-hardlink data files
(hardlink-attack guard) — local validation setups must use real copies.