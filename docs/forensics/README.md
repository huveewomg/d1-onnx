# Forensics artifacts

- `cache-da743.../` — fp32 d1 text graph: 100.000% GOPs SUPPORTED (yet CPU-speed runtime) + MatMul-only-unSupported evidence
- `cache-5321ae.../` — int8 graph: 167 MatMulInteger "Not supported"
- `cache-5cab6c.../` — int4-g128 graph: 166 MatMulNBits "Not supported"
- `probe_matmulinteger.onnx` + `repro_matmulinteger.py` + `minimal_matmulint_result.txt` — minimal repro: a 6 KB single-MatMulInteger graph; VAIML drops the subgraph ("dropped", 0 GOPs) — quantized matmul never compiles for non-zoo graphs
- `cache_reload_test.py` + `cache_reload_result.txt` — ticket-1 repro: small graphs reload fine (0.4 s); the write-only-cache failure is specific to large (~3 GB) .rai caches
