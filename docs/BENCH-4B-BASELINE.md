# 4B baseline (flag-off, pre-T2.1) - 2026-10-12

Server: workspace `build-cuda` llama-server (this tree, flags default off).
Recipe: `-m Qwen3.5-4B-UD-Q4_K_XL.gguf -c 8192 -ngl 99 --fit on` (full offload).
Host: RTX 4070 Laptop 8GB (7630 MiB free at launch), n_ctx 8192.
Method: POST /completion, n_predict 128, temp 0, cache_prompt false.

| run | prompt kind | pp tok/s | tg tok/s |
|-----|-------------|----------|----------|
| 1 | prose (sky) | 74.0 | 50.3 |
| 2 | code (fib) | 34.6 | 69.8 |
| 3 | prose (hamlet) | 187.2 | 69.3 |

Caveats: pp is noisy on tiny prompts (8-11 prompt tokens). Runs 2-3
overlapped with a concurrent `llama-cli` build (CPU contention), so treat pp
as approximate; tg is GPU-bound decode and less affected. No speculation
flags: this is the flag-off floor that T2.1 flag-on must beat (AC6-style,
4B-adjusted: thresholds were sized for 27B, so 4B numbers prove mechanics
only, not the 1.03x claim).
