# Advice: MoE decode speed on the 4070 8GB box (2026-10-13)

## Findings first

1. **The diagnosis is right in direction, but the mechanism is probably the driver's system-memory fallback, not Windows paging.** Since driver 536.40, NVIDIA lets CUDA allocations that do not fit in graphics memory (VRAM) spill into shared system RAM. The GPU then reads those weights over PCIe on every token. That fits all three observations:
   - The CUDA0 buffer line shows the full 16754 MiB, because the allocation "succeeded".
   - The GPU sits at 46% with full clocks, because it is waiting on the bus, not computing.
   - Speed is about 5 t/s at ngl99. Each token reads roughly 1.5 to 2 GB of active weights for a 3B-active model at this file size. Over a laptop-class PCIe link that is about 10 to 15 GB/s, which caps decode near 5 to 8 t/s. That matches.
   - "Paging" suggests random thrashing. This is steady, bandwidth-bound streaming, which explains why the slowdown is smooth and deep rather than erratic.
2. **Don't trust the 569 MB `nvidia-smi` reading as evidence.** Under the Windows display driver model (WDDM), `nvidia-smi` memory numbers are unreliable, and fallback memory is not counted as dedicated VRAM. The conclusion stands without it.
3. **There is a cheap test that would confirm it.** In the NVIDIA Control Panel, set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" for the llama binary. If I'm right, ngl25 then fails at load with an out-of-memory error instead of running at 8.5 t/s. Leave that setting on afterwards so the trap fails loudly instead of slowly.
4. **The cliff is where VRAM runs out, not where an "ngl rule" kicks in.** About 15 full layers, plus the compute buffers and the desktop's own use of the card (WDDM drives the display), fill 8 GB. Every layer past that streams over PCIe, and that is slower than letting the CPU read the same weights from system RAM. System RAM bandwidth is roughly 50 to 80 GB/s against 10 to 15 GB/s over PCIe.

## Next decode-speed levers in stock llama.cpp, ranked by expected gain on this box

1. **Use `-ngl 99` plus `--n-cpu-moe N` (or `-ot "ffn_.*_exps=CPU"`) instead of `-ngl 15`.** Expected gain: largest, plausibly 1.3x to 2x, but I have not measured it on this box.
   - Why: ngl15 leaves 27 complete layers on the CPU, including attention, the shared expert, the router and the key-value (KV) cache.
   - Moving only the routed experts to the CPU puts all the small, always-used weights on the GPU. The CPU then handles just the per-token expert reads, which are bandwidth-bound and do suit the CPU.
   - Lower N until VRAM is nearly full, with the fallback disabled (finding 3) so overshooting crashes instead of crawling.
   - `--fit` should do this placement automatically. It failed here, and the reason is worth checking: our fork's custom KV types or the context size may be confusing its size estimate. That is a guess I have not checked.
2. **Sweep the CPU thread count (`-t`).** Expected gain: 0 to 30%. Why: once experts live on the CPU, decode is RAM-bandwidth bound. Too many threads (hyperthreads or E-cores) often hurt. Try the physical P-core count, then plus or minus 2.
3. **Use a smaller expert quantization (for example IQ4_XS or Q3_K for the expert tensors only).** Expected gain: roughly in proportion to the bytes saved, say 15 to 30%. Why: decode in this setup is bytes per token divided by bandwidth. This is not a runtime option and it costs some quality, but it is the most predictable gain after the first item.
4. **Prompt-lookup (n-gram) drafting.** Expected gain: 0 to 1.5x, and only on repetitive output such as code edits or text quoted from the prompt. It is likely zero on fresh generation.
   - Your own numbers argue against it: S2, S2V and C2 were all equal within noise at ngl15.
   - For a mixture-of-experts (MoE) model, checking k drafted tokens touches the union of their experts. With experts on the CPU, that makes each check cost several times a single-token step.
   - Retest after item 1, because the balance shifts once attention is on the GPU.
5. **KV-cache quantization and placement.** Expected gain: about zero on short prompts, but real at 32k.
   - Why: qwen35moe is a hybrid model. Only its full-attention layers keep a KV cache, so at short context the cache is tiny.
   - At 32k, quantizing it frees VRAM for more expert layers under item 1. Keep the cache on the GPU (never use `-nkvo`). Turn flash attention (`-fa on`) on so quantized V works and buffers shrink.
6. **Batch options (`-b`, `-ub`).** Expected gain: none for decode. They only affect prompt processing. With experts on the CPU, a larger `-ub` (1024 to 2048) does speed prompt processing, because big batches copy expert weights to the GPU once per batch.
7. **`tensor_split`.** Expected gain: none. It only divides work across several GPUs, and there is one.

## Where the evidence is thin

- The ngl15 baseline itself moves between 22.7 (n=96) and 20.5 (n=128). Gaps under about 10% are not yet real, so use at least 3 interleaved runs per arm before ranking chain variants against each other.
- The C3S result (0.80x) and the "MoE best stays S2V" verdict were both measured in the ngl15 setup. That setup is no longer the target once item 1 is adopted, so re-run the S2V, C2S and C3S comparison after switching.
- The machine's RAM size and speed are not recorded anywhere I read. If the box has 16 GB of RAM, a 16.75 GB model cannot stay resident and the ceiling becomes disk speed. Record total RAM in the bench doc.

Verify manually:
- The driver version and the sysmem fallback setting. I could not read the NVIDIA Control Panel from here.
- Whether the installed build's `--n-cpu-moe` and n-gram drafting options match the names above. Check `--help`, since I did not run the binary.
