# Handoff for another coding agent (written 2026-11-20)

Read this first, then `README.md` (Hebrew, full results) and `PRIOR_ART_moe_offload.md`.
Everything lives on branch `claude/confident-fermi-7vw8db` of `arielSofer/audio-transcription`, folder `progressive-weights/`. No pull request was ever opened and none was asked for.

תקציר בעברית: ניסינו שני רעיונות להרצת מודלים על פחות RAM. הראשון (משקולות מדורגות) לא עדיף על 8 ביט רגיל. השני (הזרמת מומחי MoE מה-SSD) עובד ונותן פלט זהה בדיוק, אבל הוא רק מחיר למודל שנכנס ל-RAM, ובתחום יש כבר עבודות רבות. עוד לא נמצא רעיון פורץ דרך. הצעד הפתוח: להבין מאיפה מגיעים 13.6 ms לטוקן שלא מוסברים בסנכרון (ראו סעיף 6).

## 1. The user and what they want
- Speaks Hebrew. Reply in Hebrew, short and direct. They answer best to lettered/numbered options ("א", "1,2,3"). They get frustrated by wasted time and by overclaiming.
- Original question: "how can an AI model run on less RAM, maybe on a very fast SSD, and how do we build it". Latest stated goal: "I want to invent something that changes the reality of AI". I told them honestly that I cannot promise that, and that what is on the table so far is engineering, not a breakthrough.
- **Never answered, asked several times:** which model and which task they actually want to run. Do not assume. Ask once, plainly, before building more.
- Their Mac: Apple Silicon, 48 GB RAM, MLX on the GPU (chip model not stated). Measured: cold sequential SSD read about 6.6 GB/s, write about 7.4 GB/s, GPU weight streaming about 147 GB/s, SSD reads slow the GPU by 9-14% and the SSD by 11-25% when both run.
- On the Mac: repo clone at `~/audio-transcription-bench` (single branch), work dir `~/pw_mac_run` with a venv (`~/pw_mac_run/venv`, Python 3.14, mlx, mlx-lm, transformers, pyarrow), `models/`, converted dirs and expert files. Downloaded there so far: Qwen2.5-0.5B, Qwen2.5-3B, mlx-community/Qwen3-30B-A3B-4bit (17 GB, plus a 16 GB expert file), mlx-community/gpt-oss-20b-MXFP4-Q8 (12 GB, plus a 10 GB expert file). Free disk was roughly 150-180 GB. gpt-oss-120b (63 GB, needs about 125 GB with its expert file) has NOT been downloaded or run.

## 2. Working rules that came out of this project
- Separate **measured** from **estimated** in every message. Several of my early estimates were wrong (I predicted a 10-30% slowdown at larger models, measured 1.6x at 3B).
- Always compare against the strong baseline (plain MLX 8-bit / the resident model) before presenting anything as a win.
- Check prior art before building. I skipped this for MoE offloading and it was a mistake.
- Before writing code, say which decision the result will inform. The user does not want more building without a purpose.
- Keep personal data (hostnames, usernames, temp paths) out of committed files.
- Never run `sleep` in a foreground command in the cloud session; never kill a process by a guessed PID (I once signalled PID 1).

## 3. Idea 1: progressive weights ("Netflix for weights") - finished, negative
4-bit "nested" base in RAM plus 4-bit residual tiles on SSD, fetched by score |x_tile|*||R_tile||. Works as designed (`mlx_engine.py`, `tilefile.py`, `progressive.py`), but:
- Known idea (DecDEC OSDI 2025, StrataFlux, Any-Precision LLM, Matryoshka quantization and others; see README).
- Per-token selection is far too slow (3B: 1.7 tok/s). "Decide once from the prompt" (plan mode, 4-bit resident residual) fixes speed.
- Mac, Qwen2.5-3B, same tokens: fp16 ppl 10.87 / 22.6 tok/s / 6.17 GB; **8-bit MLX ppl 10.87 / 35.8 tok/s / 3.48 GB**; 4-bit MLX 12.49 / 50.9 / 2.10 GB; ours plan 10%: 11.65 / 38.3 / about 2.26 GB (RAM estimated). Qwen2.5-0.5B: 8-bit 18.95 ppl at 124 tok/s vs ours plan 10% 20.49 at 128 tok/s.
- Verdict: for any model that fits in 8-bit, plain 8-bit is better. Only untested niche: models where 4-bit fits and 8-bit does not (GPTQ/AWQ not compared; a GPTQ-4 baseline never ran).

## 4. Idea 2: MoE expert streaming (`moe_stream.py`) - works, exact, but only helps models larger than RAM
How it works: `convert` copies every expert to a 4 KB-aligned block of one file (handles per-expert checkpoints and stacked tensors, MXFP4 and affine 4-bit). At run time each `SwitchGLU` is wrapped: router top-k is read to the CPU (one sync per layer), missing experts are read with `pread` (F_NOCACHE, 8 workers), kept in a per-layer LRU of MLX arrays, stacked, and the original `SwitchGLU` runs with local indices.
Runner: `run_moe_on_mac.sh` (download, convert, bench at several cache sizes, routing trace, `moe_reuse_analysis.py`). Env: `MODEL`, `CACHE_FRACS`, `VERIFY`, `NEED_GB`, `ENGINE_T`, `ENGINE_WINDOWS`, `ENGINE_START`.

**Exactness (measured):** perplexity identical to the resident model at every cache size on Qwen3-30B-A3B (14.1495) and gpt-oss-20b (160.7278; high only because it is a chat model fed raw text). OLMoE logits identical (max diff 0.000) in the container.

**Qwen3-30B-A3B-4bit** (48 layers, 128 experts, top-8, 2.65 MB/expert, 16.3 GB experts; WikiText-2, 2x256 tokens, cold cache per window):

| cache share | tok/s | ms/tok | hit rate | MB read/tok |
|---|---|---|---|---|
| resident | 34.9 | 28.7 | - | - |
| 1.0 | 27.5 | 36.4 | 0.992 | 8.3 |
| 0.5 | 24.0 | 41.7 | 0.971 | 29.7 |
| 0.25 | 13.8 | 72.3 | 0.822 | 181.7 |
| 0.125 | 6.8 | 148.0 | 0.644 | 363.3 |

**gpt-oss-20b** (24 layers, 32 experts, top-4, 13.24 MB/expert, 10.17 GB experts):

| cache share | tok/s | ms/tok | hit rate | MB read/tok | peak MLX GB |
|---|---|---|---|---|---|
| resident | 76.5 | 13.1 | - | - | - |
| 1.0 | 32.3 | 31.0 | 0.997 | 3.9 | 11.3 |
| 0.5 | 20.4 | 49.1 | 0.921 | 100.9 | 8.4 |
| 0.25 | 11.8 | 84.7 | 0.729 | 344.4 | 5.8 |
| 0.125 | 7.4 | 134.4 | 0.518 | 612.7 | 4.5 |

Other measured facts: effective SSD read rate 5-6 GB/s (2.6 MB experts) and about 8 GB/s (13 MB experts); cost per missed expert about 2.2 ms per 13 MB across cache sizes; a fixed expert set chosen once from the prompt hits about as often as LRU on Qwen3 (0.814 vs 0.822 at 25%, 0.951 vs 0.971 at 50%) but 2-3 points worse on gpt-oss-20b; peak memory is about non-expert weights (1.1 GB) + cache + a constant 2.2 GB (cause not investigated).
Full raw output: `results/moe_qwen3_30b_mac.txt`, `results/moe_gpt_oss_20b_mac.txt`.

**Estimates, NOT measurements:** gpt-oss-120b (36 layers, 128 experts, top-4, same 13.24 MB expert) at 8-18 tok/s depending on hit rate (formula in README: about 46 ms + 2.2 ms per missed expert, 144 expert activations per token), assuming a cache of about 45% of the experts fits in 48 GB (assumes macOS lets the GPU use about 36 GB). Time to first token on a long prompt of at least about 10 s (reads all 61 GB). Hit rates on code/chat/other languages are unknown; WikiText is a favourable single domain.

## 5. Prior art on MoE offloading (details and links in `PRIOR_ART_moe_offload.md`)
Already exists, including on Apple Silicon: Flash-MoE (C/Metal, Qwen3.5-397B on a 48 GB laptop, 4.36 tok/s), Edge0 (MLX, trained predictor, 15-18 tok/s on a 24 GB Mac mini, not output-identical), mlx-flash, plus CUDA work (SeqMoE, Speculating Experts, ProMoE, HOBBIT, MoE-Infinity). llama.cpp with mmap runs gpt-oss-120b on a 24 GB Mac at about 4.5 tok/s peak (one write-up, CPU only). Not found: chunked speculative execution with exact verification on unified memory. Caveat: pages were read by a helper agent, arXiv PDFs unreadable, some content only via summaries. Verify before quoting.

## 6. Where things stand (the user picked "A": cut the per-layer sync tax)
`sync_tax.py` ran on the Mac (`results/sync_tax_gpt_oss_20b_mac.txt`): forcing a sync after every layer of the NORMAL gpt-oss-20b costs 17.5 vs 13.4 ms, i.e. **0.18 ms per sync, 4.3 ms per token for 24 syncs**. The streamed engine with everything cached loses 17.9 ms/token vs resident. So **syncs explain only about 4.3 of 17.9 ms**; removing all of them would give at most about 1.16x. Idea A as stated (fewer syncs via speculative chunked execution) has little to win and was effectively disproved by this measurement. I told the user this has not been communicated in full yet: they pasted the result and this handoff was requested right after.

**Open: the other about 13.6 ms/token.** Candidate causes, none tested:
1. `mx.stack` copies the active experts on the GPU every layer and token (gpt-oss-20b: about 1.27 GB read + 1.27 GB written per token; the `stack_ms` timer only measures graph construction, the GPU copy shows up in `router_sync_ms`). Does not fit Qwen3 well (its whole tax is 7.7 ms for 48 layers, and 48 x 0.18 ms of syncs would already be 8.6 ms), so treat as a hypothesis.
2. CPU readback and `np.unique`, creating the local-index array, setting arrays into modules each call.
3. Different kernel behaviour on freshly created small stacked arrays.

Proposed next experiment (about an hour, informs whether a faster engine is worth it): a script like `sync_tax.py` that wraps each `SwitchGLU` of the RESIDENT model in four modes: normal; eval after each layer; plus `np.array(indices)` readback; plus `mx.take(original_stacked_array, unique_ids)` copies with local indices (mirrors the streamed path without any I/O). The differences between modes attribute the 13.6 ms. If the copy dominates, the fix is a per-layer slab of slots read in place by `gather_qmm` with slot indices (no per-token copy, one copy of the cache in memory). That is engineering, not new research, but it would move gpt-oss-120b from the pessimistic to the optimistic end of the estimate. Reuse `PROJS`, `ARRS`, `wrap_model` patterns from `moe_stream.py`.

## 7. Other directions I offered (B and C), not started
- B: KV cache on SSD for very long context / agents. Crowded field, value unknown, not searched.
- C: architecture built around SSD (huge lookup-table / memory-layer parameters, so SSD speed is an asset). Highest ceiling, lowest odds, needs training; a Mac can only do small-scale proofs. Prior art (memory layers, product-key memory) must be searched first.
Honest status: nothing found so far is a breakthrough. If the user still wants one, the honest path is: pick a real bottleneck, search prior art first, define the baseline, run kill tests quickly.

## 8. Files
- `moe_stream.py`, `moe_reuse_analysis.py`, `moe_verify.py`, `run_moe_on_mac.sh`, `sync_tax.py`: MoE streaming and its measurements.
- `mlx_engine.py`, `tilefile.py`, `progressive.py`, `run_on_mac.sh`, `run_sweep.py`, `summarize.py`, `decide_early_test.py`, `gptq_baseline.py`, `kv_fusion_test.py`, `onelayer_tests.py`, `diffusion_deepcache.py`, `sim_reference_*.py`: progressive-weights work and side tests.
- `mac_ssd_bench.py`, `mac_contention_bench.py`: SSD and SSD-vs-GPU contention benchmarks.
- `results/`: raw outputs. `README.md`: Hebrew write-up. `PRIOR_ART_moe_offload.md`: prior art with caveats.
- Model notes: loading `trust_remote_code=True` was blocked in the cloud session; use standard classes. `mlx_lm.utils.load_model(path, lazy=True)` is used to avoid reading experts that are replaced by streaming.
