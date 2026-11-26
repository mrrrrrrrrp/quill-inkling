# Speed findings, 2026-11-26

What was tried to make decode faster on the reference machine (RTX 4070 SUPER 12 GB, Ryzen 7 5800X3D,
32 GB DDR4-3000), and what came of it. The short version: two changes ship on by default in v0.1.2. The author
sees about **10% faster decode** in everyday use, but their A/B runs were drowned out by desktop noise, so the
controlled tests neither confirm that figure nor say which change it comes from.

The measurements come from one teacher-forced trace: 12 prompts, 4,463 speculative rounds, 9,210 tokens. "Item N"
in `trace_sim*.py` and `ntbench.cpp` refers to the numbers in the first column.

## Ideas tested

| Item | Idea | Verdict | Evidence |
|---|---|---|---|
| 0 | Record every routing decision and simulate ideas offline | **Kept** | Simulated unique misses match the CPU tier's own call log (64.09 per round). Tracing overhead is inside the noise |
| 1 | Cut verify rows that would miss the GPU's experts | **Rejected** | Even a perfect choice saves 1.0% of wall time: the 8 verify rows are nearby drafts and share nearly all their top-8 experts |
| 2a | Score residency by the union of experts per round | **Rejected** | 3.9% fewer misses in the model, 0.7% with real accounting |
| 2b | Repack all resident experts when idle | **Rejected** | The repack itself contributes about nothing. A better *ranking* would matter, not the repack |
| 3 | Non-temporal loads, so loop passes 2 and 3 re-read experts from L3 | **Shipped, A/B inconclusive** | Microbenchmark: a 60 MB set stays readable at 151 GB/s after 300 MB of streaming (+238%). 46% of loop pass 2/3 misses are re-reads. End to end: off 91.4, on 86.8, off 103.7 tok/s, so no verdict |
| 4 | Place expert pages in faster RAM address ranges | **Rejected** | A 12 GiB scan reads 29.7–33.3 GB/s in every band. There is no slow range to avoid |
| 5 | Higher GPU scheduling priority than the desktop | **Shipped, A/B inconclusive** | 83.4, 85.3 and 95.9 tok/s across the three runs: no verdict |
| 6 | Reclaim wasted VRAM | **Nothing to reclaim** | Expert slots have no padding and the build already targets only the card's architecture. Closing other GPU apps still helps: about +2.5% per 100 MiB freed |
| 14 | Precompute part of the drafter's input during idle time | **Possible** | The input is one linear layer. Estimated 1–3% |
| 16 | Is the drafter's confidence calibrated? | **Yes** | Below 0.5 confidence 17–29% of drafts are accepted, at 0.9 and above 91–95% |

## Measurement facts

- The CPU tier moves 287 MB per round (64 unique misses, 22.8 calls), at about 33.4 GB/s while busy.
- Median round time: 19.1 ms at 2 verified tokens, 27.1 ms at 4, 42.4 ms at 8.
- The first ~16 rounds after each prompt miss about 3× as often as later ones (158 vs 46 per round), so decode is
  slower at the start of a reply.
- Optimal (Belady) residency would miss only 18.5% as often as observed, but needs foresight that no available
  signal gives. Realistic online policies reach 52–54 misses per round against 59.5 observed.
- There are windows of about 1.5 ms at the loop-pass boundaries where the CPU tier, and so system RAM, is idle.
- Teacher-forced runs are not bit-deterministic (2.06–2.09 tokens per verification on identical text). The cause is
  not known yet.

## Chat page

The live tok/s figure read about 6–7% low, and lower still early in a reply, because it averaged from the first
token and assumed 3.8 characters per token. It now uses a window of about 2 seconds and a characters-per-token
value that is corrected by each reply's real token count. This changes what is displayed, not how fast the engine
decodes.

## Next

1. Repeat the two inconclusive A/Bs on a quiet desktop, then keep or revert them.
2. Prefetch experts for the next loop pass into the GPU during the idle windows (estimated +3–4%).
3. Measure drafter acceptance with and without the abliteration.
4. Find the cause of the teacher-forced nondeterminism.

## Reproducing

```powershell
cd bench
.\rp.ps1 -Root $env:LOCALAPPDATA\Inkling -Label trace -Seed -1 -Env @{QUILL_TIER_TRACE="$PWD\trace_t1.txt"}
python trace_sim.py trace_t1.txt summary       # also: residency, truncation, l3, calib
.\ntbench.cmd; .\ntbench.exe                    # L3 and RAM microbenchmark, no GPU needed
```

`ntbench` is tied to the reference CPU: it pins threads to its cores and sizes its buffers for a 96 MiB L3.
