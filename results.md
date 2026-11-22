# MedLA Reproduction — Results

**Run:** MedLA (released `alexander2618/MedLA` implementation) on a fixed, stratified 60-item MedDDx sample.
**Backbone (paid arm):** OpenAI `gpt-4o-mini-2024-07-18`, temperature 0, seed 42, 17 M-agents, concurrent.
**Sample:** `sample_ids.json` — 20 items/tier (basic / intermediate / expert), stratified random, seed 42.
**Guardrails:** on-disk cache, per-call cost ledger, $4 hard abort (not triggered), max_output_tokens 1024, retries=2.
**Date:** 2026-11-21.

## Accuracy

| Arm | Basic | Interm. | Expert | Overall / AVE |
|---|---|---|---|---|
| **Paper — MedLA + LLaMA-3.1-8B** (Table 1) | 48.2 | 43.0 | 41.7 | 44.3 |
| Paper — LLaMA-3.1-8B baseline (Table 1) | 43.4 | 36.8 | 30.6 | 36.9 |
| **This run — MedLA + gpt-4o-mini** | **50.0** (10/20) | **50.0** (10/20) | **60.0** (12/20) | **53.3** (32/60) |
| Groq free-tier arm | — | — | — | *skipped per instruction* |

The column labels above are **MedLA's tier names**, which are **swapped relative to the benchmark's author (KGARevion, Su et al. ICLR 2025)**. MedLA's loader names the lowest-std 483-item group "Expert," but by KGARevion's construction (tiers cut on the std of candidate-answer semantic similarity; **lower std = more similar distractors = harder**; Expert = std 0–0.02 = 245 items) that 483-group is actually the **easiest** tier. The item-count fingerprint is decisive: 483 = KGARevion **Basic**, 245 = KGARevion **Expert**. (Both MedLA's paper and our run use the same swapped labels, so the paper-vs-ours rows remain apples-to-apples.) Re-mapped to true difficulty:

| KGARevion tier (true difficulty) | MedLA/our label | N | Our MedLA acc |
|---|---|---|---|
| **Basic** (easiest) | our "Expert" | 483 | **60.0%** |
| **Intermediate** | our "Inter" | 1041 | **50.0%** |
| **Expert** (hardest) | our "Basic" | 245 | **50.0%** |

So accuracy **decreases monotonically with true difficulty** (easiest 60% ≥ intermediate 50% ≥ hardest 50%) — the ordering is correct, not inverted.

### Gaps vs the paper's MedLA number (flagged where > 3 points)
| Tier | Ours − Paper-MedLA | Flag |
|---|---|---|
| Basic | +1.8 | within noise |
| Intermediate | **+7.0** | ⚑ |
| Expert | **+18.3** | ⚑⚑ |
| Overall | **+9.0** | ⚑ |

**Why we exceed the paper (expected, not a reproduction defect):** this is a **cross-backbone** comparison. The paper's backbone is **LLaMA-3.1-8B**; ours is **gpt-4o-mini**, a substantially stronger model. We deliberately used an OpenAI model (no OpenAI model is 8B-class) to probe *where MedLA still fails even with a strong backbone*. The gap is largest on the tier MedLA calls **"Expert"** (+18.3) — but note (see the relabel above) that tier is actually KGARevion's **easiest** (Basic) tier, so the strong backbone is helping most on the *easiest* items, which is sensible. Once mapped to true difficulty, our accuracy is **monotonic** (easiest 60% ≥ hardest 50%); the earlier "inverted ordering / sampling noise" reading was an artifact of MedLA's swapped tier labels, now corrected.

## Arm comparison — the headline (MedDDx, gpt-4o-mini, same fixed 60 items, temp 0)

Tier columns use MedLA's labels (our "Expert" = KGARevion's *easiest*; see relabel above).

| Arm | Basic | Interm. | Expert | **Overall** |
|---|---|---|---|---|
| S1 — single call, **direct** (no CoT) | 10/20 (50.0) | 8/20 (40.0) | 8/20 (40.0) | **43.3** |
| S2 — single call, **CoT** | 10/20 (50.0) | 7/20 (35.0) | 12/20 (60.0) | **48.3** |
| Majority vote of 17 agents' first answers | 7/20 (35.0) | 10/20 (50.0) | 12/20 (60.0) | **48.3** |
| **MedLA — full, 17 agents** | 10/20 (50.0) | 10/20 (50.0) | 12/20 (60.0) | **53.3** |

**Reading:** On MedDDx, the full MedLA pipeline (~71 calls/item) beats a **single CoT call (+5.0)** and a majority vote of the agents' first answers (+5.0). CoT alone lifts a direct single call by +5.0 (43.3→48.3), exactly matching majority vote; MedLA's debate/selection layer adds another +5.0 on top. So the entire multi-agent apparatus buys **+5 points over one CoT call** here — and, per `failure_analysis.md`, 57% of its remaining errors are the pipeline discarding a correct answer it already had. Whether that selection bottleneck is intrinsic to MedLA or specific to MedDDx is what the MedQA-US run (Task 3/4) tests.

## MedQA-US arms (Task 3) — the scaffolding *hurts* a strong backbone

Fixed seed-42 60-item sample of the 1,273 MedQA-US test items (`sample_ids_medqa.json`), gpt-4o-mini, temp 0, `max_output_tokens=1536`.

| Arm | Ours (gpt-4o-mini, 60 items) | Paper (LLaMA-3.1-8B, full test) |
|---|---|---|
| Single call, direct | — (not run) | baseline **56.3** |
| **Arm A** — single call, CoT | **80.0%** (48/60) | CoT 54.9 |
| **Arm B** — majority vote of 17 first answers | **76.7%** (46/60) | majority voting 46.8 |
| **Arm C** — full MedLA, 17 agents | **75.0%** (45/60) | MedLA **62.6** |
| (other baselines) | | MedAgents 53.2 · MDAgents 53.4 · DyLAN 51.6 |

**The inter-arm gap *reverses* across backbones** (this is the backbone-robust quantity — and it is not robust here):
- Paper (LLaMA-3.1-8B, weak): MedLA − CoT = **+7.7**, MedLA − majority = **+15.8** → the scaffolding *helps* a weak model.
- Ours (gpt-4o-mini, strong): MedLA − CoT = **−5.0**, MedLA − majority = **−1.7** → the scaffolding *hurts* a strong model.

Interpretation: MedLA's elimination/debate/arbitration machinery adds value when the backbone is weak enough that cross-checking recovers errors, but on a strong backbone it mostly injects noise that pulls a good single-call answer off target. On MedQA, agents disagreed on only **28/60** (vs 50/60 on MedDDx — MedQA options are genuinely distinct, not near-synonyms), discussion changed 6 answers (**3 right / 3 wrong**, net 0), and arbitration fired **0/60**.

**Selection bottleneck is intrinsic, not a MedDDx artifact (preview of Task 4):** of the 15 Arm-C failures, **11 (73%)** had the gold answer proposed by ≥1 agent and then discarded — the same aggregation-dominant pattern seen on MedDDx (57%). And Arm B (majority vote, 76.7%) *beats* Arm C (full pipeline, 75.0%), so the debate+decision layer is net-negative on this backbone. Full error-type breakdown in Task 4.

**Caveats:** 60 items (wide CI); cross-backbone, so these cannot confirm/refute the paper's absolute numbers — what is measured is the **gap between arms on one backbone**, and the finding is that that gap's *sign* is backbone-dependent. Cost: Arm C $2.32, Arm A $0.015; cumulative **$5.06** of $6.00.

## Caveats (read before citing these numbers)
1. **Cross-backbone.** Paper = LLaMA-3.1-8B; ours = gpt-4o-mini. Not a like-for-like reproduction of the paper's absolute numbers; it reproduces the *method/pipeline*.
2. **Small sample.** 20/tier → wide confidence intervals; treat tier differences cautiously.
3. **Call-count vs paper Table 5 does NOT line up — by upstream design.** The released code's M-agent does **iterative elimination (n−1 LLM calls) + 1 logic call ≈ 4 calls/agent**, whereas the paper's prose says each M-agent emits its tree in **one** call. This discrepancy is in `alexander2618/MedLA` itself, not our fork (verified against the upstream `util/agent.py` and our base commit `5c12cfc`). Measured here: **~71.5 calls/item** at 17 agents.
4. **Truncation.** 59 / 4,287 calls (1.38%) hit the 1024-token cap, almost all long LogicAgent TSV outputs; final answers are parsed from an early `<Answer>`/`<Eliminate>` tag, so impact on the extracted answer is minimal. A 1536 cap would remove it.
5. **Determinism.** temp 0 + seed 42 + per-agent seeded RNG make prompts reproducible; residual variation is OpenAI's best-effort temp-0 nondeterminism (not controllable).
6. **Concurrency is an API-backend adaptation, not an algorithm change.** We run the M-agents' independent calls concurrently because each call here is a network round-trip to a remote API. The paper's "M-agents run in parallel/independently" refers to *logical* independence; MedLA ran on 8×A100 under vLLM, where a serial Python loop still gets hardware parallelism from server-side request batching. Their serial loop was correct for that deployment; our threading only recovers the same wall-clock behavior over HTTP. It does not change results.
7. **Tier labels are swapped vs the benchmark author** (see the relabel table above): MedLA's "Expert" = KGARevion's Basic (easiest, 483 items); MedLA's "Basic" = KGARevion's Expert (hardest, 245).

## Cost
| | |
|---|---|
| **Total spend** | **$2.7128** (ceiling $4.00, not hit) |
| Calls | 4,287 (7 cache hits) |
| Avg calls/item | 71.5 |
| Avg cost/item | $0.0452 |
| Cost by tier | basic $0.85 · inter $0.93 · expert $0.93 |

Full per-call detail in `cost_ledger.csv`; per-item records in `per_item.jsonl`; failure taxonomy in `failure_analysis.md`.

## Headline framework findings (detail in failure_analysis.md)
- **MedLA's full machinery beats a naive majority-vote of the agents' first answers by only +5 pts (53.3% vs 48.3%).**
- **The debate step is barely net-positive:** it changed the final answer 18/60 times → 10 correct, 8 wrong (net +2).
- **Arbitration is essentially unused:** the DecisionMakersAgent fired on 1/60 items.
- **57% of failures (16/28) had the correct answer proposed by ≥1 agent but lost it in aggregation** — the bottleneck is *selection*, not *generation*.
