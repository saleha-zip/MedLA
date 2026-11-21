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

### Gaps vs the paper's MedLA number (flagged where > 3 points)
| Tier | Ours − Paper-MedLA | Flag |
|---|---|---|
| Basic | +1.8 | within noise |
| Intermediate | **+7.0** | ⚑ |
| Expert | **+18.3** | ⚑⚑ |
| Overall | **+9.0** | ⚑ |

**Why we exceed the paper (expected, not a reproduction defect):** this is a **cross-backbone** comparison. The paper's backbone is **LLaMA-3.1-8B**; ours is **gpt-4o-mini**, a substantially stronger model. We deliberately used an OpenAI model (no OpenAI model is 8B-class) to probe *where MedLA still fails even with a strong backbone*. The gap is largest on the **Expert** tier (+18.3), consistent with a stronger backbone helping most on the hardest items. The Expert>Inter>Basic ordering (inverted vs the paper) is **within sampling noise** at n=20/tier (95% CI ≈ ±21 pts).

## Caveats (read before citing these numbers)
1. **Cross-backbone.** Paper = LLaMA-3.1-8B; ours = gpt-4o-mini. Not a like-for-like reproduction of the paper's absolute numbers; it reproduces the *method/pipeline*.
2. **Small sample.** 20/tier → wide confidence intervals; treat tier differences cautiously.
3. **Call-count vs paper Table 5 does NOT line up — by upstream design.** The released code's M-agent does **iterative elimination (n−1 LLM calls) + 1 logic call ≈ 4 calls/agent**, whereas the paper's prose says each M-agent emits its tree in **one** call. This discrepancy is in `alexander2618/MedLA` itself, not our fork (verified against the upstream `util/agent.py` and our base commit `5c12cfc`). Measured here: **~71.5 calls/item** at 17 agents.
4. **Truncation.** 59 / 4,287 calls (1.38%) hit the 1024-token cap, almost all long LogicAgent TSV outputs; final answers are parsed from an early `<Answer>`/`<Eliminate>` tag, so impact on the extracted answer is minimal. A 1536 cap would remove it.
5. **Determinism.** temp 0 + seed 42 + per-agent seeded RNG make prompts reproducible; residual variation is OpenAI's best-effort temp-0 nondeterminism (not controllable).

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
