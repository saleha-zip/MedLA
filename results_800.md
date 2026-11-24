# MedLA reproduction at scale — 800 MedQA-US items

**Backbone:** `gpt-4o-mini-2024-07-18` (temp 0, seed 42) · **Benchmark:** MedQA-US test, 800-item seed-42 sample (nested: the original 60 ⊂ these 800 ⊂ the full 1,273).
**Run:** overnight, item-concurrent (4 items in flight × 17 agents), resumable, fsync'd ledger + cache.
**Spend:** **\$31.23** total (\$0.20 Arm A + \$31.03 Arm C) · **0 rate-limit (429) events** · 266 cache hits (resume-test + original-60 re-use) · ~2.7 h wall-clock.

> **Cross-backbone caveat.** The paper uses LLaMA-3.1-8B; we use gpt-4o-mini. This is **not** a replication of the paper's absolute accuracies — it is a test of the *sign and size of the gap between arms* on a stronger backbone. That gap reverses relative to the paper (see §4).

---

## 1. Arms

| Arm | What it is | Cost model |
|-----|-----------|-----------|
| **A — CoT single** | one chain-of-thought call per question | paid directly (\$0.20 / 800) |
| **B — Majority-17** | plurality vote of the 17 agents' *initial* answers (first-seen wins ties) | **derived** from Arm C logs, \$0 extra |
| **C — Full MedLA** | 17 elimination agents + LogicAgent + up to 3 discussion rounds + arbitration | paid (\$31.03 / 800) |

Arm B is computed from Arm C's `initial_answers`, so it isolates *"what the raw ensemble already knew"* from *"what MedLA's discussion/arbitration did with it."*

---

## 2. Accuracy (all 800) with 95% Wilson intervals

| Arm | Correct | Accuracy | 95% CI |
|-----|---------|----------|--------|
| **A — CoT single** | 637 / 800 | **79.6 %** | [76.7, 82.3] |
| **C — Full MedLA** | 599 / 800 | **74.9 %** | [71.8, 77.8] |
| **B — Majority-17** | 583 / 800 | **72.9 %** | [69.7, 75.8] |

**Ordering: CoT > MedLA > Majority.** A single well-prompted CoT call beats the full 17-agent framework by **+4.7 points**. MedLA's discussion+arbitration *does* recover ground over naive majority voting (+2.0), but not enough to overcome the single-call baseline.

### Restricted to the original 60 (representativeness check)

| Arm | 60-item (this run) | 800-item | Prior 60-item report |
|-----|-----|-----|-----|
| A | 80.0 % [68.2, 88.2] | 79.6 % | 80.0 % |
| C | 76.7 % [64.6, 85.6] | 74.9 % | 76.7 % (prev. "75.0%") |
| B | 75.0 % [62.8, 84.2] | 72.9 % | 76.7 % |

The 60-item subset reproduces the earlier numbers and its point estimates sit inside the 800-item CIs. The earlier small-sample conclusion (**MedLA < CoT on a strong backbone**) holds at scale, now with tight intervals — the 60-item run was slightly optimistic for C (76.7 → 74.9) but not misleading.

---

## 3. Significance — exact McNemar (two-sided, on all 800)

| Comparison | wins / losses (discordant) | p |
|-----------|-----|-----|
| **C (MedLA) vs A (CoT)** | A wins 69, C wins 31 (100 discordant) | **0.00018** |
| **C (MedLA) vs B (Majority)** | C wins 29, B wins 13 (42 discordant) | **0.020** |
| A (CoT) vs B (Majority) | A wins 87, B wins 33 (120 discordant) | 8.7e-07 |

- MedLA **significantly underperforms** single CoT (p ≈ 2e-4): on the 100 questions the two disagree, CoT is right more than 2× as often.
- MedLA **significantly beats** raw majority voting (p ≈ 0.02): the discussion/arbitration machinery is not useless — it adds signal over the bare ensemble — it just starts from and stays below the CoT baseline.

---

## 4. Where MedLA loses: the aggregation bottleneck

**Arm C process stats (800 items):**

| metric | value |
|---|---|
| items where the 17 agents initially disagreed | 356 / 800 (44.5 %) |
| discussion rounds run (0 / 1 / 2 / 3) | 444 / 315 / 29 / 12 |
| items escalated to the arbitration agent | **6 / 800 (0.8 %)** |
| final answer ≠ first agent's answer | 99 / 800 (12.4 %) |
| &nbsp;&nbsp;of those: wrong→right (fixed) | 50 |
| &nbsp;&nbsp;right→wrong (broke) | 31 |
| &nbsp;&nbsp;wrong→wrong | 18 |

Discussion changes 99 answers for a **net of +19 correct** (50 fixed − 31 broken) — real but small, and swamped by the baseline gap. Arbitration almost never fires (0.8 %); in practice MedLA's aggregation *is* convergence-to-consensus during discussion, not the DecisionMakersAgent.

**The decisive statistic — selection failure:**

> On **118 of 201 Arm-C failures (58.7 %)** the correct answer **was proposed by at least one of the 17 agents and then discarded** by the consensus/aggregation step.

The right answer is already inside the ensemble on a majority of losses; MedLA's mechanism cannot surface a minority-correct agent against a confident wrong majority. This is the structural bottleneck — not a knowledge gap — and it is intrinsic to consensus-style multi-agent aggregation.

---

## 5. Failure taxonomy

201 of 800 Arm-C items are wrong (>60), so the 5-bucket taxonomy is applied to a **seed-42 random 60** of the 201 failures (`taxonomy_sample_ids.json`).

| Bucket | n / 60 | % | What it means |
|---|---|---|---|
| **AGGREGATION** | **31** | **51.7 %** | gold was among the 17 initial answers but the framework discarded it |
| **INFERENTIAL** | 23 | 38.3 % | all 17 agents converged on a plausible-but-wrong answer; multi-step clinical reasoning error (gold never proposed) |
| **FACTUAL** | 3 | 5.0 % | single-fact recall miss shared by every agent (e.g. rouleaux in myeloma; nucleus pulposus = notochord; glucagon for β-blocked anaphylaxis) |
| **LABEL / unanswerable** | 3 | 5.0 % | answer requires an unviewable figure/graph the text-only pipeline cannot see (PV-loop image, acid-secretion graph, CT with arrows) |
| **GENERATION** | 0 | 0.0 % | malformed/unparseable output — none occurred |

Over half of sampled failures are aggregation-stage, consistent with §4. The next-largest bucket (INFERENTIAL) is *consensus* reasoning error — every agent wrong the same way — which multi-agent debate cannot fix because there is no dissenting correct voice to amplify. Only ~10 % are failures outside the framework's control (pure fact gaps or image-dependent items). **~90 % of failures are things more agents or more debate structurally cannot repair.**

---

## 6. Takeaway for the proposed framework

1. On a capable backbone, **MedLA costs ~155× a single CoT call (\$31 vs \$0.20) to lose 4.7 points** of accuracy.
2. The loss is not ignorance: the gold answer is in the agent pool on **59 %** of failures. The defect is **selection/aggregation**, confirmed by both the per-item diagnostic (§4) and the taxonomy (§5).
3. Discussion helps vs. naive voting (+2.0, p=0.02) but the gain is small and arbitration is nearly inert (0.8 %).
4. This motivates a framework whose aggregation can **recover a minority-correct agent** rather than regress to the confident majority — the measured bottleneck here.

---

## 7. Reproducibility & deliverables

- Nested sample built by `build_samples_800.py` (invariants asserted: 60 ⊂ 800 ⊂ 1273, prefix-preserving).
- Determinism: temp 0, seed 42, per-agent seeded shuffles, per-item NumPy-seeded few-shots → cache-stable (a re-run of any item is a \$0 cache hit; resume re-pays nothing — verified).
- `guarded_create` hard-aborts at `BUDGET_USD`; 429 waits are capped separately and never consume the retry budget (0 fired this run).

**Files** (in `week4/MedLA/`):
`results_800.md` · `per_item_medqa_800.jsonl` (800) · `arm_a_medqa_800.jsonl` (800) · `arm_b_medqa_800.jsonl` (derived) · `traces_wrong_800.jsonl` (201 failure traces) · `taxonomy_sample_ids.json` + `taxonomy_buckets.json` · `cost_ledger_800.csv` · `progress_800.log`. The original 60-item outputs are retained unchanged.
