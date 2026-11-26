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

## 6. Robustness — did our edits or the setup degrade MedLA unfairly?

Stress-tested from every angle that could penalize MedLA relative to the CoT baseline. **It did not.** But one angle reframes the finding (see §6.3) and must be stated honestly.

### 6.1 Code-edit audit (diff vs pristine upstream `82e2769`)
Every change to the reasoning path is infra, not algorithm:
- **Provider/key routing + guardrail funnel** (cache → budget → bounded retries → ledger). No change to prompts or decisions.
- **Deterministic per-agent shuffle** (`random.Random(f"{seed}-{agent_index}")` replacing global `random.shuffle`): same distribution, now reproducible. **Diversity preserved** — per-slot accuracy spans 69.1–73.1 %, agents disagree initially on 44.5 % of items, and the oracle ceiling is 89.4 % (if seeding had collapsed the ensemble, oracle ≈ mean agent ≈ 71 %).
- **Crash-fix var-inits + `except BudgetExceeded: raise`**: these paths never fired — 0/800 invalid final answers, 0/13,600 invalid agent answers.
- **`select_agent_index = 0`** (deterministic discussion representative, was `randint`): `agents_talk` collapses agents to one representative *per distinct answer*, so this only changes which reasoning-text represents an answer cluster, never which answers are on the table.
- The elimination / discussion / arbitration logic is otherwise identical to upstream.

### 6.2 No mechanical answer-loss, no unfair asymmetry
- **Parsing:** 0 invalid final predictions; 0/13,600 (0.00 %) invalid agent answers. MedLA is not penalized by a fragile parser.
- **Truncation:** `max_tokens=1536` identical to Arm A. Only 61/65,323 calls (0.09 %) hit the length cap, and every one still produced a valid answer. Not material.
- **Matched settings:** both arms use the same model, temp 0, seed 42, and max_tokens 1536. The *only* differences between Arm A and Arm C are MedLA's method itself (elimination prompt, few-shot, multi-agent, aggregation) — i.e. exactly what we intend to measure.
- **Few-shot sanity:** the injected exemplars are well-formed worked CoT examples with correct labels, not noise.

### 6.3 The decomposition that refines the thesis

| Stage | Accuracy | vs previous |
|---|---|---|
| Single CoT call (Arm A) | **79.6 %** | — |
| **Single MedLA agent (mean of 17)** | **71.4 %** | **−8.2** (method/prompt cost) |
| Majority-17 (Arm B) | 72.9 % | +1.5 over mean agent |
| **Full MedLA (Arm C)** | **74.9 %** | **+3.5 over mean agent** |
| Oracle — any of 17 agents right | **89.4 %** | +14.5 unrealized by aggregation |

Two facts that sharpen (and partly correct) the headline:

1. **The ensemble is *not* destroying a good signal.** MedLA's 17-agent aggregation *beats its own single agent* (74.9 % vs 71.4 %, +3.5) and beats naive majority (+2.0). Within MedLA's own method, more agents + discussion **help**.
2. **The dominant cause of MedLA < CoT is upstream of aggregation:** each MedLA agent, using option-elimination + few-shot scaffolding, is **~8 points weaker than a plain CoT call** on this strong backbone. gpt-4o-mini does not need the elimination scaffolding and is hurt by it. Aggregation then recovers +3.5 but cannot cover the 8-point member deficit, so the full method still lands below CoT.
3. **The selection gap is real and large:** the ensemble contains the right answer 89.4 % of the time but realizes only 74.9 % — 14.5 points left on the table by consensus aggregation.

**Honest confound (stated, not hidden):** Arm A is zero-shot CoT; a MedLA agent is elimination + 1 few-shot. So the cross-method gap (CoT vs MedLA) mixes *"ensemble overhead"* with *"prompt design."* The comparison that cleanly isolates the ensemble — single MedLA agent vs the MedLA ensemble, same prompt — shows the ensemble **winning** (+3.5). Open question still worth one cheap control run: is the 8-point per-agent deficit from the elimination framing, the few-shot, or both?

### 6.4 What you can and cannot conclude
- **Supported, robustly, not an artifact:** the full MedLA framework underperforms a simple CoT call on a strong backbone (−4.7, p = 2e-4), and this is due to MedLA's method, not our implementation.
- **State with nuance:** it is *not* that "adding agents hurts" — MedLA's ensemble beats its own single agent. The precise claim is: **on a strong backbone, MedLA's method (elimination-style per-agent prompting + consensus aggregation) costs more and scores lower than plain CoT, because the scaffolding degrades each agent and consensus recovers only a fraction of the ensemble's 89 % oracle ceiling.** That 89 %→75 % selection gap is the opening for your framework.

---

## 7. Takeaway for the proposed framework

1. On a capable backbone, **MedLA costs ~155× a single CoT call (\$31 vs \$0.20) to lose 4.7 points** of accuracy.
2. The loss is not ignorance: the gold answer is in the agent pool on **59 %** of failures. The defect is **selection/aggregation**, confirmed by both the per-item diagnostic (§4) and the taxonomy (§5).
3. Discussion helps vs. naive voting (+2.0, p=0.02) but the gain is small and arbitration is nearly inert (0.8 %).
4. This motivates a framework whose aggregation can **recover a minority-correct agent** rather than regress to the confident majority — the measured bottleneck here.

---

## 8. Reproducibility & deliverables

- Nested sample built by `build_samples_800.py` (invariants asserted: 60 ⊂ 800 ⊂ 1273, prefix-preserving).
- Determinism: temp 0, seed 42, per-agent seeded shuffles, per-item NumPy-seeded few-shots → cache-stable (a re-run of any item is a \$0 cache hit; resume re-pays nothing — verified).
- `guarded_create` hard-aborts at `BUDGET_USD`; 429 waits are capped separately and never consume the retry budget (0 fired this run).

**Files** (in `week4/MedLA/`):
`results_800.md` · `per_item_medqa_800.jsonl` (800) · `arm_a_medqa_800.jsonl` (800) · `arm_b_medqa_800.jsonl` (derived) · `traces_wrong_800.jsonl` (201 failure traces) · `taxonomy_sample_ids.json` + `taxonomy_buckets.json` · `cost_ledger_800.csv` · `progress_800.log`. The original 60-item outputs are retained unchanged.
