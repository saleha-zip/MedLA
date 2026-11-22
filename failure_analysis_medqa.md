# MedQA-US Failure Analysis — Arm C (full MedLA), gpt-4o-mini, 60-item fixed sample

Arm C accuracy 45/60 (75.0%); **15 failures**. Each failure assigned to exactly one of the five
requested buckets, from the logged per-item traces (`per_item_medqa.jsonl`). AGGREGATION is
assigned first by its structural definition (gold proposed by ≥1 agent, then discarded); the
remaining items are read individually for FACTUAL / INFERENTIAL / GENERATION / LABEL.

## Counts

| Bucket | Count | Share of 15 |
|---|---|---|
| **AGGREGATION** — ≥1 agent proposed gold, pipeline discarded it | **11** | **73%** |
| FACTUAL — a premise the agent asserted is false | 2 | 13% |
| INFERENTIAL — premises sound, conclusion doesn't follow | 1 | 7% |
| GENERATION — no agent proposed gold, reasoning not obviously wrong (knowledge gap) | 1 | 7% |
| LABEL — model's answer clinically defensible, gold questionable | 0 | 0% |

**The single number: 73% of Arm-C failures are AGGREGATION** (11/15) — *higher* than on MedDDx (57%). On a clean benchmark with genuinely distinct options, the selection bottleneck not only persists but dominates, so it is **a property of MedLA, not an artifact of MedDDx**. Corroborating: Arm B (majority vote of first answers, 76.7%) **beats** Arm C (full pipeline, 75.0%), so the elimination/debate/decision layer is net-negative on this backbone.

## Examples per bucket

### AGGREGATION (11) — the answer was in the room
- **medqa_us:228** (gold A *Chikungunya*, final B *Dengue*). SE-Asia travel + fever + maculopapular rash + prominent symmetric MCP/wrist arthralgia = chikungunya. An agent proposed A, but the cohort fixated on leukopenia/thrombocytopenia and converged on dengue. Gold proposed, then discarded.
- **medqa_us:501** (gold D *adaptation in binding proteins*, final B *cleave β-lactam rings*). GPC in clusters, **dicloxacillin failed** → dicloxacillin is already β-lactamase-resistant, so failure implies MRSA (altered PBP2a) = D. An agent held D; the cohort selected the β-lactamase answer (which dicloxacillin would resist). Gold present and dropped.
- (others: 93, 285, 393, 541, 689, 778, 1098, 1149, 1206 — e.g., 1149 physiologic genu varum → gold *reassurance*, pipeline chose *x-ray*; 393 gastroschisis → gold *sterile bowel bag*, pipeline chose *immediate surgery*.)

### FACTUAL (2) — the chosen option asserts a false fact
- **medqa_us:178** (gold D *right-sided analgesia*, final A *spastic paralysis at the level of lesion*). Left C5 hemicord = Brown-Séquard; the "other finding" is contralateral (right) pain/temperature loss. Option A is false: **at** the lesion level weakness is flaccid/LMN — spastic (UMN) is *below* the level. The pipeline selected an anatomically false statement.
- **medqa_us:191** (gold C *maternal diabetes*, final A *maternal oligohydramnios*). Absent sacrum + lumbar vertebrae = caudal regression syndrome → maternal diabetes. The agents asserted oligohydramnios as the cause; oligohydramnios (Potter) causes limb/pulmonary findings, **not sacral agenesis** — a false causal premise.

### INFERENTIAL (1) — sound premises, conclusion doesn't follow
- **medqa_us:198** (gold D *prostate cancer*, final C *PCOS*). Two-step stem: a fertility drug given *pulsatile/intermittently* (a GnRH agonist, e.g., leuprolide) — "if administered **differently** [continuously], what else does it treat?" → continuous GnRH agonist downregulates the axis → prostate cancer. The agents correctly saw a fertility drug but collapsed the question to "the woman's diagnosis = PCOS," never tracing the drug's alternate (continuous) indication. Premises fine; conclusion answers the wrong question.

### GENERATION (1) — gold never produced, no clear false premise
- **medqa_us:928** (gold B *glucagon*, final D *norepinephrine*). Anaphylaxis in a patient on **labetalol** (β-blocker), refractory to epinephrine → **glucagon**. Norepinephrine is a defensible pressor for persistent shock (no outright false claim), but no agent ever produced glucagon — the β-blocker→glucagon link was simply absent. Knowledge gap.

### LABEL (0)
None. Unlike MedDDx — where label artifacts appeared (e.g., all agents chose Permethrin, the standard head-lice drug, over the gold Tetramethrin) — the USMLE-style MedQA golds held up on inspection; every miss is a real model error, not a questionable label.

## Behavior stats (Arm C, all 60 items)

- **Agents disagreed:** 28/60 (vs 50/60 on MedDDx — MedQA options are genuinely distinct, so the cohort agrees far more often).
- **Discussion changed the final answer:** 6/60 → **3 toward correct, 3 toward wrong** (net 0). Among the 15 failures, discussion flipped the answer on 3 (689, 778, 1149).
- **Arbitration (DecisionMakersAgent) fired:** 0/60.

## Takeaway

On MedQA, MedLA's remaining errors are **dominated by aggregation (73%)**, and the full pipeline underperforms both a single CoT call (80.0%) and its own majority vote (76.7%). The selection/debate machinery is the bottleneck, and it is **intrinsic to MedLA** — the same pattern seen on MedDDx, only stronger on the cleaner benchmark. The remaining non-aggregation errors (4/15) are ordinary single-model failures (2 factual, 1 inferential, 1 knowledge-gap) with no label artifacts.
