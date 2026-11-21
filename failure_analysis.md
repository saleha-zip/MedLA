# MedLA Failure Analysis — MedDDx, 60-item fixed sample

**Backbone:** gpt-4o-mini-2024-07-18 · **MedLA accuracy:** 32/60 (53.3%) · **Wrong items:** 28.
Taxonomy built bottom-up: every wrong item was classified from (a) a structural signal —
was the gold answer ever proposed by any of the 17 agents? did discussion change the final
answer? — and (b) manual reading of the agents' elimination/rebuttal/decision reasoning traces
in `per_item.jsonl`.

## The single most important finding

**The bottleneck is selection, not generation.** In **16 of 28 failures (57%)**, at least one
agent *did* propose the correct answer, and the framework then discarded it. A naive
**majority vote of the agents' first answers scores 48.3% (29/60)**; MedLA's full pipeline
(elimination + logic + debate + arbitration) reaches **53.3%** — a **+5-point** gain for ~71
LLM calls per item. And the debate step itself is **barely net-positive**: it changed the final
answer on 18/60 items, ending **correct 10 / wrong 8** (net +2). Arbitration fired on **1/60**.

→ For a proposed framework, this is the lever: better **aggregation / answer-selection**, not
more agents or more debate rounds.

## Primary split

| | Count | Meaning |
|---|---|---|
| **A. Aggregation failure** — gold proposed but lost | 16 / 28 | the answer was "in the room"; the pipeline picked a wrong sibling |
| **B. Generation failure** — gold never proposed | 12 / 28 | no agent produced the gold entity at all |

## A. Aggregation failures (16) — "the answer was in the room"

All 16 involved agent disagreement and a discussion phase. Two mechanisms:

- **A1 — Discussion-induced degradation (6 items: basic:31, basic:104, inter:453, expert:26, expert:27, expert:446).**
  The debate step changed the answer and still missed the gold that was present initially.
  *Example — expert:26* ("conditions linked to femoral vein thrombophlebitis… hereditary"):
  initial agents split **9×D (gold) vs 8×A**; rebuttal argued the specific "iliac vein
  thrombophlebitis (A)" over the general "thrombophlebitis (D)", and the final flipped to **A (wrong)**.
  The debate talked the cohort *out of* the correct plurality.
- **A2 — Static selection of a wrong sibling (10 items).** No flip; when agents disagreed the
  elimination/consensus survivor was a plausible-but-wrong sibling.
  *Example — expert:328* ("gene not expressed in appendix AND medial globus pallidus"; options
  NOX5/NOXO1/NOX1/NOX4): agents scattered across all four; gold **A (NOX5)** was proposed by 3
  agents but the cohort settled on **D (NOX4)**.

## B. Generation failures (12) — gold never proposed

- **B1 — Knowledge-graph granularity / synonymy (≈11 items).** The option set is near-synonyms,
  ontology siblings, or an umbrella-vs-specific pair, and the gold is fixed by a specific
  knowledge-graph edge rather than by reasoning. The model reasons plausibly and picks a sibling.
  *Examples:*
  - inter:121 / inter:117 — gold is the **umbrella** term "gallbladder disease (C)"; every agent
    chose a *specific* stone ("choledocholithiasis (A)" / "cholelithiasis (D)").
  - expert:386 — gold "diabetes mellitus (disease) (B)" (umbrella); agents chose "type 2 diabetes (D)".
  - inter:631 — gold SERPINA6 (C) among the SERPINA1/3/5/6 family; expert:176 — gold "cerebellar
    vermis (A)" vs the whole "cerebellum (D)".
- **B2 — Specialized knowledge gap (subset of B1's hardest).** Obscure gene-expression-atlas or
  PPI facts no amount of reasoning recovers (e.g., NOX isoform expression across two specific
  tissues). Here the benchmark tests *memorized KG edges*, not clinical logic.

## Cross-cutting confound — benchmark label artifacts

Some "failures" are the model being **clinically right but marked wrong**.
*Clearest — basic:113* ("recommended treatments for head lice"): **all 17 agents unanimously
chose Permethrin (D)** — the standard first-line head-lice treatment — but the gold is
**Tetramethrin (B)**. This is a gold-label artifact, not a reasoning error. Because MedDDx options
are overwhelmingly near-synonyms whose "correct" answer is a database edge, **raw accuracy
understates the backbone's clinical/biological reasoning**, and the task rewards KG recall over logic.

## Where in the pipeline failures originate

| Locus | Role in the 28 failures |
|---|---|
| Logic tree / elimination reasoning | Produces plausible-but-wrong siblings (drives most of B). The per-step logic reports rarely flag the decisive error. |
| Discussion / debate | **Net-neutral-to-harmful**: 10 helped, 8 hurt across the run; flipped away from a present gold in 6 failures (A1). |
| Arbitration (DecisionMakersAgent) | **Essentially inert** — fired once in 60 items; not a failure locus, but contributes nothing. |
| Backbone knowledge | Hard limit on B2 (obscure KG facts); not fixable by orchestration. |

## Quantitative summary

- Agents disagreed on **50/60** items (83%) — the near-synonym option design makes almost every item contested.
- Avg discussion rounds **1.08**; discussion changed the answer 18×, **10 right / 8 wrong**.
- Gold present-but-lost **16/28**; gold never proposed **12/28**.
- Majority-vote-of-first-answers baseline **48.3%** vs MedLA **53.3%** (+5.0).

## Top 3 failure modes by frequency

1. **Aggregation loses a correct candidate (16/28)** — selection/debate discards a gold answer that was proposed.
2. **KG granularity/synonymy generation miss (≈11/28)** — gold is an ontology sibling/umbrella the agents never produce.
3. **Discussion-induced degradation (6/28)** — the debate step actively flips a correct plurality to a wrong answer.

## Implication for a proposed framework

The data points one way: MedLA spends ~71 calls/item but its debate/arbitration layer adds only
+5 points over majority vote and sometimes *degrades* correct answers. The highest-value
intervention is a **principled answer-selection/aggregation mechanism** (e.g., confidence- or
evidence-weighted selection that cannot drop a well-supported candidate), plus **synonym/ontology
normalization** so near-duplicate options stop splitting the vote. More agents, more rounds, or the
current arbitration are not where the points are.
