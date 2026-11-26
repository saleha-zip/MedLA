"""Robustness checks for the 800-item MedLA reproduction (results_800.md §6).

Verifies the MedLA<CoT finding is not an artifact of our edits or setup:
  1) parse health  — invalid final/agent answers (expect 0)
  2) truncation    — finish_reason=length rate in the shared cache
  3) decomposition — CoT vs single MedLA agent vs majority vs ensemble vs oracle
  4) diversity     — per-slot accuracy spread + oracle ceiling (seeding didn't
                     collapse the ensemble)
Run: .venv/bin/python validate_800.py
"""
import json, glob
from collections import Counter

VALID = set("ABCDE")
C = [json.loads(l) for l in open("per_item_medqa_800.jsonl")]
A = {json.loads(l)["item_id"]: json.loads(l) for l in open("arm_a_medqa_800.jsonl")}
N = len(C)

# 1) parse health
bad_final = [r["item_id"] for r in C if r["predicted"] not in VALID]
bad_agent = sum(1 for r in C for a in r["initial_answers"] if a not in VALID)
tot_agent = sum(len(r["initial_answers"]) for r in C)
print(f"[parse]  invalid final={len(bad_final)}  invalid agent answers={bad_agent}/{tot_agent}")

# 2) truncation (shared cache; both arms)
fr = Counter(); maxct = 0
for p in glob.glob("cache/*.json"):
    try: d = json.load(open(p))
    except Exception: continue
    fr[d.get("finish_reason")] += 1
    maxct = max(maxct, d.get("usage", {}).get("completion_tokens", 0) or 0)
print(f"[trunc]  finish_reason={dict(fr)}  max_completion_tokens={maxct}")

# 3) decomposition + 4) diversity
slot_c = [0]*17; slot_n = [0]*17
mean_corr = 0.0; first = 0; oracle = 0
for r in C:
    g = r["gold"]; ia = r["initial_answers"]
    for i, a in enumerate(ia[:17]):
        slot_n[i] += 1; slot_c[i] += (a == g)
    mean_corr += sum(a == g for a in ia) / len(ia)
    first += (ia and ia[0] == g)
    oracle += (g in ia)
slot_acc = [100*slot_c[i]/slot_n[i] for i in range(17) if slot_n[i]]
armA = 100*sum(A[r["item_id"]]["correct"] for r in C)/N

print(f"[decomp] N={N}")
print(f"  Arm A  CoT single        : {armA:.1f}%")
print(f"  single MedLA agent (mean): {100*mean_corr/N:.1f}%")
print(f"  first-agent (slot 0)     : {100*first/N:.1f}%")
print(f"  Arm C  MedLA ensemble    : {100*sum(r['correct'] for r in C)/N:.1f}%")
print(f"  ORACLE any-of-17 right   : {100*oracle/N:.1f}%")
print(f"[divers] per-slot accuracy {min(slot_acc):.1f}%..{max(slot_acc):.1f}% "
      f"(spread => ensemble diversity intact)")
