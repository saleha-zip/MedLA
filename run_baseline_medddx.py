"""Single-agent baseline arms on the fixed 60-item MedDDx sample.

Arm S1: one call, direct answer, no CoT  (MedLA's own baseline_prompt).
Arm S2: one call, chain-of-thought then answer.

Same model (gpt-4o-mini-2024-07-18), temp 0, seed 42, same question formatting
MedLA gives its agents (Question: / Option: + <Answer>Answer: X</Answer>),
minus the multi-agent scaffolding. All calls go through the shared guardrail
ledger (cumulative budget continues; abort at BUDGET_USD).
"""
import json, os
from dotenv import load_dotenv
load_dotenv("/home/saleha/Documents/RAship/week4/MedLA/.env")
from openai import OpenAI
from util.llm_infra import guarded_create, set_item, running_total, item_summary
from util.load_data import load_data_medddx
from util.agent import parse_content, find_answer

MODEL = "gpt-4o-mini-2024-07-18"
TIERMAP = {"basic": "basic", "inter": "inter", "expert": "hard"}
SYS = "You are a medical expert. Answer the question based on the knowledge you have."

S1 = '''Answer my questions as requested without any other extraneous content. You must choose one of these options as the answer.
Question: {q}
Option: {o}
Follow the format below:
<Answer>Answer: A/B/C/D/E</Answer>.
(don't forget the <Answer> </Answer> tags.)'''

S2 = '''You must choose one of these options as the answer.
Question: {q}
Option: {o}
Let's think step by step: reason through each option, then give your final choice.
End your response with the format below:
<Answer>Answer: A/B/C/D/E</Answer>.
(don't forget the <Answer> </Answer> tags.)'''


def load_items():
    spec = json.load(open("sample_ids.json"))
    cache = {}
    def tier_items(t):
        if t not in cache: cache[t], _ = load_data_medddx(TIERMAP[t])
        return cache[t]
    out = []
    for iid in spec["ids"]:
        pre, idx = iid.split(":"); tier = pre[len("medddx_"):]
        it = dict(tier_items(tier)[int(idx)]); it["item_id"] = iid; it["tier"] = tier
        out.append(it)
    return out


def extract(content):
    try:
        return parse_content(content)
    except Exception:
        a = find_answer(content or "")
        return a or ""


def run_arm(client, items, arm, prompt, max_tokens):
    recs = []
    for it in items:
        set_item(f"{arm}|{it['item_id']}")
        msg = [{"role": "system", "content": SYS},
               {"role": "user", "content": prompt.format(q=it["question"], o=str(it["options"]))}]
        out = guarded_create(client=client, provider="openai", model=MODEL, messages=msg,
                             temperature=0.0, max_tokens=max_tokens, seed=42, max_retries=2)
        pred = extract(out)
        recs.append({"item_id": it["item_id"], "tier": it["tier"], "gold": it["answer_idx"],
                     "predicted": pred, "correct": int(pred == it["answer_idx"]), "arm": arm})
    return recs


def report(recs, arm):
    from collections import defaultdict
    t = defaultdict(lambda: [0, 0])
    for r in recs:
        t[r["tier"]][1] += 1; t[r["tier"]][0] += r["correct"]
    tot_c = sum(r["correct"] for r in recs); n = len(recs)
    line = {tier: f'{t[tier][0]}/{t[tier][1]}' for tier in ["basic", "inter", "expert"]}
    print(f"[{arm}] basic={line['basic']} inter={line['inter']} expert={line['expert']} "
          f"OVERALL={tot_c}/{n}={100*tot_c/n:.1f}%")
    return tot_c, n


if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    items = load_items()
    print(f"loaded {len(items)} items; spend before ${running_total():.4f}")
    all_recs = []
    all_recs += run_arm(client, items, "S1", S1, 64)
    print(f"S1 done; spend ${running_total():.4f}")
    all_recs += run_arm(client, items, "S2", S2, 1536)
    print(f"S2 done; spend ${running_total():.4f}")
    with open("baseline_medddx.jsonl", "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print("\n=== RESULTS (MedDDx, gpt-4o-mini, 60 items) ===")
    report([r for r in all_recs if r["arm"] == "S1"], "S1 direct ")
    report([r for r in all_recs if r["arm"] == "S2"], "S2 CoT    ")
    print(f"total cumulative spend: ${running_total():.4f}")
