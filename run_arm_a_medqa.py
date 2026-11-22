"""MedQA-US Arm A: single call, chain-of-thought, on the fixed 60-item sample.

Same model/temp/seed and the same CoT prompt used for the MedDDx S2 arm, so the
MedQA arms are directly comparable. Shares the cumulative guardrail ledger.
Run this only when no other ledger-writing process is active (avoid two
processes appending cost_ledger.csv at once).
"""
import json, os
from dotenv import load_dotenv
load_dotenv("/home/saleha/Documents/RAship/week4/MedLA/.env")
from openai import OpenAI
from util.llm_infra import guarded_create, set_item, running_total
from util.load_data import load_data_medqa_us
from util.agent import parse_content, find_answer

MODEL = "gpt-4o-mini-2024-07-18"
SYS = "You are a medical expert. Answer the question based on the knowledge you have."
COT = '''You must choose one of these options as the answer.
Question: {q}
Option: {o}
Let's think step by step: reason through each option, then give your final choice.
End your response with the format below:
<Answer>Answer: A/B/C/D/E</Answer>.
(don't forget the <Answer> </Answer> tags.)'''

def extract(c):
    try:
        return parse_content(c)
    except Exception:
        return find_answer(c or "") or ""

if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qa, _ = load_data_medqa_us()
    spec = json.load(open("sample_ids_medqa.json"))
    print(f"Arm A (MedQA CoT): {len(spec['ids'])} items; spend before ${running_total():.4f}")
    recs = []
    for iid in spec["ids"]:
        it = qa[int(iid.split(":")[1])]
        set_item(f"A|{iid}")
        msg = [{"role": "system", "content": SYS},
               {"role": "user", "content": COT.format(q=it["question"], o=str(it["options"]))}]
        out = guarded_create(client=client, provider="openai", model=MODEL, messages=msg,
                             temperature=0.0, max_tokens=1536, seed=42, max_retries=2)
        pred = extract(out)
        recs.append({"item_id": iid, "gold": it["answer_idx"], "predicted": pred,
                     "correct": int(pred == it["answer_idx"]), "arm": "A_cot"})
    with open("arm_a_medqa.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tot = sum(r["correct"] for r in recs); n = len(recs)
    unparse = sum(1 for r in recs if r["predicted"] == "")
    print(f"Arm A (MedQA, single CoT): {tot}/{n} = {100*tot/n:.1f}%  (unparseable={unparse})")
    print(f"cumulative spend: ${running_total():.4f}")
