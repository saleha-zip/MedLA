"""Arm A (single CoT) on the 800-item MedQA sample — resumable + concurrent.

Reference arm; run FIRST. Shares the 800-run ledger (LEDGER_PATH) and cache dir.
Ledger item_id is prefixed "A|" to separate it from Arm C rows.
Env: SAMPLE, OUT, CONC, and llm_infra's LEDGER_PATH / CACHE_DIR / BUDGET_USD.
"""
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv("/home/saleha/Documents/RAship/week4/MedLA/.env")
from openai import OpenAI
from util.llm_infra import guarded_create, running_total, BudgetExceeded, rate_limit_count
from util.load_data import load_data_medqa_us
from util.agent import parse_content, find_answer

MODEL = "gpt-4o-mini-2024-07-18"
SAMPLE = os.getenv("SAMPLE", "sample_ids_medqa_800.json")
OUT = os.getenv("OUT", "arm_a_medqa_800.jsonl")
CONC = int(os.getenv("CONC", "8"))
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

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sample = json.load(open(SAMPLE))["ids"]
    completed = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                completed.add(json.loads(l)["item_id"])
            except Exception:
                pass
    mq, _ = load_data_medqa_us()
    by_id = {f"medqa_us:{i}": mq[i] for i in range(len(mq))}
    todo = [iid for iid in sample if iid not in completed]
    print(f"Arm A: sample={len(sample)} completed={len(completed)} todo={len(todo)} CONC={CONC}")
    lock = threading.Lock(); stop = threading.Event(); done = [len(completed)]

    def work(iid):
        if stop.is_set():
            return None
        it = by_id[iid]
        msg = [{"role": "system", "content": SYS},
               {"role": "user", "content": COT.format(q=it["question"], o=str(it["options"]))}]
        out = guarded_create(client=client, provider="openai", model=MODEL, messages=msg,
                             temperature=0.0, max_tokens=1536, seed=42, max_retries=2,
                             item_id=f"A|{iid}")
        pred = extract(out)
        return {"item_id": iid, "gold": it["answer_idx"], "predicted": pred,
                "correct": int(pred == it["answer_idx"]), "arm": "A_cot"}

    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = {ex.submit(work, iid): iid for iid in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except BudgetExceeded as e:
                print("BUDGET ABORT:", e); stop.set(); continue
            except Exception as e:
                print(f"item {futs[fut]} failed: {e}"); continue
            if rec is None:
                continue
            with lock:
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
                done[0] += 1
    recs = [json.loads(l) for l in open(OUT)]
    tot = sum(r["correct"] for r in recs); n = len(recs)
    print(f"Arm A done: {tot}/{n} = {100*tot/n:.1f}% | spend ${running_total():.4f} | 429s={rate_limit_count()}")

if __name__ == "__main__":
    main()
