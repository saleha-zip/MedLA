"""Arm C (full MedLA) scale runner — resumable, item-concurrent, 429-aware.

Overnight-safe:
  * RESUME: items already in PER_ITEM are skipped (no re-pay). A crash mid-item
    leaves no record for that item, so it re-runs from the shared cache (free).
  * ITEM CONCURRENCY: CONC items in flight, each with its own 17-agent pool.
  * BUDGET: llm_infra aborts at BUDGET_USD; in-flight items may be lost but are
    cached; completed items are intact on disk.
  * PROGRESS: a line appended to PROGRESS every 25 completed items.

Per-item cost is NOT written live (it is recoverable from the ledger by
item_id); enrich post-hoc for the report.

Env: SAMPLE, PER_ITEM, TRACES, PROGRESS, CONC, MAX_ITEMS (0=all),
and llm_infra's LEDGER_PATH / CACHE_DIR / BUDGET_USD.
"""
import json, os, time, threading, hashlib
import numpy as np
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
load_dotenv("/home/saleha/Documents/RAship/week4/MedLA/.env")

from util.load_data import load_data_medqa_us
from util.rag import QADB
from util.util import alpha_lower, check_answer_same
from util.agent import MedAgent_Eliminate, LogicAgent, DecisionMakersAgent
from util.llm_infra import running_total, BudgetExceeded, rate_limit_count
from main import agents_talk  # concurrent rebuttals, deterministic representative

SAMPLE   = os.getenv("SAMPLE",   "sample_ids_medqa_800.json")
PER_ITEM = os.getenv("PER_ITEM", "per_item_medqa_800.jsonl")
TRACES   = os.getenv("TRACES",   "traces_wrong_800.jsonl")
PROGRESS = os.getenv("PROGRESS", "progress_800.log")
CONC     = int(os.getenv("CONC", "4"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "0"))

ARGS = SimpleNamespace(model="openai", llm_name="gpt-4o-mini-2024-07-18", url=None,
                       max_tokens=1536, reasoning_effort=None, seed="42", debug=False,
                       temp=0.0, few_shots=1, knowledges=0, baseline_bool=0,
                       num_rounds=3, num_agents=17, dataset="medqa_us")

# qdb.get_few_shot() uses global NumPy randomness; seed it per-item (under a
# lock) so every item's prompts are deterministic regardless of thread order.
# This makes the cache stable -> a re-run of any item (incl. in-flight at a
# crash) is a free cache hit, so resume never re-pays.
_NP_LOCK = threading.Lock()

def _few_shot_for(item_id, qdb):
    with _NP_LOCK:
        np.random.seed(int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16))
        return qdb.get_few_shot(ARGS.few_shots, ARGS.num_agents)


def process_item(item, qdb):
    """Full MedLA pipeline for one item (mirrors main.run_single_exp, minus
    wandb/save_log). Returns (compact_record, wrong_trace_or_None)."""
    item_id = item["item_id"]
    question = (item["question"], item["options"])
    label = item["answer_idx"]
    fewshots_data = _few_shot_for(item_id, qdb)
    log_list = []

    agent_dict = {}
    for i in range(ARGS.num_agents):
        name = f"[{i}]{alpha_lower(f'Agent {i}')}"
        a = MedAgent_Eliminate(args=ARGS, instruction=f"you are {name}. ",
                               agent_name=name, logger=log_list, agent_index=i)
        a.item_id = item_id
        agent_dict[name] = a

    def first_answer(idx, agent):
        agent.answer_question(question, fewshots_data[idx],
                              baseline_bool=ARGS.baseline_bool, retry_count=[0])
        la = LogicAgent(args=ARGS, instruction='you are a logic agent.',
                        agent_name='LogicAgent', logger=log_list)
        la.item_id = item_id
        agent.logic_report = la.check_logic(agent.reason, retry_count=[0])

    pairs = list(agent_dict.items())
    with ThreadPoolExecutor(max_workers=len(pairs)) as ex:
        for f in [ex.submit(first_answer, i, a) for i, (n, a) in enumerate(pairs)]:
            f.result()  # propagate exceptions incl. BudgetExceeded

    initial_answers = [agent_dict[n].ans for n in agent_dict]
    first_val = initial_answers[0] if initial_answers else None
    rounds_run = 0
    for _ in range(ARGS.num_rounds):
        if check_answer_same([agent_dict[n].ans for n in agent_dict]):
            break
        agent_dict = agents_talk(agent_dict, item_id)
        rounds_run += 1

    making_decision = 0
    ans_list = [agent_dict[n].ans for n in agent_dict]
    if check_answer_same(ans_list):
        final = ans_list[0]
    else:
        mod = DecisionMakersAgent(args=ARGS, logger=log_list)
        mod.item_id = item_id
        final = mod.make_decision(agent_dict, question, ARGS.num_rounds, retry_count=[0])
        making_decision = 1

    rec = {"item_id": item_id, "tier": "medqa_us", "gold": label, "predicted": final,
           "correct": int(final == label), "num_agents": len(initial_answers),
           "agents_after_discussion": len(agent_dict), "initial_answers": initial_answers,
           "first_agent_answer": first_val,
           "agents_disagreed": int(len(set(initial_answers)) > 1),
           "rounds_run": rounds_run, "made_arbitration": making_decision,
           "discussion_changed_answer": int(final != first_val)}
    trace = None
    if not rec["correct"]:
        trimmed = [{"agent": e.get("agent_name"), "step": e.get("step"),
                    "answer": e.get("answer"), "output": e.get("info")} for e in log_list]
        trace = {"item_id": item_id, "question": item["question"], "options": item["options"],
                 "gold": label, "predicted": final, "initial_answers": initial_answers,
                 "logic_tree": trimmed}
    return rec, trace


def main():
    sample = json.load(open(SAMPLE))["ids"]
    completed = set()
    if os.path.exists(PER_ITEM):
        for l in open(PER_ITEM):
            try:
                completed.add(json.loads(l)["item_id"])
            except Exception:
                pass
    mq, _ = load_data_medqa_us()
    by_id = {f"medqa_us:{i}": mq[i] for i in range(len(mq))}
    todo = [{"item_id": iid, "question": by_id[iid]["question"],
             "options": by_id[iid]["options"], "answer_idx": by_id[iid]["answer_idx"]}
            for iid in sample if iid not in completed]
    if MAX_ITEMS:
        todo = todo[:MAX_ITEMS]

    qdb = QADB(db_size=100)
    start_completed = len(completed)
    print(f"sample={len(sample)} completed={start_completed} todo={len(todo)} CONC={CONC}")
    t0 = time.time()
    lock = threading.Lock()
    stop = threading.Event()
    done = [start_completed]
    processed = [0]

    def work(it):
        if stop.is_set():
            return None
        return process_item(it, qdb)

    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except BudgetExceeded as e:
                print("BUDGET ABORT:", e)
                stop.set()
                continue
            except Exception as e:
                print(f"item {futs[fut]['item_id']} failed: {type(e).__name__}: {e}")
                continue
            if res is None:
                continue
            rec, trace = res
            with lock:
                with open(PER_ITEM, "a") as f:
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
                if trace:
                    with open(TRACES, "a") as f:
                        f.write(json.dumps(trace) + "\n")
                done[0] += 1
                processed[0] += 1
                if done[0] % 25 == 0 or done[0] == len(sample):
                    el = time.time() - t0
                    tot = running_total()
                    rate = tot / processed[0] if processed[0] else 0.0
                    proj = tot + rate * (len(todo) - processed[0])
                    with open(PROGRESS, "a") as pf:
                        pf.write(f"{datetime.now().isoformat()} done={done[0]}/{len(sample)} "
                                 f"new={processed[0]}/{len(todo)} cost=${tot:.4f} "
                                 f"elapsed={el:.0f}s 429s={rate_limit_count()} "
                                 f"proj_run_total=${proj:.2f}\n")
                        pf.flush()
    tot = running_total()
    print(f"DONE done={done[0]}/{len(sample)} new_this_run={processed[0]} "
          f"cost=${tot:.4f} 429s={rate_limit_count()} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
