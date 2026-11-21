import argparse
from datetime import datetime
from loguru import logger
import multiprocessing
import sys
import os
import pandas as pd
import json

import wandb
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
# from search.utils.utils import get_args
from random import randint

load_dotenv()

from util.agent import (
    LogicAgent,
    MedAgent_Eliminate,
    DecisionMakersAgent,
    Agent,
    parse_content
)
from util.base import base
from util.llm_infra import set_item, item_summary, BudgetExceeded, running_total
from util.util import alpha_lower, find_answer, check_answer_same, cal_acc
from util.prompt import baseline_template, EXCEPTION_PROMPT, simple_answer_constrain_prompt


def agents_talk(agent_dict, index, retry_count=[0]):

    last_ans_reason_dict = {}
    ans_dict_angent = {}
    for agent_name, agent in agent_dict.items():
        
        if agent.ans in ans_dict_angent:
            ans_dict_angent[agent.ans].append(agent_name)
        else:
            ans_dict_angent[agent.ans] = [agent_name]
        
        last_ans_reason_dict[agent_name] = {
            'name': agent.agent_name, 
            'ans': agent.ans, 
            'reason': agent.reason + 'logic_report' + agent.logic_report}
    
    random_select_agent = []
    for ans, agent_name_list in ans_dict_angent.items():
        if len(agent_name_list) > 1:
            select_agent_index = randint(0, len(agent_name_list) - 1)
            random_select_agent.append(agent_name_list[select_agent_index])
        else: 
            random_select_agent.append(agent_name_list[0])
    
        
    # import pdb; pdb.set_trace()
    agent_dict_new = {}
    for agent_name in random_select_agent:
        agent_dict_new[agent_name] = agent_dict[agent_name]
    
    # Rebuttals within a round are independent across agents -> run concurrently.
    def _rebut(agent_name, agent):
        agent_ans = agent.ans
        ans, reason_str = agent.rebuttal(last_ans_reason_dict, retry_count=retry_count)
        logger.warning(f"problem: {index}: discussion rebuttal {agent_name}:{agent_ans}->{ans}")

    if agent_dict_new:
        with ThreadPoolExecutor(max_workers=len(agent_dict_new)) as ex:
            futs = [ex.submit(_rebut, n, a) for n, a in agent_dict_new.items()]
            for f in futs:
                f.result()  # propagate exceptions, incl. BudgetExceeded
    return agent_dict_new


def save_log(
    final_decision, label, question, log_list, data_index, output_file, debug=False,retry_count=0,retry_pro=0,make_decision=0,make_decision_num=[0]):
    json.dump(
        {
            "final_decision": final_decision,
            "label": label,
            "question": question,
        },
        open(output_file, "+a"),
    )
    make_decision_num[0] += make_decision
    num_right, numn_total, acc = cal_acc(open(output_file, "r").readline())
    right_str = "Right" if final_decision == label else "Wrong"
    
    log = {
            "accuracy": acc,
            "right_bool": int(final_decision == label),
            "num_right": num_right,
            "numn_total": numn_total,
            "talkTimes": len(log_list),
            "data_index": data_index,
            "retry_count": retry_count,
            "retry_pro": retry_pro/numn_total,
            "make_decision": make_decision,
            "make_decision_num": make_decision_num[0]/numn_total,
    }
    
    if data_index < 10 or final_decision != label:
        log.update({
            f"{right_str}/talkTimes{len(log_list)}/log_{data_index}_predict{final_decision}_label{label}": pd.DataFrame(
                log_list
            ),
        })
    
    wandb.log(
        log
    )


def run_single_exp(question, fewshots_data, args, label, problem_index, output_path,retry_all=[0],make_decision_num=[0], item_id=None, tier=None):
    set_item(item_id)  # attribute every LLM call of this item in the ledger
    retry_count=[ 0]
    log_list = []
    final_decision = None

    agents_data = []
    for index_agent, agent in enumerate(fewshots_data):
        agents_data.append(f"Agent {index_agent}")

    agent_dict = {}
    for index_agent, agent in enumerate(agents_data):
        agent_role = agent
        agent_name = f"[{index_agent}]{alpha_lower(agent_role)}"
        inst_prompt = f"you are {agent_name}. "
        
        agent_new = MedAgent_Eliminate(
            args=args,
            instruction=inst_prompt,
            agent_name=agent_name,
            logger=log_list,
        )
        # logging.info(agent_new.temp)
        agent_dict[agent_name] = agent_new

    # The N agents' answer+logic calls are independent -> run them concurrently.
    # Each worker uses its OWN LogicAgent so the shared instance isn't raced.
    def _first_answer(agent_idx, agent_name, agent_current):
        agent_current.answer_question(
            question, fewshots_data[agent_idx],
            baseline_bool=args.baseline_bool, retry_count=retry_count)
        local_logic = LogicAgent(
            args=args, instruction='you are a logic agent.',
            agent_name='LogicAgent', logger=log_list)
        agent_current.logic_report = local_logic.check_logic(
            agent_current.reason, retry_count=retry_count)
        logger.warning(
            f"problem: {problem_index}: first ans, {agent_name}({agent_current.ans})")

    agent_items = list(agent_dict.items())
    with ThreadPoolExecutor(max_workers=len(agent_items)) as ex:
        futs = [ex.submit(_first_answer, i, name, agent)
                for i, (name, agent) in enumerate(agent_items)]
        for f in futs:
            f.result()  # propagate exceptions, incl. BudgetExceeded
    initial_answers = [agent_dict[n].ans for n in agent_dict.keys()]
    first_answer = initial_answers[0] if initial_answers else None
    rounds_run = 0
    num_rounds = args.num_rounds
    for round_n in range(num_rounds):
        ans_list_c = [agent_dict[agent_name].ans for agent_name in agent_dict.keys()]
        bool_ans_same = check_answer_same(ans_list_c)

        if not bool_ans_same:
            agent_dict = agents_talk(agent_dict, problem_index,retry_count=retry_count)
            rounds_run += 1
        else:
            break
        logger.warning(
            f"problem: {problem_index}: discusstion, Round {round_n}"
            + f"({[agent_dict[agent_name].ans for agent_name in agent_dict.keys()]})"
        )
    making_decision = 0
    ans_list_c = [agent_dict[agent_name].ans for agent_name in agent_dict.keys()]
    bool_ans_same = check_answer_same(ans_list_c)
    if not bool_ans_same:
        moderator = DecisionMakersAgent(
            args=args,
            logger=log_list,
        )
        final_decision = moderator.make_decision(agent_dict, question, num_rounds,retry_count=retry_count)
        making_decision = 1
    else:
        final_decision = ans_list_c[0]
    logger.warning(f"problem: {problem_index}: final_decision {final_decision}")
    retry_all[0]+= retry_count[0]
    save_log(
        final_decision,
        label,
        question,
        log_list,
        problem_index,
        output_path,
        args.debug,
        retry_count=retry_count[0],
        retry_pro=retry_all[0],
        make_decision=making_decision,
        make_decision_num=make_decision_num,
    )

    # per-item deliverable record (checkpointed immediately after each item)
    try:
        trimmed = [
            {"agent": e.get("agent_name"), "step": e.get("step"),
             "answer": e.get("answer"), "output": e.get("info")}
            for e in log_list
        ]
        rec = {
            "item_id": item_id, "tier": tier,
            "question": question[0], "options": question[1],
            "gold": label, "predicted": final_decision,
            "correct": int(final_decision == label),
            "num_agents": len(initial_answers),
            "agents_after_discussion": len(agent_dict),
            "initial_answers": initial_answers,
            "first_agent_answer": first_answer,
            "agents_disagreed": int(len(set(initial_answers)) > 1),
            "rounds_run": rounds_run,
            "made_arbitration": making_decision,
            "discussion_changed_answer": int(final_decision != first_answer),
            "cost": item_summary(item_id),
            "logic_tree": trimmed,
        }
        per_item_path = getattr(args, "per_item", None) or "per_item.jsonl"
        with open(per_item_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"per_item write failed for {item_id}: {e}")

    return log_list, final_decision

def run_single_basiline(question, fewshots_data, args, label, problem_index, output_path,retry_all=[0],make_decision_num=[0]):
    log_list = []
    agent = Agent(
        args=args,
        system_instruction="You are a medical expert. Answer the question based on the knowledge you have.",
        agent_name='BaselineAgent',
        logger=log_list,
    )
    
    message = baseline_template.format(
        question=question[0],
        option=question[1],
    )
    retry_count = 0
    while 1:
        try:
            if retry_count > 3:
                raise Exception("Retry limit reached. Exiting...")
            opinion = agent.chat(message)
            final_decision = parse_content(opinion)
            break
        except (ValueError, IndexError) as e:
            logger.warning(
                f"{agent.agent_name} retry {retry_count}: parse error - {str(e)}; opinion: {opinion}"
            )
            message = EXCEPTION_PROMPT + simple_answer_constrain_prompt
            retry_count += 1
    
    save_log(
        final_decision,
        label,
        question,
        log_list,
        problem_index,
        output_path,
        args.debug,
        retry_count=0,
        retry_pro=0,
        make_decision=0,
        make_decision_num=[0],
    )


def cot_rag(args, wandb_logger):
    instance = base(args, wandb_logger, "test")
    if args.baseline_bool:
        instance.invoke(run_single_basiline)
    else:
        instance.invoke(run_single_exp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="medddx_expert", choices=[
        'medqa_us', 'mmlu', 'bioasq',
        "medddx_basic", "medddx_inter", "medddx_expert",])
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek",
        choices=[
            "vllm",
            "siliconflow",
            "openai",
            "deepseek",
            "zhipuai",
            "groq",
        ],
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="deepseek-chat",
        choices=[
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "qwq10k",
            "gpt-4o-mini-2024-07-18",
            "deepseek-chat",
            "deepseek-reasoner",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "DSR1_10k70b",
            "llama3370B13K",
            "llama3370B10K",
            "llama3.3_8192",
            "gpt-4o-mini",
            "llama3.3_4096",
            "DS-r1:70b",
            "qwen2.5:72b_6000",
            "llama3.3",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "llama3.1-8b",
            "llama-3.3-70b",
        ],
    )
    parser.add_argument("--url", type=int, default=None)
    
    parser.add_argument("--few_shots", type=int, default=1)
    parser.add_argument("--knowledges", type=int, default=0)
    parser.add_argument("--db_size", type=int, default=100)
    parser.add_argument("--num_agents", type=int, default=17)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_times", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--reasoning_effort", type=str, default=None)

    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--baseline_bool", type=int, default=0, choices=[1, 0])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--num_threads", type=int, default=10)
    parser.add_argument(
        "--tag", type=str, default="recover"
    )
    
    parser.add_argument("--wandb", type=str, default="MedLA")
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--sample_ids", type=str, default=None,
                        help="Path to sample_ids.json; selects the fixed stratified sample across MedDDx tiers.")
    parser.add_argument("--per_item", type=str, default="per_item.jsonl",
                        help="Path to append one JSON record per evaluated item.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Provider switch: read from env (PROVIDER=groq|openai), default groq.
    # Overrides --model so one env var flips the whole run between arms.
    args.model = os.environ.get("PROVIDER", args.model if args.model else "groq")

    args.num_threads = int(os.environ.get("ADAMA_NUM_THREADS", args.num_threads))
    
    if args.model == "vllm":
        if os.environ.get("ADAMA_PORT", args.url) is None:
            raise ValueError("Please set ADAMA_PORT environment variable or pass --url argument for vllm model.")
        else:
            try:
                args.url = f"http://127.0.0.1:{int(os.environ.get("ADAMA_PORT"))}"
            except:
                args.url = os.environ.get("ADAMA_PORT")

    if args.debug:
        args.num_samples = 1
        args.times = 1
        args.num_threads = 1

    return args


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    
    # print the command line arguments
    print("Command line arguments:", sys.argv)
    print("Number of arguments:", len(sys.argv))
    print("Arguments:", sys.argv[1:])

    args = get_args()

    # Seed RNG for reproducibility (few-shot selection in rag.py uses numpy;
    # option shuffling in agent.py uses random). temperature=0 + these seeds
    # make a single-threaded run deterministic and cache-friendly.
    import random as _random
    try:
        import numpy as _np
        _np.random.seed(int(args.seed))
    except Exception:
        pass
    _random.seed(int(args.seed))

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb_logger = wandb.init(
        project=args.wandb,
        name=f"{args.model}_{args.dataset}_{args.tag}_{time_str}",
        config=args,
    )

    try:
        cot_rag(args, wandb_logger)
    except BudgetExceeded as e:
        logger.error(f"HARD BUDGET ABORT: {e}. "
                     f"Partial results are checkpointed in {args.per_item} and the ledger.")
    finally:
        logger.info(f"Cumulative spend this project: ${running_total():.4f}")
        wandb.finish()
