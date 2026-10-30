# MedLA — Groq Setup & Change Log

## What this repo does

MedLA is a reproduction of the paper *"MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models"* ([arXiv:2509.23725](https://arxiv.org/abs/2509.23725)). It answers knowledge-intensive medical multiple-choice questions (datasets: MedDDx basic/inter/expert, and — when the MIRAGE data is present — MedQA, MMLU-medical, BioASQ) using a **collaborative multi-agent pipeline** rather than a single model call. For each question it spins up *N* `MedAgent_Eliminate` agents that each reason through the options and eliminate the least-likely ones; a `LogicAgent` audits every reasoning chain and emits a structured logic report; if the agents disagree they run several rounds of rebuttal/discussion (`agents_talk`); and if consensus still fails a `DecisionMakersAgent` arbitrates and picks the final answer. A single-agent `--baseline_bool 1` mode is provided for comparison. The entry point is `main.py`; data loading/orchestration is in `util/base.py` + `util/load_data.py`, the agent classes and model backends in `util/agent.py`, prompts in `util/prompt.py`, and few-shot logic exemplars in `util/rag.py`. Accuracy and debate metrics are logged to Weights & Biases, and per-question results are written to `output/`.

All work below was done to run the reproduction for free on **Groq** (an OpenAI-API-compatible provider). Changes were committed to `main` of the fork `saleha-zip/MedLA` as commit `0a1b875`.

---

## Environment setup (one-time)

| Command | Why |
|---|---|
| `source .venv/bin/activate` | Use the repo's existing virtualenv (deps already installed). |
| created `.env` with `API_KEY=<groq key>` and `WANDB_MODE=offline` | Keep the API key out of the shell/source, and run W&B offline (no `wandb login` needed). **`.env` is git-ignored and never committed.** |

---

## Code changes to the original code

### `main.py`
| Change | Why |
|---|---|
| `from dotenv import load_dotenv` + `load_dotenv()` near the top | Load `API_KEY` / `WANDB_MODE` from the git-ignored `.env` automatically, so vars don't have to be exported each run. |
| Added `"groq"` to the `--model` choices | Allow selecting the new Groq backend. |
| Added model IDs to `--llm_name` choices (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, …) | argparse rejects unknown model names, so the Groq model IDs had to be registered. |
| Added `--max_tokens` (default 8000) | Cap completion length to stay under Groq free-tier output-tokens-per-minute limits. |
| Added `--reasoning_effort` (default `None`) | Let reasoning models (gpt-oss) emit a final answer under a tight token budget; only sent to the API when explicitly set. |

### `util/agent.py`
| Change | Why |
|---|---|
| Added a `groq` branch in `AgentHelpler.__init__` (OpenAI client with `base_url="https://api.groq.com/openai/v1"`) | Groq is OpenAI-API-compatible, so it reuses the existing OpenAI client against Groq's endpoint. |
| `max_tokens=8000` → `max_tokens=getattr(self, "max_tokens", 8000)` in `invoke()` | Honor the new `--max_tokens` flag instead of a hardcoded value. |
| Built `extra_kwargs` to pass `reasoning_effort` only when set, and splatted it into the chat-completions call | Send `reasoning_effort` to models that support it (gpt-oss) while omitting it for models that reject it (e.g. Llama). |
| Set `self.max_tokens` and `self.reasoning_effort` in `Agent.__init__` from `args` | Thread the two new flags down to every agent instance. |
| Initialize `opinion`/`reason` before the retry loops in `check_logic`, `answer_question`, `rebuttal`, `make_decision` | A failed API call raises before those vars are assigned; the `except` handlers logged them → `UnboundLocalError` that crashed the whole run. Now a transient failure degrades gracefully instead of aborting. |

### `.gitignore`
| Change | Why |
|---|---|
| Added `.env` | Prevent the file holding the API key from being committed. |

### New (uncommitted) file
| File | Why |
|---|---|
| `.env` | Holds `API_KEY` and `WANDB_MODE=offline`; git-ignored, local only. |

> **Note — bundled pre-existing fixes:** commit `0a1b875` also includes fixes authored by the repo owner (saleha-zip) before this session, separate from the Groq work: an exponential-backoff retry wrapper around the API call in `invoke()`, a safer `getattr(AgentHelpler, "temp", 0.7)`, the `run_single_basiline` `Agent(...)` constructor call fix, the `--wandb` argument, the vllm `url.endswith` typo fix, and a `mkdir` so the debug output dir exists. They were committed together because they shared the working tree.

---

## Key commands run (diagnostics → working run)

| Command | Why / result |
|---|---|
| `python main.py --help` | Smoke-test that imports and argparse (incl. new flags) load. |
| `python main.py --dataset medddx_expert --model groq --llm_name llama-3.1-8b-instant --num_agents 3 --max_tokens 2048 --debug` | First attempt. **Failed: 404 model-not-found** — this Groq key has no Llama models. |
| `curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $KEY"` | Listed the key's real catalog: a **regional (Saudi/SDAIA) Groq deployment** — `allam-2-7b`, `orpheus-arabic-saudi`, `gpt-oss-*`, `qwen3.8-27b`, whisper, prompt-guard. **No general Llama.** |
| probe of `qwen/qwen3.8-27b` | Free-tier output limit is only **1000 OTPM**; too small, and it's a reasoning model. |
| probe of `openai/gpt-oss-120b` with `reasoning_effort=low` | **Works:** 8000 OTPM, returns clean `<Answer>`-tagged output. Chosen as the pipeline-proof model. |
| `python main.py --dataset medddx_expert --model groq --llm_name openai/gpt-oss-120b --reasoning_effort low --num_agents 3 --max_tokens 1500 --debug` | **✅ Full pipeline ran end-to-end**, 1/1 correct on the sample. This is the current verified command. |
| `git add .gitignore main.py util/agent.py && git commit && git push origin main` | Committed the changes to `main` of `saleha-zip/MedLA` (commit `0a1b875`). |

---

## Current status & next step

- **Verified working command** (regional Groq key, pipeline proof):
  ```bash
  python main.py --dataset medddx_expert --model groq --llm_name openai/gpt-oss-120b \
    --reasoning_effort low --num_agents 3 --max_tokens 1500 --debug
  ```
- **Known limitation:** the current Groq key is a regional catalog with **no Llama-3.1-8B**, so numbers from it are *not* directly comparable to the paper (which uses LLaMA-3.1-8B). A standard global `groq.com` key (or OpenRouter's free `llama-3.1-8b-instruct`) would be needed for a paper-comparable run.
- **Multi-sample results on gpt-oss-120b (MedDDx-expert, 3 agents, max_tokens 1500):**
  - First attempt (10 samples, **3 threads**) **crashed**: gpt-oss-120b's free limit is **8000 tokens/minute total**, so 3 threads constantly hit `429`, the retries exhausted into a `ValueError`, and the unguarded `except` handlers threw `UnboundLocalError` — aborting the run after only 2 samples. Fixed by (a) the `opinion`/`reason` guards above and (b) dropping to 1 thread.
  - Rerun (**5 samples, 1 thread**): completed cleanly — **3/5 = 60% accuracy**, 0 rate-limit errors, 0 crashes, ~1.8 min/sample. Output in `output/20261030/medddx_expert/`.
- **Lesson:** on this free tier `--num_threads 1` is required (8000 TPM is the bottleneck, not parallelism). More threads just waste time in `429` backoff.
- **Next:** these numbers are gpt-oss-120b, not the paper's LLaMA-3.1-8B, and n is small. For a comparable figure, obtain a Llama-3.1-8B source (standard global Groq key or OpenRouter) and rerun at 1 thread with a larger `--num_samples`.
