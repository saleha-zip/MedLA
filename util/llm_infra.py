"""Guardrail infrastructure for the MedLA paid-run reproduction.

Centralises everything that must hold for *every* LLM call, regardless of
provider or model:

  * on-disk response cache keyed by (model, prompt, temperature, max_tokens,
    extra) -> a cache hit never costs money;
  * a cost ledger (CSV) with one row per call and a running cumulative total;
  * a HARD budget abort: once cumulative spend >= BUDGET_USD the next paid
    call raises BudgetExceeded instead of spending more;
  * bounded retries (max_retries) with exponential backoff, then give up and
    return None so the caller logs and moves on (never retry indefinitely).

All state lives on disk (cache/ dir + cost_ledger.csv at the repo root), so a
crash or an abort loses nothing and a rerun resumes for free from the cache.
"""

import csv
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone

from loguru import logger

# --- paths & config (all relative to the process CWD = repo root) -----------
CACHE_DIR = os.path.join(os.getcwd(), "cache")
LEDGER_PATH = os.path.join(os.getcwd(), "cost_ledger.csv")
BUDGET_USD = float(os.getenv("BUDGET_USD", "4.00"))

# USD per 1,000,000 tokens as (input, output).
# Groq free-tier models -> 0. OpenAI prices MUST be re-confirmed live before
# the paid run (see GATE 2); these are placeholders pending that check.
PRICING = {
    # OpenAI (to be verified at GATE 2)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    # Groq free tier
    "openai/gpt-oss-120b": (0.0, 0.0),
    "openai/gpt-oss-20b": (0.0, 0.0),
    "qwen/qwen3.8-27b": (0.0, 0.0),
}

LEDGER_FIELDS = [
    "timestamp", "provider", "model",
    "prompt_tokens", "completion_tokens",
    "cost_usd", "cache_hit", "cumulative_usd",
]


class BudgetExceeded(Exception):
    """Raised when cumulative spend has reached BUDGET_USD."""


_lock = threading.Lock()
_total = {"usd": None}  # lazily initialised from the ledger on first use


def _price(model, prompt_tokens, completion_tokens):
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout


def _prompt_key(model, messages, temperature, max_tokens, extra):
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens, "extra": extra or {}},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key):
    return os.path.join(CACHE_DIR, key + ".json")


def _load_total_locked():
    """Populate the cached running total from the ledger once. Call under _lock."""
    if _total["usd"] is not None:
        return
    total = 0.0
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    total += float(row["cost_usd"])
                except (KeyError, ValueError):
                    pass
    _total["usd"] = total


def running_total():
    with _lock:
        _load_total_locked()
        return _total["usd"]


def _append_ledger_locked(row):
    """Append one ledger row and fsync. Call under _lock."""
    new_file = not os.path.exists(LEDGER_PATH)
    with open(LEDGER_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def guarded_create(client, provider, model, messages, temperature, max_tokens,
                   extra_kwargs=None, seed=42, max_retries=2):
    """Cache-first, budget-guarded chat-completion.

    Returns the message content string, or None if all attempts failed.
    Raises BudgetExceeded when the budget ceiling has been hit.
    """
    extra_kwargs = extra_kwargs or {}
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _prompt_key(model, messages, temperature, max_tokens, extra_kwargs)
    cpath = _cache_path(key)

    # 1) cache hit -> free, log with cache_hit=True and unchanged cumulative.
    if os.path.exists(cpath):
        with open(cpath) as f:
            cached = json.load(f)
        with _lock:
            _load_total_locked()
            _append_ledger_locked({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": provider, "model": model,
                "prompt_tokens": cached.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": cached.get("usage", {}).get("completion_tokens", 0),
                "cost_usd": 0.0, "cache_hit": True,
                "cumulative_usd": round(_total["usd"], 6),
            })
        return cached["content"]

    # 2) budget gate BEFORE spending.
    with _lock:
        _load_total_locked()
        if _total["usd"] >= BUDGET_USD:
            raise BudgetExceeded(
                f"Budget ceiling reached: ${_total['usd']:.4f} >= ${BUDGET_USD:.2f}")

    # 3) call with bounded retries + exponential backoff.
    call_kwargs = dict(model=model, messages=messages,
                       temperature=temperature, max_tokens=max_tokens)
    if provider in ("openai", "groq"):
        call_kwargs["seed"] = seed
    call_kwargs.update(extra_kwargs)

    for attempt in range(max_retries + 1):  # 1 initial try + max_retries retries
        try:
            resp = client.chat.completions.create(**call_kwargs)
        except Exception as e:
            wait = 2 ** attempt * 5
            logger.warning(
                f"API error ({type(e).__name__}) attempt {attempt + 1}/"
                f"{max_retries + 1}, waiting {wait}s: {e}")
            time.sleep(wait)
            continue

        content = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = _price(model, pt, ct)

        with _lock:
            _load_total_locked()
            _total["usd"] += cost
            cumulative = _total["usd"]
            _append_ledger_locked({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": provider, "model": model,
                "prompt_tokens": pt, "completion_tokens": ct,
                "cost_usd": round(cost, 6), "cache_hit": False,
                "cumulative_usd": round(cumulative, 6),
            })

        # cache only successful, non-empty content.
        if content is not None:
            with open(cpath, "w") as f:
                json.dump({"content": content,
                           "usage": {"prompt_tokens": pt, "completion_tokens": ct},
                           "model": model, "provider": provider}, f)

        # hard abort if this call pushed us to/over the ceiling.
        if cumulative >= BUDGET_USD:
            raise BudgetExceeded(
                f"Budget ceiling reached after call: ${cumulative:.4f} >= ${BUDGET_USD:.2f}")

        if content is not None:
            return content
        # empty content: fall through to retry
        time.sleep(2 ** attempt * 5)

    logger.error(f"All {max_retries + 1} attempts failed for model {model}; moving on.")
    return None
