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

# --- paths & config (env-overridable so parallel runs can use a dedicated
#     ledger while SHARING the cache dir for free cross-run cache hits) --------
CACHE_DIR = os.getenv("CACHE_DIR") or os.path.join(os.getcwd(), "cache")
LEDGER_PATH = os.getenv("LEDGER_PATH") or os.path.join(os.getcwd(), "cost_ledger.csv")
BUDGET_USD = float(os.getenv("BUDGET_USD", "4.00"))

# Count of 429 / rate-limit waits across the process (not errors, just waits).
_rate_limit_hits = {"n": 0}


def rate_limit_count():
    return _rate_limit_hits["n"]

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
    "timestamp", "item_id", "provider", "model",
    "prompt_tokens", "completion_tokens",
    "cost_usd", "cache_hit", "cumulative_usd",
]

# Which item (question) the current call belongs to, for per-item attribution.
# Set by the orchestrator before each item; single-thread runs make this safe.
_current = {"item_id": None}


def set_item(item_id):
    _current["item_id"] = item_id


def item_summary(item_id):
    """Aggregate ledger rows for one item: calls, tokens, cost, cache hits."""
    s = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0,
         "completion_tokens": 0, "cost_usd": 0.0}
    if not os.path.exists(LEDGER_PATH):
        return s
    with open(LEDGER_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("item_id") != str(item_id):
                continue
            s["calls"] += 1
            s["cache_hits"] += 1 if row.get("cache_hit") == "True" else 0
            s["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
            s["completion_tokens"] += int(row.get("completion_tokens") or 0)
            s["cost_usd"] += float(row.get("cost_usd") or 0.0)
    s["cost_usd"] = round(s["cost_usd"], 6)
    return s


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


try:
    from openai import RateLimitError as _RateLimitError
except Exception:  # pragma: no cover
    class _RateLimitError(Exception):
        pass

# Cap on rate-limit (429) waits per call so a stuck quota can't loop forever.
_MAX_RATELIMIT_WAITS = 15


def guarded_create(client, provider, model, messages, temperature, max_tokens,
                   extra_kwargs=None, seed=42, max_retries=2, item_id=None):
    """Cache-first, budget-guarded chat-completion.

    item_id: attributes the call in the ledger. Passed explicitly so it is
    correct under item-level concurrency (the module global _current is only a
    fallback for the legacy single-thread path).

    Returns the message content string, or None if all (non-rate-limit) retries
    failed. Raises BudgetExceeded when the budget ceiling has been hit.
    429/rate-limit waits do NOT consume the retry budget (capped separately).
    """
    extra_kwargs = extra_kwargs or {}
    iid = item_id if item_id is not None else _current["item_id"]
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
                "item_id": iid,
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

    call_kwargs = dict(model=model, messages=messages,
                       temperature=temperature, max_tokens=max_tokens)
    if provider in ("openai", "groq"):
        call_kwargs["seed"] = seed
    call_kwargs.update(extra_kwargs)

    # 3) call: bounded retries for real errors; unbounded-but-capped backoff
    #    for 429s (rate-limit waits do not consume the retry budget).
    attempt = 0          # counts non-rate-limit attempts against max_retries
    rl_waits = 0         # counts rate-limit waits against _MAX_RATELIMIT_WAITS
    while attempt <= max_retries:
        try:
            resp = client.chat.completions.create(**call_kwargs)
        except _RateLimitError as e:
            with _lock:
                _rate_limit_hits["n"] += 1
            rl_waits += 1
            if rl_waits > _MAX_RATELIMIT_WAITS:
                logger.error(f"Rate-limit waits exhausted ({rl_waits}) for {model}; moving on.")
                return None
            wait = min(60, 2 ** min(rl_waits, 6) * 2)  # 4,8,16,32,60,60,...
            retry_after = getattr(getattr(e, "response", None), "headers", {})
            try:
                wait = max(wait, float(retry_after.get("retry-after", 0)))
            except Exception:
                pass
            logger.warning(f"429 rate limit (wait #{rl_waits}, {wait:.0f}s) for {model}")
            time.sleep(wait)
            continue  # does NOT increment attempt
        except Exception as e:
            wait = 2 ** attempt * 5
            logger.warning(
                f"API error ({type(e).__name__}) attempt {attempt + 1}/"
                f"{max_retries + 1}, waiting {wait}s: {e}")
            time.sleep(wait)
            attempt += 1
            continue

        content = resp.choices[0].message.content
        finish = getattr(resp.choices[0], "finish_reason", None)
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
                "item_id": iid,
                "provider": provider, "model": model,
                "prompt_tokens": pt, "completion_tokens": ct,
                "cost_usd": round(cost, 6), "cache_hit": False,
                "cumulative_usd": round(cumulative, 6),
            })

        if content is not None:
            with open(cpath, "w") as f:
                json.dump({"content": content,
                           "usage": {"prompt_tokens": pt, "completion_tokens": ct},
                           "finish_reason": finish,
                           "model": model, "provider": provider}, f)

        if cumulative >= BUDGET_USD:
            raise BudgetExceeded(
                f"Budget ceiling reached after call: ${cumulative:.4f} >= ${BUDGET_USD:.2f}")

        if content is not None:
            return content
        attempt += 1
        time.sleep(2 ** attempt * 5)

    logger.error(f"All {max_retries + 1} attempts failed for model {model}; moving on.")
    return None
