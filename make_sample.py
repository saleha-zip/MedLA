"""Draw the FIXED stratified sample for the MedLA reproduction.

Stratified random sample, seed 42, equal allocation across the three MedDDx
difficulty tiers. Item IDs are stable (tier + deterministic file-order index
from util.load_data.load_data_medddx), so every run -- Groq, paid, our own
framework, evaluation -- uses exactly this sample. Never resample.

Writes sample_ids.json. Re-running reproduces the identical set.
"""

import json
import random

from util.load_data import load_data_medddx

# tier label -> the `seed` arg load_data_medddx expects ("hard" == expert)
TIERS = {"basic": "basic", "inter": "inter", "expert": "hard"}
PER_TIER = 20
SEED = 42


def main():
    out = {"seed": SEED, "per_tier": PER_TIER, "tiers": list(TIERS), "ids": [], "by_tier": {}}
    for tier, load_key in TIERS.items():
        qa, _ = load_data_medddx(load_key)
        n = len(qa)
        # per-tier RNG derived from the global seed -> stable, tier-independent
        rng = random.Random(f"{SEED}-{tier}")
        picks = sorted(rng.sample(range(n), PER_TIER))
        ids = [f"medddx_{tier}:{i}" for i in picks]
        out["by_tier"][tier] = {"n_available": n, "ids": ids}
        out["ids"].extend(ids)
    with open("sample_ids.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote sample_ids.json: {len(out['ids'])} ids "
          f"({PER_TIER}/tier x {len(TIERS)} tiers, seed {SEED})")
    for tier, d in out["by_tier"].items():
        print(f"  {tier}: {len(d['ids'])} of {d['n_available']} -> {d['ids'][:3]} ...")


if __name__ == "__main__":
    main()
