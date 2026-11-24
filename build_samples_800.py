"""Build nested MedQA-US samples so 60 ⊂ 800 ⊂ 1273.

- sample_ids_medqa_800.json: the existing 60 ids (original order) + 740 more
  drawn seed-42 without replacement from the remaining 1,213, in draw order.
- sample_ids_medqa_full.json: the 800 above, then the remaining 473 in the
  same seed-42 draw order -> extending later is an increment, not a re-run.
"""
import json, random
from util.load_data import load_data_medqa_us

SEED = 42

def main():
    qa, _ = load_data_medqa_us()
    n = len(qa)                      # 1273
    orig = json.load(open("sample_ids_medqa.json"))["ids"]   # 60, keep order
    orig_set = set(orig)
    remaining = [f"medqa_us:{i}" for i in range(n) if f"medqa_us:{i}" not in orig_set]
    assert len(remaining) == n - len(orig), (len(remaining), n, len(orig))

    rng = random.Random(SEED)
    perm = rng.sample(remaining, len(remaining))   # seed-42 permutation of the 1213
    ext740 = perm[:740]
    ext_rest = perm[740:]                           # 473

    ids800 = orig + ext740
    ids_full = ids800 + ext_rest

    # invariants
    assert len(ids800) == 800 and len(set(ids800)) == 800
    assert len(ids_full) == n and len(set(ids_full)) == n
    assert orig == ids800[:60]                       # original 60 first, in order
    assert orig_set.issubset(set(ids800))            # strict subset
    assert ids800 == ids_full[:800]                  # 800 is a prefix of full

    json.dump({"seed": SEED, "dataset": "medqa_us", "n": 800,
               "nests_from": "sample_ids_medqa.json (first 60)", "ids": ids800},
              open("sample_ids_medqa_800.json", "w"), indent=2)
    json.dump({"seed": SEED, "dataset": "medqa_us", "n": n,
               "nests_from": "sample_ids_medqa_800.json (first 800)", "ids": ids_full},
              open("sample_ids_medqa_full.json", "w"), indent=2)

    print(f"800: {len(ids800)} ids | full: {len(ids_full)} ids")
    print(f"original 60 are a strict subset of 800: {orig_set.issubset(set(ids800))}")
    print(f"original 60 == first 60 of 800 (same order): {orig == ids800[:60]}")
    print(f"800 == first 800 of full: {ids800 == ids_full[:800]}")
    print(f"first new id (#61): {ids800[60]} | last 800 id: {ids800[-1]}")

if __name__ == "__main__":
    main()
