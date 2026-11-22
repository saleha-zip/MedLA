"""Draw the FIXED MedQA-US sample (seed 42, 60 items, no stratification).

MedQA-US has no difficulty tiers, so this is a plain random sample over the
1,273 test items (stable file-order index). Writes sample_ids_medqa.json;
re-running reproduces the identical set.
"""
import json, random
from util.load_data import load_data_medqa_us

SEED = 42
N = 60

def main():
    qa, _ = load_data_medqa_us()
    rng = random.Random(SEED)
    picks = sorted(rng.sample(range(len(qa)), N))
    ids = [f"medqa_us:{i}" for i in picks]
    json.dump({"seed": SEED, "dataset": "medqa_us", "n_available": len(qa), "ids": ids},
              open("sample_ids_medqa.json", "w"), indent=2)
    print(f"Wrote sample_ids_medqa.json: {len(ids)} ids from {len(qa)} (seed {SEED})")
    print("first 5:", ids[:5])

if __name__ == "__main__":
    main()
