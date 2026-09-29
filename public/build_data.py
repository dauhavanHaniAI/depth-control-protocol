"""Build disjoint calibration / evaluation splits for the public-model DCP runs.

math : MATH test set (problem + worked solution); only solution tokens are scored.
web  : FineWeb sample-10BT documents; all tokens are scored.
Each domain yields 200 evaluation and 200 calibration documents with no overlap.
"""
import json, random
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).parent / "data"
rng = random.Random(20260923)

subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
rows = []
for s in subjects:
    for r in load_dataset("EleutherAI/hendrycks_math", s, split="test"):
        if 400 <= len(r["solution"]) <= 3000:
            rows.append({"prompt": f"Problem: {r['problem']}\nSolution:",
                         "text": " " + r["solution"], "level": r["level"], "subject": s})
rng.shuffle(rows)
math = rows[:400]

web = []
for r in load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True):
    t = r["text"]
    if 1500 <= len(t) <= 8000 and r.get("language", "en") == "en":
        web.append({"prompt": "", "text": t[:4000]})
    if len(web) >= 1200:
        break
rng.shuffle(web)
web = web[:400]

for name, docs in (("math", math), ("web", web)):
    for split, part in (("eval", docs[:200]), ("cal", docs[200:400])):
        with open(OUT / f"{name}_{split}.jsonl", "w") as f:
            for d in part:
                f.write(json.dumps(d) + "\n")
        print(name, split, len(part))
