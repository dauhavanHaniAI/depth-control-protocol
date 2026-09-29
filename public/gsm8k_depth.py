#!/usr/bin/env python3
"""gsm8k_depth.py — task-level check of DCP schedules on the public checkpoint of our architecture.

Greedy generation on the GSM8K test set under each execution schedule (full, repeat(4), repeat(1), prefix(4), ...),
batched with right padding (the model is causal, verified to 2e-5 in fp32), stopping each row at </answer>.
Per-item results are appended to out/gsm8k_depth/<schedule>.jsonl, so interrupted runs resume.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

import sona_adapter as S

HERE = Path(__file__).parent
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def pred_answer(text):
    m = re.search(r"<answer>(.*?)(</answer>|$)", text, re.S)
    seg = m.group(1) if m else text
    nums = NUM.findall(seg)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same(a, b):
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


@torch.no_grad()
def generate(ad, tok, prompts, cfg, max_new, stop_id, eos_id, pad_id):
    rows = [list(p) for p in prompts]
    done = [False] * len(rows)
    for _ in range(max_new):
        act = [i for i, d in enumerate(done) if not d]
        if not act:
            break
        L = max(len(rows[i]) for i in act)
        x = torch.full((len(act), L), pad_id, dtype=torch.long, device="cuda")
        for r, i in enumerate(act):
            x[r, :len(rows[i])] = torch.tensor(rows[i], device="cuda")
        logits = ad.run_batch(x, cfg)
        for r, i in enumerate(act):
            nxt = int(logits[r, len(rows[i]) - 1].argmax())
            rows[i].append(nxt)
            if nxt in (stop_id, eos_id):
                done[i] = True
    return [r[len(p):] for r, p in zip(rows, prompts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="<checkpoint>.pt")
    ap.add_argument("--schedules", default="full,repeat4,repeat1,prefix4,prefix1")
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--max_new", type=int, default=512)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = list(load_dataset("openai/gsm8k", "main", split="test"))[:args.n]
    tok, m, step = S.load(args.ckpt, "cuda")
    ad = S.SonaAdapter(m, tok)
    cf = dict(ad.configs())
    stop_id, eos_id, pad_id = tok.t.token_to_id("</answer>"), tok.t.token_to_id("<eos>"), tok.t.token_to_id("<pad>")
    out = HERE / "out" / "gsm8k_depth"
    out.mkdir(parents=True, exist_ok=True)
    for sched in args.schedules.split(","):
        f = out / f"{sched}.jsonl"
        seen = {json.loads(l)["idx"] for l in f.open()} if f.exists() else set()
        todo = [i for i in range(len(ds)) if i not in seen]
        t0 = time.time()
        for b in range(0, len(todo), args.bs):
            idx = todo[b:b + args.bs]
            prompts = [[tok.bos_token_id] + tok(f"<problem>\n{ds[i]['question']}\n</problem>\n<think>\n").input_ids
                       for i in idx]
            gens = generate(ad, tok, prompts, cf[sched], args.max_new, stop_id, eos_id, pad_id)
            with f.open("a") as fh:
                for i, g in zip(idx, gens):
                    text = tok.t.decode(g, skip_special_tokens=False)
                    p, gold = pred_answer(text), gold_answer(ds[i]["answer"])
                    fh.write(json.dumps({"idx": i, "pred": p, "gold": gold, "correct": same(p, gold),
                                         "parsed": p is not None, "n_tokens": len(g),
                                         "closed": "</answer>" in text}) + "\n")
            n_done = len(seen) + b + len(idx)
            print(f"[{sched}] {n_done}/{len(ds)}  {time.time() - t0:.0f}s", flush=True)
        rs = [json.loads(l) for l in f.open()]
        print(f"== {sched}: acc {sum(r['correct'] for r in rs) / len(rs):.4f}  parse {sum(r['parsed'] for r in rs) / len(rs):.3f}  n={len(rs)}", flush=True)


if __name__ == "__main__":
    main()
