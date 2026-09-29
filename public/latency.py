#!/usr/bin/env python3
"""latency.py — wall-clock time and peak memory of each execution schedule (public checkpoint, bf16, one GPU).
Teacher-forced forward pass over a batch of 8 x 1024 tokens; median of 10 timed runs after 3 warm-up runs."""
import json, time
from pathlib import Path
import torch
import sona_adapter as S

tok, m, step = S.load("<checkpoint>.pt", "cuda")
ad = S.SonaAdapter(m, tok)
cf = dict(ad.configs())
g = torch.Generator().manual_seed(0)
x = torch.randint(20, 31000, (8, 1024), generator=g).cuda()
res = {}
with torch.no_grad():
    for name in ("prefix1", "prefix4", "repeat1", "repeat4", "suffix4", "full", "extend16"):
        for _ in range(3):
            ad.run_batch(x, cf[name])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(10):
            torch.cuda.synchronize(); t = time.perf_counter()
            ad.run_batch(x, cf[name])
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        ts.sort()
        res[name] = {"applications": len(cf[name]["plan"]), "ms_median": 1000 * ts[5],
                     "tokens_per_s": 8 * 1024 / ts[5], "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        print(name, {k: round(v, 2) for k, v in res[name].items()}, flush=True)
out = Path(__file__).parent / "out" / "public-sft" / "reviewer_c" / "latency.json"
out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "batch": [8, 1024], "results": res}, indent=1))
