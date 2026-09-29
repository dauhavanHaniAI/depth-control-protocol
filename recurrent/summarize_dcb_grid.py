#!/usr/bin/env python3
"""Collect the DCB calibration grid into one table.

The point of the grid is that the verdict for each arm is known BEFORE the
run, so this prints the expectation next to the measurement. A framework that
only ever confirms itself is not calibrated; the arm that must come out
"present" (C) is what separates "sampling was done" from "the output saw the
configuration being measured".
"""
from __future__ import annotations
import json, pathlib

ARMS = [("dcb_A_fixed",   "cố định 8",    "CÓ"),
        ("dcb_B_full",    "U{1..8}",      "KHÔNG"),
        ("dcb_C_partial", "U{5..8}",      "CÓ")]
STEPS = [1500, 3000, 6000]


def load(arm, step, mode):
    f = pathlib.Path(f"exp/dcb_{arm}_{step}_{mode}/results.json")
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())["results"]["depth_ablation"]
    except Exception:
        return None


def shares(arm, step):
    pre, rep = load(arm, step, "prefix"), load(arm, step, "repeat")
    if not pre or "1" not in pre or "8" not in pre:
        return None
    naive = pre["1"]["nll"] - pre["8"]["nll"]
    if abs(naive) < 1e-9:
        return None
    out = {"naive": naive}
    if "cal_nll" in pre["1"] and "cal_nll" in pre["8"]:
        cal = pre["1"]["cal_nll"] - pre["8"]["cal_nll"]
        out["cal_share"] = (naive - cal) / naive
    T = [v["cal_temperature"] for v in pre.values() if "cal_temperature" in v]
    if T:
        out["T_span"] = max(T) - min(T)
        out["T_full"] = pre["8"].get("cal_temperature")
    if rep and "1" in rep:
        out["apply_share"] = (pre["1"]["nll"] - rep["1"]["nll"]) / naive
        if "4" in rep:
            out["distinct_share"] = (rep["1"]["nll"] - rep["4"]["nll"]) / naive
    return out


def main():
    print()
    print(f"{'nhánh':16s} {'lịch độ sâu':11s} {'bước':>5s} {'Δ thô':>7s} "
          f"{'hiệu chuẩn':>10s} {'biên T':>7s} {'T@8':>6s} "
          f"{'áp dụng':>8s} {'khác biệt':>10s}  dự đoán")
    print("-" * 104)
    rows = []
    for arm, sched, expect in ARMS:
        for st in STEPS:
            s = shares(arm, st)
            if not s:
                continue
            rows.append((arm, expect, st, s))
            f = lambda k, w=9, p=1, pc=True: (
                f"{100*s[k]:>{w}.{p}f}%" if k in s and pc else
                f"{s[k]:>{w}.{p}f}" if k in s else f"{'-':>{w+1}s}")
            print(f"{arm:16s} {sched:11s} {st:>5d} {s['naive']:>7.3f} "
                  f"{f('cal_share',9)} {f('T_span',6,3,False)} "
                  f"{f('T_full',5,2,False)} {f('apply_share',7)} "
                  f"{f('distinct_share',9)}  {expect}")

    print("\n── phán quyết ở ngân sách cao nhất ──")
    ok, have_any = True, False
    for arm, sched, expect in ARMS:
        got = None
        for st in reversed(STEPS):
            s = shares(arm, st)
            if s and "cal_share" in s:
                got = (st, s["cal_share"]); break
        if got is None:
            print(f"  {arm:16s} chưa có dữ liệu"); ok = False; continue
        have_any = True
        st, cs = got
        # ngưỡng 5% lấy từ Huginn-0125 (-0,6%) so với Sona độ sâu cố định (25,6%)
        verdict = "CÓ" if cs > 0.05 else "KHÔNG"
        mark = "khớp" if verdict == expect else "LỆCH"
        if verdict != expect:
            ok = False
        print(f"  {arm:16s} step {st}: tỉ trọng hiệu chuẩn {100*cs:5.1f}%  "
              f"-> {verdict:6s} (dự đoán {expect})  [{mark}]")
    if not have_any:
        msg = "Chưa có nhánh nào hoàn tất; không có gì để phán quyết."
    elif ok:
        msg = "Công cụ khôi phục đúng cả ba trường hợp."
    else:
        msg = ("Có nhánh chưa xong hoặc không khớp dự đoán. Kết quả lệch là "
               "thứ cần báo cáo, không phải lỗi cần sửa.")
    print("\n" + msg)


if __name__ == "__main__":
    main()
