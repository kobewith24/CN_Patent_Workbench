# -*- coding: utf-8 -*-
"""Candidate-depth sweep for hybrid retrieval (factoid subset).

Decides the service default `candidates` by overall R@5 / latency, not by the
four hand-picked hard cases.

Usage:
    python eval/sweep_candidates.py
"""
import json
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service.retrieval import Retriever  # noqa: E402

DEPTHS = [50, 100, 200]
MODES = [("hybrid", False), ("hybrid_rerank", True)]
WATCH = ["CN201510084776.0", "CN201510373239.8", "CN201520352400.9", "CN201520442192.1"]

def boot_ci(deltas, n=2000, seed=0):
    if not deltas:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    m = len(deltas)
    vals = []
    for _ in range(n):
        vals.append(sum(deltas[rnd.randrange(m)] for _ in range(m)) / m)
    vals.sort()
    return (vals[int(0.025 * n)], vals[int(0.975 * n)])

def main():
    gold = [json.loads(line) for line in open(os.path.join(ROOT, "eval", "gold.jsonl"), encoding="utf-8")]
    gold = [g for g in gold if g["type"] == "factoid"]
    r = Retriever()

    rows = {}
    for depth in DEPTHS:
        for name, _rr in MODES:
            rows["%s@%d" % (name, depth)] = {"ranks": [], "ms": [], "by_qid": {}}

    for g in gold:
        for depth in DEPTHS:
            for name, rr in MODES:
                t0 = time.time()
                hits, _ = r.search(g["question"], k=10, candidates=depth,
                                   mode="hybrid", rerank=rr)
                ms = (time.time() - t0) * 1000
                ids = [h["doc_id"] for h in hits]
                gold_id = g["gold_doc_ids"][0]
                rank = (ids.index(gold_id) + 1) if gold_id in ids else None
                key = "%s@%d" % (name, depth)
                rows[key]["ranks"].append(rank)
                rows[key]["ms"].append(ms)
                rows[key]["by_qid"][g["qid"]] = rank
        print("done", g["qid"], flush=True)

    summary = {}
    hit5_by_key = {}
    for key, v in rows.items():
        rs = v["ranks"]
        hit5 = [1.0 if x and x <= 5 else 0.0 for x in rs]
        hit10 = [1.0 if x and x <= 10 else 0.0 for x in rs]
        mrr = sum(1.0 / x for x in rs if x) / len(rs)
        ms = sorted(v["ms"])
        hit5_by_key[key] = hit5
        summary[key] = {
            "R@5": round(sum(hit5) / len(hit5), 4),
            "R@10": round(sum(hit10) / len(hit10), 4),
            "MRR": round(mrr, 4),
            "median_ms": round(statistics.median(ms), 1),
            "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1),
        }

    base = hit5_by_key["hybrid@50"]
    deltas = {}
    for key, hit5 in hit5_by_key.items():
        d = [hit5[i] - base[i] for i in range(len(base))]
        lo, hi = boot_ci(d)
        deltas[key] = {"delta": round(sum(d) / len(d), 4), "ci95": [round(lo, 4), round(hi, 4)]}

    watch = {}
    for g in gold:
        if g["gold_doc_ids"][0] in WATCH:
            watch[g["qid"]] = {k: rows[k]["by_qid"].get(g["qid"]) for k in rows}

    out = {"depths": DEPTHS, "n": len(gold), "summary": summary,
           "vs_hybrid@50": deltas, "watch_cases": watch}
    outdir = os.path.join(ROOT, "eval", "reports")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "sweep_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = ["# 候选深度扫描（30 道 factoid）", "",
             "同一套 hybrid 检索，只改 rerank 前的候选数；k=10。", "",
             "| 配置 | R@5 | R@10 | MRR | 中位延迟 | p95 延迟 | ΔR@5 vs hybrid@50 [95% CI] |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for depth in DEPTHS:
        for name, _rr in MODES:
            key = "%s@%d" % (name, depth)
            s = summary[key]
            d = deltas[key]
            lines.append("| %s@%d | %.3f | %.3f | %.3f | %.0f ms | %.0f ms | %+.3f [%+.3f, %+.3f] |" % (
                name, depth, s["R@5"], s["R@10"], s["MRR"], s["median_ms"], s["p95_ms"],
                d["delta"], d["ci95"][0], d["ci95"][1]))
    lines += ["", "## 4 道难题的 gold 排名（k=10 内，None = 未进前 10）", "",
              "| qid | " + " | ".join("%s@%d" % (n, d) for d in DEPTHS for n, _ in MODES) + " |",
              "|---|" + "---|" * (len(DEPTHS) * len(MODES))]
    for qid, ranks in watch.items():
        lines.append("| %s | %s |" % (qid, " | ".join(
            "-" if ranks["%s@%d" % (n, d)] is None else str(ranks["%s@%d" % (n, d)])
            for d in DEPTHS for n, _ in MODES)))
    with open(os.path.join(outdir, "sweep_candidates.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(watch, ensure_ascii=False))
    print("saved")

if __name__ == "__main__":
    main()
