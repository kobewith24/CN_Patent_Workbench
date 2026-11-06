# -*- coding: utf-8 -*-
"""Evaluate the B set and the planner set.

B: dense / BM25 / hybrid / hybrid+rerank at candidates=100, recall + nDCG + CI.
Planner: does `_plan` split the question, and does splitting help retrieve both
gold patents compared with a single query.

Usage:
    python eval/run_eval.py
"""
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service.retrieval import META_PATH, Retriever  # noqa: E402

CONFIGS = ["dense", "bm25", "hybrid", "hybrid_rerank"]
K = 100

def load_label2id():
    out = []
    with open(META_PATH, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["doc_id"])
    return out

def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def ndcg(ranked_flags, ideal, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(ranked_flags[:k]) if f)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(ideal, k)))
    return dcg / idcg if idcg else 0.0

def boot_ci(deltas, n=2000, seed=0):
    if not deltas:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    m = len(deltas)
    vals = sorted(sum(deltas[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return (vals[int(0.025 * n)], vals[int(0.975 * n)])

def eval_b(r, items, label2id):
    per_q = []
    for it in items:
        q = it["question"]
        golds = set(it["gold_doc_ids"])
        d = r.dense(q, K)
        s = r.sparse(q, K)
        h = r.rrf([d, s], top=K)
        hr = r.rerank(q, h)
        lists = {
            "dense": [label2id[x] for x, _ in d],
            "bm25": [label2id[x] for x, _ in s],
            "hybrid": [label2id[x] for x, _ in h],
            "hybrid_rerank": [label2id[x] for x, _ in hr],
        }
        row = {"qid": it["qid"], "golds": sorted(golds), "ranks": {}}
        for cfg in CONFIGS:
            ids = lists[cfg]
            ranks = [ids.index(g) + 1 for g in golds if g in ids]
            row["ranks"][cfg] = min(ranks) if ranks else None
            row.setdefault("top10", {})[cfg] = ids[:10]
        per_q.append(row)
        print("B", it["qid"], {c: row["ranks"][c] for c in CONFIGS}, flush=True)

    metrics = {}
    for cfg in CONFIGS:
        n = len(per_q)
        r5 = sum(1 for x in per_q if x["ranks"][cfg] and x["ranks"][cfg] <= 5) / n
        r10 = sum(1 for x in per_q if x["ranks"][cfg] and x["ranks"][cfg] <= 10) / n
        r50 = sum(1 for x in per_q if x["ranks"][cfg] and x["ranks"][cfg] <= 50) / n
        mrr = sum(1.0 / x["ranks"][cfg] for x in per_q if x["ranks"][cfg]) / n
        nd = sum(ndcg([1.0 if d in set(x["golds"]) else 0.0 for d in x["top10"][cfg]],
                      min(len(x["golds"]), 10)) for x in per_q) / n
        metrics[cfg] = {"n": n, "R@5": round(r5, 4), "R@10": round(r10, 4),
                        "R@50": round(r50, 4), "MRR": round(mrr, 4), "nDCG@10": round(nd, 4)}

    base = [1.0 if x["ranks"]["dense"] and x["ranks"]["dense"] <= 5 else 0.0 for x in per_q]
    cis = {}
    for cfg in CONFIGS:
        cur = [1.0 if x["ranks"][cfg] and x["ranks"][cfg] <= 5 else 0.0 for x in per_q]
        d = [cur[i] - base[i] for i in range(len(base))]
        lo, hi = boot_ci(d)
        cis[cfg] = {"delta": round(sum(d) / len(d), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return per_q, metrics, cis

def eval_planner(r, items):
    from service.app import _plan  # noqa: E402

    rows = []
    for it in items:
        q = it["question"]
        subs = _plan(q)
        golds = set(it["gold_doc_ids"])
        plans_ok = len(subs) > 1
        union = []
        for sq in subs:
            hits, _ = r.search(sq, k=5, candidates=100, mode="hybrid", rerank=True)
            union.extend(h["doc_id"] for h in hits)
        both_plan = golds.issubset(set(union))
        single, _ = r.search(q, k=5, candidates=100, mode="hybrid", rerank=True)
        both_single = golds.issubset({h["doc_id"] for h in single})
        rows.append({"qid": it["qid"], "n_sub": len(subs), "both_plan": both_plan,
                     "both_single": both_single})
    n = len(rows)
    return {
        "n": n,
        "split_rate": round(sum(1 for x in rows if x["n_sub"] > 1) / n, 4),
        "both_plan@5": round(sum(1 for x in rows if x["both_plan"]) / n, 4),
        "both_single@5": round(sum(1 for x in rows if x["both_single"]) / n, 4),
    }, rows

def main():
    b = load(os.path.join(ROOT, "eval", "gold_b.jsonl"))
    p = load(os.path.join(ROOT, "eval", "gold_planner.jsonl"))
    print("B=%d planner=%d" % (len(b), len(p)), flush=True)
    r = Retriever()
    label2id = load_label2id()

    t0 = time.time()
    per_q, metrics, cis = eval_b(r, b, label2id)
    planner_metrics, planner_rows = eval_planner(r, p) if p else ({}, [])
    print("elapsed %.0fs" % (time.time() - t0), flush=True)

    out = {"b": {"metrics": metrics, "vs_dense": cis, "queries": per_q},
           "planner": {"metrics": planner_metrics, "queries": planner_rows}}
    outdir = os.path.join(ROOT, "eval", "reports")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = ["# 评测（Tier B + planner）", "",
             "B 层：%d 题，多选题 gold，gold 由问题自身的稀有词定义；候选深度 100。" % len(b), "",
             "| 配置 | R@5 | R@10 | R@50 | MRR | nDCG@10 | ΔR@5 vs dense [95% CI] |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for cfg in CONFIGS:
        m = metrics[cfg]
        c = cis[cfg]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f [%+.3f, %+.3f] |" % (
            cfg, m["R@5"], m["R@10"], m["R@50"], m["MRR"], m["nDCG@10"],
            c["delta"], c["ci95"][0], c["ci95"][1]))
    if planner_metrics:
        lines += ["", "## planner（%d 题非模板对比问法）" % planner_metrics["n"], "",
                  "| 指标 | 值 |", "|---|---:|",
                  "| 成功拆成多子查询 | %.3f |" % planner_metrics["split_rate"],
                  "| 两篇 gold 都进 top5（拆解后） | %.3f |" % planner_metrics["both_plan@5"],
                  "| 两篇 gold 都进 top5（单查询） | %.3f |" % planner_metrics["both_single@5"]]
    with open(os.path.join(outdir, "eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(planner_metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()
