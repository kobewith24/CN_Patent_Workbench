# -*- coding: utf-8 -*-
"""Retrieval baseline on the frozen 60-question set.

Rows: bm25 / dense / dense+rerank / hybrid / hybrid+rerank.
All rows share the same candidates (top-50) so the only variable is the
retrieval stack.

Usage:
    python eval/run_retrieval.py
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service.retrieval import Retriever  # noqa: E402

CONFIGS = ["bm25", "dense", "dense_rerank", "hybrid", "hybrid_rerank"]
LABELS = {
    "bm25": "BM25 (FTS5)",
    "dense": "dense (bge-m3)",
    "dense_rerank": "dense + rerank",
    "hybrid": "hybrid (RRF)",
    "hybrid_rerank": "hybrid + rerank",
}

def load_gold(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_id_map(meta_path):
    m = {}
    with open(meta_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc_id = json.loads(line)["doc_id"]
            m[doc_id] = i
    return m

def first_rank(lst, gold):
    for i, lab in enumerate(lst, 1):
        if lab == gold:
            return i
    return None

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join(ROOT, "eval", "gold.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "reports"))
    args = ap.parse_args()

    gold = load_gold(args.gold)
    id_map = build_id_map(os.path.join(ROOT, "data", "index", "meta_compact.jsonl"))
    r = Retriever()

    per_query = []
    for row in gold:
        if row["type"] == "unanswerable":
            continue
        q = row["question"]
        golds = [id_map[g] for g in row["gold_doc_ids"] if g in id_map]
        d = r.dense(q, 50)
        s = r.sparse(q, 50)
        h = r.rrf([d, s], top=50)
        dr = r.rerank(q, d)
        hr = r.rerank(q, h)

        lists = {
            "bm25": [lab for lab, _ in s],
            "dense": [lab for lab, _ in d],
            "dense_rerank": [lab for lab, _ in dr],
            "hybrid": [lab for lab, _ in h],
            "hybrid_rerank": [lab for lab, _ in hr],
        }
        ranks = {}
        for cfg in CONFIGS:
            ranks[cfg] = [first_rank(lists[cfg], g) for g in golds]
        per_query.append({
            "qid": row["qid"], "type": row["type"], "question": q,
            "golds": golds, "ranks": ranks,
        })
        print("%s %s | %s" % (
            row["qid"], " ".join("%s=%s" % (c, ranks[c]) for c in ("dense", "bm25", "hybrid_rerank")),
            q[:40]), flush=True)

    # metrics
    metrics = {}
    fact = [x for x in per_query if x["type"] == "factoid"]
    comp = [x for x in per_query if x["type"] == "comparison"]
    for cfg in CONFIGS:
        hit5 = [1.0 if (x["ranks"][cfg][0] or 10 ** 6) <= 5 else 0.0 for x in fact]
        hit10 = [1.0 if (x["ranks"][cfg][0] or 10 ** 6) <= 10 else 0.0 for x in fact]
        hit50 = [1.0 if (x["ranks"][cfg][0] or 10 ** 6) <= 50 else 0.0 for x in fact]
        mrr = [1.0 / x["ranks"][cfg][0] if x["ranks"][cfg][0] else 0.0 for x in fact]
        all5 = []
        any5 = []
        for x in comp:
            rs = x["ranks"][cfg]
            all5.append(1.0 if rs and all(v is not None and v <= 5 for v in rs) else 0.0)
            any5.append(1.0 if any(v is not None and v <= 5 for v in rs) else 0.0)
        metrics[cfg] = {
            "factoid_n": len(fact),
            "recall@5": round(sum(hit5) / len(hit5), 4),
            "recall@10": round(sum(hit10) / len(hit10), 4),
            "recall@50": round(sum(hit50) / len(hit50), 4),
            "mrr": round(sum(mrr) / len(mrr), 4),
            "comparison_n": len(comp),
            "all_gold@5": round(sum(all5) / len(all5), 4),
            "any_gold@5": round(sum(any5) / len(any5), 4),
        }

    # paired bootstrap vs dense on factoid recall@5
    base = [1.0 if (x["ranks"]["dense"][0] or 10 ** 6) <= 5 else 0.0 for x in fact]
    cis = {}
    for cfg in CONFIGS:
        cur = [1.0 if (x["ranks"][cfg][0] or 10 ** 6) <= 5 else 0.0 for x in fact]
        delta = [cur[i] - base[i] for i in range(len(base))]
        lo, hi = boot_ci(delta)
        cis[cfg] = {"delta_vs_dense": round(sum(delta) / len(delta), 4),
                    "ci95": [round(lo, 4), round(hi, 4)]}

    hard = json.load(open(os.path.join(ROOT, "eval", "hard_cases.json"), encoding="utf-8"))
    hard_rows = []
    by_qid = {x["qid"]: x for x in per_query}
    for case in hard["cases"]:
        x = by_qid.get(case["qid"])
        if not x:
            continue
        hard_rows.append({
            "qid": case["qid"], "mechanism": case["mechanism"],
            "ranks": {c: x["ranks"][c][0] for c in CONFIGS},
        })

    report = {
        "gold": args.gold,
        "n": len(per_query),
        "configs": CONFIGS,
        "metrics": metrics,
        "vs_dense": cis,
        "hard_cases": hard_rows,
        "queries": per_query,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "baseline_60.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = ["# 检索基线（60 题）", "",
             "同一批题、同一批 top-50 候选，只换检索栈。", "",
             "| 配置 | R@5 | R@10 | R@50 | MRR | 对比 all@5 | 对比 any@5 | ΔR@5 vs dense [95% CI] |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for cfg in CONFIGS:
        m = metrics[cfg]
        d = cis[cfg]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f [%+.3f, %+.3f] |" % (
            LABELS[cfg], m["recall@5"], m["recall@10"], m["recall@50"], m["mrr"],
            m["all_gold@5"], m["any_gold@5"],
            d["delta_vs_dense"], d["ci95"][0], d["ci95"][1]))
    lines += ["", "## 4 道难题的 gold 排名", "",
              "| qid | 机制 | " + " | ".join(CONFIGS) + " |",
              "|---|---|" + "---|" * len(CONFIGS)]
    for h in hard_rows:
        lines.append("| %s | %s | %s |" % (
            h["qid"], h["mechanism"],
            " | ".join("-" if h["ranks"][c] is None else str(h["ranks"][c]) for c in CONFIGS)))
    with open(os.path.join(args.out, "baseline_60.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved", args.out)

if __name__ == "__main__":
    main()
