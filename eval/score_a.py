# -*- coding: utf-8 -*-
"""Score the A set (pooled, LLM-judged) and calibrate the judge against human labels.

Usage:
    python eval/score_a.py
"""
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "eval", "annotations")
OUT = os.path.join(ROOT, "eval", "reports")

def boot(vals, n=2000, seed=0):
    if not vals:
        return [0.0, 0.0]
    rnd = random.Random(seed)
    m = len(vals)
    xs = sorted(sum(vals[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return [round(xs[int(0.025 * n)], 4), round(xs[int(0.975 * n)], 4)]

def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8")]

def kappa(a, b):
    n = len(a)
    if not n:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p1 = sum(a) / n
    p2 = sum(b) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 0.0 if pe >= 1 else (po - pe) / (1 - pe)

def ndcg(flags, ideal, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags[:k]) if f)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(ideal, k)))
    return dcg / idcg if idcg else 0.0

def main():
    pairs = load(os.path.join(ANN, "pairs_all.jsonl"))
    human = load(os.path.join(ANN, "human.jsonl"))
    if not pairs:
        raise SystemExit("no pairs_all.jsonl")

    labelled = [p for p in pairs if p.get("llm") is not None]
    by_q = {}
    for p in labelled:
        if p.get("rank_hybrid"):
            by_q.setdefault(p["qid"], []).append(p)

    ndcgs, p5 = [], []
    for qid, rows in by_q.items():
        rows.sort(key=lambda x: x["rank_hybrid"])
        flags = [1.0 if r["llm"] == 1 else 0.0 for r in rows]
        ndcgs.append(ndcg(flags, sum(1 for f in flags if f)))
        p5.append(sum(flags[:5]) / min(5, len(flags)))

    hmap = {(h["qid"], h["doc_id"]): h["human"] for h in human}
    both = [(p["llm"], hmap[(p["qid"], p["doc_id"])]) for p in pairs
            if (p["qid"], p["doc_id"]) in hmap and p.get("llm") is not None]
    k = kappa([x[0] for x in both], [x[1] for x in both]) if both else None
    agree = sum(1 for x, y in both if x == y) / len(both) if both else None

    rerank_pairs = [(p["rerank"], hmap[(p["qid"], p["doc_id"])]) for p in pairs
                    if (p["qid"], p["doc_id"]) in hmap and p.get("rerank") is not None]
    rerank_acc = None
    if rerank_pairs:
        # threshold at median as a crude classifier
        vals = sorted(v for v, _ in rerank_pairs)
        thr = vals[len(vals) // 2]
        rerank_acc = sum(1 for v, h in rerank_pairs if (1 if v >= thr else 0) == h) / len(rerank_pairs)

    summary = {
        "pairs": len(pairs), "llm_labelled": len(labelled),
        "human_n": len(human),
        "nDCG@10_llm": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else None,
        "nDCG@10_ci95": boot(ndcgs),
        "P@5_llm": round(sum(p5) / len(p5), 4) if p5 else None,
        "P@5_ci95": boot(p5),
        "kappa_human_vs_llm": None if k is None else round(k, 4),
        "agreement": None if agree is None else round(agree, 4),
        "human_pos_rate": None if not both else round(sum(y for _, y in both) / len(both), 4),
        "llm_pos_rate": None if not both else round(sum(x for x, _ in both) / len(both), 4),
        "rerank_threshold_acc": None if rerank_acc is None else round(rerank_acc, 4),
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "eval_a.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

    lines = ["# A 层（池化 + 判定）", "",
             "| 指标 | 值 |", "|---|---:|",
             "| 池化 pair 数 | %d |" % summary["pairs"],
             "| 人工标注数 | %d |" % summary["human_n"],
             "| nDCG@10（LLM 判定） | %s |" % summary["nDCG@10_llm"],
             "| P@5（LLM 判定） | %s |" % summary["P@5_llm"],
             "| 人工 vs LLM kappa | %s |" % summary["kappa_human_vs_llm"],
             "| 一致率 | %s |" % summary["agreement"],
             "| 人工正例率 / LLM 正例率 | %s / %s |" % (summary["human_pos_rate"], summary["llm_pos_rate"]),
             "| reranker 阈值分类准确率 | %s |" % summary["rerank_threshold_acc"]]
    with open(os.path.join(OUT, "eval_a.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
