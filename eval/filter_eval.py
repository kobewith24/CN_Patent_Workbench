# -*- coding: utf-8 -*-
"""Filtered retrieval: old post-filter vs mask + exact scoring.

Ground truth = exact top-10 over the full eligible set. `old` is the previous
behaviour (HNSW top-5000 then drop). `new` is the production path
(Retriever.search), which scores exactly when the eligible set is small and
falls back to deeper HNSW when it is large.

Usage:
    python eval/filter_eval.py
"""
import json
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service.filters import exact_topk  # noqa: E402
from service.retrieval import Retriever  # noqa: E402

N_Q = 40

def pct(vals, p):
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]

def main():
    gold = [json.loads(line) for line in open(os.path.join(ROOT, "eval", "gold_b.jsonl"), encoding="utf-8")]
    random.seed(3)
    questions = random.sample([g["question"] for g in gold], min(N_Q, len(gold)))

    r = Retriever()
    cities = [c for c in r.meta.values["city"] if c]
    sections = [s for s in r.meta.values["ipc_section"] if s]
    types = [t for t in r.meta.values["patent_type"] if t]

    kinds = {}
    for q in questions:
        kinds.setdefault("narrow(city+year)", []).append((q, {
            "city": random.choice(cities), "year_from": 2019, "year_to": 2021}))
        kinds.setdefault("medium(section+year)", []).append((q, {
            "ipc_section": random.choice(sections), "year_from": 2018}))
        kinds.setdefault("broad(type)", []).append((q, {"patent_type": random.choice(types)}))

    out = {}
    for name, items in kinds.items():
        old_recall, old_empty, old_ms, new_recall, new_ms, elig = [], 0, [], [], [], []
        no_data = 0
        for q, f in items:
            mask = r.meta.mask(f)
            e = int(mask.sum())
            if e == 0:
                no_data += 1
                continue
            elig.append(e)

            v = r.encode(q)
            truth_labels = [lab for lab, _ in exact_topk(r.emb, v, mask, 10)]
            truth_ids = {m["doc_id"] for m in r._read_metas(truth_labels)}

            t0 = time.time()
            hits, _ = r.search(q, k=10, candidates=100, filters=f, mode="dense", rerank=False)
            new_ms.append((time.time() - t0) * 1000)
            new_ids = {h["doc_id"] for h in hits}
            new_recall.append(len(truth_ids & new_ids) / min(10, len(truth_ids)))

            t1 = time.time()
            hnsw = r.dense(q, 5000)
            old_labels = [lab for lab, _ in hnsw if mask[lab]][:10]
            old_ms.append((time.time() - t1) * 1000)
            old_ids = {m["doc_id"] for m in r._read_metas(old_labels)} if old_labels else set()
            if not old_ids:
                old_empty += 1
            old_recall.append(len(truth_ids & old_ids) / min(10, len(truth_ids)))

        valid = len(new_recall)
        out[name] = {
            "n": valid,
            "no_data_rate": round(no_data / len(items), 4),
            "eligible_median": int(statistics.median(elig)) if elig else 0,
            "old_empty_rate": round(old_empty / valid, 4) if valid else 0.0,
            "old_recall@10": round(sum(old_recall) / valid, 4) if valid else 0.0,
            "new_recall@10": round(sum(new_recall) / valid, 4) if valid else 0.0,
            "old_p50_ms": round(pct(old_ms, 0.5), 1),
            "old_p95_ms": round(pct(old_ms, 0.95), 1),
            "new_p50_ms": round(pct(new_ms, 0.5), 1),
            "new_p95_ms": round(pct(new_ms, 0.95), 1),
        }
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)

    outdir = os.path.join(ROOT, "eval", "reports")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "filter_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = ["# 过滤检索：旧事后筛 vs 新掩码+精确", "",
             "真值 = 完整可用集上的精确 top-10。旧 = HNSW 5000 后丢弃不合格项；新 = 生产路径。",
             "只统计可用集非空的组合；“无数据”是过滤条件本身没有命中任何专利，不算检索失败。", "",
             "| 过滤类型 | 可用条数(中位) | 无数据率 | 旧空结果率 | 旧 recall@10 | 新 recall@10 | 旧 p50/p95 | 新 p50/p95 |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for k, v in out.items():
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.0f/%.0f ms | %.0f/%.0f ms |" % (
            k, v["eligible_median"], v["no_data_rate"], v["old_empty_rate"], v["old_recall@10"],
            v["new_recall@10"], v["old_p50_ms"], v["old_p95_ms"], v["new_p50_ms"], v["new_p95_ms"]))
    lines += ["", "可用集 ≤150k 时新路径是精确解（recall@10 = 1.0、空结果率 0）；",
              "超过阈值退回 HNSW 并自适应加深，避免大过滤集下的全量扫描。"]
    with open(os.path.join(outdir, "filter_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
