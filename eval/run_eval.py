# -*- coding: utf-8 -*-
"""Run the seed eval set against /retrieve and /answer.

Usage:
    python run_eval.py                 # retrieval for all 60 + answers for 18
    python run_eval.py --answer-n 6    # quick smoke
"""
import argparse
import json
import os
import time

import requests

ROOT = r"D:\Projects\CN_Patent_RAG"
GOLD = os.path.join(ROOT, "eval", "gold.jsonl")
OUT = os.path.join(ROOT, "eval", "report.json")
RETRIEVE = "http://127.0.0.1:8001/retrieve"
ANSWER = "http://127.0.0.1:8001/answer"
EXTRA = {}


def load_gold(path=GOLD):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def retrieve(q, k=5):
    r = requests.get(RETRIEVE, params={"q": q, "k": k, **EXTRA}, timeout=120)
    r.raise_for_status()
    return r.json()


def answer(q, k=5):
    r = requests.get(ANSWER, params={"q": q, "k": k, **EXTRA}, timeout=240)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-n", type=int, default=18)
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-title-boost", action="store_true")
    ap.add_argument("--no-multistep", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    global EXTRA, OUT
    EXTRA = {}
    if args.no_rerank:
        EXTRA["use_rerank"] = "false"
    if args.no_title_boost:
        EXTRA["use_title_boost"] = "false"
    if args.no_multistep:
        EXTRA["use_multistep"] = "false"
    if args.no_verify:
        EXTRA["use_verify"] = "false"
    if args.tag:
        OUT = os.path.join(ROOT, "eval", "report_%s.json" % args.tag)

    rows = load_gold(args.gold)
    details = []
    fact_hits = []
    comp_all = []
    comp_any = []

    t0 = time.time()
    for row in rows:
        d = {"qid": row["qid"], "type": row["type"], "question": row["question"],
             "gold": row["gold_doc_ids"]}
        if row["type"] == "unanswerable":
            d["retrieval"] = None
            details.append(d)
            continue
        try:
            res = retrieve(row["question"], k=5)
            got = [h["doc_id"] for h in res.get("hits", [])]
            d["retrieved"] = got
            d["elapsed_ms"] = res.get("elapsed_ms")
            if row["type"] == "factoid":
                hit = row["gold_doc_ids"][0] in got
                fact_hits.append(hit)
                d["hit"] = hit
            else:
                all_hit = all(g in got for g in row["gold_doc_ids"])
                any_hit = any(g in got for g in row["gold_doc_ids"])
                comp_all.append(all_hit)
                comp_any.append(any_hit)
                d["all_hit"] = all_hit
                d["any_hit"] = any_hit
        except Exception as exc:  # noqa: BLE001
            d["error"] = str(exc)
        details.append(d)
    retrieval_seconds = time.time() - t0

    # Answer a stratified subset.
    subsets = {"factoid": [], "comparison": [], "unanswerable": []}
    if args.answer_n > 0:
        for row in rows:
            if len(subsets[row["type"]]) < max(1, args.answer_n // 3):
                subsets[row["type"]].append(row)
    answer_rows = [r for v in subsets.values() for r in v]

    ans_details = []
    fact_cite = []
    comp_cite_any = []
    faith = []
    refusals = []
    t1 = time.time()
    for row in answer_rows:
        d = {"qid": row["qid"], "type": row["type"]}
        try:
            res = answer(row["question"], k=5)
            d["answer"] = res.get("answer", "")
            d["faithfulness"] = res.get("faithfulness")
            src = [s["doc_id"] for s in res.get("sources", [])]
            d["sources"] = src
            if row["answerable"]:
                if row["type"] == "factoid":
                    fact_cite.append(row["gold_doc_ids"][0] in src)
                else:
                    comp_cite_any.append(any(g in src for g in row["gold_doc_ids"]))
                if res.get("faithfulness") is not None and res["faithfulness"] >= 0:
                    faith.append(res["faithfulness"])
            else:
                txt = res.get("answer", "")
                refused = any(t in txt for t in (
                    "资料不足", "无法回答", "未提及", "未涉及", "未披露",
                    "无法确认", "没有相关", "没有提及", "无法判断"))
                refusals.append(refused)
                d["refused"] = refused
        except Exception as exc:  # noqa: BLE001
            d["error"] = str(exc)
        ans_details.append(d)
    answer_seconds = time.time() - t1

    report = {
        "n": len(rows),
        "retrieval": {
            "factoid_n": len(fact_hits),
            "factoid_recall@5": round(sum(fact_hits) / len(fact_hits), 4) if fact_hits else None,
            "comparison_n": len(comp_all),
            "comparison_all_hit@5": round(sum(comp_all) / len(comp_all), 4) if comp_all else None,
            "comparison_any_hit@5": round(sum(comp_any) / len(comp_any), 4) if comp_any else None,
            "seconds": round(retrieval_seconds, 1),
        },
        "answer": {
            "n": len(ans_details),
            "factoid_citation_hit": round(sum(fact_cite) / len(fact_cite), 4) if fact_cite else None,
            "comparison_citation_any": round(sum(comp_cite_any) / len(comp_cite_any), 4) if comp_cite_any else None,
            "avg_faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
            "unanswerable_refusal": round(sum(refusals) / len(refusals), 4) if refusals else None,
            "seconds": round(answer_seconds, 1),
        },
        "details": details,
        "answers": ans_details,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("retrieval", "answer")},
                     ensure_ascii=False, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
