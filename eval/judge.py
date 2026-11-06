# -*- coding: utf-8 -*-
"""Pool candidates for the A set and judge relevance with the local LLM.

For each A question: dense top-8 + BM25 top-8 + hybrid top-10, deduped. Each
(question, patent) pair gets an LLM relevance label and a reranker score. A
stratified 50-pair sample is written for human calibration.

Usage:
    python eval/judge.py                 # pool + judge (slow)
    python eval/judge.py --sample-only   # resample from pairs_all.jsonl
"""
import argparse
import json
import os
import random
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service.retrieval import Retriever  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
OUTDIR = os.path.join(ROOT, "eval", "annotations")

PROMPT = """判断下面这件专利能否回答该问题（部分回答也算）。只输出 JSON：{{"r":1}} 或 {{"r":0}}。
问题：{q}
标题：{title}
摘要：{abstract}
"""


def judge(q, title, abstract):
    body = PROMPT.format(q=q, title=title, abstract=(abstract or "")[:600])
    try:
        r = requests.post(OLLAMA + "/api/generate", json={
            "model": MODEL, "prompt": body, "stream": False, "think": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 16, "num_ctx": 2048},
        }, timeout=90)
        r.raise_for_status()
        data = json.loads((r.json().get("response") or "").strip())
        return 1 if int(data.get("r", 0)) == 1 else 0
    except Exception:  # noqa: BLE001
        return None


def sample_pairs(pairs, n=50, seed=7):
    """Alternate positive/negative, one pair per question."""
    pos = [p for p in pairs if p["llm"] == 1]
    neg = [p for p in pairs if p["llm"] == 0]
    rnd = random.Random(seed)
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    sample, used_q = [], set()
    pi = ni = 0
    while len(sample) < n and (pi < len(pos) or ni < len(neg)):
        use_pos = len(sample) % 2 == 0
        src = pos if use_pos else neg
        i = pi if use_pos else ni
        picked = None
        while i < len(src):
            cand = src[i]
            i += 1
            if cand["qid"] not in used_q:
                picked = cand
                break
        if use_pos:
            pi = i
        else:
            ni = i
        if picked is not None:
            sample.append(picked)
            used_q.add(picked["qid"])
    if len(sample) < n:
        for p in pairs:
            if len(sample) >= n:
                break
            if p["qid"] not in used_q:
                sample.append(p)
                used_q.add(p["qid"])
    return sample


def write_sample(pairs):
    sample = sample_pairs(pairs)
    with open(os.path.join(OUTDIR, "pairs_human.jsonl"), "w", encoding="utf-8") as f:
        for p in sample:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("sample=%d (pos=%d neg=%d)" % (
        len(sample), sum(1 for p in sample if p["llm"] == 1),
        sum(1 for p in sample if p["llm"] == 0)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.sample_only:
        pairs = [json.loads(line) for line in open(os.path.join(OUTDIR, "pairs_all.jsonl"), encoding="utf-8")]
        write_sample(pairs)
        return

    items = [json.loads(line) for line in open(os.path.join(ROOT, "eval", "gold_v1_a.jsonl"), encoding="utf-8")]
    print("A items", len(items), flush=True)
    r = Retriever()

    pairs = []
    for it in items:
        q = it["question"]
        d = r.dense(q, 8)
        s = r.sparse(q, 8)
        h = r.rrf([d, s], top=10)
        order = {lab: rank for rank, (lab, _) in enumerate(h, 1)}
        seen = set()
        labels = []
        for lab, _ in d:
            if lab not in seen:
                seen.add(lab)
                labels.append((lab, "dense"))
        for lab, _ in s:
            if lab not in seen:
                seen.add(lab)
                labels.append((lab, "bm25"))

        metas = r._read_metas([lab for lab, _ in labels])
        fulls = r._read_fulls(metas)
        texts = [(q, rec.get("title", "") + "\n" + rec.get("abstract", "")) for rec in fulls]
        try:
            rr = r.reranker.predict(texts, batch_size=8, show_progress_bar=False)
        except Exception:  # noqa: BLE001
            rr = [None] * len(texts)

        for i, ((lab, src), m, rec) in enumerate(zip(labels, metas, fulls)):
            pairs.append({
                "qid": it["qid"], "question": q, "doc_id": m["doc_id"],
                "title": m["title"], "abstract": rec.get("abstract", ""),
                "src": src, "rank_hybrid": order.get(lab),
                "llm": judge(q, m["title"], rec.get("abstract", "")),
                "rerank": None if rr[i] is None else round(float(rr[i]), 4),
            })
        print("judged", it["qid"], len(labels), flush=True)

    with open(os.path.join(OUTDIR, "pairs_all.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    write_sample(pairs)
    print("pairs=%d" % len(pairs), flush=True)


if __name__ == "__main__":
    main()
