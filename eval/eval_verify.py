# -*- coding: utf-8 -*-
"""Generate answers for the A set and compare three citation verifiers.

Each answer is split into claims (all sentences, v2 semantics). Every claim is
judged by: local LLM, Chinese NLI, and the reranker. A 40-claim stratified
sample is written for human calibration.

Usage:
    python eval/eval_verify.py
"""
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service.app import _build_prompt  # noqa: E402
from service.retrieval import Retriever  # noqa: E402
from service.verify import (NLIJudge, build_claims, judge_llm, judge_nli,  # noqa: E402
                            judge_rerank, summarize)

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
ANN = os.path.join(ROOT, "eval", "annotations")


def generate(prompt, timeout=240):
    r = requests.post(OLLAMA + "/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0.1, "num_predict": 384, "num_ctx": 4096},
    }, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def main():
    items = [json.loads(line) for line in open(os.path.join(ROOT, "eval", "gold_v1_a.jsonl"), encoding="utf-8")]
    os.makedirs(ANN, exist_ok=True)
    r = Retriever()
    try:
        nli = NLIJudge()
    except Exception as exc:  # noqa: BLE001
        print("NLI unavailable, continuing without it:", exc, flush=True)
        nli = None

    rows = []
    t0 = time.time()
    for it in items:
        q = it["question"]
        hits, _ = r.search(q, k=5, candidates=100, mode="hybrid", rerank=True)
        if not hits:
            continue
        by_rank = {h["rank"]: h for h in hits}
        prompt = _build_prompt(q, [SimpleNamespace(**h) for h in hits])
        try:
            answer = generate(prompt)
        except Exception as exc:  # noqa: BLE001
            print("gen failed", it["qid"], exc, flush=True)
            continue
        claims = build_claims(answer, len(hits))
        if not claims:
            continue

        c_llm = judge_llm([dict(c) for c in claims], by_rank)
        if nli is not None:
            c_nli = judge_nli([dict(c) for c in claims], by_rank, nli)
        else:
            c_nli = [{"supported": None, "score": None} for _ in claims]
        c_rr = judge_rerank([dict(c) for c in claims], by_rank, r.reranker)

        for i, c in enumerate(claims):
            rows.append({
                "qid": it["qid"], "question": q,
                "claim_id": c["id"], "sentence": c["sentence"], "clean": c["clean"],
                "refs": c["refs"],
                "ref_abstracts": [by_rank[x]["abstract"] for x in c["refs"]],
                "answer": answer,
                "llm": c_llm[i]["supported"], "nli": c_nli[i]["supported"],
                "rr": c_rr[i]["supported"],
                "llm_score": c_llm[i]["score"], "nli_score": c_nli[i].get("score"),
                "rr_score": c_rr[i].get("score"),
            })
        print("done", it["qid"], "claims", len(claims), "%.0fs" % (time.time() - t0), flush=True)

    with open(os.path.join(ANN, "claims_all.jsonl"), "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    def macro(method):
        by_q = {}
        for x in rows:
            by_q.setdefault(x["qid"], []).append({"refs": x["refs"], "supported": x[method]})
        vals = [summarize(v) for v in by_q.values()]
        out = {}
        for k in ("citation_coverage", "support_rate", "unsupported_rate"):
            out[k] = round(sum(v[k] for v in vals) / len(vals), 4)
        return out

    methods = ["llm", "rr"] + (["nli"] if nli is not None else [])
    summary = {m: macro(m) for m in methods}
    summary["n_claims"] = len(rows)
    summary["n_answers"] = len({x["qid"] for x in rows})

    # stratified 40: cited claims only, alternate llm-supported / llm-unsupported
    cited = [x for x in rows if x["refs"]]
    pos = [x for x in cited if x["llm"]]
    neg = [x for x in cited if not x["llm"]]
    rnd = random.Random(11)
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    sample, used_q = [], set()
    pi = ni = 0
    while len(sample) < 40 and (pi < len(pos) or ni < len(neg)):
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
    for x in cited:
        if len(sample) >= 40:
            break
        if x["qid"] not in used_q:
            sample.append(x)
            used_q.add(x["qid"])

    with open(os.path.join(ANN, "claims_human.jsonl"), "w", encoding="utf-8") as f:
        for x in sample:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    outdir = os.path.join(ROOT, "eval", "reports", "v1")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "verify_v2.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "sample_size": len(sample)}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("sample", len(sample), "pos", sum(1 for x in sample if x["llm"]), flush=True)


if __name__ == "__main__":
    main()
