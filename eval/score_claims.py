# -*- coding: utf-8 -*-
"""Compare LLM / NLI / reranker as citation verifiers against human labels."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "eval", "annotations")
OUT = os.path.join(ROOT, "eval", "reports")

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

def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))

def best_threshold(scores, labels):
    cands = sorted(set(scores))
    best = (0.0, 0.5)
    for t in cands:
        pred = [1 if s >= t else 0 for s in scores]
        acc = sum(1 for p, y in zip(pred, labels) if p == y) / len(labels)
        if acc > best[0]:
            best = (acc, t)
    return {"acc": round(best[0], 4), "threshold": round(best[1], 4)}

def main():
    rows = load(os.path.join(ANN, "claims_all.jsonl"))
    labels = load(os.path.join(ANN, "claims_human_labels.jsonl"))
    hmap = {(x["qid"], x["claim_id"]): x["human"] for x in labels}
    pairs = [x for x in rows if (x["qid"], x["claim_id"]) in hmap]
    y = [hmap[(x["qid"], x["claim_id"])] for x in pairs]
    n = len(pairs)

    out = {"n_human": n, "human_pos": sum(y)}
    for m in ("llm", "nli", "rr"):
        sub = [x for x in pairs if x.get(m) is not None]
        if not sub:
            continue
        yy = [hmap[(x["qid"], x["claim_id"])] for x in sub]
        pred = [1 if x[m] else 0 for x in sub]
        out[m] = {
            "n": len(sub),
            "kappa": round(kappa(pred, yy), 4),
            "acc": round(sum(1 for p, t in zip(pred, yy) if p == t) / len(sub), 4),
        }
    for m in ("nli_score", "rr_score"):
        sc = [x.get(m) for x in pairs]
        if any(v is None for v in sc):
            continue
        out[m + "_auc"] = round(auc(sc, y), 4) if auc(sc, y) is not None else None
        out[m + "_best"] = best_threshold(sc, y)

    # v2 metrics, macro over answers
    by_q = {}
    for x in rows:
        by_q.setdefault(x["qid"], []).append(x)
    def macro(key):
        vals = [sum(1 for c in v if c[key]) / len(v) for v in by_q.values()]
        return round(sum(vals) / len(vals), 4) if vals else None
    mac = {
        "citation_coverage": round(sum(1 for x in rows if x["refs"]) / len(rows), 4) if rows else None,
        "llm_unsupported_rate": round(1 - macro("llm"), 4) if rows else None,
        "rr_unsupported_rate": round(1 - macro("rr"), 4) if rows else None,
    }
    if any(x.get("nli") is not None for x in rows):
        mac["nli_unsupported_rate"] = round(1 - macro("nli"), 4)
    out["macro"] = mac

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "verify_compare.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
