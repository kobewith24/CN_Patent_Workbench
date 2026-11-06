# -*- coding: utf-8 -*-
"""Evidence audit over the already-generated claims (no regeneration, no human labels).

Usage:
    python eval/audit.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service.verify import audit_claims  # noqa: E402

def main():
    path = os.path.join(ROOT, "eval", "annotations", "claims_all.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    claims = [{"id": r["claim_id"], "refs": r["refs"], "clean": r["clean"], "sentence": r["sentence"],
               "qid": r["qid"], "question": r["question"]} for r in rows]

    # reconstruct per-claim abstract lookup from the stored fields
    by_qid = {}
    for r in rows:
        by_qid.setdefault(r["qid"], {})
        for rank, abstract in zip(r["refs"], r["ref_abstracts"]):
            by_qid[r["qid"]][rank] = abstract

    audited = []
    for c in claims:
        sub, _ = audit_claims([c], by_qid[c["qid"]])
        audited.append(sub[0])

    total = len(audited)
    body = [c for c in audited if not c["meta"]]
    cited = [c for c in body if c["refs"]]
    num = [c for c in cited if c["numbers"]]
    missing_rows = [c for c in num if c["numbers_missing"]]

    summary = {
        "claims": total,
        "meta_rate": round(1 - len(body) / total, 4),
        "citation_coverage": round(len(cited) / len(body), 4) if body else 0.0,
        "numeric_checked": len(num),
        "numeric_unsupported_rate": round(len(missing_rows) / len(num), 4) if num else 0.0,
        "term_coverage": round(sum(c["term_coverage"] for c in cited) / len(cited), 4) if cited else 0.0,
        "grounded_rate": round(sum(1 for c in body if c["grounded"]) / len(body), 4) if body else 0.0,
    }

    examples = [{
        "qid": c["qid"], "claim": c["clean"], "missing": c["numbers_missing"],
        "refs": c["refs"],
    } for c in missing_rows[:12]]

    outdir = os.path.join(ROOT, "eval", "reports")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "audit.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": examples}, f, ensure_ascii=False, indent=2)

    lines = ["# 证据审计（424 条 claim，无需人工）", "",
             "| 指标 | 值 |", "|---|---:|",
             "| claim 总数 | %d |" % summary["claims"],
             "| 元陈述占比 | %.3f |" % summary["meta_rate"],
             "| 引用覆盖率 | %.3f |" % summary["citation_coverage"],
             "| 含数值的已引用 claim | %d |" % summary["numeric_checked"],
             "| **数值无出处率** | **%.3f** |" % summary["numeric_unsupported_rate"],
             "| 术语覆盖率 | %.3f |" % summary["term_coverage"],
             "| **有据可查率 grounded** | **%.3f** |" % summary["grounded_rate"],
             "", "## 数值无出处样例", ""]
    for e in examples:
        lines.append("- `%s` 缺 %s：%s" % (e["qid"], e["missing"], e["claim"][:90]))
    with open(os.path.join(outdir, "audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("examples:")
    for e in examples[:6]:
        print(" ", e["qid"], e["missing"], e["claim"][:70])

if __name__ == "__main__":
    main()
