# -*- coding: utf-8 -*-
"""Recompute the refusal metric on an existing report with the widened detector."""
import json
import os

ROOT = r"D:\Projects\CN_Patent_RAG"
REPORT = os.path.join(ROOT, "eval", "report.json")
TERMS = ("资料不足", "无法回答", "未提及", "未涉及", "未披露",
         "无法确认", "没有相关", "没有提及", "无法判断")


def main():
    with open(REPORT, encoding="utf-8") as f:
        rep = json.load(f)
    vals = []
    for d in rep.get("answers", []):
        if d.get("type") == "unanswerable":
            txt = d.get("answer", "")
            d["refused"] = any(t in txt for t in TERMS)
            vals.append(d["refused"])
    rep["answer"]["unanswerable_refusal"] = round(sum(vals) / len(vals), 4) if vals else None
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep["answer"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
