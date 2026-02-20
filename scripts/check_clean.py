# -*- coding: utf-8 -*-
"""Check the cleaned JSONL for residual ad text and basic integrity."""
import json
import re
from collections import Counter

SRC = r"D:\Projects\CN_Patent_RAG\data\clean\patents.jsonl"
OUT = r"D:\Projects\CN_Patent_RAG\data\stats\residual_ads.json"

# Only ad phrases, not the legitimate word "马克" (马克笔/托卡马克/马克思主义...).
PATTERNS = {
    "mark_data": r"马\s*克\s*数\s*据",
    "macrodatas": r"macrodatas",
    "baidu": r"百度搜索",
    "wechat": r"微信公众号|关注公众号",
    "source_shell": r"来源\s*[：:]?\s*[“\"'‘’]?\s*[”\"'‘’]?\s*$",
    "url": r"https?://|www\.",
    "dangling": r"(?:关注|数据由|该数据由)\s*$",
}


def main():
    counts = Counter()
    examples = {}
    n = 0
    empty_abs = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            text = rec["abstract"]
            if not text:
                empty_abs += 1
            for name, pat in PATTERNS.items():
                if re.search(pat, text, re.IGNORECASE):
                    counts[name] += 1
                    examples.setdefault(name, text[:160])

    out = {
        "rows": n,
        "empty_abstract": empty_abs,
        "residual": dict(counts),
        "examples": examples,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"rows": n, "empty_abstract": empty_abs,
                      "residual": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
