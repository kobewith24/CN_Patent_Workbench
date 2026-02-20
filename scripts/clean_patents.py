# -*- coding: utf-8 -*-
"""Clean the 520k Chinese patent CSV into a RAG-ready JSONL corpus.

Usage:
    python clean_patents.py            # full run
    python clean_patents.py --limit 5000

Outputs:
    data/clean/patents.jsonl
    data/clean/sample_20.jsonl
    data/stats/clean_stats.json
"""
import argparse
import html
import json
import os
import re
from collections import Counter

import pandas as pd

RAW = r"D:\Projects\CN_Patent_RAG\data\raw\patents_sample_500k.csv"
CLEAN_DIR = r"D:\Projects\CN_Patent_RAG\data\clean"
STATS_DIR = r"D:\Projects\CN_Patent_RAG\data\stats"

# Promotional junk injected by the data reseller. Keep patterns narrow so we
# do not delete legitimate technical text.
AD_PATTERNS = [
    r"马\s*克\s*数\s*据\s*网",
    r"马\s*克\s*团\s*队",
    r"macrodatas\.cn",
    r"https?://\S+",
    r"www\.\S+",
    r"百度搜索\s*[“\"]?\s*马\s*克\s*数\s*据\s*网\s*[”\"]?",
    r"百度搜索\s*[“\"][^”\"]{0,20}[”\"]",
    r"关注公众号\s*[“\"]?[^”\"\s]{0,20}[”\"]?",
    r"微信公众号\s*[“\"]?[^”\"\s]{0,20}[”\"]?",
    r"更多数据[：:]\s*\S*",
    r"详见官网[：:]\s*\S*",
    r"数据由\s*[【\[](?:[^】\]]{0,20})[】\]]\s*整理",
    r"该数据由\s*[【\[](?:[^】\]]{0,20})[】\]]\s*整理",
    r"[（(]\s*来\s*自[^）)]{0,30}(?:马\s*克|数据网)[^）)]{0,10}[）)]",
    r"来\s*自\s*[：:]?\s*马\s*克\s*数\s*据\s*网",
    r"来源[：:]\s*(?:百度搜索|马\s*克|macrodatas)[^。]{0,40}",
    r"【[^】]{0,20}(?:马\s*克|数据网)[^】]{0,20}】",
    r"数据由\s*[“\"']?\s*马\s*克\s*数\s*据\s*(?:网)?\s*[”\"']?\s*整理",
    r"该数据由\s*[“\"']?\s*马\s*克\s*数\s*据\s*(?:网)?\s*[”\"']?\s*整理",
    r"来自\s*[：:]\s*www\.?",
    r"来自\s*[：:]\s*$",
]
AD_RE = [re.compile(p, re.IGNORECASE) for p in AD_PATTERNS]

COL_MAP = {
    "专利类型": "patent_type",
    "专利名称": "title",
    "摘要文本": "abstract_raw",
    "申请人类型": "applicant_type_raw",
    "申请人城市": "city",
    "申请号": "doc_id",
    "申请日": "apply_date",
    "申请年份": "year",
    "IPC主分类号": "ipc_main_raw",
    "SampleYear": "sample_year",
}


def clean_text(text):
    """Strip reseller ads/URLs and normalize whitespace. Returns (text, hits)."""
    if not isinstance(text, str):
        return "", 0
    t = html.unescape(text).replace("\u3000", " ")
    hits = 0
    for pat in AD_RE:
        t, n = pat.subn("", t)
        hits += n
    t = re.sub(r"[ \t\r\n]+", " ", t)
    t = re.sub(r"。{2,}", "。", t)
    t = re.sub(r"[，,]\s*[，,]", "，", t)
    t = re.sub(r"(?:来源|来自|数据由|该数据由|更多数据|详见官网|微信公众号|关注公众号)\s*[：:]?\s*$", "", t)
    t = re.sub(r"[（(【\[]\s*[）)】\]]?\s*$", "", t)
    # artifacts left after removing ads: empty quotes/brackets, dangling words
    t = re.sub(r"[“\"'‘’]{2}", "", t)
    t = re.sub(r"[（(]\s*[）)]", "", t)
    t = re.sub(r"[【\[]\s*[】\]]", "", t)
    t = re.sub(r"(?:来源|来自|关注|数据由|该数据由|更多数据|详见官网)\s*[：:]?\s*[“\"'‘’]?\s*[”\"'‘’]?\s*$", "", t)
    t = re.sub(r"^[\s。，；;、,：:“\"'‘’]+", "", t)
    t = re.sub(r"[\s，；;、,：:“\"'‘’]+$", "", t)
    return t, hits


def split_multi(value):
    parts = re.split(r"[,，、/]", str(value or ""))
    return [p.strip() for p in parts if p.strip()]


def ipc_parts(value):
    s = str(value or "").strip().upper()
    if not s or s.lower() == "nan":
        return "", ""
    section = s[0] if s[0] in "ABCDEFGH" else ""
    subclass = s[:4] if len(s) >= 4 else s
    return section, subclass


def length_stats(series):
    s = series[series > 0]
    if s.empty:
        return {}
    return {
        "count": int(s.count()),
        "mean": round(float(s.mean()), 1),
        "min": int(s.min()),
        "p25": int(s.quantile(0.25)),
        "p50": int(s.quantile(0.50)),
        "p75": int(s.quantile(0.75)),
        "p95": int(s.quantile(0.95)),
        "max": int(s.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only read N rows (smoke test)")
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()

    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    df = pd.read_csv(args.raw, encoding="utf-8-sig", nrows=args.limit or None,
                     dtype={"申请号": str, "申请日": str})
    raw_rows = len(df)
    df = df.rename(columns=COL_MAP)

    # ---- clean text ----
    cleaned, hits = [], []
    for v in df["abstract_raw"].tolist():
        t, n = clean_text(v)
        cleaned.append(t)
        hits.append(n)
    df["abstract"] = cleaned
    df["ad_hits"] = hits

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["patent_type"] = df["patent_type"].fillna("").astype(str).str.strip()
    df["doc_id"] = df["doc_id"].fillna("").astype(str).str.strip()
    df["city"] = df["city"].fillna("").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # ---- IPC ----
    ipc = df["ipc_main_raw"].map(ipc_parts)
    df["ipc_section"] = [x[0] for x in ipc]
    df["ipc_subclass"] = [x[1] for x in ipc]
    df["ipc_main"] = df["ipc_main_raw"].fillna("").astype(str).str.strip().str.upper()
    df.loc[df["ipc_main"].str.lower() == "nan", "ipc_main"] = ""

    # ---- applicant types ----
    df["applicant_types"] = df["applicant_type_raw"].map(split_multi)
    df["applicant_type"] = df["applicant_types"].map(lambda x: x[0] if x else "")

    # ---- lengths / text ----
    df["abstract_len"] = df["abstract"].str.len()
    df["text"] = df["title"] + "\n" + df["abstract"]
    df["text_len"] = df["text"].str.len()

    # ---- filters ----
    before = len(df)
    df = df[df["title"].str.len() > 0]
    f_title = before - len(df)
    before = len(df)
    df = df[df["abstract_len"] >= 20]
    f_abstract = before - len(df)
    before = len(df)
    df = df[df["year"].between(2015, 2024)]
    f_year = before - len(df)
    before = len(df)
    df = df[df["doc_id"].str.len() > 0]
    f_id = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset=["doc_id"], keep="first")
    f_dup = before - len(df)

    # ---- write jsonl ----
    out_path = os.path.join(CLEAN_DIR, "patents.jsonl")
    sample_path = os.path.join(CLEAN_DIR, "sample_20.jsonl")
    fields = ["doc_id", "title", "abstract", "patent_type", "ipc_main",
              "ipc_section", "ipc_subclass", "year", "applicant_types",
              "applicant_type", "city", "text", "abstract_len", "text_len"]
    with open(out_path, "w", encoding="utf-8") as f, \
         open(sample_path, "w", encoding="utf-8") as fs:
        for i, row in enumerate(df[fields].itertuples(index=False)):
            rec = row._asdict()
            rec["year"] = int(rec["year"])
            line = json.dumps(rec, ensure_ascii=False)
            f.write(line + "\n")
            if i < 20:
                fs.write(line + "\n")

    # ---- stats ----
    stats = {
        "raw_rows": raw_rows,
        "clean_rows": int(len(df)),
        "dropped": {
            "empty_title": int(f_title),
            "short_abstract": int(f_abstract),
            "year_out_of_range": int(f_year),
            "empty_doc_id": int(f_id),
            "duplicate_doc_id": int(f_dup),
        },
        "ad_cleaned_rows": int((df["ad_hits"] > 0).sum()),
        "ad_replacements": int(df["ad_hits"].sum()),
        "abstract_len": length_stats(df["abstract_len"]),
        "text_len": length_stats(df["text_len"]),
        "year_counts": {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()},
        "patent_type_counts": {str(k): int(v) for k, v in df["patent_type"].value_counts().items()},
        "ipc_section_counts": {str(k): int(v) for k, v in df["ipc_section"].value_counts().items()},
        "ipc_missing": int((df["ipc_main"] == "").sum()),
        "applicant_type_counts": {str(k): int(v) for k, v in df["applicant_type"].value_counts().head(20).items()},
        "top_cities": {str(k): int(v) for k, v in df["city"].value_counts().head(30).items()},
    }
    with open(os.path.join(STATS_DIR, "clean_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: stats[k] for k in ("raw_rows", "clean_rows", "dropped",
                                            "ad_cleaned_rows", "ipc_missing")},
                     ensure_ascii=False, indent=2))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
