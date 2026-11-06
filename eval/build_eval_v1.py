# -*- coding: utf-8 -*-
"""Build the v1 eval sets.

B set: 240 topic questions. A question is generated from the abstract (never the
title); the gold set is then defined by the *question's own* rare terms — patents
in the same IPC subclass sharing two of them — so gold is relevant to the question,
not to the seed's wording.

A set: 60 seeds for pooled judging (top-k from dense / BM25 / hybrid).
Planner set: 30 comparison questions built from real same-topic patent pairs,
with phrasings the current regex does not cover.

Usage:
    python eval/build_eval_v1.py --b 240 --a 60 --planner 30
"""
import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
META = os.path.join(ROOT, "data", "index", "meta_compact.jsonl")
FULL = os.path.join(ROOT, "data", "clean", "patents.jsonl")
FTS = os.path.join(ROOT, "data", "index", "bm25_fts5.db")
OUT = os.path.join(ROOT, "eval")

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"

PROMPT = """下面是一件中国专利的摘要。请根据摘要写一个中文技术问题，要求：
1. 是工程师或研发人员会问的问题，必须读了摘要才能回答；
2. 用你自己的话，不要照抄摘要里的整句，不要出现“本实用新型/本发明”；
3. 问题要围绕这些技术点：{terms}；
4. 不要提专利号，不要提及“摘要”；
5. 只输出问题本身，15 到 40 字，不要编号、不要解释、不要引号。

摘要：{abstract}
"""

COMPARE_TEMPLATES = [
    "同样是{s}领域，{a}和{b}各自的技术路线是什么？",
    "{a} 相比 {b}，在实现方式上有什么不同？",
    "如果要在这两件专利里选一个，{a}与{b}各自的优缺点是什么？",
    "请说明 {a} 相对于 {b} 的改进点。",
    "{b} 和 {a} 哪个更适合大规模部署？",
    "从技术实现看，{a}、{b} 有何不同？",
    "这两件专利（{a} / {b}）分别解决了什么问题？",
    "结合 {a} 与 {b}，说明它们在结构上的差异。",
    "{a} 和 {b} 分别适用于什么场景？",
    "如果要复现 {a} 的方案，能从 {b} 里借鉴什么？",
]


def load_meta():
    doc_ids, subclasses, sections, years, offsets = [], [], [], [], []
    with open(META, "rb") as f:
        for line in f:
            m = json.loads(line.decode("utf-8"))
            doc_ids.append(m["doc_id"])
            subclasses.append(m["ipc_subclass"])
            sections.append(m["ipc_section"])
            years.append(m["year"])
            offsets.append(m["off"])
    return doc_ids, subclasses, sections, years, offsets


def read_rec(off):
    with open(FULL, "rb") as f:
        f.seek(off)
        return json.loads(f.readline().decode("utf-8"))


def ask_ollama(prompt, timeout=120):
    for attempt in range(2):
        try:
            r = requests.post(OLLAMA + "/api/generate", json={
                "model": MODEL, "prompt": prompt, "stream": False, "think": False,
                "options": {"temperature": 0.7, "num_predict": 80, "num_ctx": 2048},
            }, timeout=timeout)
            r.raise_for_status()
            text = (r.json().get("response") or "").strip()
            text = re.sub(r"^(问题|问：|Q[:：])\s*", "", text).split("\n")[0].strip().strip('"“”')
            if 8 <= len(text) <= 100:
                return text
        except Exception as exc:  # noqa: BLE001
            print("ollama retry", attempt, exc, flush=True)
            time.sleep(1)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=240)
    ap.add_argument("--a", type=int, default=60)
    ap.add_argument("--planner", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()

    random.seed(args.seed)
    doc_ids, subclasses, sections, years, offsets = load_meta()
    print("meta", len(doc_ids), flush=True)

    by_sub = {}
    for i, sc in enumerate(subclasses):
        if sc:
            by_sub.setdefault(sc, []).append(i)

    conn = sqlite3.connect(FTS)
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_vocab USING fts5vocab('docs', 'row')")

    def df(term):
        row = conn.execute("SELECT doc FROM docs_vocab WHERE term=?", (term,)).fetchone()
        return row[0] if row else 0

    from service.text import tokenize  # noqa: E402

    def find_gold(seed_idx, tokens):
        """Same-subclass patents sharing two rare tokens of the question."""
        sc = subclasses[seed_idx]
        pool = set(by_sub.get(sc, []))
        cand = [(df(t), t) for t in set(tokens) if len(t) >= 2 and 2 <= df(t) <= 500]
        cand.sort()
        picked = []
        for _d, t in cand:
            if any(t in p or p in t for p in picked):
                continue
            picked.append(t)
        terms = picked[:8]
        for i in range(len(terms) - 1):
            for j in range(i + 1, min(i + 5, len(terms))):
                t1, t2 = terms[i], terms[j]
                expr = '"%s" AND "%s"' % (t1.replace('"', '""'), t2.replace('"', '""'))
                try:
                    rows = conn.execute(
                        "SELECT rowid FROM docs WHERE docs MATCH ? LIMIT 300", (expr,)).fetchall()
                except sqlite3.OperationalError:
                    continue
                gold = [r[0] - 1 for r in rows if (r[0] - 1) in pool]
                if 2 <= len(gold) <= 20:
                    return gold, [t1, t2]
        for d, t in cand:
            if d > 80:
                continue
            expr = '"%s"' % t.replace('"', '""')
            try:
                rows = conn.execute(
                    "SELECT rowid FROM docs WHERE docs MATCH ? LIMIT 300", (expr,)).fetchall()
            except sqlite3.OperationalError:
                continue
            gold = [r[0] - 1 for r in rows if (r[0] - 1) in pool]
            if 2 <= len(gold) <= 20:
                return gold, [t]
        return [], []

    by_sec = {}
    for i, s in enumerate(sections):
        by_sec.setdefault(s or "NA", []).append(i)
    need = args.b + args.a
    quota = need * 5 // max(1, len(by_sec)) + 60
    seeds = []
    for s in sorted(by_sec):
        picks = [i for i in by_sec[s] if years[i] and years[i] >= 2015]
        seeds.extend(random.sample(picks, min(quota, len(picks))))
    random.shuffle(seeds)
    print("seeds", len(seeds), flush=True)

    fb = open(os.path.join(OUT, "gold_v1_b.jsonl"), "w", encoding="utf-8")
    fa = open(os.path.join(OUT, "gold_v1_a.jsonl"), "w", encoding="utf-8")
    nb = na = 0
    b_items = []
    t0 = time.time()

    for idx in seeds:
        if nb >= args.b and na >= args.a:
            break
        rec = read_rec(offsets[idx])
        abstract = rec.get("abstract", "") or ""
        if len(abstract) < 80:
            continue
        toks = [t for t in tokenize(rec.get("title", "") + " " + abstract, ngrams=(), max_tokens=80)
                if not t.isascii()]
        gold, used = find_gold(idx, toks)
        if not gold:
            continue
        q = ask_ollama(PROMPT.format(abstract=abstract[:600], terms="、".join(used)))
        if not q:
            continue

        want_b = nb < args.b
        item = {
            "qid": ("b%03d" % (nb + 1)) if want_b else ("a%03d" % (na + 1)),
            "question": q,
            "gold_doc_ids": [doc_ids[g] for g in gold][:40],
            "seed_doc_id": doc_ids[idx],
            "ipc_subclass": subclasses[idx],
            "terms": used,
            "synthetic": True,
        }
        if want_b:
            fb.write(json.dumps(item, ensure_ascii=False) + "\n")
            fb.flush()
            b_items.append(item)
            nb += 1
        else:
            fa.write(json.dumps(item, ensure_ascii=False) + "\n")
            fa.flush()
            na += 1
        if (nb + na) % 20 == 0:
            print("b=%d a=%d %.0fs" % (nb, na, time.time() - t0), flush=True)

    fb.close()
    fa.close()
    print("wrote b=%d a=%d" % (nb, na), flush=True)

    # planner: pairs drawn from tight B clusters (2-8 docs), varied phrasing
    planner = []
    tpl_i = 0
    id2idx = {d: i for i, d in enumerate(doc_ids)}
    for item in b_items:
        if len(planner) >= args.planner:
            break
        if not (2 <= len(item["gold_doc_ids"]) <= 8):
            continue
        pair = random.sample(item["gold_doc_ids"], 2)
        if pair[0] not in id2idx or pair[1] not in id2idx:
            continue
        a_rec = read_rec(offsets[id2idx[pair[0]]])
        b_rec = read_rec(offsets[id2idx[pair[1]]])
        if len(a_rec.get("title", "")) < 6 or len(b_rec.get("title", "")) < 6:
            continue
        for _ in range(2):
            if len(planner) >= args.planner:
                break
            tpl = COMPARE_TEMPLATES[tpl_i % len(COMPARE_TEMPLATES)]
            tpl_i += 1
            planner.append({
                "qid": "p%02d" % (len(planner) + 1),
                "question": tpl.format(a="《%s》" % a_rec["title"], b="《%s》" % b_rec["title"],
                                       s=item["ipc_subclass"]),
                "gold_doc_ids": [pair[0], pair[1]],
                "ipc_subclass": item["ipc_subclass"],
                "synthetic": True,
            })

    with open(os.path.join(OUT, "gold_v1_planner.jsonl"), "w", encoding="utf-8") as f:
        for r in planner:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote planner", len(planner), flush=True)


if __name__ == "__main__":
    main()
