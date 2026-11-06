# -*- coding: utf-8 -*-
"""Citation verification, v2.

Difference from v0: **every** sentence is a claim. Sentences without a `[n]`
marker cannot be supported by any source and are counted as unsupported. v0
only looked at cited sentences, so an answer could cite one safe sentence and
still score 1.0.

Metrics:
    citation_coverage = cited / all claims
    support_rate      = supported / cited          (among cited claims)
    unsupported_rate  = 1 - supported / all claims
"""
import json
import re
import unicodedata

import requests

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
NLI_DIR = r"D:\models\erlangshen-nli"

_SENT = re.compile(r"(?<=[。！？；\n])")
_REF = re.compile(r"\[(\d+)\]")

LLM_PROMPT = """你是严格的引用校验器。逐条判断下面的结论能否由它引用的专利摘要直接支持。
判定标准：摘要必须明确包含结论所需的信息；仅有主题相关不算支持；否定/排除性表述也必须有摘要依据。
只输出 JSON，不要输出其他文字：{{"results":[{{"id":1,"supported":true}}]}}

【结论与引用】
{body}
"""


def split_sentences(text):
    out = []
    for chunk in _SENT.split(text or ""):
        s = chunk.strip()
        if len(s) >= 8:
            out.append(s)
    return out


def build_claims(text, n_sources):
    claims = []
    for i, s in enumerate(split_sentences(text)):
        refs = [int(x) for x in _REF.findall(s) if 1 <= int(x) <= n_sources]
        clean = _REF.sub("", s).strip()
        if not clean:
            continue
        claims.append({"id": i, "sentence": s, "clean": clean, "refs": sorted(set(refs))})
    return claims


def judge_llm(claims, by_rank, timeout=180):
    """LLM judge; uncited claims are unsupported without a call."""
    for c in claims:
        if not c["refs"]:
            c.update(supported=False, score=0.0, by="no_citation", reason="no [n]")
    todo = [c for c in claims if c["refs"]]
    if not todo:
        return claims

    lines = []
    for i, c in enumerate(todo, 1):
        lines.append("%d. 结论：%s" % (i, c["clean"]))
        for r in c["refs"]:
            lines.append("   引用[%d]：%s" % (r, by_rank[r]["abstract"][:600]))
    prompt = LLM_PROMPT.format(body="\n".join(lines))
    try:
        resp = requests.post(OLLAMA_URL + "/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 256, "num_ctx": 4096},
        }, timeout=timeout)
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        results = {int(x["id"]): x for x in json.loads(raw).get("results", [])}
    except Exception as exc:  # noqa: BLE001
        print("llm judge failed: %s" % exc, flush=True)
        for c in todo:
            c.update(supported=False, score=0.0, by="llm_error", reason=str(exc)[:80])
        return claims

    for i, c in enumerate(todo, 1):
        r = results.get(i)
        if r is None:
            c.update(supported=False, score=0.0, by="llm", reason="judge_missing")
        else:
            c.update(supported=bool(r.get("supported")), score=1.0 if r.get("supported") else 0.0,
                     by="llm", reason=str(r.get("reason", ""))[:120])
    return claims


class NLIJudge:
    def __init__(self, path=NLI_DIR, device="cuda"):
        import os

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        try:
            self.tok = AutoTokenizer.from_pretrained(path)
        except Exception:  # noqa: BLE001
            from transformers import BertTokenizer
            self.tok = BertTokenizer(os.path.join(path, "vocab.txt"))
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        self.device = device
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next((k for k, v in labels.items() if "entail" in v or "蕴含" in v), 0)
        print("nli labels:", labels, "entail_idx:", self.entail_idx, flush=True)

    def score(self, pairs, batch=16):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch):
                chunk = pairs[i:i + batch]
                enc = self.tok([p for p, _ in chunk], [h for _, h in chunk],
                               truncation=True, max_length=512, padding=True,
                               return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, self.entail_idx].cpu().tolist()
                out.extend(float(x) for x in probs)
        return out


def judge_nli(claims, by_rank, nli, thresh=0.5):
    todo = [c for c in claims if c["refs"]]
    for c in claims:
        if not c["refs"]:
            c.update(supported=False, score=0.0, by="no_citation", reason="no [n]")
    if not todo or nli is None:
        return claims
    pairs = []
    for c in todo:
        for r in c["refs"]:
            pairs.append((by_rank[r]["abstract"], c["clean"]))
    scores = nli.score(pairs)
    k = 0
    for c in todo:
        s = max(scores[k:k + len(c["refs"])])
        k += len(c["refs"])
        c.update(supported=bool(s >= thresh), score=round(float(s), 4), by="nli")
    return claims


def judge_rerank(claims, by_rank, reranker, thresh=0.5):
    todo = [c for c in claims if c["refs"]]
    for c in claims:
        if not c["refs"]:
            c.update(supported=False, score=0.0, by="no_citation", reason="no [n]")
    if not todo or reranker is None:
        return claims
    pairs = []
    for c in todo:
        for r in c["refs"]:
            pairs.append((c["clean"], by_rank[r]["abstract"]))
    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
    k = 0
    for c in todo:
        s = max(float(x) for x in scores[k:k + len(c["refs"])])
        k += len(c["refs"])
        c.update(supported=bool(s >= thresh), score=round(s, 4), by="rerank")
    return claims


def summarize(claims):
    total = len(claims)
    cited = [c for c in claims if c["refs"]]
    supported = [c for c in claims if c["supported"]]
    supported_cited = [c for c in cited if c["supported"]]
    return {
        "claims": total,
        "cited": len(cited),
        "citation_coverage": round(len(cited) / total, 4) if total else 0.0,
        "support_rate": round(len(supported_cited) / len(cited), 4) if cited else 0.0,
        "unsupported_rate": round(1 - len(supported) / total, 4) if total else 1.0,
        "faithfulness_v1": round(len(supported_cited) / len(cited), 4) if cited else 0.0,
    }


# ---------------- evidence audit (no semantic judgement) ----------------

META_PAT = re.compile(
    r"未涉及|不涉及|未公开|未提及|未披露|未说明|未明确|未给出|未记载|未提供|"
    r"关联性不足|关联性较弱|相关性不足|证据不充分|证据不足|资料不足|"
    r"无法回答|无法判断|无法支持|不足以|不属于|缺乏|没有相关|（注|\(注")
_SECTION = re.compile(r"^(结论|明确相关的专利|相关但证据不充分的专利)\s*[：:]\s*")
_LEAD = re.compile(r"^[\s\-\u2013\u2014\u2022\u00b7\*\d]+[、.．)）]?\s*")
_NUM = re.compile(r"\d+(?:\.\d+)?%?")
_CJK = re.compile(r"[\u4e00-\u9fff]")

GENERIC_TERMS = frozenset("""
方法 装置 系统 结构 技术 问题 方案 效果 优点 结论 专利 摘要 装置 设备 包括 通过 进行 采用 实现 提供
具有 能够 可以 以及 并且 同时 此外 因此 从而 使得 需要 主要 相关 具体 一种 本发明 本实用新型 一些 部分
""".split())


def _norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    for a, b in (("～", "-"), ("~", "-"), ("—", "-"), ("–", "-"), ("％", "%")):
        s = s.replace(a, b)
    return s


def _strip(s):
    s = _SECTION.sub("", (s or "").strip())
    return _LEAD.sub("", s).strip()


def extract_numbers(text):
    s = _norm(text)
    # strip IPC codes, patent-ish identifiers and letter-hyphen-number names (Tween-20, DC53)
    s = re.sub(r"[A-H]\d{2}[A-Z]\s*\d+/\d+", " ", s)
    s = re.sub(r"[A-Za-z]+\s*[-\u2013]\s*\d+", " ", s)
    out = []
    for m in _NUM.finditer(s):
        v = m.group(0)
        if v.endswith("%"):
            out.append(v)
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if f.is_integer() and (0 <= f <= 10 or 1900 <= f <= 2100):
            continue
        out.append(v)
    return out


def content_terms(text):
    from .text import tokenize
    return [t for t in tokenize(text, ngrams=(), max_tokens=60)
            if len(t) >= 2 and not t.isdigit() and t not in GENERIC_TERMS and _CJK.search(t)]


def check_claim(clean, abstracts):
    """Mechanical grounding check for one claim against its cited abstracts."""
    text = _strip(_norm(clean))
    if not abstracts:
        return {"meta": False, "numbers": [], "numbers_missing": [], "term_coverage": 0.0,
                "grounded": False, "reason": "no_citation"}
    if META_PAT.search(text):
        return {"meta": True, "numbers": [], "numbers_missing": [], "term_coverage": 1.0,
                "grounded": False, "reason": "meta"}

    ref = _norm(" ".join(abstracts))
    nums = extract_numbers(text)
    ref_nums = set(extract_numbers(ref))
    missing = [n for n in nums if n not in ref_nums]

    terms = content_terms(text)
    hit = [t for t in terms if t in ref]
    cov = len(hit) / len(terms) if terms else 1.0
    grounded = (not missing) and cov >= 0.6
    return {"meta": False, "numbers": nums, "numbers_missing": missing,
            "term_coverage": round(cov, 4), "grounded": grounded,
            "reason": "ok" if grounded else ("number" if missing else "term")}


def audit_claims(claims, abstracts_by_rank):
    rows = []
    for c in claims:
        abstracts = [abstracts_by_rank[r] for r in c["refs"] if r in abstracts_by_rank]
        res = check_claim(c["clean"], abstracts)
        row = dict(c)
        row.update(res)
        rows.append(row)

    body = [r for r in rows if not r["meta"]]
    cited = [r for r in body if r["refs"]]
    num = [r for r in cited if r["numbers"]]
    summary = {
        "claims": len(rows),
        "meta_rate": round(1 - len(body) / len(rows), 4) if rows else 0.0,
        "citation_coverage": round(len(cited) / len(body), 4) if body else 0.0,
        "numeric_checked": len(num),
        "numeric_unsupported_rate": round(sum(1 for r in num if r["numbers_missing"]) / len(num), 4) if num else 0.0,
        "term_coverage": round(sum(r["term_coverage"] for r in cited) / len(cited), 4) if cited else 0.0,
        "grounded_rate": round(sum(1 for r in body if r["grounded"]) / len(body), 4) if body else 0.0,
    }
    return rows, summary
