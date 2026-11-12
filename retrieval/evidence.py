"""Evidence IDs resolve to exact slices of original abstracts. No semantic claims."""
import hashlib
import re
import unicodedata
from decimal import Decimal

SPLIT = re.compile(r"[^。！？；\n]+[。！？；]?|\n+")
# Counts, small values and ranges are retained. Units are not discarded.
NUMBER = re.compile(r"(?<![A-Za-z\d])[-+]?\d+(?:\.\d+)?(?:\s*(?:%|℃|°C|°|mm|cm|nm|μm|um|km|mL|ml|mg|kg|MPa|kPa|Pa|mV|kV|mA|kW|kHz|MHz|GHz|Hz|ms|m|L|g|V|A|W|s|秒|分钟|小时|毫米|厘米|米|个|组|层|倍))?", re.I)
UNIT_ALIASES = {"°c": "℃", "毫米": "mm", "厘米": "cm", "米": "m", "秒": "s", "μm": "um"}


def fragments(hit):
    text = hit.get("abstract", "")
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    out = []
    for m in SPLIT.finditer(text):
        start, end = m.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        out.append({"id": f"{hit['doc_id']}:{digest}:{start}-{end}", "doc_id": hit["doc_id"],
                    "start": start, "end": end, "text": text[start:end], "abstract_hash": digest})
    return out


def normalized_numbers(text):
    s = unicodedata.normalize("NFKC", text or "")
    s = re.sub(r"(?<=[\d%])\s*[-–—~～]\s*(?=\d)", " ", s)
    # Identifiers such as IPC, patent IDs and Tween-20 are not measurements.
    s = re.sub(r"\b(?:CN\d+[A-Z0-9.]*|[A-H]\d{2}[A-Z](?:\s*\d+/\d+)?)\b", " ", s, flags=re.I)
    s = re.sub(r"\b[A-Za-z]+[-]?\d+[A-Za-z]*\b", " ", s)
    out = []
    for m in NUMBER.finditer(s):
        value = re.match(r"[-+]?\d+(?:\.\d+)?", m.group()).group()
        unit = m.group()[len(value):].strip().lower()
        unit = UNIT_ALIASES.get(unit, unit)
        out.append((format(Decimal(value).normalize(), "f"), unit))
    return out


def audit_claims(claims, hits):
    by_doc = {h["doc_id"]: h for h in hits}
    by_id = {e["id"]: e for h in hits for e in h["evidence"]}
    rows = []
    for raw in claims:
        text = str(raw.get("text", "")).strip()[:2000]
        proposed = raw.get("evidence_ids") or []
        ids = list(dict.fromkeys(i for i in proposed if isinstance(i, str))) if isinstance(proposed, list) else []
        declared_doc = str(raw.get("doc_id") or "")
        invalid = [i for i in ids if i not in by_id]
        refs = [by_id[i] for i in ids if i in by_id]
        cross_doc = bool(declared_doc and any(e["doc_id"] != declared_doc for e in refs))
        ref_text = " ".join(e["text"] for e in refs)
        available = normalized_numbers(ref_text)
        missing = [v + u for v, u in normalized_numbers(text)
                   if (v, u) not in available and not (not u and any(a == v for a, _ in available))]
        intact = all(by_doc[e["doc_id"]]["abstract"][e["start"]:e["end"]] == e["text"] for e in refs)
        status = "evidence_located"
        if not text or not refs or invalid or cross_doc or not intact or declared_doc not in by_doc:
            status = "invalid_reference"
        elif missing:
            status = "numeric_review"
        rows.append({"text": text, "doc_id": declared_doc, "dimension": str(raw.get("dimension") or "技术方案"),
                     "evidence_ids": ids, "evidence": refs, "status": status,
                     "missing_numbers": missing, "invalid_ids": invalid,
                     "note": "原文定位与数值检查通过；语义是否支持结论仍需复核。" if status == "evidence_located"
                     else "引用无效，未纳入正式结论。" if status == "invalid_reference" else "结论中的数值或单位未在所引片段中找到，请复核。"})
    return {"claims": rows, "summary": {"total": len(rows),
            "evidence_located": sum(r["status"] == "evidence_located" for r in rows),
            "numeric_review": sum(r["status"] == "numeric_review" for r in rows),
            "invalid_reference": sum(r["status"] == "invalid_reference" for r in rows),
            "semantic_verification": False}}


def group_similar(hits, threshold=0.88):
    """Display grouping only. Never merges applications or claims a patent family."""
    def grams(text):
        s = re.sub(r"\W", "", text)
        return {s[i:i+3] for i in range(max(0, len(s)-2))}
    groups, prior = [], []
    for hit in hits:
        g = grams(hit.get("abstract", ""))
        group = hit["doc_id"]
        for old, old_group in prior:
            if g and old and len(g & old) / len(g | old) >= threshold:
                group = old_group
                break
        hit["similar_group"] = group
        prior.append((g, group))
        groups.append(group)
    return len(set(groups))
