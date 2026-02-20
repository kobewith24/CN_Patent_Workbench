# -*- coding: utf-8 -*-
"""FastAPI retrieval service: single BGE-M3 index + metadata filters.

Run:
    cd D:\Projects\CN_Patent_RAG
    python -m uvicorn service.app:app --host 127.0.0.1 --port 8001
"""
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

import hnswlib
import numpy as np
import requests
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = r"D:\Projects\CN_Patent_RAG"
INDEX_PATH = os.path.join(ROOT, "data", "index", "patent_index_1024.bin")
META_PATH = os.path.join(ROOT, "data", "index", "meta_compact.jsonl")
FULL_PATH = os.path.join(ROOT, "data", "clean", "patents.jsonl")
MODEL_DIR = r"D:\models\bge-m3"
RERANKER_DIR = r"D:\models\bge-reranker-v2-m3"

DIM = 1024
MAX_LEN = 384
EF_SEARCH = 128
THREADS = 4

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
TRACE_DIR = os.path.join(ROOT, "logs", "trace")

STATE = {}


class Hit(BaseModel):
    rank: int
    score: float
    doc_id: str
    title: str
    abstract: str
    ipc_main: str
    ipc_section: str
    ipc_subclass: str
    year: int
    patent_type: str
    applicant_type: str
    city: str


class RetrieveResponse(BaseModel):
    query: str
    filters: dict
    scanned: int
    elapsed_ms: float
    hits: list


class AnswerResponse(BaseModel):
    trace_id: str
    query: str
    answer: str
    model: str
    filters: dict
    retrieval_ms: float
    generation_ms: float
    elapsed_ms: float
    sub_queries: list
    claims: list
    faithfulness: float
    cited_ranks: list
    uncited_ranks: list
    sources: list


def _load_offsets(path):
    offs = []
    pos = 0
    with open(path, "rb") as f:
        for line in f:
            offs.append(pos)
            pos += len(line)
    return offs


def _read_metas(labels, offsets):
    out = []
    with open(META_PATH, "rb") as f:
        for lab in labels:
            f.seek(offsets[int(lab)])
            out.append(json.loads(f.readline().decode("utf-8")))
    return out


def _read_fulls(metas):
    out = []
    with open(FULL_PATH, "rb") as f:
        for m in metas:
            f.seek(m["off"])
            out.append(json.loads(f.readline().decode("utf-8")))
    return out


def _passes(m, f):
    if f["ipc_section"] and m["ipc_section"] != f["ipc_section"]:
        return False
    if f["ipc_subclass"] and not m["ipc_subclass"].startswith(f["ipc_subclass"]):
        return False
    if f["year_from"] and m["year"] < f["year_from"]:
        return False
    if f["year_to"] and m["year"] > f["year_to"]:
        return False
    if f["patent_type"] and m["patent_type"] != f["patent_type"]:
        return False
    if f["applicant_type"] and f["applicant_type"] not in m["applicant_type"]:
        return False
    if f["city"] and f["city"] not in m["city"]:
        return False
    return True


def _plan(q):
    """Heuristic decomposition for comparison questions."""
    q = q.strip()
    m = re.match(r"^(?:请)?(?:对比|比较)\s*(.+?)\s*(?:和|与|、)\s*(.+?)[？?]?$", q)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        tail = b.split("的")[-1].strip() if "的" in b else ""
        if tail and tail not in a:
            a = a + tail
        return [a, b]
    m = re.match(r"^(.+?)\s*(?:和|与)\s*(.+?)\s*(?:的)?(?:区别|差异|对比|优劣)", q)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return [q]


def _bigram_sim(a, b):
    """Character-bigram overlap, used as a light title-match rerank feature."""
    a = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", a or "")
    b = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", b or "")
    if len(a) < 2 or len(b) < 2:
        return 0.0
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / max(1, len(B))


def _strip_filters(q, filters):
    """Remove filter phrases already enforced by the API from the prompt query."""
    s = q
    s = re.sub(r"\b[A-H]\d{2}[A-Z]\s*(?:领域)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b[A-H]\s*部(?:\s*领域)?", "", s)
    if filters.get("year_from") or filters.get("year_to"):
        s = re.sub(r"(?:19|20)\d{2}\s*年?\s*(?:以后|之后|以来|以前|之前|前|后)?", "", s)
        s = re.sub(r"近\s*\d+\s*年", "", s)
    if filters.get("patent_type"):
        s = s.replace(filters["patent_type"], "")
    if filters.get("city"):
        s = s.replace(filters["city"], "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^[，。？?、\s]+|[，。？?、\s]+$", "", s)
    return s or q


def _build_prompt(q, hits, sub_queries=None, filters=None):
    lines = [
        "你是专利技术情报助手。请严格依据【资料】回答，不要编造资料以外的内容。",
        "要求：",
        "1. 直接回答，不要复述【资料】里的编号、专利号、标题、IPC 字段清单；",
        "2. 按以下结构输出：",
        "   结论：一两句话直接回答；",
        "   明确相关的专利：逐条说明，句末用 [编号]；",
        "   相关但证据不充分的专利：逐条说明，句末用 [编号]，没有就省略；",
        "3. 只有资料完全无关时才说“资料不足”，不要因为个别条件不确定就整体拒答；",
        "4. 申请人名称、诉讼、许可、权利要求范围不在资料中，遇到这类问题请明确说明无法回答；",
        "5. 正文引用只写 [编号]，不要重复专利号、标题、IPC、年份、类型等字段；",
        "6. 资料中的每一条 [编号] 都必须在答案中出现一次：明确相关或证据不充分；不要遗漏。",
    ]
    if filters:
        active = []
        if filters.get("ipc_section"):
            active.append("IPC 部类=%s" % filters["ipc_section"])
        if filters.get("ipc_subclass"):
            active.append("IPC 大类=%s" % filters["ipc_subclass"])
        if filters.get("year_from"):
            active.append("年份>=%s" % filters["year_from"])
        if filters.get("year_to"):
            active.append("年份<=%s" % filters["year_to"])
        if filters.get("patent_type"):
            active.append("专利类型=%s" % filters["patent_type"])
        if filters.get("city"):
            active.append("城市=%s" % filters["city"])
        if filters.get("applicant_type"):
            active.append("申请人类型=%s" % filters["applicant_type"])
        if active:
            lines.append("当前过滤条件：%s；以下资料均已满足这些条件，不要再逐条核对。"
                         % "，".join(active))
    if sub_queries and len(sub_queries) > 1:
        lines.append("对比要求：请分别说明两者的技术路线，再总结差异；不要只罗列专利。")
        lines.append("子问题：" + "；".join("%d) %s" % (i + 1, s) for i, s in enumerate(sub_queries)))
    lines += ["", "【资料】"]
    for h in hits:
        lines.append("[%d] %s（%d，%s）" % (h.rank, h.title, h.year, h.ipc_main))
        lines.append("摘要：" + h.abstract)
        lines.append("")
    lines.append("【问题】")
    lines.append(q)
    return "\n".join(lines)


def _extract_claims(text, hits):
    by_rank = {h.rank: h for h in hits}
    claims = []
    for sent in re.split(r"(?<=[。！？；\n])", text):
        sent = sent.strip()
        if not sent:
            continue
        refs = [int(x) for x in re.findall(r"\[(\d+)\]", sent)]
        refs = [r for r in refs if r in by_rank]
        if not refs:
            continue
        clean = re.sub(r"\[\d+\]", "", sent).strip()
        if not clean:
            continue
        claims.append({"sentence": sent, "refs": refs, "clean": clean})
    return claims, by_rank


def _claim_out(c, support, supported, verified_by, reason=""):
    return {
        "sentence": c["sentence"],
        "refs": c["refs"],
        "support": support,
        "supported": supported,
        "verified_by": verified_by,
        "reason": reason,
    }


def _llm_judge(claims, by_rank):
    """Ask the generator model to judge each claim against its cited abstracts."""
    lines = [
        "你是严格的引用校验器。逐条判断下面的结论是否能由它所引用的专利摘要直接支持。",
        "判定标准：摘要必须明确包含结论所需的信息；仅有主题相关不算支持；否定/排除性表述也必须有摘要依据。",
        '只输出 JSON，不要输出其他文字：{"results":[{"id":1,"supported":true,"reason":"..."}]}',
        "",
        "【结论与引用】",
    ]
    for i, c in enumerate(claims, 1):
        lines.append("%d. 结论：%s" % (i, c["clean"]))
        for r in c["refs"]:
            lines.append("   引用[%d]：%s" % (r, by_rank[r].abstract[:300]))
    prompt = "\n".join(lines)
    try:
        resp = requests.post(
            OLLAMA_URL + "/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_predict": 512, "num_ctx": 4096},
            },
            timeout=180,
        )
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        results = {int(x["id"]): x for x in data.get("results", [])}
    except Exception as exc:  # noqa: BLE001
        print("llm judge failed: %s" % exc, flush=True)
        return None

    out = []
    for i, c in enumerate(claims, 1):
        r = results.get(i)
        if r is None:
            out.append(_claim_out(c, 0.0, False, "llm", "judge_missing"))
            continue
        supported = bool(r.get("supported"))
        out.append(_claim_out(c, 1.0 if supported else 0.0, supported,
                              "llm", str(r.get("reason", ""))[:120]))
    return out


def _verify_reranker(claims, by_rank):
    reranker = STATE.get("reranker")
    if reranker is None:
        return []
    out = []
    for c in claims:
        pairs = [(c["clean"], by_rank[r].abstract) for r in c["refs"]]
        try:
            scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
            best = max(float(s) for s in scores)
        except Exception as exc:  # noqa: BLE001
            print("rerank verify failed: %s" % exc, flush=True)
            continue
        out.append(_claim_out(c, round(best, 4), best >= 0.5, "reranker"))
    return out


def _verify_answer(text, hits):
    claims, by_rank = _extract_claims(text, hits)
    if not claims:
        return []
    judged = _llm_judge(claims, by_rank)
    if judged is not None:
        return judged
    return _verify_reranker(claims, by_rank)


def _load():
    t0 = time.time()
    torch.set_num_threads(THREADS)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model = SentenceTransformer(MODEL_DIR, device="cuda")
    model.max_seq_length = MAX_LEN
    model.half()
    torch.cuda.empty_cache()

    try:
        reranker = CrossEncoder(RERANKER_DIR, device="cuda")
        reranker.model.half()
        print("reranker loaded on cuda (fp16)", flush=True)
    except Exception as exc:  # noqa: BLE001
        reranker = None
        print("reranker load failed: %s" % exc, flush=True)
    torch.cuda.empty_cache()

    index = hnswlib.Index(space="ip", dim=DIM)
    index.load_index(INDEX_PATH)
    index.set_ef(EF_SEARCH)

    offsets = _load_offsets(META_PATH)
    STATE.update({
        "model": model,
        "reranker": reranker,
        "index": index,
        "offsets": offsets,
        "device": str(model.device),
        "load_seconds": round(time.time() - t0, 1),
    })
    print("service ready in %.1fs | docs=%d | device=%s | reranker=%s"
          % (STATE["load_seconds"], index.get_current_count(), STATE["device"],
             "cpu" if reranker is not None else "off"), flush=True)


@asynccontextmanager
async def lifespan(app):
    _load()
    yield


app = FastAPI(title="CN Patent RAG Retrieval", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(ROOT, "service", "static", "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    if "index" not in STATE:
        return {"status": "loading"}
    return {
        "status": "ok",
        "docs": STATE["index"].get_current_count(),
        "device": STATE["device"],
        "load_seconds": STATE["load_seconds"],
    }


@app.get("/retrieve", response_model=RetrieveResponse)
def retrieve(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
    candidates: int = Query(50, ge=1, le=500),
    ipc_section: str = Query(""),
    ipc_subclass: str = Query(""),
    year_from: int = Query(0),
    year_to: int = Query(0),
    patent_type: str = Query(""),
    city: str = Query(""),
    applicant_type: str = Query(""),
    use_filters: bool = Query(True),
    use_rerank: bool = Query(True),
    use_title_boost: bool = Query(True),
):
    if "index" not in STATE:
        return {"query": q, "filters": {}, "scanned": 0, "elapsed_ms": 0, "hits": []}

    t0 = time.time()
    filters = {
        "ipc_section": ipc_section.strip().upper(),
        "ipc_subclass": ipc_subclass.strip().upper(),
        "year_from": year_from,
        "year_to": year_to,
        "patent_type": patent_type.strip(),
        "city": city.strip(),
        "applicant_type": applicant_type.strip(),
    }
    has_filter = any([
        filters["ipc_section"], filters["ipc_subclass"], filters["year_from"],
        filters["year_to"], filters["patent_type"], filters["city"],
        filters["applicant_type"],
    ])

    fetch_k = candidates if not has_filter else min(max(candidates * 20, 1000), 5000)
    fetch_k = min(fetch_k, STATE["index"].get_current_count())

    v = STATE["model"].encode([q], normalize_embeddings=True, convert_to_numpy=True)
    v = np.asarray(v, dtype="float32")
    labels, dists = STATE["index"].knn_query(v, k=fetch_k)

    metas = _read_metas(labels[0], STATE["offsets"])
    keep_n = min(max(k * 10, 30), 50)
    kept = []
    for lab, dist, m in zip(labels[0], dists[0], metas):
        if (not use_filters) or _passes(m, filters):
            kept.append((1.0 - float(dist), m))
            if len(kept) >= keep_n:
                break

    fulls = _read_fulls([m for _, m in kept])
    scores = None
    reranker = STATE.get("reranker")
    if use_rerank and reranker is not None and fulls:
        try:
            scores = reranker.predict(
                [(q, rec["title"] + "\n" + rec["abstract"]) for rec in fulls],
                batch_size=8,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            print("rerank failed: %s" % exc, flush=True)
            scores = None

    if scores is not None:
        final_scores = [
            (0.9 * float(scores[i]) + 0.1 * _bigram_sim(q, fulls[i]["title"]))
            if use_title_boost else float(scores[i])
            for i in range(len(fulls))
        ]
        order = sorted(range(len(fulls)), key=lambda i: final_scores[i], reverse=True)[:k]
    else:
        final_scores = None
        order = list(range(min(k, len(fulls))))

    hits = []
    for rank, idx in enumerate(order, 1):
        m = kept[idx][1]
        rec = fulls[idx]
        score = final_scores[idx] if final_scores is not None else kept[idx][0]
        hits.append(Hit(
            rank=rank,
            score=round(score, 4),
            doc_id=m["doc_id"],
            title=m["title"],
            abstract=rec["abstract"],
            ipc_main=m["ipc_main"],
            ipc_section=m["ipc_section"],
            ipc_subclass=m["ipc_subclass"],
            year=m["year"],
            patent_type=m["patent_type"],
            applicant_type=m["applicant_type"],
            city=m["city"],
        ))

    return RetrieveResponse(
        query=q,
        filters=filters,
        scanned=int(fetch_k),
        elapsed_ms=round((time.time() - t0) * 1000, 1),
        hits=hits,
    )


def _write_trace(payload):
    try:
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, payload["trace_id"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print("trace write failed: %s" % exc, flush=True)


@app.get("/answer", response_model=AnswerResponse)
def answer(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=10),
    ipc_section: str = Query(""),
    ipc_subclass: str = Query(""),
    year_from: int = Query(0),
    year_to: int = Query(0),
    patent_type: str = Query(""),
    city: str = Query(""),
    applicant_type: str = Query(""),
    use_filters: bool = Query(True),
    use_rerank: bool = Query(True),
    use_title_boost: bool = Query(True),
    use_multistep: bool = Query(True),
    use_verify: bool = Query(True),
):
    t0 = time.time()
    trace_id = uuid.uuid4().hex[:12]
    sub_queries = _plan(q) if use_multistep else [q]
    t_retr = time.time()
    merged = {}
    filters = {}
    if len(sub_queries) > 1:
        per_k = max(2, (k + 1) // 2)
        for sq in sub_queries:
            rr = retrieve(
                q=sq, k=per_k, candidates=50,
                ipc_section=ipc_section, ipc_subclass=ipc_subclass,
                year_from=year_from, year_to=year_to,
                patent_type=patent_type, city=city, applicant_type=applicant_type,
                use_filters=use_filters, use_rerank=use_rerank,
                use_title_boost=use_title_boost,
            )
            filters = rr.filters
            for h in rr.hits:
                if h.doc_id not in merged or h.score > merged[h.doc_id].score:
                    merged[h.doc_id] = h
        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:k]
    else:
        rr = retrieve(
            q=q, k=k, candidates=50,
            ipc_section=ipc_section, ipc_subclass=ipc_subclass,
            year_from=year_from, year_to=year_to,
            patent_type=patent_type, city=city, applicant_type=applicant_type,
            use_filters=use_filters, use_rerank=use_rerank,
            use_title_boost=use_title_boost,
        )
        filters = rr.filters
        hits = rr.hits

    retrieval_ms = round((time.time() - t_retr) * 1000, 1)
    for i, h in enumerate(hits, 1):
        h.rank = i

    if not hits:
        elapsed = round((time.time() - t0) * 1000, 1)
        text = "资料不足：没有检索到相关专利。"
        _write_trace({
            "trace_id": trace_id, "query": q, "filters": filters,
            "sub_queries": sub_queries, "hits": [],
            "retrieval_ms": retrieval_ms, "generation_ms": 0,
            "elapsed_ms": elapsed, "answer": text, "claims": [],
            "faithfulness": -1.0, "cited_ranks": [], "uncited_ranks": [],
        })
        return AnswerResponse(
            trace_id=trace_id, query=q, answer=text, model=OLLAMA_MODEL,
            filters=filters, retrieval_ms=retrieval_ms, generation_ms=0,
            elapsed_ms=elapsed,
            sub_queries=sub_queries, claims=[], faithfulness=-1.0,
            cited_ranks=[], uncited_ranks=[], sources=[],
        )

    semantic_q = _strip_filters(q, filters)
    prompt = _build_prompt(semantic_q, hits, sub_queries, filters)
    t1 = time.time()
    try:
        resp = requests.post(
            OLLAMA_URL + "/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.1, "num_predict": 384, "num_ctx": 4096},
            },
            timeout=180,
        )
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="generation failed: %s" % exc)
    generation_ms = round((time.time() - t1) * 1000, 1)
    if not text:
        text = "资料不足：模型没有生成有效答案。"

    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)
                    if 1 <= int(x) <= len(hits)})
    uncited = [h.rank for h in hits if h.rank not in cited]

    claims = _verify_answer(text, hits) if use_verify else []
    supported = sum(1 for c in claims if c["supported"])
    faithfulness = round(supported / len(claims), 4) if claims else -1.0

    elapsed = round((time.time() - t0) * 1000, 1)
    _write_trace({
        "trace_id": trace_id, "query": q, "filters": filters,
        "sub_queries": sub_queries,
        "hits": [{"rank": h.rank, "doc_id": h.doc_id, "title": h.title,
                  "score": h.score} for h in hits],
        "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
        "elapsed_ms": elapsed, "answer": text, "claims": claims,
        "faithfulness": faithfulness, "cited_ranks": cited,
        "uncited_ranks": uncited,
        "flags": {"rerank": use_rerank, "title_boost": use_title_boost,
                  "multistep": use_multistep, "verify": use_verify},
    })

    return AnswerResponse(
        trace_id=trace_id, query=q, answer=text, model=OLLAMA_MODEL, filters=filters,
        retrieval_ms=retrieval_ms, generation_ms=generation_ms,
        elapsed_ms=elapsed,
        sub_queries=sub_queries, claims=claims, faithfulness=faithfulness,
        cited_ranks=cited, uncited_ranks=uncited,
        sources=[h.model_dump() for h in hits],
    )


@app.get("/answer_stream")
def answer_stream(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=10),
    ipc_section: str = Query(""),
    ipc_subclass: str = Query(""),
    year_from: int = Query(0),
    year_to: int = Query(0),
    patent_type: str = Query(""),
    city: str = Query(""),
    applicant_type: str = Query(""),
):
    """SSE: meta -> token* -> verify -> done."""
    def sse(event, data):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))

    def gen():
        t0 = time.time()
        trace_id = uuid.uuid4().hex[:12]
        sub_queries = _plan(q)
        merged = {}
        filters = {}
        t_retr = time.time()
        if len(sub_queries) > 1:
            per_k = max(2, (k + 1) // 2)
            for sq in sub_queries:
                rr = retrieve(q=sq, k=per_k, candidates=50,
                              ipc_section=ipc_section, ipc_subclass=ipc_subclass,
                              year_from=year_from, year_to=year_to,
                              patent_type=patent_type, city=city,
                              applicant_type=applicant_type)
                filters = rr.filters
                for h in rr.hits:
                    if h.doc_id not in merged or h.score > merged[h.doc_id].score:
                        merged[h.doc_id] = h
            hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:k]
        else:
            rr = retrieve(q=q, k=k, candidates=50,
                          ipc_section=ipc_section, ipc_subclass=ipc_subclass,
                          year_from=year_from, year_to=year_to,
                          patent_type=patent_type, city=city,
                          applicant_type=applicant_type)
            filters = rr.filters
            hits = rr.hits
        retrieval_ms = round((time.time() - t_retr) * 1000, 1)
        for i, h in enumerate(hits, 1):
            h.rank = i

        yield sse("meta", {
            "trace_id": trace_id,
            "sub_queries": sub_queries,
            "retrieval_ms": retrieval_ms,
            "sources": [h.model_dump() for h in hits],
        })

        if not hits:
            text = "资料不足：没有检索到相关专利。"
            yield sse("token", text)
            yield sse("done", {"trace_id": trace_id, "elapsed_ms": round((time.time() - t0) * 1000, 1)})
            return

        prompt = _build_prompt(_strip_filters(q, filters), hits, sub_queries, filters)
        text = ""
        try:
            with requests.post(
                OLLAMA_URL + "/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "think": False,
                    "options": {"temperature": 0.1, "num_predict": 384, "num_ctx": 4096},
                },
                stream=True,
                timeout=180,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line.decode("utf-8"))
                    tok = obj.get("response") or ""
                    if tok:
                        text += tok
                        yield sse("token", tok)
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"message": str(exc)})
            return

        claims = _verify_answer(text, hits)
        supported = sum(1 for c in claims if c["supported"])
        faithfulness = round(supported / len(claims), 4) if claims else -1.0
        cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)
                        if 1 <= int(x) <= len(hits)})
        uncited = [h.rank for h in hits if h.rank not in cited]
        elapsed = round((time.time() - t0) * 1000, 1)

        yield sse("verify", {"claims": claims, "faithfulness": faithfulness,
                             "cited_ranks": cited, "uncited_ranks": uncited})
        yield sse("done", {"trace_id": trace_id, "elapsed_ms": elapsed})
        _write_trace({
            "trace_id": trace_id, "query": q, "filters": filters,
            "sub_queries": sub_queries,
            "hits": [{"rank": h.rank, "doc_id": h.doc_id, "title": h.title,
                      "score": h.score} for h in hits],
            "retrieval_ms": retrieval_ms, "elapsed_ms": elapsed,
            "answer": text, "claims": claims, "faithfulness": faithfulness,
            "cited_ranks": cited, "uncited_ranks": uncited,
        })

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    path = os.path.join(TRACE_DIR, trace_id + ".json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="trace not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
