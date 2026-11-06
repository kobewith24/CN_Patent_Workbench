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

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .retrieval import EMPTY_FILTERS, Retriever
from .verify import build_claims, judge_llm, summarize

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


class VerifyRequest(BaseModel):
    answer: str
    sources: list


class VerifyResponse(BaseModel):
    claims: list
    faithfulness: float
    citation_coverage: float = -1.0
    unsupported_rate: float = -1.0


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


def _verify_text(text, hits):
    """v2: every sentence is a claim; uncited sentences count as unsupported."""
    by_rank = {h.rank: {"abstract": h.abstract} for h in hits}
    claims = judge_llm(build_claims(text, len(hits)), by_rank)
    out = [{
        "sentence": c["sentence"],
        "refs": c["refs"],
        "support": c.get("score", 0.0),
        "supported": c["supported"],
        "verified_by": c.get("by", "llm"),
        "reason": c.get("reason", ""),
    } for c in claims]
    return out, summarize(claims)



def _load():
    t0 = time.time()
    r = Retriever()
    STATE.update({
        "retriever": r,
        "reranker": r.reranker,
        "device": str(r.model.device),
        "load_seconds": round(time.time() - t0, 1),
    })
    print("service ready in %.1fs | docs=%d | device=%s | bm25=%s"
          % (STATE["load_seconds"], r.index.get_current_count(),
             STATE["device"], bool(r.bm25)), flush=True)


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
    r = STATE.get("retriever")
    if r is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "docs": r.index.get_current_count(),
        "device": STATE["device"],
        "load_seconds": STATE["load_seconds"],
        "bm25": bool(r.bm25),
    }


@app.get("/retrieve", response_model=RetrieveResponse)
def retrieve(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
    candidates: int = Query(100, ge=1, le=500),
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
    use_bm25: bool = Query(True),
    mode: str = Query(""),
):
    r = STATE.get("retriever")
    if r is None:
        return RetrieveResponse(query=q, filters={}, scanned=0, elapsed_ms=0, hits=[])

    filters = {
        "ipc_section": ipc_section.strip().upper(),
        "ipc_subclass": ipc_subclass.strip().upper(),
        "year_from": year_from,
        "year_to": year_to,
        "patent_type": patent_type.strip(),
        "city": city.strip(),
        "applicant_type": applicant_type.strip(),
    }
    if not use_filters:
        filters = dict(EMPTY_FILTERS)

    m = mode.strip().lower() or ("hybrid" if (use_bm25 and r.bm25 is not None) else "dense")

    t0 = time.time()
    rows, scanned = r.search(
        q, k=k, candidates=candidates, filters=filters, mode=m,
        rerank=use_rerank, title_boost=use_title_boost,
    )
    hits = [Hit(**row) for row in rows]
    return RetrieveResponse(
        query=q, filters=filters, scanned=scanned,
        elapsed_ms=round((time.time() - t0) * 1000, 1), hits=hits,
    )


@app.get("/retrieve_plan", response_model=RetrieveResponse)
def retrieve_plan(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
    candidates: int = Query(100, ge=1, le=500),
    ipc_section: str = Query(""),
    ipc_subclass: str = Query(""),
    year_from: int = Query(0),
    year_to: int = Query(0),
    patent_type: str = Query(""),
    city: str = Query(""),
    applicant_type: str = Query(""),
):
    subs = _plan(q)
    if len(subs) <= 1:
        return retrieve(q=q, k=k, candidates=candidates, ipc_section=ipc_section,
                        ipc_subclass=ipc_subclass, year_from=year_from,
                        year_to=year_to, patent_type=patent_type, city=city,
                        applicant_type=applicant_type)

    t0 = time.time()
    per_k = max(2, (k + 1) // 2)
    merged = {}
    filters = {}
    scanned = 0
    for sq in subs:
        rr = retrieve(q=sq, k=per_k, candidates=candidates,
                      ipc_section=ipc_section, ipc_subclass=ipc_subclass,
                      year_from=year_from, year_to=year_to,
                      patent_type=patent_type, city=city,
                      applicant_type=applicant_type)
        filters = rr.filters
        scanned += rr.scanned
        for h in rr.hits:
            if h.doc_id not in merged or h.score > merged[h.doc_id].score:
                merged[h.doc_id] = h

    hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:k]
    for i, h in enumerate(hits, 1):
        h.rank = i
    return RetrieveResponse(
        query=q,
        filters=filters,
        scanned=scanned,
        elapsed_ms=round((time.time() - t0) * 1000, 1),
        hits=hits,
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    hits = []
    for s in req.sources:
        if not isinstance(s, dict):
            continue
        try:
            hits.append(Hit(**s))
        except Exception:  # noqa: BLE001
            continue
    if not hits:
        return VerifyResponse(claims=[], faithfulness=-1.0)
    claims, stats = _verify_text(req.answer, hits)
    return VerifyResponse(claims=claims, faithfulness=stats["support_rate"],
                          citation_coverage=stats["citation_coverage"],
                          unsupported_rate=stats["unsupported_rate"])


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

    claims, vstats = _verify_text(text, hits) if use_verify else ([], None)
    faithfulness = vstats["support_rate"] if vstats else -1.0

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

        claims, vstats = _verify_text(text, hits)
        faithfulness = vstats["support_rate"] if vstats else -1.0
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
    uvicorn.run("service.app:app", host="127.0.0.1", port=8001)
