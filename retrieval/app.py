import hmac
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence import audit_claims, fragments, group_similar
from .settings import CONFIG, ROOT, manifest

STATE = {}
LOCK = threading.Lock()


def authorize(x_internal_token: str = Header(default="")):
    path = ROOT / "data/internal.token"
    expected = os.environ.get("PATENT_INTERNAL_TOKEN") or (path.read_text().strip() if path.exists() else "")
    if not expected or not hmac.compare_digest(expected, x_internal_token):
        raise HTTPException(401, "internal authentication required")


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ipc_section: str = Field(default="", pattern=r"^[A-H]?$", max_length=1)
    ipc_subclass: str = Field(default="", pattern=r"^(?:[A-H]\d{2}[A-Z]?)?$", max_length=4)
    year_from: int = Field(default=0, ge=0, le=2100)
    year_to: int = Field(default=0, ge=0, le=2100)
    patent_type: str = Field(default="", max_length=20)
    city: str = Field(default="", max_length=40)
    applicant_type: str = Field(default="", max_length=40)

    @model_validator(mode="after")
    def order(self):
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("起始年份不得晚于结束年份")
        if self.patent_type not in ("", "发明申请", "发明授权", "实用新型"):
            raise ValueError("未知专利类型")
        if self.ipc_section and self.ipc_subclass and not self.ipc_subclass.startswith(self.ipc_section):
            raise ValueError("IPC 部类与小类冲突")
        return self


class Search(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    filters: Filters = Field(default_factory=Filters)
    k: int = Field(default=CONFIG["top_k"], ge=1, le=50)
    candidates: int = Field(default=CONFIG["candidates"], ge=10, le=200)
    mode: str = Field(default="hybrid", pattern=r"^(hybrid|dense|bm25)$")
    rerank: bool = True


class Documents(BaseModel):
    doc_ids: list[str] = Field(min_length=1, max_length=50)


class Audit(BaseModel):
    claims: list[dict] = Field(max_length=100)
    documents: list[dict] = Field(max_length=10)


@asynccontextmanager
async def lifespan(app):
    from .retrieval import Retriever
    STATE["manifest"] = manifest()
    STATE["retriever"] = Retriever()
    yield
    if STATE["retriever"].bm25:
        STATE["retriever"].bm25.close()


app = FastAPI(title="Patent Workbench Retrieval", lifespan=lifespan,
              dependencies=[Depends(authorize)])


@app.get("/health")
def health():
    return {"ready": "retriever" in STATE, "manifest": STATE.get("manifest"),
            "reranker_ready": bool(STATE.get("retriever") and STATE["retriever"].reranker is not None),
            "candidates": CONFIG["candidates"], "generation": False}


@app.post("/search")
def search(req: Search):
    if not req.query.strip():
        raise HTTPException(422, "检索词不能为空")
    started = time.perf_counter()
    with LOCK:
        r = STATE["retriever"]
        if req.rerank and r.reranker is None:
            raise HTTPException(503, "重排器不可用，未静默退回其他排序方式")
        hits, scanned = r.search(req.query.strip(), k=req.k, candidates=req.candidates,
                                 filters=req.filters.model_dump(), mode=req.mode, rerank=req.rerank)
    for h in hits:
        h["evidence"] = fragments(h)
    group_count = group_similar(hits)
    return {"query": req.query.strip(), "filters": req.filters.model_dump(), "hits": hits,
            "group_count": group_count, "scanned": scanned,
            "elapsed_ms": round((time.perf_counter()-started)*1000, 1),
            "config": {"candidates": req.candidates, "mode": req.mode, "rerank": req.rerank},
            "corpus": STATE["manifest"], "no_results": not hits,
            "notice": "没有符合当前条件的候选，未自动放宽过滤条件。" if not hits else "候选按检索相关性排序，摘要信息需结合原文复核。"}


@app.post("/validate_filters")
def validate_filters(req: Filters):
    return req.model_dump()


@app.post("/documents")
def documents(req: Documents):
    with LOCK:
        hits = STATE["retriever"].documents(req.doc_ids)
    for h in hits:
        h["evidence"] = fragments(h)
    return {"documents": hits, "missing": [d for d in req.doc_ids if d not in {h['doc_id'] for h in hits}]}


@app.post("/audit")
def audit(req: Audit):
    # Never trust a caller-provided quote, offsets or fabricated document content.
    ids = list(dict.fromkeys(str(d.get("doc_id", "")) for d in req.documents))
    if not ids:
        return audit_claims(req.claims, [])
    with LOCK:
        hits = STATE["retriever"].documents(ids)
    for h in hits:
        h["evidence"] = fragments(h)
    return audit_claims(req.claims, hits)
