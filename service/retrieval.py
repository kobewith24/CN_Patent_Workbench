# -*- coding: utf-8 -*-
"""Hybrid retrieval: dense (bge-m3 + HNSW) + sparse (FTS5 BM25) + RRF + rerank.

Single source of truth for both the FastAPI service and the offline eval, so
the two cannot drift apart.
"""
import json
import os

import hnswlib
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from .bm25 import BM25Index

ROOT = r"D:\Projects\CN_Patent_RAG"
INDEX_PATH = os.path.join(ROOT, "data", "index", "patent_index_1024.bin")
META_PATH = os.path.join(ROOT, "data", "index", "meta_compact.jsonl")
FULL_PATH = os.path.join(ROOT, "data", "clean", "patents.jsonl")
BM25_PATH = os.path.join(ROOT, "data", "index", "bm25_fts5.db")
MODEL_DIR = r"D:\models\bge-m3"
RERANKER_DIR = r"D:\models\bge-reranker-v2-m3"

DIM = 1024
MAX_LEN = 384
EF_SEARCH = 128
THREADS = 4
RRF_K = 60

FILTER_KEYS = ("ipc_section", "ipc_subclass", "year_from", "year_to",
               "patent_type", "city", "applicant_type")

EMPTY_FILTERS = {
    "ipc_section": "", "ipc_subclass": "", "year_from": 0, "year_to": 0,
    "patent_type": "", "city": "", "applicant_type": "",
}


def _load_offsets(path):
    offs = []
    pos = 0
    with open(path, "rb") as f:
        for line in f:
            offs.append(pos)
            pos += len(line)
    return offs


def _passes(meta, filters):
    if filters["ipc_section"] and meta["ipc_section"] != filters["ipc_section"]:
        return False
    if filters["ipc_subclass"] and not meta["ipc_subclass"].startswith(filters["ipc_subclass"]):
        return False
    if filters["year_from"] and meta["year"] < filters["year_from"]:
        return False
    if filters["year_to"] and meta["year"] > filters["year_to"]:
        return False
    if filters["patent_type"] and meta["patent_type"] != filters["patent_type"]:
        return False
    if filters["applicant_type"] and filters["applicant_type"] not in meta["applicant_type"]:
        return False
    if filters["city"] and filters["city"] not in meta["city"]:
        return False
    return True


def _bigram_sim(a, b):
    """Character-bigram overlap, kept for parity with the v0 rerank blend."""
    import re
    a = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", a or "")
    b = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", b or "")
    if len(a) < 2 or len(b) < 2:
        return 0.0
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / max(1, len(B))


class Retriever:
    def __init__(self, device="cuda", with_bm25=True, with_reranker=True):
        torch.set_num_threads(THREADS)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        self.model = SentenceTransformer(MODEL_DIR, device=device)
        self.model.max_seq_length = MAX_LEN
        self.model.half()
        torch.cuda.empty_cache()

        self.reranker = None
        if with_reranker:
            try:
                self.reranker = CrossEncoder(RERANKER_DIR, device=device)
                self.reranker.model.half()
            except Exception as exc:  # noqa: BLE001
                print("reranker load failed: %s" % exc, flush=True)
        torch.cuda.empty_cache()

        self.index = hnswlib.Index(space="ip", dim=DIM)
        self.index.load_index(INDEX_PATH)
        self.index.set_ef(EF_SEARCH)
        self.offsets = _load_offsets(META_PATH)

        self.bm25 = None
        if with_bm25 and os.path.exists(BM25_PATH):
            self.bm25 = BM25Index(BM25_PATH)
        print("retriever ready | docs=%d | bm25=%s | reranker=%s"
              % (self.index.get_current_count(), bool(self.bm25), bool(self.reranker)),
              flush=True)

    # ---------- io ----------
    def _read_metas(self, labels):
        out = []
        with open(META_PATH, "rb") as f:
            for lab in labels:
                f.seek(self.offsets[int(lab)])
                out.append(json.loads(f.readline().decode("utf-8")))
        return out

    def _read_fulls(self, metas):
        out = []
        with open(FULL_PATH, "rb") as f:
            for m in metas:
                f.seek(m["off"])
                out.append(json.loads(f.readline().decode("utf-8")))
        return out

    # ---------- retrieval ----------
    def dense(self, q, fetch_k=50):
        v = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        v = np.asarray(v, dtype="float32")
        k = min(int(fetch_k), self.index.get_current_count())
        labels, dists = self.index.knn_query(v, k=k)
        return [(int(lab), float(1.0 - dist)) for lab, dist in zip(labels[0], dists[0])]

    def sparse(self, q, fetch_k=200):
        if self.bm25 is None:
            return []
        return self.bm25.search(q, k=int(fetch_k))

    @staticmethod
    def rrf(lists, k=RRF_K, top=50):
        agg = {}
        for lst in lists:
            for rank, (lab, _score) in enumerate(lst, 1):
                agg[lab] = agg.get(lab, 0.0) + 1.0 / (k + rank)
        return sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top]

    def rerank(self, q, items, title_boost=True, batch=8):
        if not items or self.reranker is None:
            return list(items)
        metas = self._read_metas([lab for lab, _ in items])
        fulls = self._read_fulls(metas)
        pairs = [(q, (rec.get("title", "") + "\n" + rec.get("abstract", "")))
                 for rec in fulls]
        scores = self.reranker.predict(pairs, batch_size=batch, show_progress_bar=False)
        out = []
        for i, (lab, _old) in enumerate(items):
            s = float(scores[i])
            if title_boost:
                s = 0.9 * s + 0.1 * _bigram_sim(q, fulls[i].get("title", ""))
            out.append((lab, s))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def search(self, q, k=5, candidates=50, filters=None, mode="hybrid",
               rerank=True, title_boost=True):
        filters = dict(EMPTY_FILTERS, **(filters or {}))
        has_filter = any(filters[x] for x in FILTER_KEYS)

        fetch_k = candidates
        if has_filter:
            fetch_k = min(max(candidates * 20, 1000), 5000)
        fetch_k = min(int(fetch_k), self.index.get_current_count())

        lists = []
        if mode in ("dense", "hybrid"):
            d = self.dense(q, fetch_k)
            if has_filter and d:
                metas = self._read_metas([lab for lab, _ in d])
                d = [(lab, s) for (lab, s), m in zip(d, metas) if _passes(m, filters)]
            lists.append(d)

        if mode in ("bm25", "hybrid"):
            k_sparse = max(fetch_k, 500) if has_filter else fetch_k
            s = self.sparse(q, k_sparse)
            if has_filter and s:
                metas = self._read_metas([lab for lab, _ in s])
                s = [(lab, sc) for (lab, sc), m in zip(s, metas) if _passes(m, filters)]
            lists.append(s)

        if not lists or not any(lists):
            return [], 0

        if mode == "hybrid":
            merged = self.rrf(lists, top=candidates)
        else:
            merged = lists[0][:candidates]

        if rerank and self.reranker is not None:
            merged = self.rerank(q, merged, title_boost=title_boost)
        merged = merged[:k]

        metas = self._read_metas([lab for lab, _ in merged])
        fulls = self._read_fulls(metas)
        hits = []
        for rank, ((_lab, score), m, rec) in enumerate(zip(merged, metas, fulls), 1):
            hits.append({
                "rank": rank,
                "score": round(float(score), 4),
                "doc_id": m["doc_id"],
                "title": m["title"],
                "abstract": rec["abstract"],
                "ipc_main": m["ipc_main"],
                "ipc_section": m["ipc_section"],
                "ipc_subclass": m["ipc_subclass"],
                "year": m["year"],
                "patent_type": m["patent_type"],
                "applicant_type": m["applicant_type"],
                "city": m["city"],
            })
        return hits, fetch_k
