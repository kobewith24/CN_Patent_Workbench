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
from .filters import MetaTable, exact_topk

ROOT = r"D:\Projects\CN_Patent_RAG"
INDEX_PATH = os.path.join(ROOT, "data", "index", "patent_index_1024.bin")
META_PATH = os.path.join(ROOT, "data", "index", "meta_compact.jsonl")
FULL_PATH = os.path.join(ROOT, "data", "clean", "patents.jsonl")
BM25_PATH = os.path.join(ROOT, "data", "index", "bm25_fts5.db")
EMB_PATH = os.path.join(ROOT, "data", "index", "patent_emb.npy")
EXACT_MAX = 150000
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
        self.meta = MetaTable(META_PATH, self.index.get_current_count())
        self.emb = np.load(EMB_PATH, mmap_mode="r")

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
    def encode(self, q):
        v = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(v, dtype="float32")[0]

    def dense_vec(self, v, fetch_k=50):
        k = min(int(fetch_k), self.index.get_current_count())
        labels, dists = self.index.knn_query(v.reshape(1, -1), k=k)
        return [(int(lab), float(1.0 - dist)) for lab, dist in zip(labels[0], dists[0])]

    def dense(self, q, fetch_k=50):
        return self.dense_vec(self.encode(q), fetch_k)

    def _hnsw_filtered(self, v, mask, want, cap=20000):
        n = self.index.get_current_count()
        fetch = 1000
        kept = []
        while True:
            fetch = min(fetch, n)
            labels, dists = self.index.knn_query(v.reshape(1, -1), k=fetch)
            kept = [(int(l), float(1.0 - d)) for l, d in zip(labels[0], dists[0]) if mask[l]]
            if len(kept) >= want or fetch >= cap or fetch >= n:
                break
            fetch *= 2
        return kept[:want]

    def _sparse_filtered(self, q, mask, want):
        kept = []
        for limit in (5000, 20000, 50000):
            rows = self.sparse(q, limit)
            kept = [(lab, sc) for lab, sc in rows if mask[lab]]
            if len(kept) >= min(want, 100) or len(rows) < limit:
                break
        return kept[:want]

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
        n = self.index.get_current_count()

        if has_filter:
            mask = self.meta.mask(filters)
            eligible = int(mask.sum())
            if eligible == 0:
                return [], 0
            fetch_k = min(max(candidates * 20, 1000), 5000)
            v = self.encode(q)
            d = exact_topk(self.emb, v, mask, fetch_k) if eligible <= EXACT_MAX \
                else self._hnsw_filtered(v, mask, candidates)
            s = self._sparse_filtered(q, mask, max(candidates, 200)) \
                if mode in ("bm25", "hybrid") else []
            lists = []
            if mode in ("dense", "hybrid"):
                lists.append(d)
            if mode in ("bm25", "hybrid"):
                lists.append(s)
            scanned = eligible
        else:
            fetch_k = min(int(candidates), n)
            lists = []
            if mode in ("dense", "hybrid"):
                lists.append(self.dense(q, fetch_k))
            if mode in ("bm25", "hybrid"):
                lists.append(self.sparse(q, fetch_k))
            scanned = fetch_k

        if not lists or not any(lists):
            return [], scanned

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
        return hits, scanned
