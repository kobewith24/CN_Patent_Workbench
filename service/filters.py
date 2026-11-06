# -*- coding: utf-8 -*-
"""Metadata masks and exact filtered retrieval.

The v0 path fetched 2k-5k HNSW neighbours and dropped the ones that failed the
filter, so a narrow filter (one city + two years) could return nothing at all.
Here the filter is a boolean mask over 519k rows, and when the eligible set is
small enough we score it exactly instead of hoping HNSW surfaced it.
"""
import json

import numpy as np

FIELDS = ("year", "ipc_section", "ipc_subclass", "patent_type", "city", "applicant_type")


class MetaTable:
    def __init__(self, path, n):
        self.n = n
        self.codes = {f: np.zeros(n, dtype=np.int32) for f in FIELDS}
        self.values = {f: [] for f in FIELDS}
        self._maps = {f: {} for f in FIELDS}
        self.years = np.zeros(n, dtype=np.int16)

        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= n:
                    break
                m = json.loads(line)
                self.years[i] = int(m.get("year") or 0)
                for f in FIELDS:
                    v = m.get(f)
                    key = int(v) if f == "year" else str(v)
                    mp = self._maps[f]
                    code = mp.get(key)
                    if code is None:
                        code = len(self.values[f])
                        mp[key] = code
                        self.values[f].append(key)
                    self.codes[f][i] = code

    def _code_mask(self, field, value):
        code = self._maps[field].get(value)
        if code is None:
            return np.zeros(self.n, dtype=bool)
        return self.codes[field] == code

    def _contains_mask(self, field, value):
        ids = [i for i, v in enumerate(self.values[field]) if value in v]
        if not ids:
            return np.zeros(self.n, dtype=bool)
        return np.isin(self.codes[field], ids)

    def _prefix_mask(self, field, prefix):
        ids = [i for i, v in enumerate(self.values[field]) if v.startswith(prefix)]
        if not ids:
            return np.zeros(self.n, dtype=bool)
        return np.isin(self.codes[field], ids)

    def mask(self, f):
        m = np.ones(self.n, dtype=bool)
        if f.get("ipc_section"):
            m &= self._code_mask("ipc_section", f["ipc_section"])
        if f.get("ipc_subclass"):
            m &= self._prefix_mask("ipc_subclass", f["ipc_subclass"])
        if f.get("year_from"):
            m &= self.years >= int(f["year_from"])
        if f.get("year_to"):
            m &= self.years <= int(f["year_to"])
        if f.get("patent_type"):
            m &= self._code_mask("patent_type", f["patent_type"])
        if f.get("city"):
            m &= self._contains_mask("city", f["city"])
        if f.get("applicant_type"):
            m &= self._contains_mask("applicant_type", f["applicant_type"])
        return m


def exact_topk(emb, q, mask, k, chunk=8192):
    """Exact inner-product top-k over the rows selected by `mask`."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    scores = np.empty(idx.size, dtype=np.float32)
    for start in range(0, idx.size, chunk):
        part = idx[start:start + chunk]
        scores[start:start + chunk] = np.asarray(emb[part], dtype=np.float32) @ q
    kk = min(int(k), idx.size)
    top = np.argpartition(-scores, kk - 1)[:kk]
    top = top[np.argsort(-scores[top])]
    return [(int(idx[i]), float(scores[i])) for i in top]
