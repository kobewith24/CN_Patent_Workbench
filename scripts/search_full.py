# -*- coding: utf-8 -*-
"""Top-k sanity check against the full 519k index (no metadata filters yet)."""
import json
import time

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = r"D:\Projects\CN_Patent_RAG"
INDEX = ROOT + r"\data\index\patent_index_1024.bin"
CLEAN = ROOT + r"\data\clean\patents.jsonl"
OUT = ROOT + r"\data\stats\search_full.json"
MODEL = r"D:\models\bge-m3"

QUERIES = [
    "基于深度学习的图像去噪方法",
    "托卡马克装置中等离子体破裂的预测",
    "一种用于手机散热的均热板结构",
    "马克思主义基本原理课教学用投影装置",
    "锂离子电池热失控的早期预警",
]


def build_offsets(path):
    offs = []
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            offs.append(pos)
            pos += len(line)
    return offs


def read_at(path, off):
    with open(path, "rb") as f:
        f.seek(off)
        return json.loads(f.readline().decode("utf-8"))


def main():
    t0 = time.time()
    offsets = build_offsets(CLEAN)
    print("offsets=%d (%.1fs)" % (len(offsets), time.time() - t0), flush=True)

    index = hnswlib.Index(space="ip", dim=1024)
    index.load_index(INDEX)
    index.set_ef(128)
    print("index loaded: %d (%.1fs)" % (index.get_current_count(), time.time() - t0), flush=True)

    model = SentenceTransformer(MODEL, device="cuda")
    model.max_seq_length = 384
    model.half()

    out = []
    for q in QUERIES:
        t1 = time.time()
        v = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        v = np.asarray(v, dtype="float32")
        labels, dists = index.knn_query(v, k=5)
        hits = []
        for lab, dist in zip(labels[0], dists[0]):
            r = read_at(CLEAN, offsets[int(lab)])
            hits.append({
                "score": round(float(1.0 - dist), 4),
                "doc_id": r["doc_id"],
                "title": r["title"],
                "ipc": r["ipc_main"],
                "year": r["year"],
                "type": r["patent_type"],
            })
        out.append({"query": q, "elapsed_ms": round((time.time() - t1) * 1000, 1), "hits": hits})
        print("q=%s %.0fms" % (q, (time.time() - t1) * 1000), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
