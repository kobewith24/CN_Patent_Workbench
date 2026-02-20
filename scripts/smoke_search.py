# -*- coding: utf-8 -*-
"""Query the 2k smoke index to sanity-check BGE-M3 retrieval."""
import json

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = r"D:\Projects\CN_Patent_RAG"
INDEX = ROOT + r"\data\index\patent_index_1024_smoke2000.bin"
CLEAN = ROOT + r"\data\clean\patents.jsonl"
OUT = ROOT + r"\data\stats\smoke_query.json"
MODEL = r"D:\models\bge-m3"

QUERIES = [
    "基于深度学习的图像去噪方法",
    "托卡马克装置中等离子体破裂的预测",
    "一种用于手机散热的均热板结构",
    "马克思主义基本原理课教学用投影装置",
]


def main():
    recs = []
    with open(CLEAN, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            recs.append(json.loads(line))

    index = hnswlib.Index(space="ip", dim=1024)
    index.load_index(INDEX)
    index.set_ef(64)

    model = SentenceTransformer(MODEL, device="cuda")
    model.max_seq_length = 384
    model.half()

    out = []
    for q in QUERIES:
        v = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        v = np.asarray(v, dtype="float32")
        labels, dists = index.knn_query(v, k=5)
        hits = []
        for lab, dist in zip(labels[0], dists[0]):
            r = recs[int(lab)]
            hits.append({
                "score": round(float(1.0 - dist), 4),
                "doc_id": r["doc_id"],
                "title": r["title"],
                "ipc": r["ipc_main"],
                "year": r["year"],
            })
        out.append({"query": q, "hits": hits})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
