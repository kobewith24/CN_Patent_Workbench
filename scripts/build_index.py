# -*- coding: utf-8 -*-
"""Build a single BGE-M3 dense index over the cleaned patent corpus.

Usage:
    python build_index.py --limit 2000     # smoke
    python build_index.py                  # full 519k

Outputs (data/index/):
    patent_emb[_smokeN].npy
    patent_index_1024[_smokeN].bin
    index_meta[_smokeN].json
"""
import argparse
import json
import os
import time

import hnswlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = r"D:\Projects\CN_Patent_RAG"
CLEAN = os.path.join(ROOT, "data", "clean", "patents.jsonl")
INDEX_DIR = os.path.join(ROOT, "data", "index")
MODEL_DIR = r"D:\models\bge-m3"

DIM = 1024
MAX_LEN = 384
BATCH = 96
M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 128
THREADS = 4


def load_records(limit=0):
    recs = []
    with open(CLEAN, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    os.makedirs(INDEX_DIR, exist_ok=True)
    suffix = "" if not args.limit else "_smoke%d" % args.limit
    emb_path = os.path.join(INDEX_DIR, "patent_emb%s.npy" % suffix)
    index_path = os.path.join(INDEX_DIR, "patent_index_1024%s.bin" % suffix)
    meta_path = os.path.join(INDEX_DIR, "index_meta%s.json" % suffix)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(THREADS)

    t0 = time.time()
    recs = load_records(args.limit)
    texts = [r["text"] for r in recs]
    n = len(texts)
    print("records=%d limit=%d" % (n, args.limit), flush=True)

    model = SentenceTransformer(MODEL_DIR, device="cuda")
    model.max_seq_length = MAX_LEN
    model.half()
    print("model loaded in %.1fs | max_len=%d | half" % (time.time() - t0, MAX_LEN), flush=True)

    emb = np.empty((n, DIM), dtype="float32")
    t1 = time.time()
    step = 20000
    for start in range(0, n, step):
        end = min(n, start + step)
        emb[start:end] = model.encode(
            texts[start:end],
            batch_size=args.batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        done = end
        rate = done / max(1e-6, time.time() - t1)
        eta = (n - done) / max(1e-6, rate)
        print("encoded %d/%d (%.0fs, %.0f/s, eta %.0fs)"
              % (done, n, time.time() - t1, rate, eta), flush=True)
    encode_seconds = time.time() - t1
    np.save(emb_path, emb)
    print("emb saved: %s %.2f GB" % (emb_path, os.path.getsize(emb_path) / 1073741824), flush=True)
    del texts

    t2 = time.time()
    index = hnswlib.Index(space="ip", dim=DIM)
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=M)
    for start in range(0, n, step):
        end = min(n, start + step)
        index.add_items(emb[start:end], np.arange(start, end), num_threads=THREADS)
        print("indexed %d/%d" % (end, n), flush=True)
    index.set_ef(EF_SEARCH)
    index.save_index(index_path)
    build_seconds = time.time() - t2

    meta = {
        "records": n,
        "dim": DIM,
        "max_len": MAX_LEN,
        "batch": args.batch,
        "model": MODEL_DIR,
        "hnsw_m": M,
        "ef_construction": EF_CONSTRUCTION,
        "ef_search": EF_SEARCH,
        "encode_seconds": round(encode_seconds, 1),
        "build_seconds": round(build_seconds, 1),
        "index_size_mb": round(os.path.getsize(index_path) / 1048576, 1),
        "emb_size_mb": round(os.path.getsize(emb_path) / 1048576, 1),
        "clean_file": CLEAN,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print("DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
