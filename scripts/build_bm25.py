# -*- coding: utf-8 -*-
"""Build the FTS5 BM25 index over data/clean/patents.jsonl.

Usage:
    python scripts/build_bm25.py --limit 2000   # smoke, writes bm25_fts5_smoke2000.db
    python scripts/build_bm25.py                # full 519k
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service.bm25 import DEFAULT_DB, build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    db = args.db
    if args.limit:
        db = db.replace(".db", "_smoke%d.db" % args.limit)
    corpus = os.path.join(ROOT, "data", "clean", "patents.jsonl")

    t = build(corpus, db, limit=args.limit, batch=args.batch)
    print("rows=%d db=%s size=%.1fMB" % (t, db, os.path.getsize(db) / 1048576))


if __name__ == "__main__":
    main()
