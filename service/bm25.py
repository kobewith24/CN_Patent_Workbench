# -*- coding: utf-8 -*-
"""SQLite FTS5 backed BM25 index over patent title + abstract.

Why FTS5: 519k documents of postings would cost 2-4GB of RAM, which this
16GB laptop cannot spare next to the 2.1GB HNSW index. FTS5 keeps the index on
disk (~ hundreds of MB) and gives BM25 ranking for free.

Rowids are ``line_index + 1`` so label ``rowid - 1`` matches the HNSW label and
the offsets table exactly.
"""
import json
import os
import sqlite3
import time

from .text import match_expr, tokenize

DEFAULT_DB = r"D:\Projects\CN_Patent_RAG\data\index\bm25_fts5.db"


class BM25Index:
    def __init__(self, path=DEFAULT_DB):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA mmap_size=1073741824")
        self.conn.execute("PRAGMA cache_size=-131072")

    def close(self):
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def search(self, query, k=200, w_title=3.0, w_body=1.0):
        expr = match_expr(query)
        if not expr:
            return []
        sql = ("SELECT rowid, bm25(docs, ?, ?) AS s FROM docs "
               "WHERE docs MATCH ? ORDER BY s LIMIT ?")
        try:
            rows = self.conn.execute(sql, (w_title, w_body, expr, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() returns negative values; flip so larger is better.
        return [(int(r[0]) - 1, -float(r[1])) for r in rows]


def build(corpus_path, db_path=DEFAULT_DB, limit=0, batch=2000, log=print):
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")
    conn.execute(
        "CREATE VIRTUAL TABLE docs USING fts5("
        "title, body, content='', tokenize='unicode61')"
    )

    t0 = time.time()
    n = 0
    buf = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if limit and n >= limit:
                break
            rec = json.loads(line)
            title = tokenize(rec.get("title", ""), max_tokens=128)
            body = tokenize(rec.get("abstract", ""), max_tokens=512)
            buf.append((n + 1, " ".join(title), " ".join(body)))
            n += 1
            if len(buf) >= batch:
                conn.executemany(
                    "INSERT INTO docs(rowid, title, body) VALUES (?, ?, ?)", buf)
                conn.commit()
                buf.clear()
                if n % 20000 == 0:
                    rate = n / max(1e-6, time.time() - t0)
                    log("tokenized %d (%.0f/s, %.0fs)" % (n, rate, time.time() - t0))
    if buf:
        conn.executemany("INSERT INTO docs(rowid, title, body) VALUES (?, ?, ?)", buf)
        conn.commit()

    conn.execute("INSERT INTO docs(docs) VALUES('optimize')")
    conn.commit()
    conn.close()
    log("done: %d docs in %.1fs, db=%.1f MB"
        % (n, time.time() - t0, os.path.getsize(db_path) / 1048576))
    return n
