# -*- coding: utf-8 -*-
"""SQLite FTS5 backed BM25 index over patent title + abstract.

Why FTS5: 519k documents of postings would cost 2-4GB of RAM, which this
16GB laptop cannot spare next to the 2.1GB HNSW index. FTS5 keeps the index on
disk (~ hundreds of MB) and gives BM25 ranking for free.

Rowids are ``line_index + 1`` so label ``rowid - 1`` matches the HNSW label and
the offsets table exactly.
"""
import sqlite3
from pathlib import Path

from .text import match_expr, tokenize

from .settings import BM25_PATH
DEFAULT_DB = str(BM25_PATH)


class BM25Index:
    def __init__(self, path=DEFAULT_DB):
        self.path = path
        # The legacy corpus is an external read-only asset; never create sidecars.
        self.conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True,
                                    check_same_thread=False)
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
