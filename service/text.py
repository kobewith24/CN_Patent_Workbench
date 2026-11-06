# -*- coding: utf-8 -*-
"""Shared tokenizer for the BM25 side.

One function is used by both index build and query time, so the two can never
drift. Tokens are: jieba search-mode words, Chinese 2/3-grams, and ASCII
alnum runs kept whole (so ``FXa`` / ``G06T`` survive).
"""
import re
import unicodedata

import jieba

_ASCII = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_HAS_WORD = re.compile(r"[\w\u4e00-\u9fff]")

STOPWORDS = frozenset("""
的 了 和 与 及 或 等 中 为 是 在 对 将 有 可 以 该 其 本 被 把 从 到 由 于 而 并 且 也 就 都 还 又 很 更 最
一种 一个 所述 以及 并且 通过 进行 实现 提供 涉及 用于 能够 可以 具有 包括 包含 属于 采用 利用 使得 从而
""".split())


def _normalize(text):
    return unicodedata.normalize("NFKC", text or "").lower()


def tokenize(text, ngrams=(2, 3), max_tokens=512, stopwords=STOPWORDS):
    """Return a de-duplicated token list, order preserved."""
    s = _normalize(text)
    if not s.strip():
        return []

    out = []
    for w in jieba.cut_for_search(s):
        w = w.strip()
        if not w or w in stopwords or not _HAS_WORD.search(w):
            continue
        out.append(w)

    for run in _CJK_RUN.findall(s):
        n = len(run)
        for g in ngrams:
            if n < g:
                continue
            for i in range(n - g + 1):
                out.append(run[i:i + g])

    for m in _ASCII.finditer(s):
        out.append(m.group(0))

    seen = set()
    uniq = []
    for w in out:
        if w in seen:
            continue
        seen.add(w)
        uniq.append(w)
        if len(uniq) >= max_tokens:
            break
    return uniq


def match_expr(text, max_terms=120):
    """Build an FTS5 MATCH expression (OR of quoted tokens)."""
    terms = tokenize(text, max_tokens=max_terms)
    if not terms:
        return ""
    return " OR ".join('"%s"' % t.replace('"', '""') for t in terms)
