# -*- coding: utf-8 -*-
"""Build a compact metadata file aligned with patents.jsonl (byte offsets included).

The retrieval service scans this file for candidate filtering, then reads the
full record (abstract) only for the final top-k.
"""
import json
import os

ROOT = r"D:\Projects\CN_Patent_RAG"
SRC = os.path.join(ROOT, "data", "clean", "patents.jsonl")
OUT = os.path.join(ROOT, "data", "index", "meta_compact.jsonl")

FIELDS = ("doc_id", "title", "ipc_main", "ipc_section", "ipc_subclass",
          "year", "patent_type", "applicant_type", "city")


def main():
    n = 0
    with open(SRC, "rb") as f, open(OUT, "w", encoding="utf-8") as g:
        pos = 0
        for line in f:
            rec = json.loads(line.decode("utf-8"))
            meta = {"off": pos}
            meta.update({k: rec[k] for k in FIELDS})
            g.write(json.dumps(meta, ensure_ascii=False) + "\n")
            pos += len(line)
            n += 1
    print("wrote %d -> %s (%.1f MB)" % (n, OUT, os.path.getsize(OUT) / 1048576))


if __name__ == "__main__":
    main()
