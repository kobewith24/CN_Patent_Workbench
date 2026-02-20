# -*- coding: utf-8 -*-
"""Inspect the raw patent CSV: encoding, columns, dtypes, sample rows."""
import json
import os

import pandas as pd

RAW = r"D:\Projects\CN_Patent_RAG\data\raw\patents_sample_500k.csv"
OUT = r"D:\Projects\CN_Patent_RAG\data\stats\inspect.json"


def try_encoding(enc):
    try:
        df = pd.read_csv(RAW, nrows=5, encoding=enc)
        return df
    except Exception as exc:  # noqa: BLE001
        return exc


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    result = {"size_bytes": os.path.getsize(RAW)}
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        got = try_encoding(enc)
        if isinstance(got, pd.DataFrame):
            result["encoding"] = enc
            result["columns"] = list(got.columns)
            result["dtypes"] = {c: str(t) for c, t in got.dtypes.items()}
            head = got.astype(str).head(5).to_dict(orient="records")
            result["head"] = head
            break
        result.setdefault("errors", {})[enc] = str(got)[:300]

    # Full row count only (cheap enough at 448 MB, but avoid loading all columns).
    if "columns" in result:
        n = sum(1 for _ in open(RAW, "rb")) - 1
        result["rows_by_line_count"] = n

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
