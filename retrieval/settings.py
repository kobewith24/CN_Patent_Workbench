import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("PATENT_CONFIG", ROOT / "config/workbench.json"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
DATA_ROOT = Path(os.environ.get("PATENT_DATA_ROOT", CONFIG["data_root"]))
if not DATA_ROOT.is_absolute():
    DATA_ROOT = ROOT / DATA_ROOT
MODEL_DIR = os.environ.get("PATENT_EMBEDDING_MODEL", CONFIG["embedding_model"])
RERANKER_DIR = os.environ.get("PATENT_RERANKER_MODEL", CONFIG["reranker_model"])
INDEX_PATH = DATA_ROOT / "index/patent_index_1024.bin"
META_PATH = DATA_ROOT / "index/meta_compact.jsonl"
FULL_PATH = DATA_ROOT / "clean/patents.jsonl"
BM25_PATH = DATA_ROOT / "index/bm25_fts5.db"
EMB_PATH = DATA_ROOT / "index/patent_emb.npy"


def manifest():
    """Fingerprint index metadata and asset sizes, not the entire multi-GB corpus."""
    paths = [INDEX_PATH, META_PATH, FULL_PATH, BM25_PATH, EMB_PATH]
    assets = [{"name": str(p.relative_to(DATA_ROOT)), "size": p.stat().st_size,
               "mtime_ns": p.stat().st_mtime_ns} for p in paths]
    raw = (DATA_ROOT / "index/index_meta.json").read_bytes()
    version = hashlib.sha256(raw + json.dumps(assets, sort_keys=True).encode()).hexdigest()[:20]
    return {"version": version, "scope": CONFIG["corpus_scope"], "assets": assets,
            "records": json.loads(raw)["records"], "read_only": True}
