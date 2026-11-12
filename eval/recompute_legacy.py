"""Read-only remeasurement of stored legacy rankings; never calls a model."""
import argparse
import json
import statistics
from pathlib import Path
from metrics import metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default=str(Path(__file__).resolve().parent/"legacy_rankings.json"))
    ap.add_argument("--out",default="reports/legacy-metrics-corrected.json")
    args=ap.parse_args()
    rows=json.loads(Path(args.input).read_text(encoding="utf-8"))["b"]["queries"]
    result={"source":args.input,"n":len(rows),"gold_definition":"legacy synthetic rare-term clusters; not independently judged relevance", "new_system_measurement":False,"configs":{}}
    for cfg in ("dense","bm25","hybrid","hybrid_rerank"):
        per=[metrics(r["top10"][cfg],r["golds"],5) for r in rows]
        result["configs"][cfg]={key:round(statistics.mean(v[key] for v in per),6) for key in per[0] if key!="MRR"}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
