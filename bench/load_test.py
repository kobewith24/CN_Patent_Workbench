# -*- coding: utf-8 -*-
"""Concurrency / cancellation test for the local service.

Measures TTFT and total latency at 1/2/4 concurrent clients against a service
that allows RAG_MAX_CONCURRENT generations at a time.

Usage:
    python bench/load_test.py --url http://127.0.0.1:8001
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = [
    "如何实现卡爪在振动环境下稳定夹持工件？",
    "如何通过电机驱动实现打捞钩的自动张开？",
    "如何提升玻璃幕墙的稳定性与跨度？",
    "如何实现箱梁预应力钢束的高效布设？",
]


def one(url, q, timeout=180):
    t0 = time.time()
    first = None
    status = None
    try:
        r = requests.get(url + "/answer_stream", params={"q": q}, stream=True, timeout=timeout)
        status = r.status_code
        ev = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: ") and ev == "token" and first is None:
                first = time.time() - t0
            elif line.startswith("data: ") and ev == "done":
                break
        r.close()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": type(exc).__name__, "status": status}
    return {"ok": True, "status": status,
            "ttft": round(first * 1000, 1) if first is not None else None,
            "total": round((time.time() - t0) * 1000, 1)}


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(p * len(s)))], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    before = requests.get(args.url + "/metrics", timeout=10).json()
    one(args.url, QUERIES[0])  # warm up the model so cold start does not pollute p95
    levels = {}
    for c in (1, 2, 4):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            res = list(ex.map(lambda x: one(args.url, x), QUERIES[:args.n]))
        ok = [r for r in res if r["ok"]]
        levels["c%d" % c] = {
            "n": len(res),
            "ok": len(ok),
            "429": sum(1 for r in res if r.get("status") == 429),
            "ttft_p50": pct([r["ttft"] for r in ok if r["ttft"]], 0.5),
            "ttft_p95": pct([r["ttft"] for r in ok if r["ttft"]], 0.95),
            "total_p50": pct([r["total"] for r in ok], 0.5),
            "total_p95": pct([r["total"] for r in ok], 0.95),
            "wall_s": round(time.time() - t0, 1),
        }
        print(levels["c%d" % c], flush=True)

    # cancellation: open a stream, read a token, close, then watch active
    cancelled = {}
    def abort():
        r = requests.get(args.url + "/answer_stream", params={"q": QUERIES[0]}, stream=True, timeout=60)
        ev = None
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("event: "):
                ev = line[7:]
            if ev == "token":
                break
        r.close()
        cancelled["closed_at"] = time.time()

    th = threading.Thread(target=abort)
    th.start()
    th.join()
    t_closed = time.time()
    released_after = None
    for _ in range(int(before["hold_max_s"]) + 15):
        cur = requests.get(args.url + "/metrics", timeout=10).json()
        if cur["active"] == 0:
            released_after = round(time.time() - t_closed, 1)
            break
        time.sleep(1)
    after_abort = requests.get(args.url + "/metrics", timeout=10).json()
    levels["cancel"] = {"active_after_1s": after_abort["active"],
                        "released_after_s": released_after,
                        "forced_release": after_abort["forced_release"]}

    after = requests.get(args.url + "/metrics", timeout=10).json()
    out = {"config": {k: before[k] for k in ("max_concurrent", "queue_timeout_s")},
           "levels": levels, "server": after}

    outdir = os.path.join(ROOT, "eval", "reports", "v1")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "concurrency.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = ["# 本地并发与取消", "",
             "服务端：`RAG_MAX_CONCURRENT=%s`、`RAG_QUEUE_TIMEOUT=%ss`；单卡 8GB，生成只跑一路。"
             % (before["max_concurrent"], before["queue_timeout_s"]), "",
             "| 并发 | 成功 | 429 | TTFT p50/p95 | 总时延 p50/p95 | 墙钟 |",
             "|---|---:|---:|---|---|---:|"]
    for c in ("c1", "c2", "c4"):
        v = levels[c]
        lines.append("| %d | %d/%d | %d | %s/%s ms | %s/%s ms | %.1fs |" % (
            int(c[1:]), v["ok"], v["n"], v["429"], v["ttft_p50"], v["ttft_p95"],
            v["total_p50"], v["total_p95"], v["wall_s"]))
    c = levels["cancel"]
    lines += ["", "取消测试：断开后 `active` 在 %s 秒归零（看门狗强制释放 %d 次）。"
              % (c["released_after_s"], c["forced_release"]),
              "说明：同步生成器在客户端断开时不会被及时 finalize，槽位靠看门狗兜底，不是即时取消。"]
    with open(os.path.join(outdir, "concurrency.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
