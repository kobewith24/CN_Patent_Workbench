# -*- coding: utf-8 -*-
"""Minimal human-calibration tool for the A set.

Run:
    python eval/annotate.py --out <writable path>/human.jsonl
    open http://127.0.0.1:8002

Labels are appended to the output file; re-running resumes. The tool refuses
to advance the UI when a save fails, so a sandbox/permission error can never
silently discard labels again.
"""
import argparse
import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "eval", "annotations", "pairs_human.jsonl")
HUMAN = os.path.join(ROOT, "eval", "annotations", "human.jsonl")

app = FastAPI()

def load_pairs():
    rows = []
    if os.path.exists(PAIRS):
        with open(PAIRS, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows

def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done[(d["qid"], d["doc_id"])] = d["human"]
                except Exception:  # noqa: BLE001
                    pass
    return done

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>相关性标注</title>
<style>
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:820px;margin:40px auto;padding:0 20px;color:#222;line-height:1.7}
.q{font-size:19px;font-weight:600;margin:6px 0 18px}
.card{border:1px solid #ddd;border-radius:10px;padding:18px 20px;margin-bottom:18px}
.title{font-weight:600;margin-bottom:8px}
.abs{color:#444;font-size:15px}
.bar{height:6px;background:#eee;border-radius:3px;overflow:hidden;margin:18px 0}
.bar i{display:block;height:100%;background:#2b6cb0;width:0}
button{font-size:16px;padding:10px 22px;margin-right:10px;border-radius:8px;border:1px solid #ccc;cursor:pointer;background:#fff}
button.y{border-color:#2f855a;color:#2f855a}
button.n{border-color:#c53030;color:#c53030}
.meta{color:#888;font-size:13px;margin-top:10px}
kbd{background:#f2f2f2;border:1px solid #ddd;border-radius:4px;padding:1px 6px;font-size:12px}
#err{color:#c53030;font-weight:600;margin-top:10px}
</style></head><body>
<h2>专利相关性标注</h2>
<p>判断这件专利<b>能不能回答</b>上面的问题（部分相关也算相关）。键盘：<kbd>1</kbd> 相关，<kbd>0</kbd> 不相关，<kbd>s</kbd> 跳过。</p>
<div class="bar"><i id="bar"></i></div>
<div class="meta" id="prog"></div>
<div id="err"></div>
<div id="app"></div>
<script>
const PAIRS = __DATA__;
const DONE = __DONE__;
let idx = 0;
const todo = PAIRS.filter(p => !((p.qid + "|" + p.doc_id) in DONE));
function render(){
  if (idx >= todo.length){ document.getElementById("app").innerHTML = "<h3>全部标完，谢谢。现在可以关掉这个页面。</h3>"; return; }
  const p = todo[idx];
  document.getElementById("app").innerHTML =
    '<div class="q">' + esc(p.question) + '</div>' +
    '<div class="card"><div class="title">' + esc(p.title) + '</div>' +
    '<div class="abs">' + esc(p.abstract || "") + '</div></div>' +
    '<button class="y" onclick="send(1)">相关 (1)</button>' +
    '<button class="n" onclick="send(0)">不相关 (0)</button>' +
    '<button onclick="skip()">跳过 (s)</button>';
  document.getElementById("prog").textContent = (idx+1) + " / " + todo.length + "（已保存 " + idx + " 条）";
  document.getElementById("bar").style.width = (100*(idx)/todo.length) + "%";
}
function esc(s){return (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function send(v){
  const p = todo[idx];
  document.getElementById("err").textContent = "";
  fetch("/label", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({qid:p.qid, doc_id:p.doc_id, human:v})})
    .then(r => { if(!r.ok) throw new Error("HTTP " + r.status); idx++; render(); })
    .catch(e => { document.getElementById("err").textContent = "保存失败，未前进：" + e.message; });
}
function skip(){ idx++; render(); }
document.addEventListener("keydown", e => {
  if (e.key === "1") send(1); else if (e.key === "0") send(0); else if (e.key === "s") skip();
});
render();
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    pairs = load_pairs()
    done = load_done(HUMAN)
    payload = [{k: p.get(k) for k in ("qid", "doc_id", "question", "title", "abstract")} for p in pairs]
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    done_map = {("%s|%s" % (k[0], k[1])): v for k, v in done.items()}
    return PAGE.replace("__DATA__", data).replace("__DONE__", json.dumps(done_map))

@app.post("/label")
async def label(req: Request):
    d = await req.json()
    rec = {"qid": d["qid"], "doc_id": d["doc_id"], "human": int(d["human"]),
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        os.makedirs(os.path.dirname(HUMAN), exist_ok=True)
        with open(HUMAN, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("ANNOTATE_OUT", HUMAN))
    args = ap.parse_args()
    HUMAN = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(HUMAN), exist_ok=True)
    print("labels ->", HUMAN, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8002)
