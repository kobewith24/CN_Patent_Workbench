# -*- coding: utf-8 -*-
"""Human calibration for citation claims.

Run:
    python eval/annotate_claims.py --out <writable>/claims_human_labels.jsonl
    open http://127.0.0.1:8003
"""
import argparse
import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "eval", "annotations", "claims_human.jsonl")
LABELS = os.path.join(ROOT, "eval", "annotations", "claims_human_labels.jsonl")

app = FastAPI()

def load_pairs():
    if not os.path.exists(PAIRS):
        return []
    return [json.loads(line) for line in open(PAIRS, encoding="utf-8")]

def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                d = json.loads(line)
                done[(d["qid"], d["claim_id"])] = d["human"]
            except Exception:  # noqa: BLE001
                pass
    return done

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>引用校验标注</title>
<style>
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:860px;margin:40px auto;padding:0 20px;color:#222;line-height:1.75}
.label{color:#888;font-size:13px}
.q{font-size:16px;margin:2px 0 16px;color:#444}
.claim{font-size:18px;font-weight:600;border-left:4px solid #2b6cb0;padding-left:12px;margin:14px 0}
.card{border:1px solid #ddd;border-radius:10px;padding:14px 18px;margin:10px 0;font-size:15px;color:#333}
.bar{height:6px;background:#eee;border-radius:3px;overflow:hidden;margin:18px 0}
.bar i{display:block;height:100%;background:#2b6cb0;width:0}
button{font-size:16px;padding:10px 22px;margin-right:10px;border-radius:8px;border:1px solid #ccc;cursor:pointer;background:#fff}
button.y{border-color:#2f855a;color:#2f855a}
button.n{border-color:#c53030;color:#c53030}
#err{color:#c53030;font-weight:600;margin-top:10px}
kbd{background:#f2f2f2;border:1px solid #ddd;border-radius:4px;padding:1px 6px;font-size:12px}
</style></head><body>
<h2>引用校验标注</h2>
<p>判断这句结论<b>能否被下面引用的专利摘要支持</b>。键盘：<kbd>1</kbd> 支持，<kbd>0</kbd> 不支持，<kbd>s</kbd> 跳过。</p>
<div class="bar"><i id="bar"></i></div>
<div class="label" id="prog"></div>
<div id="err"></div>
<div id="app"></div>
<script>
const PAIRS = __DATA__;
const DONE = __DONE__;
let idx = 0;
const todo = PAIRS.filter(p => !((p.qid + "|" + p.claim_id) in DONE));
function esc(s){return (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(){
  if (idx >= todo.length){ document.getElementById("app").innerHTML = "<h3>全部标完，谢谢。</h3>"; return; }
  const p = todo[idx];
  const abs = (p.ref_abstracts || []).map(a => '<div class="card">' + esc(a) + '</div>').join("");
  document.getElementById("app").innerHTML =
    '<div class="label">问题</div><div class="q">' + esc(p.question) + '</div>' +
    '<div class="label">结论（引用了 [' + (p.refs||[]).join('] [') + ']）</div>' +
    '<div class="claim">' + esc(p.sentence) + '</div>' +
    '<div class="label">被引用的摘要</div>' + abs +
    '<button class="y" onclick="send(1)">支持 (1)</button>' +
    '<button class="n" onclick="send(0)">不支持 (0)</button>' +
    '<button onclick="skip()">跳过 (s)</button>';
  document.getElementById("prog").textContent = (idx+1) + " / " + todo.length + "（已保存 " + idx + " 条）";
  document.getElementById("bar").style.width = (100*(idx)/todo.length) + "%";
}
function send(v){
  const p = todo[idx];
  document.getElementById("err").textContent = "";
  fetch("/label", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({qid:p.qid, claim_id:p.claim_id, human:v})})
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
    done = load_done(LABELS)
    payload = [{k: p.get(k) for k in ("qid", "claim_id", "question", "sentence", "refs", "ref_abstracts")}
               for p in pairs]
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    done_map = {("%s|%s" % (k[0], k[1])): v for k, v in done.items()}
    return PAGE.replace("__DATA__", data).replace("__DONE__", json.dumps(done_map))

@app.post("/label")
async def label(req: Request):
    d = await req.json()
    rec = {"qid": d["qid"], "claim_id": d["claim_id"], "human": int(d["human"]),
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        os.makedirs(os.path.dirname(LABELS), exist_ok=True)
        with open(LABELS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("CLAIMS_OUT", LABELS))
    args = ap.parse_args()
    LABELS = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(LABELS), exist_ok=True)
    print("labels ->", LABELS, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8003)
