"""Real Java -> Python -> Ollama workflow checks. Creates its own isolated workspace."""
import json
import time
from datetime import datetime,timezone
import requests
from client import Client,ROOT


def main():
    checks=[];artifacts={}
    def check(name,condition,detail=""):
        checks.append({"name":name,"passed":bool(condition),"detail":detail})
        print(("PASS " if condition else "FAIL ")+name,flush=True)
        if not condition:raise AssertionError(name+": "+str(detail))
    c=Client();s=c.call("POST","/studies",{"title":"真实数据验收：图像去噪技术方案"});sid=s["id"]
    artifacts["study_id"]=sid
    check("API requires access token",requests.get(c.url+"/api/studies",timeout=10).status_code==401)
    found=c.call("POST",f"/studies/{sid}/search",{"query":"图像去噪","filters":{"ipc_subclass":"G06T","year_from":2018}})
    hits=found["hits"];artifacts["search"]=found
    check("Real hybrid search returns candidates",len(hits)>=2,str(len(hits)))
    check("Hard filters remain enforced",all(h["year"]>=2018 and h["ipc_subclass"].startswith("G06T") for h in hits))
    check("Evidence is exact original text",all(h["abstract"][e["start"]:e["end"]]==e["text"] for h in hits for e in h["evidence"]))
    ids=[h["doc_id"] for h in hits[:2]]
    for id in ids:c.call("PUT",f"/studies/{sid}/documents",{"doc_id":id,"note":"需结合原文复核技术路线","selected":True})
    saved=c.call("GET",f"/studies/{sid}/documents")
    check("Saved candidates and notes can be read back",len(saved)==2 and saved[0]["note"]=="需结合原文复核技术路线")
    other=Client("隔离验收工作区")
    check("Other workspace cannot read study",other.raw("GET",f"/studies/{sid}").status_code==404)
    submitted=c.call("POST",f"/studies/{sid}/jobs",{"kind":"compare","question":"只比较实现方法，每件专利用一条简短结论说明技术路线，分别引用证据。","doc_ids":ids})
    result=c.wait(submitted["id"]);artifacts["compare"]=result
    check("Real LangChain4j model comparison completes",result["status"]=="completed",result["result"].get("message",""))
    analysis=result["result"]
    check("Comparison has usable evidence-linked claims",len(analysis["claims"])>=2,str(analysis.get("audit")))
    check("Claims cite their own documents",all(e["doc_id"]==claim["doc_id"] for claim in analysis["claims"] for e in claim["evidence"]))
    check("No semantic-accuracy guarantee is reported",analysis["audit"].get("semantic_verification") is False)
    md=c.raw("GET",f"/studies/{sid}/export?format=md")
    check("Markdown export contains source IDs and original quotes",md.status_code==200 and all(id in md.text for id in ids) and "证据" in md.text)
    csv=c.raw("GET",f"/studies/{sid}/export?format=csv")
    check("CSV export contains saved candidates",csv.status_code==200 and all(id in csv.text for id in ids))
    check("Unsupported legal request is explicitly rejected",c.raw("POST",f"/studies/{sid}/jobs",{"question":"判断是否侵权","kind":"overview","doc_ids":ids}).status_code==422)
    check("Invalid year range is rejected",c.raw("POST",f"/studies/{sid}/search",{"query":"图像去噪","filters":{"year_from":2024,"year_to":2018}}).status_code==400)
    started=c.call("POST",f"/studies/{sid}/jobs",{"kind":"overview","question":"详细分析所选方案的各个技术组成和实现细节。","doc_ids":ids})
    time.sleep(.2);cancel_start=time.monotonic();c.call("POST",f"/jobs/{started['id']}/cancel",{})
    cancelled=c.wait(started["id"],30);artifacts["cancel_elapsed_seconds"]=round(time.monotonic()-cancel_start,3)
    check("Explicit cancellation reaches terminal state",cancelled["status"]=="cancelled",cancelled)
    followed=c.call("POST",f"/studies/{sid}/jobs",{"kind":"follow_up","question":"这些方案中，面向低照度图像的有哪些？"})
    follow=c.wait(followed["id"]);artifacts["follow_up"]=follow
    check("Contextual follow-up rewrites and searches",follow["status"]=="completed",follow["result"].get("message",""))
    check("Follow-up retains explicit hard filters",follow["result"]["search"]["filters"]["ipc_subclass"]=="G06T" and follow["result"]["search"]["filters"]["year_from"]==2018)
    none=c.call("POST",f"/studies/{sid}/search",{"query":"图像去噪","filters":{"city":"不存在的城市_验收"}})
    check("Empty search does not silently relax filters",not none["hits"] and none["filters"]["city"]=="不存在的城市_验收")
    # Restore a useful last search for later manual inspection.
    c.call("POST",f"/studies/{sid}/search",{"query":"图像去噪","filters":{"ipc_subclass":"G06T","year_from":2018}})
    report={"time":datetime.now(timezone.utc).isoformat(),"type":"real_stack_acceptance","checks":checks,"artifacts":artifacts,
            "limitations":["These checks validate workflow and deterministic evidence integrity, not expert semantic correctness.","Cancellation confirms application termination and client transport cancellation; GPU task cessation is not independently measured."]}
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports/acceptance-real.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    # Private local credentials are deliberately excluded from the report.
    (ROOT/"data/verification-workspace.json").write_text(json.dumps({"workspace":c.workspace,"study_id":sid}),encoding="utf-8")
    print(f"{len(checks)} acceptance checks passed",flush=True)


if __name__=="__main__":main()
