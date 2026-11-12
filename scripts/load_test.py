"""Closed-loop load against the real Java entry point, with true model token usage.

Workload: one evidence-linked method claim over one real saved patent. This is
explicitly a short-analysis benchmark, not a full comparative-report benchmark.
"""
import argparse
import concurrent.futures
import json
import math
import statistics
import time
from datetime import datetime,timezone
from client import Client,ROOT


def pct(values,p):
    ordered=sorted(values)
    return round(ordered[max(0,math.ceil(len(ordered)*p)-1)],2) if ordered else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--requests',type=int,default=100);ap.add_argument('--concurrency',default='1,2,4');args=ap.parse_args()
    c=Client('真实 Java 入口压测');s=c.call('POST','/studies',{'title':'短分析闭环压测'});sid=s['id']
    found=c.call('POST',f'/studies/{sid}/search',{'query':'图像去噪','filters':{'ipc_subclass':'G06T','year_from':2018}})
    doc=found['hits'][0]['doc_id']
    def one(i):
        start=time.perf_counter()
        try:
            submitted=c.call('POST',f'/studies/{sid}/jobs',{'kind':'overview','question':'仅概括一个最主要的实现方法，严格只输出一条简短结论和一条对应证据，不列步骤。','doc_ids':[doc]})
            job=c.wait(submitted['id']);result=job['result'];timing=result.get('timing',{})
            return {'ok':job['status']=='completed','status':job['status'],'total_ms':(time.perf_counter()-start)*1000,
                    'queue_ms':result.get('queue_ms'),'ttft_ms':timing.get('model_ttft_ms'),
                    'generation_ms':timing.get('generation_ms'),'output_tokens':timing.get('output_tokens'),
                    'claims':len(result.get('claims',[])),'message':result.get('message','')}
        except Exception as e:return {'ok':False,'status':'request_error','message':str(e),'total_ms':(time.perf_counter()-start)*1000}
    cold=one(-1);print('warmup',cold['status'],flush=True)
    levels=[]
    for concurrency in [int(v) for v in args.concurrency.split(',')]:
        started=time.perf_counter();rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            for row in executor.map(one,range(args.requests)):
                rows.append(row)
                if len(rows)%10==0:print(f'concurrency={concurrency} completed={len(rows)}/{args.requests}',flush=True)
        elapsed=time.perf_counter()-started;ok=[r for r in rows if r['ok']]
        model_ttft=[r['ttft_ms'] for r in ok if r.get('ttft_ms') is not None]
        perceived=[r['queue_ms']+r['ttft_ms'] for r in ok if r.get('ttft_ms') is not None and r.get('queue_ms') is not None]
        level={'concurrency':concurrency,'requests':args.requests,'ok':len(ok),'errors':len(rows)-len(ok),'wall_seconds':round(elapsed,2),
               'rps':round(len(ok)/elapsed,3),'total_ms_p50':pct([r['total_ms'] for r in ok],.5),'total_ms_p95':pct([r['total_ms'] for r in ok],.95),
               'model_ttft_ms_p50':pct(model_ttft,.5),'model_ttft_ms_p95':pct(model_ttft,.95),
               'queue_plus_model_ttft_ms_p95':pct(perceived,.95),'no_accepted_claims':sum(r['claims']==0 for r in ok),
               'output_tokens_sum':sum(r.get('output_tokens') or 0 for r in ok),'rows':rows}
        levels.append(level)
        report={'time':datetime.now(timezone.utc).isoformat(),'workload':'one method claim / one actual patent / same question / warm repeated prompts',
                'entry_point':'Java /api/studies/{id}/jobs; polling 200ms','warmup_excluded':cold,'levels':levels,
                'limitations':['Repeated prompts benefit from upstream prompt caching.','TTFT is model-side first nonempty content, not first audited UI result.','Client total includes 200ms polling granularity.','Quality is not expert-judged; zero accepted claims are reported separately.']}
        (ROOT/'reports/load-real.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in level.items() if k!='rows'},ensure_ascii=False),flush=True)
    md=['# 真实 Java 入口闭环压测','',f'每档 {args.requests} 次请求。工作负载：同一真实专利的一条简短方法结论，重复提示，已预热。','',
        '|并发|成功|RPS|总时延 p50 / p95 (ms)|模型 TTFT p50 / p95 (ms)|无有效结论数|','|---:|---:|---:|---|---|---:|']
    for r in levels:md.append(f"|{r['concurrency']}|{r['ok']}/{r['requests']}|{r['rps']}|{r['total_ms_p50']} / {r['total_ms_p95']}|{r['model_ttft_ms_p50']} / {r['model_ttft_ms_p95']}|{r['no_accepted_claims']}|")
    md+=['','模型 TTFT 不包含排队，且前端展示的是审计后的完整结论。总时延包含客户端 200ms 轮询粒度。',
         '重复短问题享有提示缓存收益，本报告不能代表冷启动、长摘要、多专利长报告或真实开放流量。',
         '使用模型返回的真实 token usage，不以 SSE 分块数代替 token 数。逐请求数据见 load-real.json。']
    (ROOT/'reports/load-real.md').write_text('\n'.join(md)+'\n',encoding='utf-8')


if __name__=='__main__':main()
