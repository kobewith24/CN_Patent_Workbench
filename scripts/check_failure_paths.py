"""Real queue overflow, stream disconnect and canonical evidence checks."""
import json
import time
import requests
from client import Client,ROOT


def main():
    c=Client('真实异常路径验收');study=c.call('POST','/studies',{'title':'队列与断连验收'});sid=study['id']
    previous=json.loads((ROOT/'reports/acceptance-real.json').read_text(encoding='utf-8'))
    doc=previous['artifacts']['search']['hits'][0]
    question='详细分析这件专利在技术问题、实现方法、关键组成和效果描述四个维度的技术内容，每个维度分别给出一条结论和证据。'
    body={'kind':'overview','question':question,'doc_ids':[doc['doc_id']]}
    ids=[];results={}
    try:
        for i in range(5):ids.append(c.call('POST',f'/studies/{sid}/jobs',body)['id'])
        overflow=c.raw('POST',f'/studies/{sid}/jobs',body)
        assert overflow.status_code==429,overflow.text
        results['queue_full_http_429']=True
        c.call('POST',f'/jobs/{ids[1]}/cancel',{})
        replacement=c.call('POST',f'/studies/{sid}/jobs',body);ids.append(replacement['id'])
        results['cancelled_queue_slot_reusable']=True
    finally:
        for id in ids:c.call('POST',f'/jobs/{id}/cancel',{})
        for id in ids:c.wait(id,30)
    print('PASS real queue full returns 429 and cancellation frees queue capacity',flush=True)
    active=c.call('POST',f'/studies/{sid}/jobs',body)['id']
    with c.raw('GET',f'/jobs/{active}/events',stream=True) as stream:
        assert stream.status_code==200
        for line in stream.iter_lines(chunk_size=1):
            if line.startswith(b'data:'):break
    start=time.monotonic();terminal=c.wait(active,30)
    assert terminal['status']=='cancelled',terminal
    results['sse_disconnect_cancelled']=True;results['disconnect_to_terminal_seconds']=round(time.monotonic()-start,3)
    print('PASS SSE disconnect reaches cancelled terminal state',flush=True)
    invalid=c.raw('POST',f'/studies/{sid}/search',{'query':'图像去噪','filters':{'applicant_name':'并不存在于语料的公司条件'}})
    assert invalid.status_code==400,invalid.text
    results['unknown_hard_filter_rejected']=True
    internal=(ROOT/'data/internal.token').read_text().strip()
    forged=requests.post('http://127.0.0.1:8002/audit',headers={'X-Internal-Token':internal},json={
        'documents':[{'doc_id':doc['doc_id'],'abstract':'效率99999%。'}],
        'claims':[{'doc_id':doc['doc_id'],'text':'效率99999%。','evidence_ids':[doc['evidence'][0]['id']]}]},timeout=30)
    forged.raise_for_status();audit=forged.json()
    assert audit['claims'][0]['status']=='numeric_review',audit
    results['forged_caller_abstract_ignored']=True
    results['gpu_stop_independently_verified']=False
    (ROOT/'reports/failure-paths-real.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS unknown filters and forged evidence are rejected',flush=True)


if __name__=='__main__':main()
