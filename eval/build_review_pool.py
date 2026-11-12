"""Export actual Java search candidates for review. Unknown relevance stays null."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from client import Client,ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=20);args=ap.parse_args()
    cases=[json.loads(line) for line in (ROOT/'eval/acceptance_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    c=Client('相关性复核工作区');study=c.call('POST','/studies',{'title':'独立问题候选相关性复核'});out=ROOT/'reports/review-pool.jsonl';n=0
    with out.open('w',encoding='utf-8') as fh:
        for case in [r for r in cases if r['kind']=='search'][:args.limit]:
            found=c.call('POST',f"/studies/{study['id']}/search",{'query':case['question'],'filters':case['filters']})
            for rank,hit in enumerate(found['hits'],1):
                fh.write(json.dumps({'case_id':case['case_id'],'question':case['question'],'rank':rank,'doc_id':hit['doc_id'],
                    'title':hit['title'],'abstract':hit['abstract'],'relevance':None,'rationale':'','annotator':'',
                    'rubric':'0=unrelated, 1=partially related, 2=directly useful; independently review the abstract'},ensure_ascii=False)+'\n');n+=1
            print(case['case_id'],len(found['hits']),flush=True)
    print(f'{n} candidate pairs exported. No relevance score has been calculated.',flush=True)


if __name__=='__main__':main()
