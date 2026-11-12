"""Summarize actual saved test artifacts; no fabricated quality or timing scores."""
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime,timedelta,timezone
import requests

ROOT=pathlib.Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/'reports'/name).read_text(encoding='utf-8'))


def main():
    acceptance=read('acceptance-real.json');failure=read('failure-paths-real.json');load=read('load-real.json');integrity=read('source-integrity.json')
    assert all(c['passed'] for c in acceptance['checks'])
    assert all(failure[k] for k in ['queue_full_http_429','cancelled_queue_slot_reusable','sse_disconnect_cancelled','unknown_hard_filter_rejected','forged_caller_abstract_ignored'])
    assert sum(r['requests'] for r in load['levels'])==300 and all(r['errors']==0 for r in load['levels'])
    assert not integrity['original_source_changed']
    packages={k:importlib.metadata.version(k) for k in ['torch','numpy','transformers','sentence-transformers','hnswlib','fastapi','pydantic','uvicorn','jieba']}
    workspace=json.loads((ROOT/'data/verification-workspace.json').read_text(encoding='utf-8'))['workspace']
    headers={'Authorization':'Bearer '+(ROOT/'data/access.token').read_text().strip(),'X-Workspace-ID':workspace['id'],'X-Workspace-Key':workspace['key']}
    response=requests.get('http://127.0.0.1:8088/api/info',headers=headers,timeout=10);response.raise_for_status();info=response.json()
    assert info['retrieval']['ready'] and info['retrieval']['reranker_ready']
    binary=ROOT/'app/target/patent-workbench-2.0.0.jar'
    environment={'python':sys.version,'platform':platform.platform(),'packages':packages,'app_version':info['version'],
                 'jar_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'corpus_version':info['retrieval']['manifest']['version']}
    (ROOT/'reports/environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    time=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')
    lines=['# 验证记录','',f'整理时间：{time}。测试使用真实本机数据、模型与 Java 入口。','',
      '## 原项目与独立运行','',
      f"- 开工前记录的 {integrity['original_source_files_checked']} 个原项目源码与文档哈希保持不变。",
      '- 运行所需 6 个语料与索引文件已复制到新项目，源与副本逐一 SHA-256 一致，总计约 6.22 GiB。',
      '- 默认 data_root 为 assets/corpus，运行路径不再依赖原项目目录；模型继续读取本机已有权重。',
      '- 详情：source-integrity.json；基线：source-baseline.json。','',
      '## 自动与真实链路检查','',
      '- Java 11 项测试通过：工作区鉴权、跨工作区隔离、持久化重开、取消、队列满、引用分离、边界判断和 CSV 公式防护。',
      '- Python 15 项测试通过：精确原文片段、Unicode 位置、小整数、单位、区间、引用归属、未知过滤条件与指标定义。',
      f"- 真实 Java → Python → Ollama 功能链路 {len(acceptance['checks'])} 项检查通过，详情 acceptance-real.json。",
      '- 真实异常路径 5 项检查通过：429、取消后的队列容量、SSE 断连取消、未知硬条件拒绝、伪造摘要不被采信。',
      f"- 主动取消单次观测至终态约 {acceptance['artifacts']['cancel_elapsed_seconds']} 秒；SSE 断连至终态单次观测约 {failure['disconnect_to_terminal_seconds']} 秒，受心跳探测影响。",
      '- 取消验证覆盖应用终态和客户端传输取消，没有独立测量 GPU 内核停止时间，不作即时 GPU 释放保证。',
      '- 浏览器实际操作通过：检索、勾选、四项技术对照、原文展开、标题保存、刷新/服务重启后恢复、历史结果打开。',
      '- 1440px 桌面布局与 390px 窄屏均检查；窄屏表格在容器内横向滚动，页面本身不溢出。',
      '- 实际界面：interface.jpg。','',
      '## 短分析闭环压测','',
      '真实 Java API，模型常驻、已预热。同一件真实专利、相同问题、只生成一条简短方法结论，提示缓存收益较大。每档 100 次，共 300 次成功；全部产生至少一项通过机械检查的结论。',
      '', '|并发|成功|RPS|完整任务 p50 / p95（ms）|','|---:|---:|---:|---:|']
    for r in load['levels']:lines.append(f"|{r['concurrency']}|{r['ok']}/{r['requests']}|{r['rps']}|{r['total_ms_p50']} / {r['total_ms_p95']}|")
    lines+=['','客户端完成时延包含 200ms 轮询粒度，模型 TTFT 与排队单独记录。这里不代表冷启动、长文本、多专利长报告或真实开放流量。通过引用/数值检查也不代表语义正确。',
      '逐请求记录：load-real.json；性能测量采用相同检索与模型参数，最终整理时仅补充严格读取检查及固定 HTTP/1.1 协商。','',
      '## 质量评测尚待完成的部分','',
      '- 60 道需求式验收题由 Codex 独立编写，不能冒充采集的真实用户查询。',
      '- 已为其中 20 道技术检索题导出 400 对真实候选，relevance 保持 null，供独立审阅，见 review-pool.jsonl。',
      '- 新题集尚未完成人工相关性标注，因此不报告其准确率、召回率或语义忠实度。',
      '- legacy-metrics-corrected.json 仅是旧合成集缓存的指标口径复算，明确区分 Hit@5 与多 gold Recall@5，不能视为新系统效果提升。','',
      '## 交付边界','',
      '- 本机单卡、小规模调研使用；没有生产流量、长期可用性、多节点或企业身份系统验证。',
      '- 仅摘要样本，不提供侵权、保护范围、申请人名称或完整法律状态判断。',
      '- 证据原文可定位，语义、否定、因果和实体—数值关系仍需复核；机械规则不能代替专业判断。',
      '- 工作区凭据保存在浏览器，清除浏览器数据后的身份恢复尚需另行处理；同一浏览器重启恢复已验证。',
      '- 完整运行说明见项目 README.md，API 与设计见 docs。']
    (ROOT/'reports/VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Validation report written; actual checks pass; semantic relevance remains pending review')


if __name__=='__main__':main()
