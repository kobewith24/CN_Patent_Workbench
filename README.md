# CN_Patent_RAG

面向中文专利技术情报的检索增强问答（RAG）。单语料、单索引：52 万件专利摘要 → BGE-M3 稠密索引 → 查询规划/元数据过滤 → 多步检索 → 带专利号引用的生成。

## 当前进度

- [x] 原始数据下载：`data/raw/patents_sample_500k.csv`（448 MB）
- [x] 清洗：`data/clean/patents.jsonl`（519,424 条）
- [x] 数据审计：广告短语残留 0；正常词“马克笔/托卡马克/马克思主义”保留
- [x] 全量 BGE-M3 索引：519,424 条，2.17 GB，73.9 分钟（编码 64.8 min + HNSW 8.9 min）
- [x] Top-k sanity：纯向量检索 21–30 ms（warm），相关专利排前
- [x] 检索服务：`/retrieve`，BGE-M3 + IPC/年份/类型/城市过滤 + GPU reranker 精排
- [x] 生成服务：`/answer`，Qwen3-4B 带 `[n]` 引用；对比题自动拆 1–3 个子查询
- [x] 引用校验：Qwen3 逐条判定 + reranker 兜底，返回 `claims` / `faithfulness`
- [x] 150 题合成评测：factoid recall@5 0.850、comparison all@5 0.860、faithfulness 0.606、拒答 1.000
- [x] Web UI：`http://127.0.0.1:8001/`
- [x] 消融：rerank +6.7 pt、标题 bigram +3.3 pt、多步 faithfulness +5 pt、校验约 +1.5 s/题
- [x] 请求级 trace：`/trace/{trace_id}`；SSE 流式：`/answer_stream`
- [x] 150 题合成评测集（60 factoid + 50 comparison + 40 unanswerable）
- [ ] judge 校准 / 回归脚本

## 数据

| 项 | 数值 |
|---|---|
| 原始行数 | 520,634 |
| 清洗后 | 519,424 |
| 丢弃 | 空标题 1、摘要过短 11、重复申请号 1,198 |
| 含广告并已清洗 | 155,962 行 |
| 年份 | 2015–2024 |
| 摘要长度 | 均值 260.9，P50 278，P95 304，最大 720 |
| 专利类型 | 实用新型 251,326；发明申请 211,739；发明授权 56,359 |
| IPC 缺失 | 20,585（约 4%） |
| 申请人类型 | 企业 385,955；学校 61,194；个人 47,466；科研 14,535；机关 9,895 |

**已知边界：**

- 数据集**没有申请人名称**，只有申请人类型/城市；“某公司的专利布局”类问题不可答。
- **只有摘要，没有权利要求/说明书**；“保护范围/侵权”类问题不可答。
- 摘要里混有数据商推广语（“马克数据网”等），已用正则清洗；正常词“马克笔/托卡马克/马克思主义/马克隆值”等保留。
- README 声称 1985–2024，HF 统计接口显示实际只有 **2015–2024**。

字段：`doc_id`（申请号）、`title`、`abstract`、`patent_type`、`ipc_main`、`ipc_section`、`ipc_subclass`、`year`、`applicant_types`、`applicant_type`、`city`、`text`、`abstract_len`、`text_len`。

## 目录

```
CN_Patent_RAG/
├─ data/
│  ├─ raw/patents_sample_500k.csv
│  ├─ clean/patents.jsonl
│  ├─ stats/clean_stats.json
│  └─ index/                  # BGE-M3 向量 + HNSW 索引
├─ scripts/
│  ├─ inspect_patents.py      # 原始 CSV 体检
│  ├─ clean_patents.py        # 清洗 + 统计
│  ├─ check_clean.py          # 广告残留审计
│  ├─ build_meta.py           # 紧凑元数据
│  └─ build_index.py          # BGE-M3 编码 + HNSW
├─ service/
│  ├─ app.py                  # FastAPI：/retrieve /answer /
│  └─ static/index.html       # Web 问答页
├─ eval/
│  ├─ build_eval.py           # 60/150 题生成
│  ├─ run_eval.py             # 检索 + 答案评测（支持消融）
│  ├─ ablations.md            # 消融结果
│  └─ README.md               # 指标口径与失败分析
├─ docs/
└─ README.md
```

## 环境

使用现有 conda 环境 `rag`（Python 3.11.18、torch 2.5.1+cu121、sentence-transformers 3.4.1、hnswlib）。

```powershell
conda activate rag
pip install -r requirements.txt
```

模型：`D:\models\bge-m3`（BGE-M3，1024 维）。

## 运行服务

```powershell
cd D:\Projects\CN_Patent_RAG
python -m uvicorn service.app:app --host 127.0.0.1 --port 8001
# 浏览器打开 http://127.0.0.1:8001/
```

接口：

| 接口 | 说明 |
|---|---|
| `GET /` | Web 问答页（过滤条件 / 答案 / 引用卡片 / 支撑校验） |
| `GET /retrieve?q=...&k=5&ipc_section=G&year_from=2020` | 检索 + 过滤 + rerank |
| `GET /answer?q=...&k=5&...` | 检索 + 生成 + 逐句校验 |
| `GET /answer_stream?q=...` | SSE：`meta` → `token`* → `verify` → `done` |
| `GET /trace/{trace_id}` | 请求级 trace JSON（查询/候选/答案/claims/耗时） |
| `GET /health` | 健康检查 |

## 评测基线（150 题合成集，不做人工校验）

检索 150 题；答案侧抽 18 题（详见 [eval/README.md](eval/README.md)、[eval/ablations.md](eval/ablations.md)）：

| 指标 | 数值 |
|---|---|
| factoid recall@5 | 0.850（60 题） |
| comparison all-gold / any-gold hit@5 | 0.860 / 1.000（50 题） |
| factoid 引用命中 | 0.667（6 题） |
| comparison 引用命中 | 1.000（6 题） |
| 平均 faithfulness | 0.606（12 条可答题） |
| 不可答拒答率 | 1.000（6 题） |
| 平均端到端延迟 | ~10.9 s |

> 合成题只由标题生成，同主题近邻专利无法区分，单 gold recall 是下界；未做人工校验，`faithfulness` 由同模型判定，仅作相对参考。答案侧仅 18 题，方差大（60 题子集为 0.868）。

## 数据获取

仓库不包含数据和索引。从 Hugging Face 下载
[520,634 Chinese Patents Dataset](https://huggingface.co/datasets/LiuHongwei1992/520634PatentsDataset)
的 `patents_sample_500k.csv`，放到 `data/raw/patents_sample_500k.csv`，再执行下面的清洗与建索引。

## 复现

```powershell
# 0) 下载 CSV 到 data/raw/（见上）
# 1) 清洗
python scripts/clean_patents.py

# 2) 冒烟建索引
python scripts/build_index.py --limit 2000

# 3) 全量建索引（约 60–75 分钟）
python scripts/build_index.py

# 4) 评测 150 题
python eval/build_eval.py --factoid 60 --compare 50 --unans 40 --out eval/gold_150.jsonl
python eval/run_eval.py --gold eval/gold_150.jsonl --answer-n 18 --tag 150
```

## 设计要点

- **单索引**：只对 52 万件专利摘要建一套 BGE-M3 稠密索引。
- **元数据过滤**：IPC 部类/大类、年份区间、专利类型、城市，是过滤条件，不是第二路召回。
- **查询规划**：区分事实型 / 对比型 / 技术调研型 / 不可答；复杂问题拆成 1–3 个子查询，在同一索引上分别检索后聚合证据。
- **引用校验**：生成结果按句绑定 `[n]`，先用 Qwen3 逐条判定“摘要是否明确支撑该结论”，失败时退回 cross-encoder 分数；低支撑结论标注，整体不可信时拒答。
- **评测**：150 题合成集（60 factoid + 50 comparison + 40 unanswerable）；不做人工校验，指标含 recall@5、引用命中、faithfulness、拒答率、p95，均按下界解读。

## 硬件

RTX 4060 Laptop 8GB / 16GB RAM。BGE-M3 索引约 2.1GB，编码约 50–65 分钟；服务常驻显存与现有 BGE-M3 + reranker + Qwen3-4B 方案一致（约 7GB）。
