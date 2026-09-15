# CN_Patent_RAG

面向中文专利技术情报的检索增强问答（RAG）。52 万件专利摘要 → BGE-M3 向量索引 → Spring Boot + LangChain4j 应用层 → 检索、过滤、流式生成与引用校验。

## 架构

```
浏览器
  → Spring Boot 3.5 + LangChain4j（8080）
      ├─ ChatMemory 多轮会话
      ├─ ChatModel 查询改写
      ├─ 调 Python /retrieve
      ├─ Ollama 流式生成
      ├─ 调 Python /verify
      └─ SSE：sources → token* → verify → done
  → Python FastAPI（8001）：BGE-M3 + bge-reranker + /verify
```

## 工作流程

1. **数据**：52 万件专利摘要，清洗后 519,424 条；处理推广语、空摘要、重复申请号。
2. **索引**：构建 BGE-M3 1024 维向量索引，HNSW 检索。
3. **应用层**：Spring Boot + LangChain4j 负责会话记忆、查询改写、上下文组装、流式生成与 SSE。
4. **检索**：Python 侧按 IPC/年份/类型/城市过滤，向量召回 + bge-reranker 精排，返回带专利号/摘要的证据。
5. **生成**：LangChain4j `AiServices` 编排 Ollama 流式生成，按“结论 / 明确相关 / 证据不充分”结构输出，正文只写 `[n]`。
6. **校验**：生成结束后调用 Python `/verify`，逐句判断引用摘要是否支撑结论，低支撑标记；整体不可信时拒答。
7. **评测**：150 题合成集，覆盖检索命中、引用命中、忠实度、拒答率与延迟。

## LangChain4j 用法

- `AiServices` + `PatentAssistant` 声明式定义流式对话接口；
- `ChatMemoryProvider` 按 `sessionId` 复用 `MessageWindowChatMemory`，支持多轮；
- `ChatModel` 做追问的查询改写；
- `OllamaStreamingChatModel` + `TokenStream` 逐 token 生成；
- `SseEmitter` 推送 `sources`、`token`、`verify`、`done`；
- Python `/retrieve`、`/verify` 作为检索与校验微服务。

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
├─ rag-app/
│  ├─ pom.xml                 # Spring Boot 3.5 + LangChain4j 1.23
│  └─ src/main/java/...       # AiServices / ChatMemory / SSE
├─ service/
│  ├─ app.py                  # FastAPI：/retrieve /verify
│  └─ static/index.html       # 旧版 Python 问答页
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

| 模块 | 端口 | 说明 |
|---|---|---|
| `rag-app` | 8080 | Spring Boot 3.5 + LangChain4j 应用层，Web/SSE |
| `retrieval` | 8001 | Python FastAPI，BGE-M3 + bge-reranker + `/verify` |
| Ollama | 11434 | Qwen3 生成 |

### 1. Python 检索服务

```powershell
cd D:\Projects\CN_Patent_RAG
python -m uvicorn service.app:app --host 127.0.0.1 --port 8001
```

### 2. Java 应用层

需要 JDK 21 + Maven。

```powershell
cd D:\Projects\CN_Patent_RAG\rag-app
mvn -DskipTests package
java -jar target\rag-app-1.0.0.jar
# 浏览器打开 http://127.0.0.1:8080/
```

接口：

| 服务 | 接口 | 说明 |
|---|---|---|
| Java | `GET /` | Web 问答页（过滤 / 答案 / 引用 / 支撑校验） |
| Java | `GET /api/chat?q=...&sessionId=...` | SSE：`sources` → `token`* → `verify` → `done` |
| Java | `GET /api/health` | 健康检查 |
| Python | `GET /retrieve?q=...` | 检索 + 过滤 + rerank |
| Python | `POST /verify` | 逐句引用校验 |
| Python | `GET /answer`、`/answer_stream` | 兼容旧链路 / 直接流式 |

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

RTX 4060 Laptop 8GB / 16GB RAM。BGE-M3 索引约 2.1GB，编码约 50–65 分钟；服务常驻显存约 7GB（BGE-M3 + reranker + Qwen3-4B）。
