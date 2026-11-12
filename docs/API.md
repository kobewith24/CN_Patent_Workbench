# 本机 API

除 `GET /api/health` 外，Java API 需要 `Authorization: Bearer <data/access.token>`。

先 `POST /api/workspaces {"title":"我的工作区"}` 获得一次性返回的 `id` 与 `key`。后续请求同时携带 `X-Workspace-ID` 与 `X-Workspace-Key`。数据库只保存随机密钥的 SHA-256 散列；仅有工作区 ID 无法访问。

| 方法和路径 | 用途 |
|---|---|
| GET /api/info | 版本、语料范围、检索健康及队列指标 |
| GET /api/metrics | 实际在途数、排队数、完成/取消/失败/拒绝计数 |
| POST /api/studies | 新建调研，body: title |
| GET /api/studies | 本工作区的调研列表 |
| GET /api/studies/{id} | 调研、上次查询与筛选状态 |
| PATCH /api/studies/{id} | 修改标题 |
| DELETE /api/studies/{id} | 删除单个调研及其候选、备注与分析记录；有排队/运行/核验任务时返回 409；不删除原始语料 |
| POST /api/studies/{id}/search | query 与 filters；候选深度由共享配置控制 |
| PUT /api/studies/{id}/filters | 验证并保存显式筛选条件 |
| PUT /api/studies/{id}/documents | 保存 doc_id、note、selected；忽略客户端传入的原文，以规范数据为准 |
| GET /api/studies/{id}/documents | 候选清单与备注 |
| DELETE /api/studies/{id}/documents/{doc_id} | 从候选清单移除 |
| GET /api/documents/{doc_id} | 完整摘要与证据片段 |
| POST /api/studies/{id}/jobs | 创建 overview / compare / follow_up 任务，返回 202 |
| GET /api/jobs/{id} | 状态与已保存结果 |
| GET /api/jobs/{id}/events | SSE：phase、sources、result 与心跳 |
| POST /api/jobs/{id}/cancel | 取消排队或传输请求 |
| GET /api/studies/{id}/jobs | 本次调研的分析历史 |
| GET /api/studies/{id}/export?format=md 或 csv | 调研报告 / 候选清单 |

比较请求示例：

```json
{"kind":"compare","question":"比较实现方法和关键组成，分别给出证据。","doc_ids":["实际申请号 A","实际申请号 B"]}
```

compare 选择 2–4 件，overview 选择 1–6 件；follow_up 基于本次调研上次查询与显式条件生成新检索式。无上次查询时拒绝追问。

任务终态为 completed、cancelled、timed_out、failed 或 interrupted。失败不会被当作成功回答；原文已取得时仍保留在任务记录中。队列满返回 HTTP 429，访问令牌无效 401、工作区凭据无效 403、跨工作区资源 404、超出数据范围的确定性判断 422。

Python 仅监听本机，所有端点需要 `X-Internal-Token`：`/health`、`/search`、`/documents`、`/validate_filters`、`/audit`。该服务没有生成端点，不直接供浏览器调用。
