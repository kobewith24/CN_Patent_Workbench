# v0 基线（冻结）

这份目录是 2026-11 改造前的评测快照，**只读归档，后续不再更新**。

- 来源提交：见 `env.json` 的 `git_head`
- 数据：150 题合成集（60 factoid / 50 comparison / 40 unanswerable），gold 由标题生成
- 结论口径：单 gold、合成题、同模型自评，指标只能当下界

## 已作废的说法

- 「150 题评测集上事实题 recall@5 0.85」**不再作为对外指标**。
  题面由标题生成、gold 就是该专利，且未跑过 BM25 基线，该数字不能说明检索质量。
- 「faithfulness 0.606 / 0.868」只覆盖带 `[n]` 的句子，无引用断言不进分母，
  口径偏松，v1 重算。

## 文件

| 文件 | 说明 |
|---|---|
| report.json / report_150.json | 60 / 150 题主报告 |
| report_no_*.json | 消融报告（rerank / title_boost / multistep / verify） |
| gold.jsonl / gold_150.jsonl | 对应评测题 |
| ablations.md / README.md | 旧版说明 |
| env.json | 归档时的环境指纹与索引参数 |
