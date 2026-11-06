# 消融实验（种子集 v0）

同一 60 题种子集；答案侧每类 6 题（共 18）。单次运行，样本小，数值只作方向参考。

| 配置 | factoid recall@5 | comparison all@5 | 引用命中 | faithfulness | 答案侧耗时 |
|---|---|---|---|---|---|
| 完整 | **0.700** | **0.800** | 1.000 | **0.868** | 133.6 s |
| 无 rerank（纯稠密） | 0.633 | 0.800 | - | - | - |
| 无标题 bigram | 0.667 | 0.800 | - | - | - |
| 无多步（单查询） | 0.700 | 0.800 | 1.000 | 0.818 | 182.0 s |
| 无引用校验 | 0.700 | 0.800 | 1.000 | - | 106.7 s |

## 结论

- **rerank 是 factoid 召回的主要贡献**：0.633 → 0.700（+6.7 pt）。
- **标题 bigram 再 +3.3 pt**：0.667 → 0.700；对对比题无影响。
- **多步拆解不改变检索命中**（对比题两件专利本来就能被单查询召回），但把 faithfulness 从 0.818 提到 0.868（+5 pt）。
- **引用校验约 +1.5 s/题**（133.6 vs 106.7 s / 18 题），换来 0.868 的忠实度；这是可信问答的主要成本。
- 对比题 all-gold hit@5 在所有配置下都是 0.800，说明剩余 20% 是语料/题目问题，不是检索链路问题。

## 复现

```powershell
python eval/run_eval.py --no-rerank --answer-n 0 --tag no_rerank
python eval/run_eval.py --no-title-boost --answer-n 0 --tag no_title_boost
python eval/run_eval.py --no-multistep --answer-n 18 --tag no_multistep
python eval/run_eval.py --no-verify --answer-n 18 --tag no_verify
```
