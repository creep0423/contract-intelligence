# Contract Risk Benchmark

这是合同履约风控助手的版本化评测数据包。当前 `small-synthetic-v1.0.0` 包含 10 个
合成 case，全部 `synthetic=true`；真实 UAT 合同数量为 0。

## 运行

```bash
python -m qa_core.evaluation.validate_annotations --suite full
python -m qa_core.evaluation.runner --suite fast
python -m qa_core.evaluation.runner --suite full --predictions private_uat/predictions.json
python -m qa_core.evaluation.compare_runs baseline.json candidate.json --output comparison.json
python -m qa_core.evaluation.gate benchmark_report.json --thresholds data_packs/contract_benchmark/gate.json
```

Runner 默认读取明确标记为 `offline` 的 prediction fixture，不会调用付费模型。真实模型结果
必须导出为 `PredictionBundle(mode="real_model")` 后通过 `--predictions` 显式传入。

## 添加真实 UAT

1. 将脱敏合同放在 gitignored 私有目录，例如 `private_uat/contracts/`。
2. 按 `annotations/cases.json` 增加 `synthetic=false` annotation，使用匿名 reviewer ID。
3. Gold source 使用文件名、版本、页码、条款号和短 snippet；不要使用 Milvus chunk UUID。
4. 更新私有 manifest 的 `cases_file`，无需修改 Python。
5. 先运行 annotation validator，再运行 FULL。

真实合同进入任何共享仓库前必须移除身份证号、手机号、银行账户、客户身份、商业秘密和
未公开交易信息。通用评测日志不得记录合同全文；case result 仅保存必要结构。

## 评分边界

- matcher 是确定性的，不用 LLM-as-a-Judge 作为唯一裁判。
- `AMBIGUOUS` 与未裁决语义风险进入 `needs_human_review`。
- offline baseline 只证明框架和数据可重复，不代表线上模型准确率。
- 风险 Gold 与法律判断可能存在主观性，真实 UAT 应由至少两名匿名 reviewer 独立标注并裁决。
