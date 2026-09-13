# Real UAT Gold Annotation Workflow v1

## Frozen workflow

```text
collect
↓
sanitize
↓
metadata
↓
Gold annotation
↓
secondary review
↓
adjudication
↓
freeze
↓
AI prediction
↓
benchmark
```

> AI output must not define Gold.

> Annotators must complete and freeze Gold before reviewing model predictions.

模型输出、已有摘要、行业经验和评测结果不得提前展示给 Gold 标注员。需要修改冻结后的 Gold 时，必须产生新 annotation version、记录原因并重新 adjudication，不能为了提高分数直接覆盖旧 Gold。

## Annotation focus

### Contract facts

优先标注合同金额、付款比例、付款条件、交付日期、安装条件、验收条件、质保期限、违约金和解除条件。事实状态沿用 `EXPECTED_VALUE`、`NOT_PRESENT`、`NOT_APPLICABLE` 和 `AMBIGUOUS`。

### Obligations

每项义务至少包含 responsible party、action、trigger、deadline、prerequisite 和 evidence。不能根据行业惯例补全责任方、期限或前置条件。

### TimeRule

优先覆盖：合同签署后 N 日、收到预付款后 N 日、发货前 N 日、到货后 N 日、安装完成后 N 日、调试完成后 N 日、验收合格后 N 日、收到发票后 N 日和质保期满后 N 日。

合同表达不确定时标记 `AMBIGUOUS`，不得强迫解析为确定日期。

### Risks

优先标注延迟交付、付款条件不清、验收条件不清、无限责任、高额违约金、单方解除、质保责任过重、验收与付款条件冲突及补充协议冲突。风险必须有原文证据，严重度判断与合同事实分开保存。

## Gold object boundaries

Gold 必须区分：

1. **Contract Fact**：合同明确写了什么；
2. **Risk Judgment**：基于合同事实识别的潜在风险；
3. **Recommendation**：建议进一步确认或采取的行动。

行业经验、法律常识和模型建议不能标成 Contract Fact。没有合同依据的事实不得通过 Recommendation 绕过事实约束。

## Review policy

普通 case 由一名主标注员完成，并抽取 30–50% 进行 secondary review。以下对象必须 100% secondary review：

- HIGH / CRITICAL risk；
- ambiguous clause；
- amendment conflict；
- no-evidence QA；
- event-relative deadline；
- staged payment；
- acceptance-dependent payment；
- warranty deposit release。

主标注员和复核员意见不一致时进入 adjudication。Reviewer 使用匿名 ID；annotation version、reviewer、adjudication 结果和 freeze 时间必须可追溯。

## Freeze rules

只有 `sanitization_status=COMPLETED`、`annotation_status=REVIEW` 且所需复核完成的 case 才能冻结。冻结后设置 `annotation_status=FROZEN` 和 `review_status=APPROVED`，记录版本及时间，再生成 AI prediction。预测文件与 Gold 分离保存。
