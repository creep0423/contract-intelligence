# Contract Intelligence Gold Annotation Guideline v1.0

## 1. Purpose

本规范面向 Contract Intelligence Real UAT 的人工标注人员、业务专家、合同管理员及法务 reviewer，用于建立可重复、可审计、可比较的 Gold Annotation。

Gold 的职责是记录合同原文事实和有证据支持的风险判断，不是根据模型输出、行业经验或一般法律知识补全合同内容。

## 2. Annotation Philosophy

### 2.1 Evidence First

所有关键 Gold 结果必须能够追溯到合同原文。字段、义务、时间规则、风险、问答和冲突均应优先记录稳定的文档、页码、条款和 quote。

如果原文不足以支持一个确定结论，不得通过常识补全。

### 2.2 Conservative Annotation

标注应遵循保守原则：

> 宁可明确标记“不存在”“不适用”或“有歧义”，也不要创造确定答案。

例如，合同仅约定“乙方应及时交付”时，不得标注为“30 日内交付”，应将时间状态标记为 `AMBIGUOUS`。

### 2.3 Independent Gold

标注人员不应查看当前模型预测后再决定 Gold，不得为了提升 Benchmark 得分修改 Gold，也不得删除系统难以处理的有效样本。

## 3. Annotation Objects

每组合同的 Gold 应能够表达：

```text
Contract
├── Fields
├── Obligations
├── TimeRules
├── Risks
├── Evidence
├── QA
└── Conflicts
```

- `Contract`：合同案例、文档链和版本关系；
- `Fields`：合同基础信息及关键履约字段；
- `Obligations`：合同明确要求责任方执行的事项；
- `TimeRules`：义务时间的固定、相对、事件或周期规则；
- `Risks`：规则型及语义型履约风险；
- `Evidence`：支持标注结论的稳定原文；
- `QA`：事实、风险、建议和无证据问题；
- `Conflicts`：主合同、补充协议或修订文档间的潜在冲突。

## 4. Field Annotation

### 4.1 字段状态

每个字段必须区分以下状态：

```text
EXPECTED_VALUE
NOT_PRESENT
NOT_APPLICABLE
AMBIGUOUS
```

- `EXPECTED_VALUE`：合同明确存在可确定值；
- `NOT_PRESENT`：合同中未发现该字段对应的约定；
- `NOT_APPLICABLE`：该字段对当前合同类型或案例不适用；
- `AMBIGUOUS`：合同有相关表述，但无法得到唯一、确定结果。

不得使用空字符串混淆上述状态。

### 4.2 标注示例

```json
{
  "field_name": "contract_amount",
  "status": "EXPECTED_VALUE",
  "value": {
    "currency": "CNY",
    "amount": 1000000
  },
  "evidence": {
    "document": "sanitized.pdf",
    "page": 3,
    "clause": "2.1",
    "quote": "本合同含税总价为人民币壹佰万元整。"
  }
}
```

### 4.3 常见字段规则

- 金额应同时记录规范化数值和币种；
- 百分比使用明确的比例值，不得从金额关系中无依据反推；
- 日期应区分签署、生效、终止和具体履约日期；
- 甲乙方、买卖方等角色必须依据原文，不根据行业习惯猜测；
- 合同未约定生效日期时，不能用签署日期代替。

## 5. Obligation Annotation

### 5.1 定义

Obligation 是合同中明确要求某一方完成某项行为或达到某种结果的约定。

每项义务至少包含：

- `obligation_type`：义务类型；
- `party`：义务承担方；
- `action`：需要完成的动作或结果；
- `trigger`：义务启动条件；
- `deadline`：固定日期或时间规则；
- `evidence`：支持该义务的原文。

### 5.2 支持类型

```text
PAYMENT
DELIVERY
ACCEPTANCE
WARRANTY
NOTICE
OTHER
```

若现有 Gold Schema 支持更细类型，可以使用 `INSTALLATION`、`DOCUMENT_SUBMISSION`、`MAINTENANCE`、`CONFIDENTIALITY`、`INTELLECTUAL_PROPERTY`、`INSURANCE` 等扩展值，但不得改变 Benchmark 已冻结的枚举口径。

### 5.3 标注示例

```json
{
  "gold_id": "OBL-001",
  "obligation_type": "PAYMENT",
  "party": "甲方",
  "action": "支付合同金额的30%作为首付款",
  "trigger": "CONTRACT_SIGNED",
  "deadline": {
    "rule_type": "RELATIVE_DATE",
    "offset_value": 5,
    "offset_unit": "DAY",
    "direction": "AFTER"
  },
  "evidence": {
    "document": "sanitized.pdf",
    "page": 4,
    "clause": "3.1",
    "quote": "合同签署后5日内，甲方支付合同总价的30%作为首付款。"
  }
}
```

愿景、合作意向、背景描述或没有明确责任主体的宣传性文字不应标注为义务。

## 6. TimeRule Annotation

TimeRule 应表达合同真实的时间计算方式，而不只是保存一段文本。

支持：

- `FIXED_DATE`：明确的固定日期；
- `RELATIVE_DATE`：相对于合同签署、生效或其他基准日期；
- `EVENT_TRIGGER`：由验收、交付、开票、通知等事件触发；
- `PERIODIC`：按月、季度、年度等周期重复；
- `AMBIGUOUS`：如“及时”“尽快”“适时”等无法确定期限的表述。

相对或事件触发规则应尽量标注：

- base event；
- offset value；
- offset unit；
- business day；
- direction；
- fixed date（如适用）。

合同没有明确日历日或工作日时，不得自行选择。存在多个同类事件且无法确定具体关联事件时，应标记歧义并提交 reviewer。

## 7. Risk Annotation

### 7.1 基本原则

风险标注必须同时考虑合同事实、潜在履约影响和证据。风险不是对所有“不理想”措辞的主观评价。

### 7.2 Deterministic Risk

Deterministic Risk 由确定日期、义务状态或明确规则产生，例如：

- 已逾期；
- 即将到期；
- 合同即将到期；
- 自动续约通知期临近。

标注时应记录规则计算所需的基准日期、状态和来源。

### 7.3 Semantic Risk

Semantic Risk 需要专家依据合同语义判断，例如：

- 付款条件模糊；
- 验收标准主观或缺失；
- 责任明显不对等；
- 自动续约或解除通知期风险；
- 不同文档中的约定可能冲突。

描述可以存在合理变体，但风险类型、业务对象和支持证据应明确。

### 7.4 Severity

统一风险等级：

```text
LOW
MEDIUM
HIGH
CRITICAL
```

- `LOW`：提示性问题，短期影响有限；
- `MEDIUM`：可能增加成本、争议或管理负担，需要跟进；
- `HIGH`：可能造成重要经济损失、履约失败或关键权利受损；
- `CRITICAL`：已经发生或迫近的重大损失、严重违约或关键业务中断风险。

等级具有一定主观性。标注人员应写明理由；High/Critical 分歧必须进入 reviewer/adjudication。

## 8. Evidence Annotation

Evidence 是 Gold 中的最高优先级对象。每条关键证据必须包含：

```text
document
page
clause
quote
```

要求：

- `document` 指向稳定的脱敏文档及版本；
- `page` 使用最终冻结文件的可复现页码；
- `clause` 尽量使用合同原有条款编号；
- `quote` 必须是原文中的连续文本，并足以支持结论。

禁止：

- 根据行业经验补充合同不存在的信息；
- 使用无法定位的摘要作为证据；
- 用“付款”等过短引用支持“付款期限为30日”等更具体结论；
- 使用运行时 Milvus chunk UUID 作为唯一 Gold reference；
- 根据模型生成的页码、条款或 quote 修改 Gold。

推荐稳定引用结构：

```json
{
  "gold_source_id": "SRC-001",
  "document": "amendment_01_sanitized.pdf",
  "document_version": "1.0",
  "page": 2,
  "clause": "1.2",
  "quote": "付款期限调整为验收合格后60日内。"
}
```

## 9. QA Annotation

QA 至少覆盖：

- `FACT`：金额、日期、比例、期限、主体等合同事实；
- `RISK`：基于合同约定识别风险；
- `SUGGESTION`：允许提出一般处理建议，但不得伪装成合同事实；
- `NO_EVIDENCE`：合同未明确约定的问题。

每条 QA 应记录问题、预期事实、禁止出现的断言、预期来源以及是否允许一般建议。

对于 `NO_EVIDENCE`，正确答案应明确表达“根据当前合同内容，未发现明确约定”，并且不能附加虚假来源。

## 10. Conflict Annotation

冲突标注用于主合同、补充协议和修订协议之间同一业务主题的不同约定。

典型情况：

```text
主合同：验收后30日付款
补充协议：验收后60日付款
```

应标记为 `potential_conflict`，并分别提供两个文档的 Evidence。

冲突对象至少记录：

- base document；
- amendment document；
- changed clause/topic；
- old value；
- new value；
- dual evidence。

禁止自动标注“补充协议一定优先”“某条款无效”或其他确定法律效力结论。Gold 只记录文本差异和潜在冲突；法律优先级需要法务另行判断。

## 11. Review and Adjudication

推荐流程：

```text
Annotator A
→ Annotator B
→ Reviewer
→ Adjudicated Gold
```

每次复核至少记录：

- 匿名 `reviewer_id`；
- annotation version；
- disagreement type；
- reviewer decision；
- adjudicated result；
- reviewed_at。

分歧判断优先级：

1. 原文证据；
2. 本标注规范；
3. 合同或法务专家判断；
4. 行业经验。

行业经验不能覆盖合同原文事实。

## 12. Gold Freeze Rules

Gold 冻结前必须完成：

- 所有必填对象和状态校验；
- 关键字段、义务和风险 Evidence 校验；
- 文档、页码、条款和 quote 可定位性检查；
- `NOT_PRESENT`、`NOT_APPLICABLE` 和 `AMBIGUOUS` 的区分检查；
- reviewer 复核；
- 所有重大分歧 adjudication；
- annotation version 和 dataset version 登记。

示例版本：

```text
dataset_version: contract-uat-v1.0.0
annotation_version: gold-v1.0.0
```

冻结后不得直接覆盖。修订时应创建新版本，记录 reviewer、adjudication、修订原因和受影响案例，并保持历史版本可追溯。

## 13. Prohibited Annotation Practices

禁止：

- 使用模型直接生成未经人工核验的 Gold；
- 根据当前预测结果反向修改 Gold；
- 根据法律常识或行业习惯补充合同未写明的内容；
- 修改原始合同以适配系统输出；
- 删除困难或失败案例；
- 用模糊 Evidence 支持精确金额、比例、日期或期限；
- 将风险判断伪装为合同事实。

## 14. Final Principle

Contract Intelligence Gold 的核心原则是：

> 系统只能把合同中存在并可由证据支持的内容视为合同事实。

Gold 应当可重复、可审计、可比较，并能够支持后续真实 UAT、版本回归和错误分析。
