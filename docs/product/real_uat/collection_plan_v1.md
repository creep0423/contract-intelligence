# Industrial Equipment Real UAT Collection Plan v1

## Scope

- Parent domain：Manufacturing / Supply Chain；
- Phase-1 ICP：50–300 人工业机械 / 自动化设备制造企业；
- Primary contracts：设备采购合同、设备销售合同；
- Primary users：采购负责人、财务负责人。

首批 Real UAT 不追求代表整个制造业，目标是在一个明确 ICP 中获得可解释、可重复的真实 baseline。

## Dataset size and composition

首批固定为 **20 个 contract cases**，不是 20 个文件。一个 case 可以包含主合同、技术协议、质量协议、验收协议、补充协议和变更协议。

采购合同和销售合同各至少 8 个。剩余 4 个 case 在 dataset freeze 前根据覆盖缺口分配到采购或销售类别；最终每个 case 仍必须是 `equipment_purchase` 或 `equipment_sales`，总数保持 20。

至少 4 个 case 具有 multi-document / amendment coverage，可与上述采购或销售 case 重叠，不额外增加总 case 数。变更范围至少覆盖付款条件、交付时间、验收标准、质保期限或责任条款中的两类，并至少形成 2 个可标注的 amendment conflict case。

## Complexity distribution

| Complexity | Cases |
| --- | ---: |
| SIMPLE | 4 |
| MEDIUM | 8 |
| COMPLEX | 6 |
| HIGH | 2 |
| **Total** | **20** |

## Coverage targets

- 5+ event-relative deadline cases；
- 5+ staged payment cases；
- 4+ acceptance-dependent payment cases；
- 3+ warranty deposit cases；
- 3+ ambiguous clause cases；
- 2+ amendment conflict cases；
- 2+ table-heavy cases；
- 2+ no-evidence QA situations；
- 3+ HIGH / CRITICAL risk candidate cases。

每个 case 应尽量覆盖付款、交付、安装调试、验收、质保和责任中的多个环节，但不得为了达成配额修改合同内容或 Gold。

## Case selection criteria

优先选择：

- 已完成脱敏且条款结构可读；
- 能体现本公司作为买方或卖方的设备交易；
- 包含可追溯付款、交付、验收或质保约定；
- 文件版本和主从关系可以确认；
- 业务人员可以参与事实确认，但不会接触模型预测后再定义 Gold。

第一批不选择：

- 无法合法用于 UAT 或授权边界不清的合同；
- 无法可靠脱敏的高度敏感交易；
- 纯扫描且 OCR 无法达到人工复核要求的文件；
- 大量缺页、版本不明或附件缺失导致事实无法还原的 case；
- 与制造业 / 供应链上位领域及 Phase-1 ICP 无关的行业合同。

## Dataset structure

真实数据仅存在于 Git ignored 私有边界：

```text
data/private_uat/
  documents/
  manifests/
data/private_annotations/
artifacts/private/
reports/private/
```

Git 只保存 `docs/product/real_uat/` 中的规范和 `data_packs/contract_uat_templates/` 中的空模板。

## Template enum contract

模板字段只能使用以下枚举，禁止用自由文本制造标注漂移：

- `contract_type`：`equipment_purchase`、`equipment_sales`；
- `company_role`：`buyer`、`seller`；
- `related_document_type`：`technical_agreement`、`quality_agreement`、`acceptance_agreement`、`amendment`、`change_order`；
- `complexity`：`SIMPLE`、`MEDIUM`、`COMPLEX`、`HIGH`；
- `sanitization_status`：`PENDING`、`IN_PROGRESS`、`COMPLETED`、`REJECTED`；
- `annotation_status`：`NOT_STARTED`、`IN_PROGRESS`、`REVIEW`、`FROZEN`；
- `review_status`：`NOT_STARTED`、`IN_REVIEW`、`APPROVED`、`NEEDS_REVISION`。

Real UAT 模板的 `source_type` 固定为 `real_sanitized`。`synthetic`、`snapshot` 和 `real_model` 仍由既有 Benchmark schema 表达，但不能伪装成真实脱敏合同 metadata。

## Dataset freeze rule

冻结前必须满足：20 个 case 已完成脱敏、匿名 metadata、Gold、规定比例的 secondary review 和必要 adjudication；case ID 无重复；文件 checksum、文档版本和 source reference 可追溯；真实合同与 Gold 仍位于私有目录。

冻结版本使用 `industrial-equipment-uat-vMAJOR.MINOR.PATCH`。冻结后不能静默增删 case 或修改 Gold；任何变化都产生新 dataset version，并保留变更记录。AI prediction 只能在对应 Gold version 冻结后生成。
