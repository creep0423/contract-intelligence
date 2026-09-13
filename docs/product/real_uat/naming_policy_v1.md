# Real UAT Naming Policy v1

## Purpose

所有可识别对象使用匿名、稳定、可复核的 ID。运行时 UUID、Milvus chunk ID、客户内部编号和原始合同编号不能作为 Gold 稳定标识。

## Stable identifiers

| Object | Format | Example |
| --- | --- | --- |
| Contract case | `CI-CONTRACT-NNNNNN` | `CI-CONTRACT-000001` |
| Document | `CI-DOC-NNNNNN` | `CI-DOC-000001` |
| Annotation | `CI-ANNOTATION-NNNNNN` | `CI-ANNOTATION-000001` |

编号一经分配不得复用。合同退出数据集时保留 tombstone 记录，不把其编号分配给其他合同。

## File names

推荐格式：

```text
CI-CONTRACT-000001_PURCHASE_MASTER_v1.pdf
CI-CONTRACT-000002_SALES_MASTER_v1.pdf
CI-CONTRACT-000002_AMENDMENT_01_v1.pdf
```

文档类型允许使用 `PURCHASE_MASTER`、`SALES_MASTER`、`TECHNICAL_AGREEMENT`、`QUALITY_AGREEMENT`、`ACCEPTANCE_AGREEMENT`、`AMENDMENT` 和 `CHANGE_ORDER`。

## Prohibited identifiers

文件名、目录名、metadata、Gold 和报告中禁止出现：

- 企业、客户或供应商名称；
- 项目真实名称或人名；
- 原始合同编号、内部订单号；
- CRM、ERP 或其他业务系统 ID。

显示文件名必须来自脱敏后名称，不得回退到原始上传文件名。真实源文件与匿名 ID 的映射表属于受限私有数据，不进入 Git、通用日志或公开 artifact。
