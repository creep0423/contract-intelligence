# Contract Intelligence Real UAT Dataset Specification v1.0

## 1. Purpose

本规范定义 Contract Intelligence（知契）Real UAT 数据集的范围、准入、目录、元数据、脱敏、版本和冻结要求。

Real UAT 数据集用于：

- 验证系统在真实企业脱敏合同上的字段抽取、履约义务、时间规则、风险识别、证据引用和合同问答能力；
- 作为 Contract Intelligence Benchmark 的真实业务数据输入；
- 评估模型、检索和规则在真实合同分布下的表现；
- 支持按案例和错误类型开展回归定位与错误分析。

本数据集不用于替代法务审查，也不用于评估脱离合同原文的通用法律知识。

## 2. Dataset Scope

### 2.1 首发行业方向

Pilot v1 首发聚焦：

> 制造业与供应链合同履约。

优先覆盖制造企业、供应链企业、设备采购企业和工业品采购企业的签后履约场景；可纳入少量技术服务和复杂工程合同作为邻近领域与挑战样本。

### 2.2 合同类型

| 合同类型 | 主要验证内容 |
| --- | --- |
| 采购合同 | 付款、交付、验收、违约责任 |
| 供应协议 | 持续履约、周期义务、价格与通知机制 |
| 设备采购合同 | 发货、安装、调试、验收、质保 |
| 销售合同 | 交付、收款、开票、收入履约 |
| 框架协议 | 长期义务、订单关系、续约与终止 |
| 技术服务合同 | 里程碑、服务期限、SLA、成果验收 |
| 补充协议/变更协议 | 文档版本、条款差异与潜在冲突 |

### 2.3 数据规模规划

Pilot v1 目标为 20 组合同案例：

```text
Core Domain
  制造业/供应链合同：15 组

Adjacent Domain
  IT/技术服务合同：3 组

Challenge Cases
  工程或其他复杂合同：2 组
```

一组案例可以包含主合同以及一个或多个补充协议，不得为了统计数量将同一合同链中的文档拆成多个独立案例。

## 3. Contract Selection Criteria

### 3.1 必须覆盖的履约场景

每组合同至少应包含：

- 一项付款或收款义务；
- 一项交付、实施或服务义务；
- 一项固定日期、相对日期或事件触发的时间约束；
- 一项明确的责任主体与受益方关系；
- 能够支持关键结论的稳定原文证据。

整个 Pilot 数据集还应覆盖验收、质保、通知期限、违约责任、自动续约、模糊时间以及补充协议冲突等场景。

### 3.2 推荐选择

优先选择：

- 多阶段付款或付款比例复杂的合同；
- 同时包含交付、安装、调试和验收的设备合同；
- 具有质保期、售后义务或 SLA 的合同；
- 具有自动续约、解除或提前通知期的合同；
- 包含表格型付款计划、里程碑或交付清单的合同；
- 带有补充协议、修订协议或多个版本的合同；
- 存在真实歧义但仍可由专家清晰标注的合同。

### 3.3 第一批不适合的数据

第一批数据不宜大量使用：

- 极短的 NDA；
- 仅包含基础租赁信息的简单租赁合同；
- 单页订单或缺少完整条款的报价单；
- 没有可识别履约过程的合同；
- 无法完成脱敏、页码不稳定或文本质量不可恢复的文件；
- Gold 无法通过人工复核的高度争议样本。

挑战样本可以保留，但必须明确标记，不得用低质量数据替代真实业务覆盖。

## 4. Dataset Structure

Real UAT 数据集建议存放在私有、受控且默认不进入 Git 的目录：

```text
contract_uat_v1/
├── documents/
│   └── contract_001/
│       ├── original.pdf
│       ├── sanitized.pdf
│       ├── amendment_01_sanitized.pdf
│       └── metadata.json
├── annotations/
│   └── contract_001.gold.json
├── manifests/
│   └── dataset_manifest.json
├── reviews/
│   └── adjudication/
│       └── contract_001.review.json
└── reports/
```

目录职责：

- `documents/`：原始受控文件、脱敏文件及文档级元数据；
- `annotations/`：人工 Gold Annotation；
- `manifests/`：数据集版本、案例列表、文件摘要和覆盖矩阵；
- `reviews/`：双人复核、分歧与裁定记录；
- `reports/`：该数据集产生的 UAT 或 Benchmark 报告。

真实未脱敏合同不得提交到代码仓库或公共 artifact。

## 5. Metadata Schema

每组合同必须提供 `metadata.json`。最小示例：

```json
{
  "contract_id": "CI-UAT-0001",
  "industry": "manufacturing",
  "contract_type": "procurement",
  "language": "zh-CN",
  "version": "1.0",
  "source_type": "real_sanitized",
  "contains_amendment": true,
  "documents": [
    {
      "document_id": "CI-UAT-0001-MASTER",
      "document_type": "master_contract",
      "document_version": "1.0",
      "file_name": "sanitized.pdf"
    }
  ]
}
```

必填字段：

| 字段 | 说明 |
| --- | --- |
| `contract_id` | 数据集内稳定且唯一的合同案例标识 |
| `industry` | 行业分类 |
| `contract_type` | 合同类型 |
| `language` | 文档主要语言，如 `zh-CN` |
| `version` | 当前合同案例或文档链版本 |
| `source_type` | 数据来源类型 |

`source_type` 支持：

```text
synthetic
real_sanitized
snapshot
real_model
```

含义：

- `synthetic`：人工构造的合成合同；
- `real_sanitized`：完成脱敏并获准用于 UAT 的真实合同；
- `snapshot`：固定版本的系统预测或评测快照；
- `real_model`：由真实模型运行产生的预测结果。

文档来源和预测运行模式应分别记录；不得将 `snapshot` 或 `real_model` 伪装成真实合同来源。

## 6. Data Sanitization

### 6.1 必须脱敏

企业身份：

- 公司、客户、供应商和关联方真实名称；
- 地址、联系人、部门和内部组织信息；
- 电话、邮箱和即时通信账号。

财务及个人敏感信息：

- 银行账号、开户行、税号和支付凭证信息；
- 身份证件、手机号和个人地址；
- 未获授权的真实金额、价格或折扣；
- 其他商业秘密和未公开客户信息。

### 6.2 必须尽量保留

脱敏不得破坏评测所需的业务语义。必须尽量保留：

- 章节、条款、子条款和附件结构；
- 日期、期限、先后顺序和事件触发关系；
- 责任方、受益方和权责关系；
- 付款比例、金额关系和履约节点逻辑；
- 交付、验收、质保、违约和终止逻辑；
- 主合同与补充协议之间的版本和引用关系。

替换值应保持内部一致。例如同一主体在整组文档中必须使用同一脱敏名称。

## 7. Dataset Version Control

数据集版本采用不可变语义版本，例如：

```text
contract-uat-v1.0.0
```

建议规则：

- Major：数据范围、Gold Schema 或评测口径发生不兼容变化；
- Minor：新增合同案例或新的完整标注能力；
- Patch：修正已确认的脱敏、元数据或 Gold 错误，不改变评测口径。

每次版本必须记录：

- dataset version；
- manifest checksum；
- case 数量和类型分布；
- annotation version；
- reviewer/adjudication 状态；
- 变更说明和生效日期。

不得覆盖已用于正式基线的历史版本。

## 8. Gold Freeze Rules

数据集进入正式 Benchmark 或 UAT Baseline 前，必须满足：

- 所有纳入案例均已完成脱敏检查；
- Metadata 和文件引用完整；
- Gold Annotation 已完成；
- 至少一名 reviewer 完成复核；
- 存在分歧时已形成 adjudicated result；
- dataset version 和 annotation version 已写入 manifest；
- Gold source 能映射到真实文档、页码、条款和 quote；
- 冻结内容不依赖运行时 Milvus chunk UUID。

冻结后禁止直接修改原版本。发现错误时，应创建新的 Patch 版本并记录 correction，不得为了提高评测分数修改 Gold、删除困难案例或覆盖历史报告。

## 9. Access and Lifecycle

推荐生命周期：

```text
合同获取
→ 权限确认
→ 脱敏
→ Metadata 登记
→ 人工 Gold 标注
→ Reviewer 复核
→ Adjudication
→ Dataset Freeze
→ Baseline Evaluation
→ Failure Analysis
→ 后续优化
```

真实 UAT 数据必须存放在私有目录或受控存储中，并通过最小权限访问。代码仓库只应保留规范、Schema、脱敏示例和不含真实商业信息的模板。

### 9.1 Repository and Image Boundaries

项目内的真实 UAT 数据必须使用以下私有边界：

```text
data/private_uat/                 # 脱敏合同、私有 manifest
data/private_annotations/         # Gold、review、adjudication
uploads/                          # 仅用于本地受控上传暂存
artifacts/private/                # 私有评测结果、预测和 snapshot
reports/private/                  # 其他私有验收报告
```

这些目录必须同时排除在 Git 跟踪和 Docker build context 之外。运行时合同文件应写入受控持久卷或受控外部存储，不得将合同打包进应用镜像，也不得依赖容器可写层保存。原始未脱敏合同不得放入本仓库的任何路径，即使该路径已被忽略。

私有 UAT 文件名应使用不含企业名称、合同相对方或其他敏感信息的不透明标识。评测环境建议保持 `LOG_LEVEL=INFO` 或更高。`LANGSMITH_TRACING` 默认必须保持为 `false`；只有在数据治理、外部处理授权和脱敏策略均明确批准后方可开启，因为 trace 可能包含问题、回答及来源片段。

### 9.2 Public and Private Artifact Boundary

真实 UAT 的 Benchmark 输出必须显式写入私有目录，例如：

```bash
python -m qa_core.evaluation.runner \
  --suite full \
  --manifest data/private_uat/contract_uat_v1/manifests/manifest.json \
  --predictions data/private_annotations/contract_uat_v1/predictions.json \
  --output-root artifacts/private/contract-uat-v1.0.0
```

Benchmark 的 JSON 与 Markdown 报告可能包含 case-level expected、predicted、evidence、问题、回答和来源片段，因此凡基于真实 UAT 生成的完整报告一律视为私有产物，不得发布或提交 Git。

`artifacts/public/` 仅允许存放经过单独脱敏审核的汇总指标。公开产物不得包含合同原文、quote、source text、主体名称、原始文件名、问题与回答正文、Prompt、逐案例 expected/predicted 或可反向识别合同的信息。将私有报告转换为公开摘要必须经过人工复核，不能仅依赖目录名称判定安全性。

### 9.3 Authorized Prediction Generation

Benchmark Evaluator 只负责加载 Gold 与 Prediction Bundle 并计算确定性指标。对于 `real_sanitized`、`real_model` 或 `snapshot` 数据集，Evaluator 不打开或解析 `documents` 中登记的合同文件。

真实合同预测必须先通过应用现有的可信身份入口和 DataScope 生成，再交给 Benchmark：

```text
Real Contract
→ trusted upstream
→ tenant / dataset / visibility / roles DataScope
→ Authorized Prediction Generation
→ Prediction Bundle
→ Benchmark Evaluation
```

真实数据 Prediction Bundle 必须包含 `data_access`，记录授权路径、tenant、dataset、visibility、allowed roles 和 trusted upstream 验证状态。该声明用于审计和拒绝未声明来源的 bundle，不替代上游认证，也不得由 Benchmark Evaluator 自行补写。

### 9.4 UAT Runtime Requirements

真实 UAT 运行环境必须满足：

```ini
APP_ENV=uat
CONTRACT_TRUSTED_UPSTREAM_TOKEN=<由部署系统安全注入>
LOG_LEVEL=INFO
LANGSMITH_TRACING=false
```

`uat`、`staging`、`prd`、`prod` 和 `production` 均采用可信身份语义：未配置 upstream token 时合同入口必须拒绝运行。仓库中的 `.env.uat.example` 仅是配置模板，不包含可用密钥。

MySQL 集成测试必须使用独立的 `*_test` 或 `test_*` 数据库。`.env.test.example` 与 `docker-compose.test.yml` 提供隔离测试配置；测试不得指向 Contract Intelligence UAT 或生产数据库。
