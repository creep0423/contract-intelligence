# Contract Intelligence v0.1-pre-UAT

## Checkpoint Metadata

- Date: 2026-09-11 (Asia/Shanghai)
- Repository: `Contract Intelligence`
- Git status: `GIT_REPOSITORY_CREATED`
- Commit: `7513d80e9d89fef5a8f4ef3c2cd7609c1b996611`
- Tag: `contract-intelligence-v0.1-pre-uat`
- Remote: Not configured
- Next phase: Real UAT Readiness Audit

## Current Status

- Migration Complete With Compatibility Residue
- Real UAT Documentation Complete
- Private Data Isolation Complete

## Checkpoint Scope

当前阶段纳入手工基线的文件如下：

### Added Documents

- `docs/product/real_uat_dataset_spec_v1.0.md`
- `docs/product/gold_annotation_guideline_v1.0.md`

### Modified Configuration Files

- `.gitignore`
- `.dockerignore`

上述文件已经纳入 `contract-intelligence-v0.1-pre-uat` Git 基线。后续变更应以该
commit/tag 为可追溯起点，不改写该基线。

## File Integrity

该 checkpoint 由 Git commit 和 repository tag 标识。使用
`git show contract-intelligence-v0.1-pre-uat` 可复核基线内容，不再依赖手工文件哈希。

## Limitations

该仓库是没有导入旧工程历史的独立 Git repository。基线已经提交并打 tag，但没有配置
remote，也没有 push。当前清理变更必须经过人工 review 后另行提交，不移动或覆盖本 tag。

## Recommendation

当前文档和私有数据隔离准备工作已经完成，可以进入 Real UAT Readiness Audit。进入真实数据接入前，应继续遵循 Real UAT 数据规范、Gold 标注规范和私有产物边界要求。
