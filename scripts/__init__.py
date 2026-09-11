# -*- coding: utf-8 -*-
# ============================================================================
# 项目脚本包
# ============================================================================
# 根目录只保留高频入口脚本，低频专项脚本按主题放入子包：
#
#   子包结构：
#     intent/              — 意图识别和阈值校准脚本
#       - evaluate_intent_policy.py   — 意图策略评测
#       - calibrate_thresholds.py     — 阈值候选生成
#       - train_intent_bert.py        — 本地 BERT 意图模型训练
#       - demo_intent_model.py        — 意图模型演示
#
#     quality/             — 质量周期和性能建议脚本
#       - run_v1_quality_cycle.py     — 无人值守质量周期编排
#       - recommend_runtime_params.py  — 运行时参数建议
#       - cache_acceptance_smoke.py    — 缓存验收冒烟检查
#       - check_evaluation_gate.py     — 主链路评测门禁
#       - check_ingestion_quality_gate.py — 入库质量门禁
#       - collect_performance_baseline.py — 性能基线采集
#
#     deploy/              — 部署验收脚本
#       - deploy_docker.ps1            — Docker 一键部署
#       - verify_fresh_docker_deploy.py — 新环境 Docker 部署验收
#
#     course/              — 课程资料维护脚本
#       - check_codealong_alignment.py — 跟敲代码对齐检查
#       - check_docs_consistency.py    — 讲义一致性检查
#       - sync_chapter_maps.py         — 清理旧章节地图块
#       - sync_chapter_animations.py   — 同步章节动画
#
#     demo/                — 章节演示脚本
#       - demo_ch03_langchain_components.py — 第 03 章 LangChain 组件演示
#       - demo_ch04_milvus_basics.py        — 第 04 章 Milvus 基础演示
#       - demo_query_prepare.py             — 第 06 章查询准备演示
#       - demo_query_rewrite_variants.py    — 第 07 章查询改写演示
#
#     enterprise_overlay/  — 企业仿真资料增强相关脚本
#       - analyze_dirty_enterprise_samples.py  — 脏样本治理风险分析
#       - analyze_enterprise_data_realism.py   — 资料真实度评估
#       - build_enterprise_overlay_dataset.py  — 构建 clean overlay 预览数据集
#       - check_enterprise_overlay_readiness.py — overlay 就绪门禁
#       - plan_enterprise_overlay_activation.py — 生成上线计划
#       - run_enterprise_overlay_activation.py  — 执行上线计划
#
#     kb/                  — 知识库版本对比和治理脚本
#       - compare_kb_versions.py     — 单场景版本召回对比
#       - compare_all_kb_versions.py — 全场景版本召回对比
#       - manage_kb_versions.py      — 知识库版本管理命令行工具
#       - cleanup_missing_docs.py    — 清理已删除文档对应的 manifest/chunk
#
#     ocr/                 — 离线 OCR 复核资料处理脚本
#       - run_offline_ocr.py       — 离线 OCR 清洗入口
#       - promote_ocr_candidates.py — 提升已复核 OCR 资料
#
#     tools/               — 低频本地辅助工具脚本
#       - check_langchain_stack.py       — LangChain 依赖栈检查
#       - capacity_estimate.py          — 知识库容量与参数压力估算
#       - check_local_runtime.py        — 本地运行环境诊断
#
#   首轮必会脚本（本目录）：
#     - acceptance_smoke.py           — 本地服务真实链路验收
#     - api_e2e_smoke.py             — 真实 API E2E 验收
#     - rebuild_kb_version.py        — 一次性构建完整知识库版本
#     - rebuild_scenarios.py         — 批量重建多个场景的知识库版本
#     - check_project_guardrails.py  — 项目结构和代码约束守护检查
#     - evaluate_core_chain.py       — 主问答链路工程回归评测
#     - extract_bad_cases_from_report.py — 从评测报告导出 Bad Case 草稿
#     - export_feedback_bad_cases.py — 从用户反馈导出 Bad Case 复核草稿
#     - promote_bad_cases_to_regression.py — 将复核后的 Bad Case 合并进正式回归集
#     - verify_v1_release.py         — V1 发布验收编排入口
#
#   公共基础设施（供其他脚本导入）：
#     - common.py     — 脚本层公共工具（JSON 读写、UTF-8 输出、命令执行、报告摘要）
#     - gate_utils.py — 门禁判断公共函数（阈值比较、必填项检查）
#     - eval_common.py — 评测和性能脚本共享的样本解析工具
# ============================================================================
"""项目脚本包。

根目录只保留高频入口脚本，低频专项脚本按主题放入子包。
所有脚本均通过本包导入公共设施：from scripts.common import read_json_file

典型导入方式：
    from scripts.common import configure_utf8_stdio, run_command_step
    from scripts.gate_utils import add_max_failure
    from scripts.eval_common import EvalCaseRuntime, load_eval_items
"""
