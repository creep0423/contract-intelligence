# -*- coding: utf-8 -*-
# ============================================================================
# Sub-package: enterprise_overlay — 企业仿真资料增强相关脚本
# ============================================================================
# 这些脚本用于企业仿真资料增强的完整流程：
#
#   1. 分析：识别企业资料的真实度和脏数据风险
#      - analyze_enterprise_data_realism.py   — 资料真实度评估
#      - analyze_dirty_enterprise_samples.py  — 脏样本治理风险分析
#
#   2. 预检：构建 clean overlay 并验证能否通过入库质量门禁
#      - build_enterprise_overlay_dataset.py  — 合并基础资料 + clean overlay
#      - check_enterprise_overlay_readiness.py — 就绪门禁（真实度 + dirty + 回归覆盖）
#
#   3. 上线：生成并执行激活计划
#      - plan_enterprise_overlay_activation.py  — 生成标准 rebuild 命令
#      - run_enterprise_overlay_activation.py   — 在真实环境中执行激活
#
# 整个流程保证：
#   - 基础场景保持稳定（不修改 scenarios/ 目录）
#   - 企业增强资料先做离线合并和质量门禁验证
#   - dirty_samples 只用于治理演示，绝不能进入 active 知识库
#   - 真正上线需要经过知识库版本重建 → 质量门禁 → 评测门禁三道关
# ============================================================================
"""企业仿真资料增强相关脚本。"""
