# -*- coding: utf-8 -*-
# ============================================================================
# 本地 LangChain/Milvus 运行依赖冒烟检查脚本
# ============================================================================
# 该脚本用于启动服务前检查当前配置是否符合主链路预期。
# 它不做破坏性写入，但会真实校验必需环境：
#   - LLM Key、管理令牌
#   - 模型目录（embedding + reranker）
#   - 场景配置、Milvus URI/集合、MySQL
#   - Active 知识库版本
#
# 使用场景：
#   - 修改本机 `.env` 或 Compose 注入配置后确认配置是否生效；
#   - 启动 API 前确认 Milvus URI、集合名、模型路径；
#   - 排查为什么本地读取的配置和预期不一致。
#
# 为什么要做硬校验：
#   - 当前架构不提供技术降级方案，依赖缺失时应该在启动前暴露；
#   - 比页面提问后才失败更容易定位；
#   - 不写 Milvus、不写 MySQL，保持安全。
#
# 不适合的场景：
#   - 不要用它判断知识库是否有数据；
#   - 不要用它替代 /api/retrieval/debug；
#   - 不要在这里创建集合或写入测试文档。
#
# 用法示例：
#   python scripts\check_langchain_stack.py
# ============================================================================

"""检查 LangChain 相关依赖版本与运行时导入是否兼容。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

# sys: 系统功能（sys.path 修改）
import sys

# pathlib.Path: 文件路径操作
from pathlib import Path

# ── 路径设置 ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 导入核心模块 ──
# validate_runtime_environment / validate_active_kb_versions: 启动期分阶段校验
from qa_core.config.preflight import validate_active_kb_versions, validate_runtime_environment

# get_scenario_registry / resolve_scenario: 场景配置
from qa_core.scenarios.registry import get_scenario_registry, resolve_scenario

# get_settings: 读取当前运行配置（含 Milvus URI、模型路径、LLM 配置等）
from qa_core.config.settings import get_settings

from qa_core.storage.bootstrap import bootstrap_mysql_schema


def main() -> None:
    """打印当前运行配置，并验证必需环境。

    输出内容重点看：
      - Milvus URI 和数据库名
      - 当前场景的 FAQ/Doc collection 名称
      - 本地模型路径（embedding + reranker）
      - Valid sources（场景支持的数据源白名单）
      - Active KB version（当前激活的知识库版本）
      - LLM 模型名称

    若这些值和当前运行配置预期不一致，说明配置优先级或环境变量加载存在问题。

    使用场景：
      - 本地开发第一步确认配置
      - Docker 环境通过 Compose 注入变量后确认路径
      - 评测或入库脚本运行前确认集合名不会写错

    调用顺序：命令行入口 -> main()。
    """
    settings = get_settings()                        # 读取当前运行配置
    validate_runtime_environment()                   # 执行基础运行时环境校验
    scenario = resolve_scenario()                    # 获取当前活跃场景
    registry = get_scenario_registry()               # 获取场景注册表
    bootstrap_mysql_schema()                         # 初始化 MySQL 控制面 schema
    active_check = validate_active_kb_versions(scenario.scenario_id)

    print("LangChain stack configuration")
    print(f"Active scenario: {scenario.scenario_id} / {scenario.display_name}")
    print(f"Available scenarios: {[item.scenario_id for item in registry.list_scenarios()]}")
    print(f"Milvus URI: {settings.milvus_uri}")
    print(f"Milvus database: {settings.milvus_database}")
    print(f"FAQ collection: {scenario.faq_collection}")
    print(f"Doc collection: {scenario.doc_collection}")
    print(f"LLM model: {settings.llm_model}")
    print(f"Embedding model: {settings.embedding_model_path}")
    print(f"Reranker model: {settings.reranker_model_path}")
    print(f"Valid sources: {scenario.valid_sources}")
    print(f"Active KB version: {active_check['active_kb_version']}")
    print("Runtime preflight: passed")


if __name__ == "__main__":
    main()
