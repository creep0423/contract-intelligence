# -*- coding: utf-8 -*-
# ============================================================================
# 知识库版本管理命令行工具
# ============================================================================
# 该脚本只操作 MySQL 中的知识库版本控制面，不直接访问 Milvus。
# 它用于本地查看、创建、回滚和归档知识库版本。
#
# 常用命令：
#   # 查看所有版本
#   python scripts/manage_kb_versions.py list
#
#   # 创建新版本
#   python scripts/manage_kb_versions.py create --description "2026-05-06 全量重建"
#
#   # 回滚到曾经发布过的历史版本（新版本发布请走 rebuild_kb_version.py）
#   python scripts/manage_kb_versions.py activate kb_20260506_103000_xxxxxxxx
#
#   # 归档版本（不能归档当前 active）
#   python scripts/manage_kb_versions.py archive kb_20260430_090000_xxxxxxxx
#
#   # 指定场景（默认为 ACTIVE_SCENARIO_ID）
#   python scripts/manage_kb_versions.py --scenario equipment_ops list
#
# 为什么独立成脚本：
#   - 版本切换是运维动作，不应该混在在线问答请求里；
#   - 入库脚本（rebuild_kb_version.py）负责创建版本、质量门禁和发布激活；
#   - 手工回滚时需要一个轻量入口；
#   - 不直接删除 Milvus 数据，避免误删可回滚版本。
# ============================================================================

"""提供知识库版本查询、激活、归档与回滚命令。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

# argparse: 命令行参数解析（支持子命令 list/create/activate/archive；activate 子命令只做回滚）
import argparse

# json: 标准 JSON 序列化
import json

# sys: 系统功能（sys.path 修改）
import sys

# pathlib.Path: 文件路径操作
from pathlib import Path

# ── 路径设置 ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 导入核心模块 ──
# get_kb_version_store: 获取指定场景的知识库版本 Store
#   提供 ensure_version / activate_version / archive_version / as_payload 方法
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.storage.bootstrap import bootstrap_mysql_schema


def print_json(payload) -> None:
    """以中文可读的 JSON 格式输出脚本结果。

    ensure_ascii=False 保证中文字符正常显示；
    indent=2 提供可读的缩进。

    调用顺序：命令行入口 -> print_json()。
    """
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    """解析命令并执行版本管理操作。

    支持四个子命令：
      - list:     查看所有版本（含 active_version 指针和完整版本列表）
      - create:   创建新的 staged 版本
      - activate: 回滚到曾经发布过的历史版本（前一个 active 自动回退为 staged）
      - archive:  归档指定版本（不能是 active）

    子命令的参数说明：
      - create: --kb-version（可选，不传则自动生成）、--description
      - activate: kb_version（位置参数，必传，目标版本必须曾经被激活过）
      - archive: kb_version（位置参数，必传）

    调用顺序：命令行入口 -> main()。
    """
    parser = argparse.ArgumentParser(description="Manage multi-scenario RAG knowledge base versions.")
    parser.add_argument("--scenario", default=None, help="Business scenario id. Defaults to ACTIVE_SCENARIO_ID.")

    # ── 子命令：list ──
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List knowledge base versions.")

    # ── 子命令：create ──
    create_parser = subparsers.add_parser("create", help="Create a staged knowledge base version.")
    create_parser.add_argument("--kb-version", default=None, help="Optional explicit version id.")
    create_parser.add_argument("--description", default="", help="Human readable description.")

    # ── 子命令：activate ──
    activate_parser = subparsers.add_parser("activate", help="Rollback to a previously active knowledge base version.")
    activate_parser.add_argument("kb_version")

    # ── 子命令：archive ──
    archive_parser = subparsers.add_parser("archive", help="Archive a non-active knowledge base version.")
    archive_parser.add_argument("kb_version")

    args = parser.parse_args()

    # ── 获取版本 Store ──
    bootstrap_mysql_schema()
    store = get_kb_version_store(args.scenario)

    # ── 根据子命令执行对应操作 ──
    if args.command == "list":
        # as_payload() 返回当前场景的完整版本状态（active_version + versions 列表 + metadata_store）
        print_json(store.as_payload())
        return

    if args.command == "create":
        # ensure_version() 创建或复用版本
        # create_new=True: 未指定 kb_version 时自动生成新版本
        version = store.ensure_version(
            args.kb_version,
            create_new=not bool(args.kb_version),
            description=args.description,
            created_by="manage_kb_versions",
        )
        print_json({"status": "success", "version": version.as_dict()})
        return

    if args.command == "activate":
        record = store.get(args.kb_version)
        if record is None:
            raise SystemExit(f"知识库版本不存在：{args.kb_version}")
        if record.status == "ARCHIVED":
            raise SystemExit("归档版本不能直接激活")
        if not record.activated_at:
            raise SystemExit("新版本发布必须通过 scripts/rebuild_kb_version.py --quality-gate --activate；manage_kb_versions.py activate 只用于回滚已发布版本")
        # activate_version() 回滚到曾经发布过的历史版本：
        #   1. 前一个 active 版本自动回退为 staged
        #   2. 目标版本变为 active
        #   3. 数据库事务保证一致性
        version = store.activate_version(args.kb_version)
        print_json({"status": "success", "version": version.as_dict()})
        return

    if args.command == "archive":
        # archive_version() 归档非 active 版本：
        #   1. 校验目标版本不是当前 active（否则抛异常）
        #   2. 将版本状态改为 archived
        #   3. 归档后的版本不可再激活
        version = store.archive_version(args.kb_version)
        print_json({"status": "success", "version": version.as_dict()})
        return


if __name__ == "__main__":
    # 当脚本直接运行时（python manage_kb_versions.py），__name__ == "__main__"
    # 当作为模块导入时，main() 不被调用
    main()
