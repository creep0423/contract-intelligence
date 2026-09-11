"""Pytest 运行时隔离配置。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_private_test_env() -> None:
    """只加载被 Git 忽略的 .env.test，不回退到共享运行数据库。"""
    path = PROJECT_ROOT / ".env.test"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_private_test_env()
MYSQL_TEST_ENABLED = os.getenv("CONTRACT_TEST_MYSQL_ENABLED", "").strip() == "1"
if MYSQL_TEST_ENABLED:
    database = os.getenv("TEST_MYSQL_DATABASE", "").strip()
    if not (database.endswith("_test") or database.startswith("test_")):
        raise RuntimeError("TEST_MYSQL_DATABASE 必须使用 *_test 或 test_* 独立数据库")
    required = {
        "MYSQL_HOST": "TEST_MYSQL_HOST",
        "MYSQL_PORT": "TEST_MYSQL_PORT",
        "MYSQL_USER": "TEST_MYSQL_USER",
        "MYSQL_PASSWORD": "TEST_MYSQL_PASSWORD",
        "MYSQL_DATABASE": "TEST_MYSQL_DATABASE",
    }
    for target, source in required.items():
        value = os.getenv(source, "").strip()
        if not value:
            raise RuntimeError(f"启用 MySQL 集成测试时必须配置 {source}")
        os.environ[target] = value
    os.environ["APP_ENV"] = "test"

from qa_core.config.settings import get_settings


os.environ["CACHE_ENABLED"] = "false"
get_settings.cache_clear()


def pytest_collection_modifyitems(config, items) -> None:
    if MYSQL_TEST_ENABLED:
        return
    marker = pytest.mark.skip(reason="需要显式启用隔离 MySQL：CONTRACT_TEST_MYSQL_ENABLED=1")
    for item in items:
        if "mysql_integration" in item.keywords:
            item.add_marker(marker)
