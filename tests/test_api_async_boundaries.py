"""FastAPI 路由同步与异步边界回归测试。

V1 的 MySQL、Redis、Milvus 和文件访问接口以同步客户端为主。普通 ``def``
路由由 FastAPI 在线程池中执行；只有真实等待 WebSocket 或线程桥接结果的路由
才应声明为 ``async def``。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "qa_core" / "api"
ASYNC_ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "websocket"}


def _is_route(function: ast.AsyncFunctionDef) -> bool:
    """判断异步函数是否由 FastAPI 路由装饰器注册。"""
    for decorator in function.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr in ASYNC_ROUTE_METHODS:
            return True
    return False


class ApiAsyncBoundaryTests(unittest.TestCase):
    """防止同步 I/O 被无 await 的 async 路由包裹。"""

    def test_async_routes_contain_real_await_boundary(self) -> None:
        """每个 V1 异步路由都必须包含真实的 await。"""
        violations: list[str] = []
        for path in sorted(API_DIR.glob("*.py")):
            if path.name == "v2.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(node, ast.AsyncFunctionDef) or not _is_route(node):
                    continue
                if not any(isinstance(child, ast.Await) for child in ast.walk(node)):
                    violations.append(f"{path.name}:{node.lineno} {node.name}")

        self.assertEqual(
            violations,
            [],
            "无 await 的 async 路由会阻塞事件循环，请改为 def 或接入真正的异步客户端："
            + ", ".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
