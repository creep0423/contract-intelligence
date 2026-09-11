"""
V1 企业级缓存真实链路验收脚本。

该脚本验证 Redis 语义缓存的完整闭环：
    1. 服务健康检查 —— 确认 FastAPI 服务存活
    2. 缓存状态检查 —— 确认 Redis 连接和缓存启用状态
    3. 缓存失效（可选） —— 通过 POST /api/admin/cache/invalidate 主动清缓存
    4. 首次问答（cold） —— 预期产生 cache miss，验证从头检索并缓存
    5. 二次问答（warm） —— 同一 session 同一 query 预期命中缓存（cache hit）
    6. 缓存命中验证 —— 对比两次问答前后的 retrieval_hits/retrieval_misses

验收标准：
    - service_healthy: 服务状态为 healthy
    - cache_enabled: 缓存已启用
    - redis_ok: Redis 连接成功
    - first_run_finished: 首次问答正常结束（end 事件）
    - second_run_finished: 二次问答正常结束（end 事件）
    - first_run_has_cache_miss: 首次问答出现 miss 计数增长
    - second_run_has_cache_hit: 二次问答出现 hit 计数增长
    - namespace_visible: 缓存 namespace 列表可见

用法：
    python scripts/cache_acceptance_smoke.py
    python scripts/cache_acceptance_smoke.py --no-invalidate-first --query "IT 设备采购流程"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

import websocket

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qa_core.config.settings import get_settings
from scripts.common import configure_utf8_stdio, write_optional_json


def _headers(token: str) -> dict[str, str]:
    """构造请求头：JSON Content-Type + 可选管理令牌。

    调用顺序：命令行入口 -> _headers()。
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Admin-Token"] = token
    return headers


def resolve_admin_token(raw_token: str | None) -> str:
    """解析管理接口令牌，不把令牌写入报告。

    优先使用命令行传入的 --admin-token，未传入时从运行配置中读取 ADMIN_API_TOKEN。
    返回的报告不会展示令牌本身，避免敏感信息进入日志。

    调用顺序：命令行入口 -> resolve_admin_token()。
    """
    return (raw_token or get_settings().admin_api_token or "").strip()


def fetch_json(base_url: str, path: str, token: str = "") -> dict:
    """读取 JSON 接口并返回解析后的 dict。

    超时 20 秒，适用于缓存状态查询等管理接口。

    调用顺序：命令行入口 -> fetch_json()。
    """
    req = urlrequest.Request(base_url.rstrip("/") + path, headers=_headers(token))
    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"HTTP check failed: {path}: {exc}") from exc


def post_json(base_url: str, path: str, payload: dict, token: str = "") -> dict:
    """提交 JSON 请求并返回解析后的响应 dict。

    用于缓存失效等需要 POST 的管理接口。

    调用顺序：命令行入口 -> post_json()。
    """
    req = urlrequest.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_headers(token),
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"HTTP post failed: {path}: {exc}") from exc


def websocket_events(
    *,
    base_url: str,
    token: str,
    query: str,
    scenario_id: str,
    session_id: str,
    max_events: int,
    max_seconds: float,
) -> list[dict]:
    """通过真实 WebSocket 问答入口收集流式事件。

    连接 /api/stream WebSocket，发送问题后循环接收 SSE 事件直到：
    - 收到 "end" 事件（正常结束）
    - 收到 "error" 事件（抛出 RuntimeError）
    - 超过 max_seconds 超时或超过 max_events 数量上限

    返回收集到的完整事件列表，供后续分析 cache hit/miss。

    调用顺序：命令行入口 -> websocket_events()。
    """
    parsed = urlparse(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{scheme}://{parsed.netloc}/api/stream"
    headers = [f"X-Admin-Token: {token}"] if token else []
    ws = websocket.create_connection(ws_url, timeout=90, header=headers)
    try:
        ws.send(
            json.dumps(
                {
                    "query": query,
                    "session_id": session_id,
                    "scenario_id": scenario_id,
                },
                ensure_ascii=False,
            )
        )
        events: list[dict] = []
        deadline = time.monotonic() + max_seconds
        while len(events) < max_events and time.monotonic() < deadline:
            event = json.loads(ws.recv())
            events.append(event)
            if event.get("type") == "error":
                raise RuntimeError(f"WebSocket returned error: {event.get('error')}")
            if event.get("type") == "end":
                return events
        raise RuntimeError(f"WebSocket did not finish, received {len(events)} events")
    finally:
        ws.close()


def end_cache(events: list[dict]) -> dict:
    """从 WebSocket 事件列表中提取 end 事件的 retrieval.cache 字段。

    cache 字段包含 hit_count、miss_count 等命中统计，是验证缓存闭环的核心指标。

    调用顺序：命令行入口 -> end_cache()。
    """
    end = next((event for event in reversed(events) if event.get("type") == "end"), {})
    return ((end.get("retrieval") or {}).get("cache") or {}) if end else {}


def run_check(args: argparse.Namespace) -> dict:
    """执行完整的缓存验收流程。

    流程：
        1. 健康检查 + 缓存初始状态快照（before）
        2. 可选：主动失效缓存（--invalidate-first）
        3. 首次问答（cold run）→ 预期产生 cache miss
        4. 缓存中间状态快照（middle）
        5. 二次问答（warm run）→ 预期命中 cache hit
        6. 缓存最终状态快照（after）
        7. 对比 before/middle/after 统计，验证 miss 增长 + hit 增长

    验收逻辑：
        - first_run_has_cache_miss：首次问答后 miss 计数 > 0 或 middle > before
        - second_run_has_cache_hit：二次问答后 hit 计数 > 0 或 after > middle

    调用顺序：命令行入口 -> run_check()。
    """
    admin_token = resolve_admin_token(args.admin_token)
    session_id = f"cache-smoke-{int(time.time())}"

    # 1. 初始状态：健康 + 缓存 before 快照
    health = fetch_json(args.base_url, "/health")
    before = fetch_json(args.base_url, f"/api/admin/cache/status?scenario_id={args.scenario}", admin_token)
    invalidation = {}
    if args.invalidate_first:
        # 主动失效缓存，确保首次问答走真实检索路径
        invalidation = post_json(args.base_url, "/api/admin/cache/invalidate", {"scenario_id": args.scenario}, admin_token)

    # 2. 首次问答（cold run）
    first_events = websocket_events(
        base_url=args.base_url,
        token=admin_token,
        query=args.query,
        scenario_id=args.scenario,
        session_id=session_id,
        max_events=args.max_events,
        max_seconds=args.max_seconds,
    )
    first_cache = end_cache(first_events)
    middle = fetch_json(args.base_url, f"/api/admin/cache/status?scenario_id={args.scenario}", admin_token)

    # 3. 二次问答（warm run，同一 session 同一 query 应命中缓存）
    second_events = websocket_events(
        base_url=args.base_url,
        token=admin_token,
        query=args.query,
        scenario_id=args.scenario,
        session_id=session_id,
        max_events=args.max_events,
        max_seconds=args.max_seconds,
    )
    second_cache = end_cache(second_events)
    after = fetch_json(args.base_url, f"/api/admin/cache/status?scenario_id={args.scenario}", admin_token)

    # 4. 对比统计验证
    before_stats = before.get("stats") or {}
    middle_stats = middle.get("stats") or {}
    after_stats = after.get("stats") or {}
    checks = {
        "service_healthy": health.get("status") == "healthy",
        "cache_enabled": bool(after.get("enabled")),
        "redis_ok": bool((after.get("redis") or {}).get("ok")),
        "first_run_finished": bool(first_events and first_events[-1].get("type") == "end"),
        "second_run_finished": bool(second_events and second_events[-1].get("type") == "end"),
        # 首次问答后 miss 计数应有增长（从 cold cache 检索）
        "first_run_has_cache_miss": int(first_cache.get("miss_count") or 0) > 0
        or int(middle_stats.get("retrieval_misses") or 0) > int(before_stats.get("retrieval_misses") or 0),
        # 二次问答后 hit 计数应有增长（命中 warm cache）
        "second_run_has_cache_hit": int(second_cache.get("hit_count") or 0) > 0
        or int(after_stats.get("retrieval_hits") or 0) > int(middle_stats.get("retrieval_hits") or 0),
        "namespace_visible": bool(after.get("namespaces")),
    }

    return {
        "report_type": "cache_acceptance_smoke",
        "ok": all(checks.values()),
        "base_url": args.base_url,
        "scenario_id": args.scenario,
        "query": args.query,
        "checks": checks,
        "invalidation": invalidation,
        "first_cache": first_cache,
        "second_cache": second_cache,
        "stats_before": before_stats,
        "stats_after": after_stats,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Run V1 cache acceptance checks against the live API service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--admin-token", default="", help="为空时自动读取当前运行配置中的 ADMIN_API_TOKEN。")
    parser.add_argument("--scenario", default="enterprise_knowledge")
    parser.add_argument("--query", default="新人入职流程怎么走")
    parser.add_argument("--invalidate-first", action=argparse.BooleanOptionalAction, default=True,
                        help="首次问答前先主动失效缓存，确保触发真实检索路径。")
    parser.add_argument("--max-events", type=int, default=1000, help="WebSocket 最多接收的事件数（防无限循环）。")
    parser.add_argument("--max-seconds", type=float, default=180.0, help="WebSocket 最多等待秒数（超时保护）。")
    parser.add_argument("--output", default="", help="可选 JSON 报告输出路径。")
    return parser


def main() -> None:
    """CLI 入口：执行 V1 缓存验收并输出 JSON 报告。

    调用顺序：命令行入口 -> main()。
    """
    configure_utf8_stdio()
    args = build_parser().parse_args()
    payload = run_check(args)
    write_optional_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
