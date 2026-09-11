"""Contract Intelligence 页面与健康检查路由。不参与 RAG 检索和答案生成。

包含以下路由：
- GET /                — 合同工作台
- GET /admin           — 本地状态页
- GET /health          — 容器与本地健康检查

设计原则：
- 所有页面路由返回带有 Cache-Control: no-store 头部的静态文件，禁止浏览器缓存。
- 健康检查接口返回当前活跃场景、LLM 状态和检索预热状态。
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from qa_core.config.settings import get_settings
from qa_core.llm.client import llm_runtime_status
from qa_core.retrieval.factory import retrieval_warmup_state
from qa_core.scenarios.registry import resolve_scenario


router = APIRouter()

def _static_page(path: str):
    """返回一个禁止浏览器缓存的静态文件响应。

    使用 Cache-Control: no-store 确保每次请求都从服务器获取最新文件，
    避免浏览器缓存旧版 JS 导致前端功能异常。

    参数：
        path: 静态文件路径。

    返回：
        FileResponse 对象，包含 no-store 缓存控制头部。

    调用顺序：FastAPI 路由层 -> _static_page()。
    """
    return FileResponse(path, headers={"Cache-Control": "no-store"})


def _branded_page(path: str) -> HTMLResponse:
    """渲染静态页面中的展示品牌；只替换发布版默认文案，不生成任何工程标识。"""
    settings = get_settings()
    display_name = escape(settings.product_display_name)
    english_descriptor = escape(settings.product_english_descriptor)
    contract_descriptor = escape(settings.contract_product_descriptor)
    replacements = (
        ("知契 · 企业合同履约与风险智能", escape(settings.product_browser_title)),
        ("知契 · Enterprise Intelligence", f"{display_name} · {english_descriptor}"),
        ("知契 · 运行状态", f"{display_name} · 运行状态"),
        ("面向企业合同履约、义务、风险与原文证据分析的智能工作平台。", escape(settings.product_meta_description)),
        ('data-product-display-name="知契"', f'data-product-display-name="{display_name}"'),
        (
            'data-contract-product-descriptor="Contract Intelligence"',
            f'data-contract-product-descriptor="{contract_descriptor}"',
        ),
        ('content="知契"', f'content="{display_name}"'),
        ("Contract Intelligence", contract_descriptor),
        ("Enterprise Intelligence", english_descriptor),
        ("从原文证据，到业务判断。", escape(settings.product_tagline)),
        ("看清条款，掌握履约，提前发现风险。", escape(settings.contract_tagline)),
        (">知契<", f">{display_name}<"),
    )
    content = Path(path).read_text(encoding="utf-8")
    slots: list[tuple[str, str]] = []
    for index, (default, configured) in enumerate(replacements):
        marker = f"__DISPLAY_BRAND_SLOT_{index}__"
        content = content.replace(default, marker)
        slots.append((marker, configured))
    for marker, configured in slots:
        content = content.replace(marker, configured)
    return HTMLResponse(content, headers={"Cache-Control": "no-store"})

@router.get("/")
def read_root():
    """提供合同履约风控工作台，并禁止浏览器缓存旧版 JS。

    HTTP 路由 GET /

    返回：
        禁止缓存的 index.html 页面。

    调用顺序：FastAPI 路由层 -> read_root()。
    """
    return _branded_page("static/contracts.html")


@router.get("/contracts")
def read_contracts_page():
    return _branded_page("static/contracts.html")


@router.get("/admin")
def read_admin_page():
    """提供本地状态页。

    HTTP 路由 GET /admin

    返回：
        禁止缓存的 admin.html 页面。

    调用顺序：FastAPI 路由层 -> read_admin_page()。
    """
    return _branded_page("static/admin.html")

@router.get("/health")
def health_check():
    """容器与本地健康检查接口。

    HTTP 路由 GET /health

    返回当前系统状态：引擎类型、活跃场景、LLM 连通状态和检索预热状态。
    供容器编排平台（K8s/Docker）和负载均衡器定期探测。

    返回：
        - status: "healthy"。
        - engine: 检索引擎类型。
        - active_scenario_id: 当前活跃场景 ID。
        - active_scenario_name: 当前活跃场景名称。
        - llm: LLM 运行状态。
        - retrieval_warmup: 检索预热状态。

    调用顺序：FastAPI 路由层 -> health_check()。
    """
    scenario = resolve_scenario()
    settings = get_settings()
    return {
        "status": "healthy",
        "engine": "langchain_milvus_hybrid",
        "active_scenario_id": scenario.scenario_id,
        "active_scenario_name": settings.contract_product_descriptor,
        "llm": llm_runtime_status(),
        "retrieval_warmup": retrieval_warmup_state(),
    }
