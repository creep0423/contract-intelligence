"""展示品牌迁移隔离测试。"""

from pathlib import Path

from app import app
from qa_core.api.pages import _branded_page
from qa_core.config.settings import Settings, PROJECT_ROOT, get_settings
from qa_core.contracts.service import contract_dataset_id


def test_default_display_brand_and_openapi_metadata() -> None:
    settings = Settings(_env_file=None)
    assert settings.product_display_name == "知契"
    assert settings.product_browser_title == "知契 · 企业合同履约与风险智能"
    assert app.title == "Contract Intelligence API"
    assert "contract performance" in app.description


def test_display_brand_override_only_changes_presentation(monkeypatch) -> None:
    baseline = Settings(_env_file=None)
    monkeypatch.setenv("PRODUCT_DISPLAY_NAME", "测试展示品牌")
    monkeypatch.setenv("PRODUCT_ENGLISH_DESCRIPTOR", "Test Descriptor")
    monkeypatch.setenv("CONTRACT_PRODUCT_DESCRIPTOR", "Custom Contract Intelligence")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        response = _branded_page(str(PROJECT_ROOT / "static" / "contracts.html"))
        body = response.body.decode("utf-8")
        assert "测试展示品牌 · 企业合同履约与风险智能" in body
        assert "Custom Contract Intelligence" in body
        assert "Custom Custom Contract Intelligence" not in body
        assert "Test Descriptor" not in body  # 合同页使用合同产品描述。
        assert settings.app_name == "ContractIntelligence"
        assert settings.active_scenario_id == baseline.active_scenario_id
        assert settings.cache_key_prefix == baseline.cache_key_prefix
        assert settings.mysql_database == baseline.mysql_database
        assert contract_dataset_id("contract-123") == "contract:contract-123"

        admin_page = _branded_page(str(PROJECT_ROOT / "static" / "admin.html")).body.decode("utf-8")
        assert "测试展示品牌 · 运行状态" in admin_page
    finally:
        get_settings.cache_clear()


def test_brand_does_not_change_routes_or_benchmark_identifiers() -> None:
    routes = {route.path for route in app.routes}
    assert "/api/contracts/{contract_id}/stream" in routes
    assert "/api/openapi.json" in routes
    assert "/chat" not in routes
    assert "/api/scenarios" not in routes
    assert "/api/stream" not in routes
    assert all("知契" not in route for route in routes)

    manifest = (PROJECT_ROOT / "data_packs" / "contract_benchmark" / "manifest.json").read_text(encoding="utf-8")
    assert '"benchmark_version": "contract-benchmark-v1.0.0"' in manifest
    assert '"dataset_version": "small-synthetic-v1.0.0"' in manifest
    assert "知契" not in manifest


def test_primary_html_files_use_new_display_brand() -> None:
    for relative in ("static/contracts.html", "static/admin.html"):
        html = (Path(PROJECT_ROOT) / relative).read_text(encoding="utf-8")
        assert "知契 · 企业合同履约与风险智能" in html
        assert 'meta name="application-name" content="知契"' in html
