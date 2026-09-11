"""验证启动前置校验对配置和运行依赖的约束。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from qa_core.config.preflight import _is_placeholder


def test_is_placeholder_rejects_empty_and_sample_values() -> None:
    """验证空值和示例值被识别为占位符予以拒绝。

    调用顺序：pytest/unittest 测试入口 -> test_is_placeholder_rejects_empty_and_sample_values()。
    """
    assert _is_placeholder("")
    assert _is_placeholder("replace-with-real-key")
    assert _is_placeholder("请替换为真实可用的模型服务 Key")
    assert _is_placeholder("请替换为随机长令牌")


def test_is_placeholder_accepts_realistic_values() -> None:
    """验证真实配置值不会被误判为占位符。

    调用顺序：pytest/unittest 测试入口 -> test_is_placeholder_accepts_realistic_values()。
    """
    assert not _is_placeholder("sk-prod-abc123456789")
    assert not _is_placeholder("admin-token-abc123456789")
