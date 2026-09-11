"""合同领域可预期错误，供 HTTP 与 WebSocket 统一映射。"""


class ContractNotFoundError(ValueError):
    """资源不存在或当前数据域无权访问；统一隐藏资源是否真实存在。"""


class ContractConflictError(ValueError):
    """当前资源状态与请求冲突，例如同一合同已有运行中的分析。"""
