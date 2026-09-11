"""合同履约风控 Benchmark：版本化 Gold、确定性评分、报告和回归门禁。"""

from qa_core.evaluation.loader import load_benchmark
from qa_core.evaluation.metrics import evaluate_benchmark

__all__ = ["evaluate_benchmark", "load_benchmark"]
