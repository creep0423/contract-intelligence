"""Fine-tune the local BERT intent classifier artifact.

训练工作流：加载 JSONL 训练/评测样本并校验标签分布 -> 用
format_intent_model_input 格式化输入（查询 + 是否含历史标记）-> 在
本地 bert-base-chinese 上微调三分类头（retrieval / non_retrieval / chitchat）
-> 每个 epoch 结束在评测集上计算准确率 -> 保存模型、tokenizer 与 labels.json
治理报告，并把训练/评测数据集 SHA256 指纹写入模型元数据用于审计复现。

设计决策：
- 模型只从本地目录加载（local_files_only=True），训练和推理均不联网；
- 模型产物由意图治理流程（write_intent_model_report）登记，激活与回滚必须人工审批。

Example:
    python scripts/intent/train_intent_bert.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from qa_core.config.settings import MODEL_ROOT
from qa_core.config.settings import PROJECT_ROOT as SETTINGS_PROJECT_ROOT
from qa_core.intent.model_classifier import (
    DEFAULT_EVAL_DATASET,
    DEFAULT_TRAINING_DATASET,
    LABELS_FILENAME,
    RETRIEVAL_INTENTS,
    IntentTrainingExample,
    format_intent_model_input,
    load_intent_examples,
    validate_intent_dataset_split,
)
from qa_core.intent.governance import write_intent_model_report


class IntentDataset(Dataset):
    """PyTorch Dataset 封装：将意图训练样本转为 BERT 模型输入。

    每个样本的输入文本通过 format_intent_model_input 格式化（查询 + 是否含历史标记），
    标签为 RETRIEVAL_INTENTS 中的意图类别索引（retrieval / non_retrieval / chitchat）。

    Attributes:
        examples: 意图训练样本元组
        tokenizer: HuggingFace BERT tokenizer
        max_length: token 最大长度（超长截断，不足填充）
        label2id: 意图标签 -> 整数索引的映射
    """

    def __init__(self, examples: tuple[IntentTrainingExample, ...], tokenizer, *, max_length: int) -> None:
        """初始化数据集：保存样本与 tokenizer，并构建标签到索引的映射。

        参数：
            examples: 意图训练样本元组（来自 load_intent_examples）。
            tokenizer: HuggingFace BERT tokenizer。
            max_length: token 最大长度（超长截断，不足填充）。

        调用顺序：main() -> IntentDataset()。
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 构建标签到索引的映射：retrieval→0, non_retrieval→1, chitchat→2
        # 原因：label2id 与 main() 里传给模型的 id2label 必须使用同一枚举顺序，
        # 否则训练和推理对类别索引的理解会不一致
        self.label2id = {label: index for index, label in enumerate(RETRIEVAL_INTENTS)}

    def __len__(self) -> int:
        """返回数据集样本总数。

        返回：
            训练样本数量（DataLoader 依赖该长度划分 batch）。

        调用顺序：DataLoader -> IntentDataset.__len__()。
        """
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """返回第 index 条样本的 tokenized 输入和标签 tensor。

        参数：
            index: 样本在 examples 中的下标。

        返回：
            包含 input_ids / attention_mask / labels 的 tensor 字典，
            其中 input_ids 与 attention_mask 已去掉 tokenizer 默认的 batch 维度。

        调用顺序：DataLoader -> IntentDataset.__getitem__()。
        """
        example = self.examples[index]
        # 格式化输入文本：查询 + 是否含对话历史
        encoded = self.tokenizer(
            format_intent_model_input(example.query, has_history=example.has_history),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # squeeze(0) 去掉 batch 维度（tokenizer 默认返回 [1, seq_len]）
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        # 标签为意图类别的整数索引
        item["labels"] = torch.tensor(self.label2id[example.label], dtype=torch.long)
        return item


def main() -> None:
    """执行当前脚本的完整命令行流程。

    返回：
        None；训练产物写入 --output 目录，治理报告由 write_intent_model_report() 落盘。

    异常：
        RuntimeError: --base-model 目录不存在时抛出。

    调用顺序：业务模块或命令行入口 -> main()。
    """
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Contract Intelligence retrieval intent classification.")
    parser.add_argument("--base-model", default=str(MODEL_ROOT / "bert-base-chinese"))
    parser.add_argument("--output", default=str(MODEL_ROOT / "bert_intent_classifier_v1"))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAINING_DATASET), help="JSONL training dataset path.")
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL_DATASET), help="JSONL evaluation dataset path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = Path(args.base_model)
    output = Path(args.output)
    train_data = _resolve_project_path(args.train_data)
    eval_data = _resolve_project_path(args.eval_data)
    if not base_model.exists():
        raise RuntimeError(f"base model path does not exist: {base_model}")
    training_examples = load_intent_examples(train_data)
    eval_examples = load_intent_examples(eval_data)
    validate_intent_dataset_split(training_examples, eval_examples)

    # 原因：label2id/id2label 与 IntentDataset 使用同一枚举顺序，
    # 保证保存的模型与 qa_core 加载方对类别索引的理解一致
    label2id = {label: index for index, label in enumerate(RETRIEVAL_INTENTS)}
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model),
        num_labels=len(RETRIEVAL_INTENTS),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    ).to(device)

    train_loader = DataLoader(
        IntentDataset(training_examples, tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        IntentDataset(eval_examples, tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        # 原因：train() 开启 dropout 等训练专用行为，与后面 _evaluate 的 eval 模式形成明确边界
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # 原因：batch 必须整体搬到训练设备（CUDA/CPU），否则混合设备运算直接报错
            batch = {key: value.to(device) for key, value in batch.items()}
            # 原因：PyTorch 默认累积梯度，不清零会让多个 batch 的梯度串扰
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            # 原因：loss 是计算图内的 CUDA tensor，先 detach 脱离图再搬回 CPU，
            # 才能安全转成 float 并累计平均
            total_loss += float(loss.detach().cpu())
        # 原因：每个 epoch 结束立即在评测集上打分，训练过程可观察准确率与过拟合趋势
        accuracy = _evaluate(model, eval_loader, device)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": round(total_loss / max(len(train_loader), 1), 4),
                    "eval_accuracy": round(accuracy, 4),
                },
                ensure_ascii=False,
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    base_model_label = _display_path(base_model)
    (output / LABELS_FILENAME).write_text(
        json.dumps(
            {
                "model_version": "bert-intent-v1",
                "base_model": base_model_label,
                "labels": list(RETRIEVAL_INTENTS),
                "max_length": args.max_length,
                "training_examples": len(training_examples),
                "eval_examples": len(eval_examples),
                "training_dataset": _display_path(train_data),
                "eval_dataset": _display_path(eval_data),
                "training_dataset_sha256": _sha256(train_data),
                "eval_dataset_sha256": _sha256(eval_data),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved intent BERT model to {output}")
    print(f"saved intent model governance report to {write_intent_model_report()}")


def _display_path(path: Path) -> str:
    """将模型路径转为相对于项目根目录的显示路径。

    如果路径不在项目根目录下，则返回绝对路径的 POSIX 格式。
    用于生成 labels.json 中的人类可读路径。

    调用顺序：main() -> _display_path()。
    """
    try:
        return path.resolve().relative_to(SETTINGS_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_project_path(raw_path: str) -> Path:
    """让相对数据集路径始终相对于项目根目录解析。"""

    path = Path(raw_path)
    return path if path.is_absolute() else SETTINGS_PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    """返回训练数据资产指纹，写入模型元数据用于审计复现。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate(model, eval_loader: DataLoader, device: str) -> float:
    """在 eval_loader 上计算模型准确率。

    不计算 loss，只比较预测标签与真实标签。
    返回 0.0 ~ 1.0 之间的浮点数准确率。

    调用顺序：main() -> _evaluate()。
    """
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in eval_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predicted = torch.argmax(logits, dim=-1)
            total += int(labels.numel())
            correct += int((predicted == labels).sum().item())
    return correct / total if total else 0.0


if __name__ == "__main__":
    main()
