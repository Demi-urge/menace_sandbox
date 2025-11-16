"""Fine-tune a small language model on code diff summaries.

The training data is expected to be a JSONL file where each line contains
``before_code``, ``after_code`` and ``summary`` fields as produced by
:mod:`micro_models.diff_summarizer_dataset`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except Exception:  # pragma: no cover - missing heavy deps
    load_dataset = AutoModelForCausalLM = AutoTokenizer = None  # type: ignore
    DataCollatorForLanguageModeling = Trainer = TrainingArguments = None  # type: ignore

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _prep_dataset(tokenizer, path: Path):
    ds = load_dataset("json", data_files=str(path))  # type: ignore[arg-type]

    def _format(example):
        prompt = (
            "Summarize the code change.\nBefore:\n" + example["before_code"] + "\nAfter:\n" + example["after_code"] + "\nSummary:"
        )
        input_ids = tokenizer(prompt).input_ids  # type: ignore[union-attr]
        labels = tokenizer(example["summary"]).input_ids  # type: ignore[union-attr]
        return {"input_ids": input_ids, "labels": labels}

    return ds["train"].map(_format, remove_columns=ds["train"].column_names)


def train(dataset_path: Path, output_dir: Path, model_name: str = DEFAULT_MODEL) -> None:
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers not installed")
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[union-attr]
    model = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[union-attr]
    tokenized = _prep_dataset(tokenizer, dataset_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # type: ignore[call-arg]
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)  # type: ignore[union-attr]


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Fine-tune diff summariser")
    parser.add_argument("dataset", type=Path, help="Path to JSONL dataset")
    parser.add_argument("output", type=Path, help="Directory to save the model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name")
    args = parser.parse_args()
    train(args.dataset, args.output, args.model)


if __name__ == "__main__":  # pragma: no cover
    main()
