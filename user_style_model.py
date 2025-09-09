from __future__ import annotations

"""Train a lightweight language model mirroring the user's style."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TextDataset,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TextDataset = None  # type: ignore
    DataCollatorForLanguageModeling = None  # type: ignore
    TrainingArguments = None  # type: ignore
    Trainer = None  # type: ignore

from .mirror_bot import MirrorDB
from local_model_wrapper import LocalModelWrapper

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback stubs
    ContextBuilder = Any  # type: ignore


@dataclass
class StyleModelConfig:
    """Configuration for fine tuning."""

    base_model: str = "distilgpt2"
    epochs: int = 1
    batch_size: int = 2


class UserStyleModel:
    """Fine tune a text generation model on the conversation log."""

    def __init__(self, db: MirrorDB | None = None, config: StyleModelConfig | None = None) -> None:
        self.db = db or MirrorDB()
        self.config = config or StyleModelConfig()
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    def _prepare_dataset(self, path: Path) -> tuple | None:
        records = self.db.fetch(200)
        if not records or AutoTokenizer is None or TextDataset is None:
            return None
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(f"{rec.user}\n{rec.response}\n")
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        dataset = TextDataset(tokenizer=tokenizer, file_path=str(path), block_size=64)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        return tokenizer, dataset, collator

    def train(self) -> None:
        if AutoModelForCausalLM is None or Trainer is None:
            raise RuntimeError("transformers library required")
        tmp = Path("style_data.txt")
        prepared = self._prepare_dataset(tmp)
        if prepared is None:
            return
        tokenizer, dataset, collator = prepared
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
        args = TrainingArguments(
            output_dir="style_model",
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            logging_steps=10,
            save_steps=0,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
        )
        trainer.train()
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, text: str, *, context_builder: ContextBuilder) -> str:
        """Generate text in the user's style with contextual snippets."""

        if not self.model or not self.tokenizer:
            return text

        wrapper = LocalModelWrapper(self.model, self.tokenizer)
        return wrapper.generate(
            text,
            context_builder=context_builder,
            cb_input=text,
            max_length=50,
            num_return_sequences=1,
        )


__all__ = ["StyleModelConfig", "UserStyleModel"]
