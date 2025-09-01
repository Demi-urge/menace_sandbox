# PromptDB dataset export

The `tools/prompt_dataset_cli.py` helper converts logged prompts and responses
into flat files suitable for fine‑tuning.

## Usage

```bash
python -m tools.prompt_dataset_cli output.jsonl --tag helpful --min-confidence 0.7
python -m tools.prompt_dataset_cli dataset.csv --format csv
```

* `--tag` may be repeated to require at least one matching tag.
* `--min-confidence` filters by the `vector_confidence` column.
* `--db` defaults to the `PROMPT_DB_PATH` environment variable.

Each exported record has `prompt` and `completion` fields.

## Feeding the dataset into fine‑tuning

### vLLM

Exported JSONL files can be passed directly to vLLM's LoRA trainer.

```bash
pip install vllm  # if not already installed
python -m vllm.trainers.lora_finetune \
  --model facebook/opt-125m \
  --train-dataset output.jsonl \
  --save-dir ./ft-model
```

### Ollama

Ollama accepts the same JSONL structure when training adapters.

```bash
ollama create mymodel --from llama2
ollama run mymodel --train output.jsonl
```

The resulting fine‑tuned model can then be served with the respective
framework.
