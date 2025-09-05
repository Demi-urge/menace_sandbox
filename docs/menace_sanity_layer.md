# Menace Sanity Layer

The Sanity Layer captures billing anomalies detected across Menace services and
feeds them back into future generations of bots.

## Setup

1. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure the database router before recording anomalies:
   ```python
   from db_router import init_db_router
   init_db_router("ba", "local.db", "shared.db")
   ```
3. Components listening for anomalies can subscribe via
   `UnifiedEventBus`.

## Configuration

* `record_billing_anomaly` stores events in the `billing_anomalies`
  table and publishes them on the `"billing.anomaly"` topic.
* `record_payment_anomaly` logs contextual guidance to
  `MenaceMemoryManager`, allowing behaviour to be refined over time.
* Optional `GPT_MEMORY_MANAGER` and `DiscrepancyDB` integrations provide
  additional audit trails when available.
* `record_billing_event` captures general billing issues, storing a
  `DiscrepancyRecord` and logging feedback to `GPTMemoryManager`.
* `fetch_recent_billing_issues` returns recent feedback snippets that can be
  used to bias future prompts or code generation.
* The ``write_codex`` and ``export_training`` flags determine whether anomalies
  are exported as Codex samples or appended to the training dataset.
* Supplying ``config_path`` to :func:`record_billing_event` merges suggested
  configuration updates into the referenced JSON file.

## Usage

```python
from menace_sanity_layer import record_payment_anomaly, list_anomalies

record_payment_anomaly(
    "missing_charge",
    {"id": "ch_1", "amount": 5},
    "Avoid generating bots that make Stripe charges without proper logging or central routing.",
    write_codex=True,
    export_training=False,
)

print(list_anomalies(1))
```

The ``instruction`` field provides a short guideline describing how to prevent
similar issues in the future.  These snippets are stored in GPT memory and can
be retrieved via :func:`fetch_recent_billing_issues` to bias subsequent code
generation.

## Improving Future Generations

When the Stripe watchdog or other monitors detect a billing issue they emit an
anomaly through the Sanity Layer.  Each anomaly publishes an event and records a
short instruction in `MenaceMemoryManager` describing how to avoid the problem.
The accumulated feedback helps subsequent models steer clear of the same
mistakes, continuously improving billing reliability.
