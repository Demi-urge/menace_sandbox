# Models Repository Workflow

`ImplementationPipeline` and `BotDevelopmentBot` share a single git repository where all models are developed. The path defaults to `models_repo` next to the main codebase and is created with `ensure_models_repo()` when missing. Code generation runs through the local ``SelfCodingEngine`` so no external API keys are required.

## Cloning finished models

When `ImplementationPipeline.run()` is called with a `model_id` parameter the workflow clones the completed repository into a sibling directory named after that id:

```python
from menace.implementation_pipeline import ImplementationPipeline

pipeline = ImplementationPipeline()
result = pipeline.run(tasks, model_id=123)
```

This creates a new repository `123/` alongside the main models directory containing an exact copy of the code that was just built and tested.

## Safe edits while a model is active

Concurrency is managed via the `.active_model` marker file inside the models repository. The helper :func:`model_build_lock` creates and removes this marker automatically::

    from menace.models_repo import model_build_lock

    with model_build_lock(123):
        ...    # build the model

If the file exists the pipeline waits until it disappears before starting a new build. Edits therefore pause until any running model finishes. Once testing succeeds the marker is removed and the updated repository can be cloned back safely.

## Editing an operational model

Operational models continue to live under their numeric directory once built. When
changes are required the developer waits for the `.active_model` file to disappear
which signals no build is currently running. The contents of `<model_id>/` are then
cloned back into the main `models_repo` directory using :meth:`BotDevelopmentBot.create_env`.
If the target directory already exists it is refreshed via `git fetch` and `reset`
or by deleting and cloning again. After the clone completes the
`SelfCodingEngine` generates new code snippets that are committed on top of the
copied repository.

This update mechanism allows iterative improvements of deployed models while
keeping the original history intact. The behaviour is implemented in
``create_env`` and ensures that repeated edits always start from the latest
state of the operational model.

