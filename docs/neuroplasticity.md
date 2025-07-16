# Neuroplasticity and PathwayDB

`PathwayDB` tracks execution traces of Menace workflows. Each run is logged as a `PathwayRecord` containing actions, inputs, outputs, runtime and ROI. The `outcome` field uses the `Outcome` enum with `SUCCESS`, `PARTIAL_SUCCESS` and `FAILURE`. The database keeps metadata for every pathway including how often it was executed, its success rate and a **myelination score** that decays over time.

## Myelination Score

Every execution updates the metadata table. The myelination score roughly equals the pathway's frequency multiplied by its success rate and ROI with an exponential decay based on the last activation. Higher scores indicate reliable, profitable workflows.

## Hebbian Linking

When consecutive pathways are recorded, `record_sequence()` strengthens the link between them. Heavily reinforced pairs can later merge into macro pathways. This forms a lightweight graph of actions where edges reflect historical co‑activation.

## Influence on Execution

`ModelAutomationPipeline` queries `PathwayDB` for similar action traces. If the best match exceeds the configured threshold, the pipeline treats it as a high‑trust workflow:

- Recent memory tagged with the model is **preloaded** for immediate context.
- Support bots run with a higher allocation **weight** so more resources are available.
- Planning receives a greater **trust weight** which reduces validation overhead.
- Each support bot is **primed** if it implements a `prime()` method, allowing
  it to prepare internal state before tasks begin.

Future versions may boost overall resources even further as the myelination score grows.

## Event Integration

When constructed with a `UnifiedEventBus`, `PathwayDB` emits a `"pathway:new"`
event for every logged record.  Other components such as
`SelfImprovementEngine` can subscribe to this topic to adapt models in real time.
