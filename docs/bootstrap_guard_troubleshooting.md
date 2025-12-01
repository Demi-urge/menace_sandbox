# Bootstrap guard troubleshooting

When multiple modules import bootstrap helpers at the same time, **always** call
`coding_bot_interface.prepare_pipeline_for_bootstrap` instead of instantiating a
pipeline directly. The function uses a single-flight guard and the dependency
broker to advertise a shared placeholder so that late imports attach to the
in-flight promise instead of starting another bootstrap.

Typical call sites include long-lived services such as `research_aggregator_bot`,
`watchdog`, and `prediction_manager_bot`. Each helper should:

1. Call `advertise_bootstrap_placeholder` if it owns the initial bootstrap.
2. Invoke `prepare_pipeline_for_bootstrap(...)` once and reuse the returned
   pipeline/promoter tuple for downstream components.
3. Avoid recursive or re-entrant bootstrap attempts; if a placeholder is already
   advertised, wait on the active promise instead of re-entering the bootstrap
   routine.

If tests show multiple `prepare_pipeline_for_bootstrap` log entries per process,
clear the bootstrap coordinator and dependency broker fixtures to ensure a
single-flight owner is elected before additional helpers run.
