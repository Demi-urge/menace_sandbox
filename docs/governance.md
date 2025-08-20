# Governance Rules

The central evaluation loop consults `governance.evaluate_veto` to determine
whether actions should proceed.  Two built-in rules are enforced:

- **No ship if alignment = fail** – any action with `alignment_status` of
  `"fail"` is prevented from shipping.
- **No rollback if RAROI increased in ≥3 scenarios** – per-scenario
  risk‑adjusted ROI is computed via `ROITracker.calculate_raroi`.  If three or
  more scenarios show a positive RAROI delta relative to the baseline scenario,
  rollback actions are vetoed.

These checks run after ROI calculation to halt unsafe behaviour before rewards
are dispatched.
