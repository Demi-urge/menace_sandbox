# Feedback Pipeline

The feedback pipeline closes the loop between deployed patches and future
retrieval ranking. It consists of three stages:

1. **Collection** – `MetricsDB.log_retrieval_metrics` and
   `MetricsDB.log_patch_outcome` capture per-result scores and deployment
   outcomes in `metrics.db`.
2. **Aggregation** – `metrics_aggregator.compute_retriever_stats` joins the raw
   metrics and computes win percentage, regret percentage, sample count and
   embedding staleness for each origin database. The results are stored back
   into `retriever_kpi`.
3. **Consumption** – `UniversalRetriever` queries the latest KPI values via
   `MetricsDB.latest_retriever_kpi` and biases ranking accordingly. Databases
   with higher win percentages and larger sample counts are promoted while
   stale or regretful sources are down-weighted.

Tuning is exposed through the `RetrievalWeights` dataclass. Adjust the
`win`, `regret` and `stale_cost` fields to experiment with different feedback
strengths without modifying retrieval code.
