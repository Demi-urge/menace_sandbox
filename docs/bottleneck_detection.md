# Bottleneck Detection Bots

Performance isn’t a luxury; it’s a multiplier on ROI. Each critical function is decorated with an `@perf_monitor` that records wall-clock time via `time.perf_counter` and stores `(function_signature, runtime_ms, cpu_pct, mem_mb)` into `PerfDB`. A nightly cron triggers `bottleneck_scanner.py`, ranking the P95 latencies per module. When a spike exceeds a configurable threshold, the scanner opens a Git issue via API, tagging the responsible bot and auto-assigning the Enhancement Bot.

Technically, this layer relies on `psutil` for cross-platform metrics and `sqlite3` for zero-setup storage. For heavier loads, export to Prometheus + Grafana so you can watch Menace’s pulse in real time. The scanner feeds its findings into the Resource Allocation Optimizer, ensuring slow code is either optimized or throttled.
