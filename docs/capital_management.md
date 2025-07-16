# Capital Management Enhancements

`CapitalManagementBot` now enforces optional daily and weekly budgets. Spending
is tracked in a ledger and `check_budget()` flags the bot in safe mode via
`ErrorBot` when limits are exceeded.

```python
bot = CapitalManagementBot(daily_budget=100.0,
                           weekly_budget=500.0,
                           error_bot=my_error_bot)

bot.log_expense(25.0, "api usage")
```

Calling `log_expense()` automatically checks the current totals. When the daily
or weekly budget is surpassed the bot enters safe mode and autoscaling is
halted.

`spending_anomalies()` detects unusual spikes using the same anomaly detection
helpers as `DataBot` and `auto_rollback()` can invoke a callback when the energy
score or ROI trend collapses.
