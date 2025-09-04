# Menace Auto-Reinvestment Logic

Menace periodically reinvests a portion of its billing balance into expanding its own capabilities. This process is fully automated and adjusts spending based on predicted ROI.

## 1. Billing Integration
- **Read Balance**: Menace reads the current available balance via `stripe_billing_router`.
- **Available for Reinvestment**: `reinvestable_amount = balance * cap_percentage`.
- **Execute Spend**: Funds can be used to buy services (compute, data, proxies) or hire freelancers.

## 2. Predictive Spend Engine
- Forecasts ROI per dollar reinvested and detects the point of diminishing returns.
- Calculates `optimal_spend` and only spends up to the lesser of `optimal_spend` and `reinvestable_amount`.

## 3. Dynamic Cap System
- `cap_percentage` defaults to **50%** but adjusts over time based on historical ROI.
- Caps are lowered if ROI flattens and raised in small steps if ROI compounds.

## 4. Reinvestment Execution
```python
from stripe_billing_router import get_balance

balance = get_balance("finance:finance_router_bot:monetization")
reinvestable = balance * cap_percentage
predicted = predict_optimal_spend()
amount_to_spend = min(predicted, reinvestable)
if amount_to_spend >= minimum_threshold:
    execute_spending(amount_to_spend)
    log_investment(amount_to_spend, predicted_roi, timestamp)
```
- Spending is aborted if it would drop the balance below a safety reserve.

## 5. Logging and Learning
- All transactions are logged with timestamp, amount, predicted ROI and actual ROI.
- The logs train the ROI prediction model to improve future decisions.

## 6. Safeguards
- Manual override options and per-transaction limits.
- `if balance - amount_to_spend < safety_reserve: abort()`

Example: With a $100,000 balance and a 50% cap, if predicted ROI plateaus after $14,000, only $14,000 is spent and $86,000 stays in reserve.

Implementation lives in `investment_engine.AutoReinvestmentBot` which reads the balance through `stripe_billing_router`, predicts optimal spend and logs each transaction via `InvestmentDB`.
