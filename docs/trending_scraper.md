# Trending Scraper

`TrendingScraper` collects popular product names or business ideas from
multiple sources such as Reddit, Shopify, Gumroad, Fiverr and Google Trends.
When an `energy` value is provided, the scraper filters results using preset
keywords so that low energy states favour quick wins while higher energy seeks
scalable opportunities.

```python
from menace.trending_scraper import TrendingScraper

scraper = TrendingScraper()
items = scraper.scrape_reddit()
for item in items:
    print(item.product_name)
```

These results can be passed to `BotCreationBot` to influence how new workflows
are ranked.

## Continuous Microtrend Tracking

`MicrotrendService` wraps the scraper and can run continuously in a background
thread.  Use `run_continuous()` to schedule regular scans and supply a callback
or planning component that receives new microtrends as they appear.

```python
from menace.microtrend_service import MicrotrendService

service = MicrotrendService()
service.run_continuous(interval=3600)
```
