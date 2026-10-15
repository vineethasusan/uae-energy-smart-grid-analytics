# UAE Energy & Smart Grid Analytics: results

Test period 2025-07-01 to 2025-12-31.

## Day-ahead network load (hourly)

| Method | MAPE | Summer afternoon peak hours |
|---|---|---|
| **Gradient boosting with temperature forecast** | **3.7%** | **2.9%** |
| Same model without weather | 6.7% | |
| Same hour last week | 8.5% | 6.5% |

Daily peak forecast error: 3.2%.

## Day-ahead rooftop solar (daylight hours, error as % of capacity)

| Method | All days | Dust days |
|---|---|---|
| **Gradient boosting on weather forecast** | 1.0% | 2.8% |
| Yesterday, same hour | 4.9% | 14.4% |
| Clear-sky physical model | 6.2% | 29.4% |
| Physical model with forecast irradiance | 3.5% | 5.2% |

## Meter tampering (400 meters, 20 inspections this month)

| Inspection list ranked by | Hit rate | Tampered meters found | Monthly loss recovered (AED) |
|---|---|---|---|
| **Weather-adjusted expected-vs-actual model** | **75%** | **15 / 24** | **10,696** |
| Year-on-year drop rule | 60% | 12 / 24 | 9,136 |