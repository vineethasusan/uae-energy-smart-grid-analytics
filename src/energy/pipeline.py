"""Train on Jan 2024 - Jun 2025, test on Jul - Dec 2025 (load and solar); tamper detection as of 31 Dec 2025."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import models, simulate

OUT = Path("outputs")
SPLIT = "2025-07-01"
TARIFF_AED_PER_KWH = 0.38          # illustrative average residential/commercial slab rate
INSPECTIONS = 20                   # field inspections the team can do this month
KWP = 5000.0


def run() -> dict:
    w = simulate.weather()
    load, daily = simulate.meters(w)
    df = w.merge(load, on="time")
    df["solar_kw"] = simulate.solar(w, KWP).to_numpy()
    train, test = df[df["time"] < SPLIT], df[df["time"] >= SPLIT]

    # load
    m_w = models.fit_load(train, True)
    m_nw = models.fit_load(train, False)
    Xt = models.load_features(df, True).loc[test.index]
    Xn = models.load_features(df, False).loc[test.index]
    y = test["load_kw"]
    peak_mask = test["time"].dt.hour.between(13, 17) & test["time"].dt.month.isin([7, 8, 9])
    load_res = {"model_with_weather_mape": models.mape(y, m_w.predict(Xt)),
                "model_without_weather_mape": models.mape(y, m_nw.predict(Xn)),
                "same_hour_last_week_mape": models.mape(y, Xt["lag_168"]),
                "summer_afternoon_peak_mape_model": models.mape(y[peak_mask], m_w.predict(Xt[peak_mask])),
                "summer_afternoon_peak_mape_last_week": models.mape(y[peak_mask], Xt.loc[peak_mask, "lag_168"])}
    daily_peak = test.assign(fc=m_w.predict(Xt), d=test["time"].dt.normalize()).groupby("d").agg(peak=("load_kw", "max"), fc_peak=("fc", "max"))
    load_res["daily_peak_mape"] = models.mape(daily_peak["peak"], daily_peak["fc_peak"])

    # solar
    sm = models.fit_solar(train)
    day = test["clear_sky_ghi"] > 50
    ys = test.loc[day, "solar_kw"]
    preds = {"ml_model": sm.predict(models.solar_features(test[day])),
             "yesterday_same_hour": df["solar_kw"].shift(24).loc[ys.index],
             "clear_sky_physics": KWP * test.loc[day, "clear_sky_ghi"] / 1000 * 0.82 * 0.93,
             "physics_with_forecast_irradiance": KWP * test.loc[day, "ghi_fc"] / 1000 * 0.82 * 0.93}
    dust = test.loc[day, "dust"] > 0
    solar_res = {k: {"nmae_pct_capacity": float(np.mean(np.abs(v - ys)) / KWP),
                     "dust_days_nmae": float(np.mean(np.abs(np.asarray(v)[dust.to_numpy()] - ys[dust])) / KWP)} for k, v in preds.items()}

    # tampering
    s = models.tamper_scores(daily, w)
    s["yoy"] = models.yoy_rule(daily).to_numpy()
    s["is_tamper"] = (s["tamper"] != "").astype(int)
    def at_k(col):
        top = s.nlargest(INSPECTIONS, col)
        return {"precision": float(top["is_tamper"].mean()), "found": int(top["is_tamper"].sum()), "total": int(s["is_tamper"].sum()),
                "monthly_loss_recovered_aed": round(float(top.loc[top["is_tamper"] == 1, "kwh_shortfall_recent"].sum() / 2 * TARIFF_AED_PER_KWH))}
    tamper_res = {"meters": len(s), "inspections": INSPECTIONS, "model": at_k("score"), "yoy_rule": at_k("yoy"),
                  "found_by_type": s.nlargest(INSPECTIONS, "score").query("is_tamper == 1")["tamper"].value_counts().to_dict()}
    s.sort_values("score", ascending=False).head(50).to_csv(OUT.mkdir(exist_ok=True) or OUT / "inspection_list.csv", index=False)

    m = {"load": {k: round(v, 4) for k, v in load_res.items()}, "solar": {k: {a: round(b, 4) for a, b in v.items()} for k, v in solar_res.items()},
         "tampering": tamper_res, "test_period": f"{SPLIT} to 2025-12-31", "network_peak_kw": round(float(df["load_kw"].max()))}
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2))
    (OUT / "report.md").write_text(report(m))
    return m


def report(m: dict) -> str:
    L, S, T = m["load"], m["solar"], m["tampering"]
    names = {"ml_model": "**Gradient boosting on weather forecast**", "yesterday_same_hour": "Yesterday, same hour",
             "clear_sky_physics": "Clear-sky physical model", "physics_with_forecast_irradiance": "Physical model with forecast irradiance"}
    lines = ["# UAE Energy & Smart Grid Analytics: results", "", f"Test period {m['test_period']}.", "",
             "## Day-ahead network load (hourly)", "", "| Method | MAPE | Summer afternoon peak hours |", "|---|---|---|",
             f"| **Gradient boosting with temperature forecast** | **{L['model_with_weather_mape']:.1%}** | **{L['summer_afternoon_peak_mape_model']:.1%}** |",
             f"| Same model without weather | {L['model_without_weather_mape']:.1%} | |",
             f"| Same hour last week | {L['same_hour_last_week_mape']:.1%} | {L['summer_afternoon_peak_mape_last_week']:.1%} |", "",
             f"Daily peak forecast error: {L['daily_peak_mape']:.1%}.", "",
             "## Day-ahead rooftop solar (daylight hours, error as % of capacity)", "", "| Method | All days | Dust days |", "|---|---|---|"]
    lines += [f"| {names[k]} | {v['nmae_pct_capacity']:.1%} | {v['dust_days_nmae']:.1%} |" for k, v in S.items()]
    lines += ["", f"## Meter tampering ({T['meters']} meters, {T['inspections']} inspections this month)", "",
              "| Inspection list ranked by | Hit rate | Tampered meters found | Monthly loss recovered (AED) |", "|---|---|---|---|",
              f"| **Weather-adjusted expected-vs-actual model** | **{T['model']['precision']:.0%}** | **{T['model']['found']} / {T['model']['total']}** | **{T['model']['monthly_loss_recovered_aed']:,}** |",
              f"| Year-on-year drop rule | {T['yoy_rule']['precision']:.0%} | {T['yoy_rule']['found']} / {T['yoy_rule']['total']} | {T['yoy_rule']['monthly_loss_recovered_aed']:,} |"]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
    print((OUT / "report.md").read_text())
