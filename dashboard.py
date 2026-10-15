"""Grid operator view: tomorrow's load and solar, and this month's inspection list."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="UAE Energy & Smart Grid Analytics", layout="wide")
st.title("UAE Energy & Smart Grid Analytics")
m = json.loads(Path("outputs/metrics.json").read_text())
c = st.columns(4)
c[0].metric("Day-ahead load error", f"{m['load']['model_with_weather_mape']:.1%}", f"{(m['load']['model_with_weather_mape'] - m['load']['same_hour_last_week_mape']) * 100:+.1f} pts vs last week", delta_color="inverse")
c[1].metric("Daily peak error", f"{m['load']['daily_peak_mape']:.1%}")
c[2].metric("Solar error (% of capacity)", f"{m['solar']['ml_model']['nmae_pct_capacity']:.1%}")
c[3].metric("Inspection hit rate", f"{m['tampering']['model']['precision']:.0%}", f"{(m['tampering']['model']['precision'] - m['tampering']['yoy_rule']['precision']) * 100:+.0f} pts vs rule")
st.subheader("Inspection list")
st.dataframe(pd.read_csv("outputs/inspection_list.csv").drop(columns=["tamper"]), hide_index=True, use_container_width=True)
st.markdown(Path("outputs/report.md").read_text())
