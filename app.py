import streamlit as st
from tabs.raceStats import raceStatsTab
from tabs.driverStats import driverStatsTab

st.set_page_config(
    page_title="F1 Dashboard",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏁 F1 Dashboard")

tabs = st.tabs(["Race Stats", "Driver Stats"])
with tabs[0]:
    raceStatsTab()
with tabs[1]:
    driverStatsTab()