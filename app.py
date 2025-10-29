import streamlit as st
from tabs.raceStats import raceStatsTab
from tabs.driverStats import driverStatsTab
from tabs.overallStats import homeTab
from tabs.constructorStats import constructorStatsTab
from tabs.weatherAnalysis import weatherAnalysisTab
from tabs.forecasting import forecastingTab

st.set_page_config(
    page_title="F1 Dashboard",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏁 F1 Dashboard")

tabs = st.tabs(["Home", "Races", "Drivers", "Constructors", "Forecasting", "Weather Analysis"])
with tabs[0]:
    homeTab()
with tabs[1]:
    raceStatsTab()
with tabs[2]:
    driverStatsTab()
with tabs[3]:
    constructorStatsTab()
with tabs[4]:
    forecastingTab()
with tabs[5]:
    weatherAnalysisTab()

