import streamlit as st
from tabs.raceStats import raceStatsTab
from tabs.driverStats import driverStatsTab
from tabs.overallStats import homeTab
from tabs.constructorStats import constructorStatsTab
from tabs.weatherAnalysis import weatherAnalysisTab
from tabs.tireAnalysis import tireAnalysisTab

st.set_page_config(
    page_title="F1 Dashboard",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏁 F1 Dashboard")

tabs = st.tabs(["Home", "Races", "Drivers", "Constructors", "Forecasting (?)", "Weather Analysis","Tire Analysis"])
with tabs[0]:
    homeTab()
with tabs[1]:
    raceStatsTab()
with tabs[2]:
    driverStatsTab()
with tabs[3]:
    constructorStatsTab()
with tabs[4]:
    st.write("do forecasting???")
    st.link_button(label='reference link', url="https://www.kaggle.com/code/jalelgmiza1/f1-2025-season-analytics#%F0%9F%8F%8E%EF%B8%8F-PREPARE-2025-SEASON-FEATURES-AND-PREDICT-RACE-WINNERS")
with tabs[5]:
    weatherAnalysisTab()
with tabs[6]:
    tireAnalysisTab()

