import streamlit as st
from tabs.raceStats import raceStatsTab
from tabs.driverStats import driverStatsTab

st.set_page_config(
    page_title="F1 Dashboard",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏁 F1 Dashboard")

tabs = st.tabs(["Races", "Drivers", "Constructors", "Forecasting (?)"])
with tabs[0]:
    raceStatsTab()
with tabs[1]:
    driverStatsTab()
with tabs[2]:
    st.write('constructor stats')
with tabs[3]:
    st.write("do forecasting???")
    st.link_button(label='reference link', url="https://www.kaggle.com/code/jalelgmiza1/f1-2025-season-analytics#%F0%9F%8F%8E%EF%B8%8F-PREPARE-2025-SEASON-FEATURES-AND-PREDICT-RACE-WINNERS")