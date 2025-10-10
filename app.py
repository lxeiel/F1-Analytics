import streamlit as st
import fastf1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os
import warnings
from tabs.raceStats import raceStatsTab
from tabs.driverStats import driverStatsTab

warnings.filterwarnings('ignore')
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