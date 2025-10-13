import streamlit as st
from utils.loadDatasets import *

@st.cache_data
def get_overall_data():
    return load_merged_dateset()

def driverStatsTab():
    st.header("👨‍✈️ Driver Statistics")

st.write("TODO: ADD IN FILTER")
st.write("TODO: ADD DRIVER STATS")