import streamlit as st
import pandas as pd
import plotly.express as px
from utils.loadDatasets import *

@st.cache_data
def load_data():
    return load_merged_dateset()

@st.cache_data
def get_recent_races(df):
    return load_recent_races_results(df)

@st.cache_data
def display_recent_10_races(df):
    # Group by race and year, then iterate
    for (race_name, date), group in df.groupby(['name_race', 'date']):
        label = f"**{race_name}** - *{date.date()}*"
        with st.expander(label=label):
            for idx, row in group.iterrows():
                with st.container(border=True, vertical_alignment='top', horizontal_alignment='left'):
                    driver_markdown = f"""
                    **🏎️ Driver:** {row['driver']}
                    **🏁 Position:** {row['positionOrder']}  
                    **💯 Points:** {row['points']}  
                    **⏱️ Race Time:** {row['time_results']}  
                    **⚡ Fastest Lap:** {row['fastestLapTime']}  
                    **🚀 Lap Speed:** {row['fastestLapSpeed']}
                    """
                    st.markdown(driver_markdown)


@st.cache_data
def display_constructor_points_over_time(df):
    fig = px.line(
        df,
        x='year',
        y='points',
        color='name',
        title='Constructor Points by Year',
        labels={'points': 'Total Points', 'year': 'Year', 'name': 'Constructor'}
    )

    st.plotly_chart(fig, use_container_width=True)

def overallStatsTab():
    df = load_data()
    df_recent_races = get_recent_races(df)
    st.image("https://media.formula1.com/image/upload/c_lfill,w_440/q_auto/d_common:f1:2025:fallback:driver:2025fallbackdriverright.webp/v1740000000/common/f1/2025/alpine/fracol01/2025alpinefracol01right.webp")
    display_recent_10_races(df_recent_races)

    # Line chart of total points per year by constructor
    display_constructor_points_over_time(df)