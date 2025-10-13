import streamlit as st
import fastf1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os
from utils.loadDatasets import *

@st.cache_data(show_spinner="Loading session data...")
def load_f1_session(year: int, location: str):
    """Loads and caches the F1 session data"""
    cache_dir = "./f1_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, location, 'R')
    session.load()
    return session

@st.cache_data
def get_overall_data():
    return load_merged_dateset()

@st.cache_data
def get_race_locations(df_circuit_race):
    return list(df_circuit_race['location'].unique())

@st.cache_data
def get_race_years(df_circuit_race):
    return sorted(list(df_circuit_race['year'].unique()))

import streamlit as st
import pandas as pd

def driver_selection_component(df_top5: pd.DataFrame) -> str:
    if "selected_driver" not in st.session_state:
        st.session_state.selected_driver = df_top5.iloc[0]['code']

    def select_driver(driver_code):
        st.session_state.selected_driver = driver_code

    for idx, row in df_top5.iterrows():
        selected = st.session_state.selected_driver == row["code"]

        with st.container(border=True, horizontal_alignment="left"):
            # Highlight selected driver
            st.markdown(f"### {row['Driver']} ({row['code']})")
            
            # Display driver info
            col1, col2, col3 = st.columns([2,2,1], vertical_alignment="top")
            with col1:
                st.write(f"**Team:** {row['name'] if 'name' in row else ''}")  # adjust as needed
                st.write(f"**Position:** {row['positionOrder']}")
                st.write(f"**Points:** {row['points']}")
                st.write(f"**Time:** {row['time_results']}")
            with col2:
                st.write(f"**Fastest Lap Time:** {row['fastestLapTime']}")
                st.write(f"**Fastest Lap Speed:** {row['fastestLapSpeed']} km/h")
            with col3:
                if st.button("Select", key=f"select_{row['code']}"):
                    select_driver(row["code"])

    return st.session_state.selected_driver


@st.cache_data(show_spinner=False)
def plot_speed_heatmap(selected_year, selected_location, driver_code = 'VER'):

    session = load_f1_session(selected_year, selected_location)
    driver_laps = session.laps.pick_drivers(driver_code)
    fastest_lap = driver_laps.pick_fastest()
    tel = fastest_lap.get_telemetry()

    # Prepare track coordinate and speed data
    points = np.array([tel['X'], tel['Y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    speeds = tel['Speed'][:-1]

    # Create figure and line collection
    fig_width = 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.8))
    norm = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=5, alpha=0.85)
    lc.set_array(speeds)
    line = ax.add_collection(lc)

    # Set aspect and limits
    ax.set_aspect('equal')
    ax.set_xlim(tel['X'].min() - 100, tel['X'].max() + 100)
    ax.set_ylim(tel['Y'].min() - 100, tel['Y'].max() + 100)

    # Add colorbar
    cbar = plt.colorbar(line, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Speed (km/h)', rotation=270, labelpad=12)

    # Mark start and finish points
    ax.plot(tel['X'].iloc[0], tel['Y'].iloc[0], 'go', markersize=10, label='Start')
    ax.plot(tel['X'].iloc[-1], tel['Y'].iloc[-1], 'ro', markersize=10, label='Finish')

    # Clean up and finalize
    ax.set_axis_off()
    ax.set_title(
        f"{selected_location} Circuit - Speed Heatmap\n{fastest_lap['Driver']} Fastest Lap ({selected_year})",
        fontsize=10, fontweight='bold', pad=20
    )

    plt.tight_layout()
    return fig


def raceStatsTab():
    st.header("🏎️ Race Statistics")

    st.markdown("### 🔍 Select Race")

    df_overall = get_overall_data()
    available_years = get_race_years(df_overall)
    available_locations = get_race_locations(df_overall)

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Select Year", available_years, index=len(available_years) - 1)
    with col2:
        selected_location = st.selectbox("Select Race Location", available_locations, index=0)

    st.markdown("---")
    
    df_overall_filtered = df_overall.loc[
        (df_overall['location'] == selected_location) &
        (df_overall['year'] == selected_year)
    ].sort_values(by='positionOrder', ascending=True)

    st.subheader(df_overall_filtered.iloc[0]['name_race'])

    col1, col2 = st.columns([2,1], vertical_alignment="top")

    with col1:
        df_overall_filtered['Driver'] = df_overall_filtered['forename'] + " " + df_overall_filtered['surname']
        list_display_cols = ['Driver', 'code', 'name', 'positionOrder', 'points', 'time_results', 'fastestLapTime', 'fastestLapSpeed']
        df_overall_filtered_top5 = df_overall_filtered[list_display_cols].iloc[:5]

        selected_driver = driver_selection_component(df_overall_filtered_top5)

    with col2:
        with st.container(width=500, height=800, border=False, horizontal_alignment="right"):
            fig = plot_speed_heatmap(selected_year, selected_location, driver_code=st.session_state.selected_driver)
            st.pyplot(fig, width='content')

    st.write("Top 5 drivers DF (for testing)")
    st.write(df_overall_filtered_top5)