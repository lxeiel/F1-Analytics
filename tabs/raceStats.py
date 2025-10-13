import streamlit as st
import fastf1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os
from utils.loadDatasets import *

@st.cache_data(show_spinner="Loading session data... This may take a moment ⏳")
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
def get_race_circuits_data():
    '''Get merged race and circuit dataframe'''
    df_circuits = load_circuits_dataset()
    df_races = load_race_dataset()
    df_circuit_race = pd.merge(df_races, df_circuits, on='circuitId', how='left', suffixes=['_race', '_circuits'])
    return df_circuit_race

@st.cache_data
def get_race_locations(df_circuit_race):
    return list(df_circuit_race['location'].unique())

@st.cache_data(show_spinner=False)
def plot_speed_heatmap(selected_year, selected_location):
    """
    Generate and display a speed heatmap for the fastest lap of a given race.

    Parameters:
    - selected_year (int): Race year
    - selected_location (str): Race location (e.g., 'Monza')
    - load_f1_session (function): Function to load the F1 session (cached elsewhere)
    """
    # Load session and telemetry data
    session = load_f1_session(selected_year, selected_location)
    fastest_lap = session.laps.pick_fastest()
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
        fontsize=14, fontweight='bold', pad=20
    )

    plt.tight_layout()
    return fig


def raceStatsTab():
    st.header("🏎️ Race Statistics")

    st.markdown("### 🔍 Select Race")

    available_years = list(range(2020, 2025))
    available_locations = get_race_locations()

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Select Year", available_years, index=len(available_years) - 1)
    with col2:
        selected_location = st.selectbox("Select Race Location", available_locations, index=0)

    st.markdown("---")
    
    with st.container(width=500, height=800, border=False):
        fig = plot_speed_heatmap(selected_year, selected_location)
        st.pyplot(fig, width='stretch')