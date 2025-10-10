import streamlit as st
import fastf1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os

@st.cache_data(show_spinner="Loading session data... This may take a moment ⏳")
def load_f1_session(year: int, location: str):
    """Loads and caches the F1 session data for performance."""
    session = fastf1.get_session(year, location, 'R')
    session.load()
    return session

def raceStatsTab():
    st.header("🏎️ Race Statistics")

    st.markdown("### 🔍 Select Race")

    available_years = list(range(2020, 2025))
    available_locations = [
        "Monaco", "Monza", "Silverstone", "Singapore", "Bahrain", "Abu Dhabi"
    ]

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Select Year", available_years, index=len(available_years) - 1)
    with col2:
        selected_location = st.selectbox("Select Race Location", available_locations, index=0)

    st.markdown("---")

    cache_dir = "./f1_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)

    try:
        session = load_f1_session(selected_year, selected_location)
        fastest_lap = session.laps.pick_fastest()
        tel = fastest_lap.get_telemetry()

        points = np.array([tel['X'], tel['Y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        speeds = tel['Speed'][:-1]

        fig_width = 8 
        fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.8))
        norm = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())
        lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=5, alpha=0.85)
        lc.set_array(speeds)
        line = ax.add_collection(lc)

        ax.set_aspect('equal')
        ax.set_xlim(tel['X'].min() - 100, tel['X'].max() + 100)
        ax.set_ylim(tel['Y'].min() - 100, tel['Y'].max() + 100)

        # Add colorbar
        cbar = plt.colorbar(line, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('Speed (km/h)', rotation=270, labelpad=12)

        ax.plot(tel['X'].iloc[0], tel['Y'].iloc[0], 'go', markersize=10, label='Start')
        ax.plot(tel['X'].iloc[-1], tel['Y'].iloc[-1], 'ro', markersize=10, label='Finish')

        ax.set_axis_off()

        ax.set_title(
            f"{selected_location} Circuit - Speed Heatmap\n{fastest_lap['Driver']} Fastest Lap ({selected_year})",
            fontsize=14, fontweight='bold', pad=20
        )

        plt.tight_layout()
        with st.container(width=600):
            st.pyplot(fig, width='content')

    except Exception as e:
        st.error(f"❌ Could not load data for {selected_location} {selected_year}. Error: {e}")
        st.info("Try another race or check your internet connection.")