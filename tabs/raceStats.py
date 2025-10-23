import streamlit as st
import fastf1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os
import pandas as pd
from utils.loadDatasets import load_merged_dateset

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def create_fastf1_session(year, location):
    cache_dir = "./f1_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f'loading session for {location} in {year}')
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, location, 'R')
    session.load()  # must load to access telemetry
    return session

@st.cache_data(show_spinner="Loading telemetry data...")
def load_driver_telemetry(year: int, location: str, driver_code: str):
    """
    Load telemetry for the fastest lap of a given driver, year, and race.
    Returns a df with all telemetry columns
    """
    session = create_fastf1_session(year, location)
    driver_laps = session.laps.pick_drivers(driver_code)
    if driver_laps.empty:
        raise ValueError(f"No laps found for driver {driver_code} in {location} {year}.")
    fastest_lap = driver_laps.pick_fastest()
    tel = fastest_lap.get_telemetry()

    return pd.DataFrame({
        "X": tel["X"],
        "Y": tel["Y"],
        "Speed": tel["Speed"],
        "Throttle": tel["Throttle"],
        "Brake": tel["Brake"],
        "RPM": tel["RPM"],
        "nGear": tel["nGear"],
        "DRS": tel["DRS"]
    })


def plot_telemetry_heatmap(selected_year, selected_location, driver_code='VER', metric='Speed'):
    """
    Create figure for chosen driver's fastest lap telemetry heatmap.
    
    Parameters:
    -----------
    metric : str
        Telemetry metric to display. Options:
        - 'Speed': Car speed in km/h
        - 'Throttle': Throttle position (0-100%)
        - 'Brake': Brake application (0-100% or boolean)
        - 'RPM': Engine revolutions per minute
        - 'Gear': Current gear (1-8)
        - 'DRS': DRS status (0=inactive, >0=active)
    """
    tel = load_driver_telemetry(selected_year, selected_location, driver_code)

    points = np.array([tel['X'], tel['Y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Map metric to df columns
    metric_column_map = {
        'Speed': 'Speed',
        'Throttle': 'Throttle',
        'Brake': 'Brake',
        'RPM': 'RPM',
        'Gear': 'nGear',
        'DRS': 'DRS'
    }
    
    column_name = metric_column_map[metric]
    metric_data = tel[column_name][:-1]
    
    # config for different metrics
    metric_config = {
        'Speed': {
            'cmap': 'plasma',
            'label': 'Speed (km/h)',
            'use_minmax': True
        },
        'Throttle': {
            'cmap': 'YlGn',
            'label': 'Throttle (%)',
            'use_minmax': False,
            'vmin': 0,
            'vmax': 100
        },
        'Brake': {
            'cmap': 'Reds',
            'label': 'Brake Pressure',
            'use_minmax': True  # Brake can be boolean or percentage
        },
        'RPM': {
            'cmap': 'inferno',
            'label': 'RPM',
            'use_minmax': True
        },
        'Gear': {
            'cmap': 'viridis',
            'label': 'Gear',
            'use_minmax': False,
            'vmin': 1,
            'vmax': 8
        },
        'DRS': {
            'cmap': 'RdYlGn',
            'label': 'DRS (0=Off, >0=On)',
            'use_minmax': False,
            'vmin': 0,
            'vmax': 14
        }
    }
    
    config = metric_config[metric]
    
    # Set up normalization
    if config['use_minmax']:
        norm = plt.Normalize(vmin=metric_data.min(), vmax=metric_data.max())
    else:
        norm = plt.Normalize(vmin=config['vmin'], vmax=config['vmax'])

    fig_width = 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.8))
    
    lc = LineCollection(segments, cmap=config['cmap'], norm=norm, linewidth=5, alpha=0.85)
    lc.set_array(metric_data)
    line = ax.add_collection(lc)

    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlim(tel['X'].min() - 100, tel['X'].max() + 100)
    ax.set_ylim(tel['Y'].min() - 100, tel['Y'].max() + 100)

    # Add colorbar
    cbar = plt.colorbar(line, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(config['label'], rotation=270, labelpad=12)

    # Add start/finish markers
    ax.plot(tel['X'].iloc[0], tel['Y'].iloc[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(tel['X'].iloc[-1], tel['Y'].iloc[-1], 'ro', markersize=10, label='Finish', zorder=5)
    ax.legend(loc='upper right', fontsize=8)

    ax.set_axis_off()
    ax.set_title(
        f"{selected_location} Circuit - {metric} Heatmap\n{driver_code} Fastest Lap ({selected_year})",
        fontsize=10, fontweight='bold', pad=20
    )
    plt.tight_layout()
    
    return fig


@st.cache_data
def get_overall_data():
    return load_merged_dateset()

@st.cache_data
def get_race_locations(df_circuit_race):
    return sorted(list(df_circuit_race['location'].unique()))

@st.cache_data
def get_race_years(df_circuit_race):
    return sorted(list(df_circuit_race['year'].unique()), reverse=True)


def driver_selection_component(df_top5: pd.DataFrame) -> str:
    if "selected_driver" not in st.session_state:
        st.session_state.selected_driver = df_top5.iloc[0]['code']

    def select_driver(driver_code):
        st.session_state.selected_driver = driver_code

    for idx, row in df_top5.iterrows():
        selected = st.session_state.selected_driver == row["code"]

        with st.container(border=True):
            st.markdown(f"### {row['Driver']} ({row['code']})")

            col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="top")
            with col1:
                st.write(f"**Team:** {row.get('name','')}")
                st.write(f"**Position:** {row['positionOrder']}")
                st.write(f"**Points:** {row['points']}")
                st.write(f"**Time:** {row['time_results']}")
            with col2:
                st.write(f"**Fastest Lap Time:** {row['fastestLapTime']}")
                st.write(f"**Fastest Lap Speed:** {row['fastestLapSpeed']} km/h")
            with col3:
                with st.container(horizontal_alignment='right'):
                    if st.button("Select", key=f"select_{row['code']}", type="primary"):
                        select_driver(row["code"])

    return st.session_state.selected_driver


# ============================================================
#  MAIN RACE STATS TAB
# ============================================================

def raceStatsTab():
    st.header("🏎️ Race Statistics")
    st.markdown("### 🔍 Select Race")

    df_overall = get_overall_data()
    available_years = get_race_years(df_overall)
    available_locations = get_race_locations(df_overall)

    if "selected_year" not in st.session_state:
        st.session_state.selected_year = available_years[0]

    if "selected_location" not in st.session_state:
        # choose the first valid location for that year
        st.session_state.selected_location = df_overall.loc[
            df_overall["year"] == st.session_state.selected_year, "location"
        ].iloc[0]

    #  Filter options dynamically
    filtered_locations = df_overall.loc[
        df_overall["year"] == st.session_state.selected_year, "location"
    ].unique()

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox(
            "Select Year",
            available_years,
            index=available_years.index(st.session_state.selected_year),
            key="selected_year",
            on_change=lambda: st.session_state.update({"selected_location": None})
        )

    with col2:
        selected_location = st.selectbox(
            "Select Location",
            filtered_locations,
            index=0 if st.session_state.selected_location not in filtered_locations else list(filtered_locations).index(st.session_state.selected_location),
            key="selected_location"
        )

    # --- Filter data for selected race ---
    df_overall_filtered = df_overall.loc[
        (df_overall['location'] == selected_location) &
        (df_overall['year'] == selected_year)
    ].sort_values(by='positionOrder', ascending=True)

    if df_overall_filtered.empty:
        st.warning("No race data for that selection.")
        return

    st.subheader(df_overall_filtered.iloc[0]['name_race'])

    col1, col2 = st.columns([2, 1], vertical_alignment="top")

    with col1:
        df_overall_filtered['Driver'] = df_overall_filtered['forename'] + " " + df_overall_filtered['surname']
        list_display_cols = ['Driver', 'code', 'name', 'positionOrder', 'points', 'time_results', 'fastestLapTime', 'fastestLapSpeed']
        df_overall_filtered_top5 = df_overall_filtered[list_display_cols].iloc[:5]

        selected_driver = driver_selection_component(df_overall_filtered_top5)

    with col2:
        # Metric selector
        metric_options = ['Speed', 'Throttle', 'Brake', 'RPM', 'Gear', 'DRS']
        metric_descriptions = {
            "Speed": "🏎️ Car speed in km/h at each point on track",
            "Throttle": "⚡ Throttle position percentage (0-100%)",
            "Brake": "🛑 Brake application pressure",
            "RPM": "🔄 Engine revolutions per minute",
            "Gear": "⚙️ Current gear selection (1-8)",
            "DRS": "💨 DRS activation zones and usage"
        }
        
        selected_metric = st.selectbox(
            "📊 Select Telemetry Metric",
            metric_options,
            index=0,
            help="Choose which telemetry data to visualize on the track map"
        )
        
        st.info(metric_descriptions[selected_metric])
        
        driver_code = st.session_state.get('selected_driver', df_overall_filtered_top5.iloc[0]['code'])
        
        with st.spinner(f"Loading {selected_metric} telemetry..."):
            try:
                fig = plot_telemetry_heatmap(
                    selected_year, 
                    selected_location, 
                    driver_code=driver_code,
                    metric=selected_metric
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig) 
            except Exception as e:
                st.error("⚠️ Could not load telemetry data.")
                st.exception(e)

    st.dataframe(df_overall_filtered_top5, use_container_width=True)