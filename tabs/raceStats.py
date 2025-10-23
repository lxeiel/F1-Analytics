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

@st.cache_data(show_spinner="Loading Telemetry data...")
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
def format_time_delta(time_str, winner_time_str):
    """Calculate time delta from winner"""
    if pd.isna(time_str) or time_str == '' or time_str == winner_time_str:
        return ""
    try:
        # Simple string comparison for display
        return f"+{time_str}"
    except:
        return ""

def get_position_badge(position):
    """Return styled position indicator"""
    if position == 1:
        return "🥇"
    elif position == 2:
        return "🥈"
    elif position == 3:
        return "🥉"
    else:
        return f"P{position}"

def raceStatsTab():
    st.header("🏎️ Race Statistics")
    
    # Load data
    df_overall = get_overall_data()
    available_years = get_race_years(df_overall)
    available_locations = get_race_locations(df_overall)

    # Initialize session state
    if "selected_year" not in st.session_state:
        st.session_state.selected_year = available_years[0]
    if "selected_location" not in st.session_state:
        st.session_state.selected_location = df_overall.loc[
            df_overall["year"] == st.session_state.selected_year, "location"
        ].iloc[0]
    if "selected_driver" not in st.session_state:
        st.session_state.selected_driver = None

    # Filter available locations
    filtered_locations = df_overall.loc[
        df_overall["year"] == st.session_state.selected_year, "location"
    ].unique()

    # Race selector
    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox(
            "📅 Select Year",
            available_years,
            index=available_years.index(st.session_state.selected_year),
            key="selected_year"
        )
    with col2:
        selected_location = st.selectbox(
            "🏁 Select Circuit",
            filtered_locations,
            index=0 if st.session_state.selected_location not in filtered_locations 
                  else list(filtered_locations).index(st.session_state.selected_location),
            key="selected_location"
        )

    # Filter race data
    df_race = df_overall.loc[
        (df_overall['location'] == selected_location) &
        (df_overall['year'] == selected_year)
    ].sort_values(by='positionOrder', ascending=True).copy()

    if df_race.empty:
        st.warning("⚠️ No race data available for this selection.")
        return

    # Race header with key info
    race_name = df_race.iloc[0]['name_race']
    winner = df_race.iloc[0]
    
    st.markdown(f"## {race_name}")
    
    # Quick stats bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Winner", f"{winner['forename']} {winner['surname']}")
    with col2:
        st.metric("⏱️ Winning Time", winner['time_results'])
    with col3:
        st.metric("🚀 Fastest Lap", f"{winner['fastestLapTime']}")
    with col4:
        st.metric("💨 Top Speed", f"{winner['fastestLapSpeed']} km/h")
    
    st.divider()

    # Main content: Results table + Telemetry
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader("📊 Race Results")
        
        # Prepare display dataframe
        df_race['Driver'] = df_race['forename'] + " " + df_race['surname']
        df_race['Pos'] = df_race['positionOrder'].apply(get_position_badge)
        df_race['Gap'] = df_race.apply(
            lambda row: format_time_delta(row['time_results'], winner['time_results']), 
            axis=1
        )
        
        # Display table with custom formatting
        display_df = df_race[['Pos', 'Driver', 'name', 'points', 'Gap', 
                               'fastestLapTime', 'fastestLapSpeed']].head(10)
        display_df.columns = ['Pos', 'Driver', 'Team', 'Points', 'Gap', 
                               'Fastest Lap', 'Top Speed (km/h)']
        
        # Driver selector (place BEFORE styling to avoid lag)
        st.markdown("##### 🎯 Select Driver for Telemetry Analysis")
        driver_options = df_race.head(10).apply(
            lambda row: f"{row['Driver']} ({row['code']})", axis=1
        ).tolist()
        
        # Get previously selected driver or default to first
        if "selected_driver_code" not in st.session_state:
            st.session_state.selected_driver_code = df_race.iloc[0]['code']
        
        # Find index of previously selected driver
        default_index = 0
        for idx, option in enumerate(driver_options):
            if st.session_state.selected_driver_code in option:
                default_index = idx
                break
        
        selected_driver_display = st.selectbox(
            "Driver",
            driver_options,
            index=default_index,
            label_visibility="collapsed",
            key="driver_selector"
        )
        
        # Extract and store driver code
        driver_code = selected_driver_display.split('(')[1].split(')')[0]
        st.session_state.selected_driver_code = driver_code
        
        # Style the dataframe (now uses updated driver_code)
        def highlight_selected(row):
            # Get driver code from the display name for comparison
            driver_name = row['Driver']
            row_driver_code = df_race[df_race['Driver'] == driver_name]['code'].values
            if len(row_driver_code) > 0 and row_driver_code[0] == st.session_state.selected_driver_code:
                return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_selected, axis=1)
        
        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            height=400
        )

    with col_right:
        st.subheader("📡 Telemetry Analysis")
        
        metric_categories = {
            "Speed & Performance": {
                "Speed": {"icon": "🏎️", "desc": "Car speed at each point"},
                "RPM": {"icon": "🔄", "desc": "Engine revolutions"},
            },
            "Driver Inputs": {
                "Throttle": {"icon": "⚡", "desc": "Throttle application"},
                "Brake": {"icon": "🛑", "desc": "Brake pressure"},
                "Gear": {"icon": "⚙️", "desc": "Gear selection"},
            },
            "Aerodynamics": {
                "DRS": {"icon": "💨", "desc": "DRS zones & usage"},
            }
        }
        
        all_metrics = []
        for category, metrics in metric_categories.items():
            all_metrics.extend(metrics.keys())
        
        selected_metric = st.selectbox(
            "📊 Telemetry Metric",
            all_metrics,
            index=0
        )
        
        for category, metrics in metric_categories.items():
            if selected_metric in metrics:
                st.info(f"{metrics[selected_metric]['icon']} {metrics[selected_metric]['desc']}")
                break
        
        # Generate telemetry plot
        with st.spinner(f"Loading {selected_metric} data..."):
            try:
                fig = plot_telemetry_heatmap(
                    selected_year,
                    selected_location,
                    driver_code=driver_code,
                    metric=selected_metric
                )
                st.pyplot(fig, width='stretch')
                plt.close(fig)
            except Exception as e:
                st.error("⚠️ Telemetry data unavailable")
                with st.expander("View error details"):
                    st.exception(e)
        
        # Additional stats for selected driver
        driver_data = df_race[df_race['code'] == driver_code].iloc[0]
        
        with st.container(border=True):
            st.markdown(f"**{driver_data['Driver']}** - {driver_data['name']}")
            
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric("Position", driver_data['positionOrder'])
                st.metric("Points", driver_data['points'])
            with stat_col2:
                st.metric("Fastest Lap", driver_data['fastestLapTime'])
                st.metric("Top Speed", f"{driver_data['fastestLapSpeed']} km/h")

    # Optional: Expandable section for full results
    with st.expander("📋 View Complete Race Results"):
        full_results = df_race[['positionOrder', 'Driver', 'name', 'points', 
                                 'time_results', 'fastestLapTime', 'fastestLapSpeed']]
        full_results.columns = ['Position', 'Driver', 'Team', 'Points', 
                                'Time', 'Fastest Lap', 'Top Speed (km/h)']
        st.dataframe(full_results, width='stretch', hide_index=True)