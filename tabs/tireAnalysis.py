import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.loadDatasets import load_merged_dataset
import fastf1
import fastf1.plotting
import os

# Create cache directory if it doesn't exist
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'f1_cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable FastF1 cache
fastf1.Cache.enable_cache(cache_dir)

@st.cache_data
def create_fastf1_session(year, location):
    """Create and load FastF1 session"""
    try:
        session = fastf1.get_session(year, location, 'R')
        session.load()
        return session
    except Exception as e:
        st.error(f"Error loading session: {str(e)}")
        return None

@st.cache_data
def get_tire_strategy_data(year, location):
    """Get tire strategy data for all drivers in a race"""
    try:
        session = create_fastf1_session(year, location)
        if session is None:
            return None
        
        laps = session.laps
        if laps is None or laps.empty:
            return None
        
        # Get driver information
        drivers_info = session.drivers
        driver_names = {}
        driver_teams = {}
        
        for driver_num in drivers_info:
            driver_info = session.get_driver(driver_num)
            driver_code = driver_info['Abbreviation']
            driver_full_name = f"{driver_info['FirstName']} {driver_info['LastName']}"
            driver_names[driver_code] = driver_full_name
            driver_teams[driver_code] = driver_info.get('TeamName', 'Unknown')
        
        # Process tire strategy data
        strategy_data = []
        for driver_code in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver_code].copy()
            
            # Group consecutive laps with same compound
            driver_laps['CompoundChange'] = driver_laps['Compound'] != driver_laps['Compound'].shift(1)
            driver_laps['StintNumber'] = driver_laps['CompoundChange'].cumsum()
            
            for stint_num, stint_laps in driver_laps.groupby('StintNumber'):
                if pd.isna(stint_laps['Compound'].iloc[0]):
                    continue
                
                strategy_data.append({
                    'Driver': driver_code,
                    'DriverName': driver_names.get(driver_code, driver_code),
                    'Team': driver_teams.get(driver_code, 'Unknown'),
                    'Compound': stint_laps['Compound'].iloc[0],
                    'StintNumber': stint_num,
                    'StartLap': stint_laps['LapNumber'].min(),
                    'EndLap': stint_laps['LapNumber'].max(),
                    'StintLength': len(stint_laps),
                    'TyreLife': stint_laps['TyreLife'].max()
                })
        
        return pd.DataFrame(strategy_data)
    except Exception as e:
        st.error(f"Error processing tire strategy: {str(e)}")
        return None

def tireAnalysisTab():
    st.markdown("### 🏎️ Tire Strategy Analysis")
    st.markdown("Compare tire strategies across drivers in a race using FastF1 telemetry data.")
    
    # Load overall data for race selection
    df = load_merged_dataset()
    
    # Year and location selection
    col1, col2 = st.columns(2)
    
    with col1:
        available_years = sorted(df['year'].unique(), reverse=True)
        recent_years = [y for y in available_years if y >= 2018]
        selected_year = st.selectbox("Select Year", recent_years, index=0, key="tire_year")
    
    with col2:
        races_in_year = df[df['year'] == selected_year]
        locations = sorted(races_in_year['location'].unique())
        selected_location = st.selectbox("Select Race", locations, key="tire_location")
    
    st.divider()
    
    # Load tire strategy data
    with st.spinner("Loading tire strategy data..."):
        strategy_df = get_tire_strategy_data(selected_year, selected_location)
    
    if strategy_df is None or strategy_df.empty:
        st.warning("⚠️ Tire strategy data not available for this race. Try selecting a different race from 2018 onwards.")
        return
    
    # Strategy Overview
    st.markdown("#### 📊 Strategy Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_drivers = strategy_df['Driver'].nunique()
        st.metric("👥 Drivers", total_drivers)
    with col2:
        compounds_used = strategy_df['Compound'].nunique()
        st.metric("🔴 Compounds Used", compounds_used)
    with col3:
        avg_stops = strategy_df.groupby('Driver')['StintNumber'].max().mean()
        st.metric("🔧 Avg Pit Stops", f"{avg_stops:.1f}")
    with col4:
        most_common = strategy_df['Compound'].mode()[0] if not strategy_df['Compound'].mode().empty else 'N/A'
        st.metric("🏆 Most Used", most_common)
    
    st.divider()
    
    # Tire Strategy Visualization
    st.markdown("#### 🎨 Tire Strategy Comparison")
    
    # Compound color mapping
    compound_colors = {
        'SOFT': '#FF0000',
        'MEDIUM': '#FDB913',
        'HARD': '#FFFFFF',
        'INTERMEDIATE': '#39B54A',
        'WET': '#0067AD'
    }
    
    # Sort drivers by final position or alphabetically
    driver_order = strategy_df.groupby('Driver')['StartLap'].min().sort_values().index.tolist()
    
    # Create strategy chart
    fig = go.Figure()
    
    for driver in driver_order:
        driver_stints = strategy_df[strategy_df['Driver'] == driver].sort_values('StartLap')
        driver_name = driver_stints['DriverName'].iloc[0]
        
        for _, stint in driver_stints.iterrows():
            fig.add_trace(go.Bar(
                name=stint['Compound'],
                x=[stint['StintLength']],
                y=[driver_name],
                orientation='h',
                marker=dict(
                    color=compound_colors.get(stint['Compound'], '#808080'),
                    line=dict(color='black', width=1)
                ),
                text=stint['Compound'],
                textposition='inside',
                hovertemplate=(
                    f"<b>{driver_name}</b><br>"
                    f"Compound: {stint['Compound']}<br>"
                    f"Stint: {stint['StintNumber']}<br>"
                    f"Laps: {stint['StartLap']}-{stint['EndLap']}<br>"
                    f"Length: {stint['StintLength']} laps<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))
    
    fig.update_layout(
        title=f"Tire Strategy - {selected_location} {selected_year}",
        xaxis_title="Lap Number",
        yaxis_title="Driver",
        barmode='stack',
        height=max(600, len(driver_order) * 30),
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Strategy Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Stint Length Distribution")
        
        fig_stint = px.box(
            strategy_df,
            x='Compound',
            y='StintLength',
            color='Compound',
            color_discrete_map=compound_colors,
            labels={'StintLength': 'Stint Length (Laps)', 'Compound': 'Tire Compound'}
        )
        fig_stint.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_stint, use_container_width=True)
    
    with col2:
        st.markdown("##### 🔄 Pit Stop Distribution")
        
        stops_per_driver = strategy_df.groupby('Driver')['StintNumber'].max().reset_index()
        stops_per_driver.columns = ['Driver', 'PitStops']
        
        fig_stops = px.histogram(
            stops_per_driver,
            x='PitStops',
            nbins=10,
            labels={'PitStops': 'Number of Pit Stops', 'count': 'Number of Drivers'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig_stops.update_layout(height=400)
        st.plotly_chart(fig_stops, use_container_width=True)
    
    # Detailed Strategy Table
    st.markdown("##### 📋 Detailed Strategy Breakdown")
    
    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_drivers = st.multiselect(
            "Filter by Drivers",
            options=sorted(strategy_df['DriverName'].unique()),
            default=[]
        )
    with col2:
        selected_compounds = st.multiselect(
            "Filter by Compounds",
            options=sorted(strategy_df['Compound'].unique()),
            default=[]
        )
    
    # Apply filters
    filtered_df = strategy_df.copy()
    if selected_drivers:
        filtered_df = filtered_df[filtered_df['DriverName'].isin(selected_drivers)]
    if selected_compounds:
        filtered_df = filtered_df[filtered_df['Compound'].isin(selected_compounds)]
    
    # Display table
    display_df = filtered_df[[
        'DriverName', 'Team', 'StintNumber', 'Compound', 
        'StartLap', 'EndLap', 'StintLength'
    ]].sort_values(['DriverName', 'StintNumber'])
    
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            'DriverName': 'Driver',
            'StintNumber': 'Stint #',
            'StartLap': 'Start Lap',
            'EndLap': 'End Lap',
            'StintLength': 'Laps'
        }
    )
    
    st.divider()
    st.caption(f"📊 Tire strategy data for {selected_location} {selected_year} | Powered by FastF1")