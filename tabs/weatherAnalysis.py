
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import fastf1
from utils.loadDatasets import load_merged_dataset

@st.cache_resource
def create_fastf1_session(year, location):
    import os
    cache_dir = "./f1_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, location, 'R')
    session.load()
    return session

@st.cache_data
def get_weather_data(year, location):
    """Extract weather data from FastF1 session"""
    try:
        session = create_fastf1_session(year, location)
        weather = session.weather_data
        
        if weather is None or weather.empty:
            return None
            
        # Convert to DataFrame and process
        weather_df = weather.copy()
        
        # Convert timedelta to seconds for plotting
        if 'Time' in weather_df.columns:
            # Time is a timedelta, convert to total seconds from session start
            weather_df['TimeSeconds'] = weather_df['Time'].dt.total_seconds()
            # Create a readable time label (minutes:seconds)
            weather_df['TimeLabel'] = weather_df['TimeSeconds'].apply(
                lambda x: f"{int(x//60)}:{int(x%60):02d}"
            )
        
        return weather_df
    except Exception as e:
        st.error(f"Error loading weather data: {str(e)}")
        return None

@st.cache_data
def get_lap_times_with_weather(year, location):
    """Get lap times correlated with weather conditions"""
    try:
        session = create_fastf1_session(year, location)
        laps = session.laps
        weather = session.weather_data
        
        if laps is None or laps.empty or weather is None or weather.empty:
            return None
        
        # Merge laps with weather data
        laps_df = laps.copy()
        laps_df['LapTime_seconds'] = laps_df['LapTime'].dt.total_seconds()
        
        # Convert Time to seconds for both dataframes
        laps_df['TimeSeconds'] = laps_df['Time'].dt.total_seconds()
        weather_df = weather.copy()
        weather_df['TimeSeconds'] = weather_df['Time'].dt.total_seconds()
        
        # Get driver full names
        drivers_info = session.drivers
        driver_names = {}
        for driver_num in drivers_info:
            driver_info = session.get_driver(driver_num)
            driver_code = driver_info['Abbreviation']
            driver_full_name = f"{driver_info['FirstName']} {driver_info['LastName']}"
            driver_names[driver_code] = driver_full_name
        
        # Get weather conditions for each lap
        laps_with_weather = []
        for idx, lap in laps_df.iterrows():
            lap_time_seconds = lap['TimeSeconds']
            # Find closest weather reading by comparing seconds
            time_diffs = (weather_df['TimeSeconds'] - lap_time_seconds).abs()
            closest_idx = time_diffs.idxmin()
            weather_at_lap = weather_df.loc[closest_idx]
            
            driver_code = lap['Driver']
            driver_name = driver_names.get(driver_code, driver_code)
            
            lap_data = {
                'Driver': driver_code,
                'DriverName': driver_name,
                'LapNumber': lap['LapNumber'],
                'LapTime': lap['LapTime_seconds'],
                'AirTemp': weather_at_lap['AirTemp'],
                'TrackTemp': weather_at_lap['TrackTemp'],
                'Humidity': weather_at_lap['Humidity'],
                'Pressure': weather_at_lap['Pressure'],
                'Rainfall': weather_at_lap['Rainfall'],
                'WindSpeed': weather_at_lap['WindSpeed']
            }
            laps_with_weather.append(lap_data)
        
        return pd.DataFrame(laps_with_weather)
    except Exception as e:
        st.error(f"Error processing lap and weather data: {str(e)}")
        return None

@st.cache_data
def get_tire_strategy_with_weather(year, location):
    """Get tire compound usage correlated with weather conditions"""
    try:
        session = create_fastf1_session(year, location)
        laps = session.laps
        weather = session.weather_data
        
        if laps is None or laps.empty or weather is None or weather.empty:
            return None
        
        # Get driver full names
        drivers_info = session.drivers
        driver_names = {}
        for driver_num in drivers_info:
            driver_info = session.get_driver(driver_num)
            driver_code = driver_info['Abbreviation']
            driver_full_name = f"{driver_info['FirstName']} {driver_info['LastName']}"
            driver_names[driver_code] = driver_full_name
        
        # Process laps with tire and weather data
        laps_df = laps.copy()
        laps_df['TimeSeconds'] = laps_df['Time'].dt.total_seconds()
        weather_df = weather.copy()
        weather_df['TimeSeconds'] = weather_df['Time'].dt.total_seconds()
        
        tire_weather_data = []
        for idx, lap in laps_df.iterrows():
            if pd.isna(lap['Compound']):
                continue
                
            lap_time_seconds = lap['TimeSeconds']
            time_diffs = (weather_df['TimeSeconds'] - lap_time_seconds).abs()
            closest_idx = time_diffs.idxmin()
            weather_at_lap = weather_df.loc[closest_idx]
            
            driver_code = lap['Driver']
            driver_name = driver_names.get(driver_code, driver_code)
            
            tire_data = {
                'Driver': driver_code,
                'DriverName': driver_name,
                'LapNumber': lap['LapNumber'],
                'Compound': lap['Compound'],
                'TyreLife': lap['TyreLife'],
                'TrackTemp': weather_at_lap['TrackTemp'],
                'AirTemp': weather_at_lap['AirTemp'],
                'Humidity': weather_at_lap['Humidity'],
                'Rainfall': weather_at_lap['Rainfall']
            }
            tire_weather_data.append(tire_data)
        
        return pd.DataFrame(tire_weather_data)
    except Exception as e:
        st.error(f"Error processing tire strategy data: {str(e)}")
        return None

def weatherAnalysisTab():
    st.markdown("### 🌤️ Weather Impact on Performance")
    st.markdown("Analyze how weather conditions affect race performance using FastF1 telemetry data.")
    
    # Load overall data for race selection
    df = load_merged_dataset()
    
    # Year and location selection
    col1, col2 = st.columns(2)
    
    with col1:
        available_years = sorted(df['year'].unique(), reverse=True)
        # Filter to recent years where FastF1 data is more reliable
        recent_years = [y for y in available_years if y >= 2018]
        selected_year = st.selectbox("Select Year", recent_years, index=0)
    
    with col2:
        races_in_year = df[df['year'] == selected_year]
        locations = sorted(races_in_year['location'].unique())
        selected_location = st.selectbox("Select Race", locations)
    
    st.divider()
    
    # Load weather data
    with st.spinner("Loading weather data..."):
        weather_df = get_weather_data(selected_year, selected_location)
    
    if weather_df is None:
        st.warning("⚠️ Weather data not available for this race. Try selecting a different race from 2018 onwards.")
        return
    
    # Weather Overview and Trends - Side by Side
    col_overview, col_trends = st.columns([1, 2])
    
    with col_overview:
        st.markdown("#### 🌤️ Weather Overview")
        
        # Key weather metrics in compact format
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🌡️ Air", f"{weather_df['AirTemp'].mean():.1f}°C")
            st.metric("💧 Humidity", f"{weather_df['Humidity'].mean():.1f}%")
            st.metric("🌧️ Rain", "Yes" if weather_df['Rainfall'].any() else "No")
        with col_b:
            st.metric("🏁 Track Temperature", f"{weather_df['TrackTemp'].mean():.1f}°C")
            st.metric("💨 Wind", f"{weather_df['WindSpeed'].mean():.1f} m/s")
        
    
    with col_trends:
        st.markdown("#### 📈 Weather Trends During Race")
        
        tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "💨 Wind & Pressure", "💧 Humidity & Rain"])
        
        with tab1:
            fig_temp = go.Figure()
            
            fig_temp.add_trace(go.Scatter(
                x=weather_df['Time'],
                y=weather_df['AirTemp'],
                mode='lines',
                name='Air Temperature',
                line=dict(color='#FF6B6B', width=2)
            ))
            
            fig_temp.add_trace(go.Scatter(
                x=weather_df['Time'],
                y=weather_df['TrackTemp'],
                mode='lines',
                name='Track Temperature',
                line=dict(color='#4ECDC4', width=2)
            ))
            
            fig_temp.update_layout(
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                hovermode='x unified',
                height=350,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab2:
            fig_wind = go.Figure()
            
            fig_wind.add_trace(go.Scatter(
                x=weather_df['Time'],
                y=weather_df['WindSpeed'],
                mode='lines',
                name='Wind Speed',
                line=dict(color='#95E1D3', width=2),
                yaxis='y'
            ))
            
            fig_wind.add_trace(go.Scatter(
                x=weather_df['Time'],
                y=weather_df['Pressure'],
                mode='lines',
                name='Pressure',
                line=dict(color='#FFD93D', width=2),
                yaxis='y2'
            ))
            
            fig_wind.update_layout(
                xaxis_title="Time",
                yaxis=dict(title="Wind Speed (m/s)", side='left'),
                yaxis2=dict(title="Pressure (mbar)", side='right', overlaying='y'),
                hovermode='x unified',
                height=350,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_wind, use_container_width=True)
        
        with tab3:
            fig_rain = go.Figure()
            
            fig_rain.add_trace(go.Scatter(
                x=weather_df['Time'],
                y=weather_df['Humidity'],
                mode='lines',
                name='Humidity',
                line=dict(color='#6C5CE7', width=2),
                fill='tozeroy'
            ))
            
            # Add rainfall indicator
            if weather_df['Rainfall'].any():
                rain_times = weather_df[weather_df['Rainfall'] == True]['Time']
                for rain_time in rain_times:
                    fig_rain.add_vline(
                        x=rain_time,
                        line=dict(color='blue', width=1, dash='dash'),
                        annotation_text="Rain"
                    )
            
            fig_rain.update_layout(
                xaxis_title="Time",
                yaxis_title="Humidity (%)",
                hovermode='x unified',
                height=350,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_rain, use_container_width=True)
    
    st.divider()
    
    # Lap times vs weather correlation
    st.markdown("#### 🏎️ Driver Performance vs Weather Conditions")
    
    with st.spinner("Analyzing lap times with weather conditions..."):
        laps_weather_df = get_lap_times_with_weather(selected_year, selected_location)
    
    if laps_weather_df is not None and not laps_weather_df.empty:
        # Create a mapping of driver names to codes for selection
        driver_mapping = laps_weather_df[['DriverName', 'Driver']].drop_duplicates()
        driver_mapping = driver_mapping.sort_values('DriverName')
        driver_name_to_code = dict(zip(driver_mapping['DriverName'], driver_mapping['Driver']))
        
        # Select drivers to analyze
        available_driver_names = sorted(driver_mapping['DriverName'].unique())
        selected_driver_names = st.multiselect(
            "Select Drivers to Compare",
            available_driver_names,
            default=available_driver_names[:3] if len(available_driver_names) >= 3 else available_driver_names
        )
        
        if selected_driver_names:
            # Convert selected names back to codes for filtering
            selected_driver_codes = [driver_name_to_code[name] for name in selected_driver_names]
            filtered_laps = laps_weather_df[laps_weather_df['Driver'].isin(selected_driver_codes)]
            
            # Use DriverName for display in charts
            filtered_laps_display = filtered_laps.copy()
            
            # Correlation analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Lap Time vs Track Temperature")
                fig_temp_corr = px.scatter(
                    filtered_laps_display,
                    x='TrackTemp',
                    y='LapTime',
                    color='DriverName',
                    trendline='ols',
                    labels={
                        'TrackTemp': 'Track Temperature (°C)', 
                        'LapTime': 'Lap Time (seconds)',
                        'DriverName': 'Driver'
                    }
                )
                fig_temp_corr.update_layout(height=400)
                st.plotly_chart(fig_temp_corr, use_container_width=True)
            
            with col2:
                st.markdown("##### Lap Time vs Humidity")
                fig_humidity_corr = px.scatter(
                    filtered_laps_display,
                    x='Humidity',
                    y='LapTime',
                    color='DriverName',
                    trendline='ols',
                    labels={
                        'Humidity': 'Humidity (%)', 
                        'LapTime': 'Lap Time (seconds)',
                        'DriverName': 'Driver'
                    }
                )
                fig_humidity_corr.update_layout(height=400)
                st.plotly_chart(fig_humidity_corr, use_container_width=True)
            
            # Statistical summary
            st.markdown("##### 📊 Weather Impact Statistics")
            
            # Calculate correlations
            correlations = filtered_laps[['LapTime', 'AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']].corr()['LapTime'].drop('LapTime')
            
            corr_df = pd.DataFrame({
                'Weather Factor': correlations.index,
                'Correlation with Lap Time': correlations.values
            }).sort_values('Correlation with Lap Time', key=abs, ascending=False)
            
            st.dataframe(corr_df, hide_index=True, use_container_width=True)
            
            st.caption("💡 Positive correlation means higher values lead to slower lap times. Negative correlation means higher values lead to faster lap times.")
    else:
        st.warning("⚠️ Unable to load lap time data for this race.")
    
    st.divider()
    
    # Tire Strategy Analysis
    st.markdown("##### 🔍 Tire Strategy vs Conditions")
    
    with st.spinner("Analyzing tire strategies..."):
        tire_weather_df = get_tire_strategy_with_weather(selected_year, selected_location)
    
    if tire_weather_df is not None and not tire_weather_df.empty:
            
        # Calculate key metrics
        avg_track_temp = tire_weather_df['TrackTemp'].mean()
        most_used_compound = tire_weather_df['Compound'].mode()[0] if not tire_weather_df['Compound'].mode().empty else 'N/A'
        rain_laps = tire_weather_df['Rainfall'].sum()
        total_laps = len(tire_weather_df)
        
        metric_row1_col1, metric_row1_col2, metric_row1_col3, metric_row1_col4 = st.columns(4)
        with metric_row1_col1:
            st.metric("🏆 Most Used Compound", most_used_compound)
        with metric_row1_col2:
            st.metric("🌡️ Avg Track Temp", f"{avg_track_temp:.1f}°C")
        with metric_row1_col3:
            rain_percentage = (rain_laps / total_laps * 100) if total_laps > 0 else 0
            st.metric("🌧️ Rain Affected Laps", f"{rain_percentage:.1f}%")
        with metric_row1_col4:
            avg_humidity = tire_weather_df['Humidity'].mean()
            st.metric("💧 Avg Humidity", f"{avg_humidity:.1f}%")
        
        st.markdown("---")
        
        # Tire compound distribution by weather
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Tire Compound Usage by Track Temperature")
            
            # Group by compound and calculate average track temp
            compound_temp = tire_weather_df.groupby('Compound').agg({
                'TrackTemp': 'mean',
                'LapNumber': 'count'
            }).reset_index()
            compound_temp.columns = ['Compound', 'Avg Track Temp (°C)', 'Laps Used']
            
            # Create color mapping for compounds
            compound_colors = {
                'SOFT': '#FF0000',
                'MEDIUM': '#FDB913',
                'HARD': '#FFFFFF',
                'INTERMEDIATE': '#39B54A',
                'WET': '#0067AD'
            }
            
            fig_compound = px.bar(
                compound_temp,
                x='Compound',
                y='Laps Used',
                color='Compound',
                color_discrete_map=compound_colors,
                text='Avg Track Temp (°C)',
                labels={'Laps Used': 'Number of Laps'}
            )
            fig_compound.update_traces(texttemplate='%{text:.1f}°C', textposition='outside')
            fig_compound.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_compound, use_container_width=True)
        
        with col2:
            st.markdown("##### Tire Performance by Weather")
            
            # Check if rainfall occurred
            if tire_weather_df['Rainfall'].any():
                rain_condition = tire_weather_df.groupby('Rainfall').agg({
                    'Compound': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A',
                    'LapNumber': 'count'
                }).reset_index()
                rain_condition['Condition'] = rain_condition['Rainfall'].map({True: 'Wet', False: 'Dry'})
                
                fig_rain_tire = px.pie(
                    rain_condition,
                    values='LapNumber',
                    names='Condition',
                    title='Laps by Weather Condition',
                    color='Condition',
                    color_discrete_map={'Dry': '#FDB913', 'Wet': '#0067AD'}
                )
                fig_rain_tire.update_layout(height=400)
                st.plotly_chart(fig_rain_tire, use_container_width=True)
            else:
                # Show compound distribution by humidity
                fig_humidity_tire = px.scatter(
                    tire_weather_df,
                    x='Humidity',
                    y='TyreLife',
                    color='Compound',
                    color_discrete_map=compound_colors,
                    labels={'Humidity': 'Humidity (%)', 'TyreLife': 'Tire Life (Laps)'},
                    title='Tire Life vs Humidity'
                )
                fig_humidity_tire.update_layout(height=400)
                st.plotly_chart(fig_humidity_tire, use_container_width=True)
        
        # Detailed tire usage table
        st.markdown("##### 📊 Detailed Tire Usage Statistics")
        tire_stats = tire_weather_df.groupby('Compound').agg({
            'LapNumber': 'count',
            'TyreLife': 'mean',
            'TrackTemp': 'mean',
            'AirTemp': 'mean',
            'Humidity': 'mean'
        }).reset_index()
        
        tire_stats.columns = [
            'Compound',
            'Total Laps',
            'Avg Tire Life',
            'Avg Track Temp (°C)',
            'Avg Air Temp (°C)',
            'Avg Humidity (%)'
        ]
        
        tire_stats = tire_stats.round(2)
        st.dataframe(tire_stats, hide_index=True, use_container_width=True)
    else:
        st.warning("⚠️ Unable to load tire strategy data for this race.")
        
    st.divider()
    st.caption(f"📊 Weather data for {selected_location} {selected_year} | Powered by FastF1")
