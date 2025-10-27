import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loadDatasets import *#load_merged_dataset

@st.cache_data
def load_data():
    return load_merged_dataset()


def homeTab():
    st.markdown("### Welcome to Formula 1 Data Analytics")
    
    # Load data
    df = load_data()

    # Get available years for filtering
    available_years = sorted(df['year'].unique())
    min_year, max_year = min(available_years), max(available_years)
    
    # Year range filter at top of page
    st.markdown("#### 📅 Filter Data by Year Range")
    year_range = st.slider(
        "Select Years",
        min_value=min_year,
        max_value=max_year,
        value=(max_year - 5, max_year),
        help="Adjust to focus on specific seasons",
        label_visibility="collapsed"
    )
    
    st.caption(f"Showing data from **{year_range[0]}** to **{year_range[1]}**")
    
    st.divider()
    
    # Get available years for filtering
    available_years = sorted(df['year'].unique())
    min_year, max_year = min(available_years), max(available_years)
    
    # Filter data by year range
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Key Metrics Row
    st.markdown("### 📊 Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_races = df_filtered['raceId'].nunique()
        st.metric(
            "🏆 Total Races",
            f"{total_races:,}",
            help="Number of races in selected period"
        )
    
    with col2:
        total_drivers = df_filtered['driverId'].nunique()
        st.metric(
            "👤 Unique Drivers",
            f"{total_drivers:,}",
            help="Number of different drivers"
        )
    
    with col3:
        total_teams = df_filtered['constructorId'].nunique()
        st.metric(
            "🏢 Teams",
            f"{total_teams:,}",
            help="Number of constructor teams"
        )
    
    with col4:
        # avg_speed = df_filtered['fastestLapSpeed'].mean()
        avg_speed = pd.to_numeric(df_filtered['fastestLapSpeed'], errors='coerce').dropna().mean()
        st.metric(
            "⚡ Avg Fastest Lap",
            f"{avg_speed:.1f} km/h",
            help="Average fastest lap speed"
        )
    
    st.divider()
    
    # Two-column layout for charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        # === CHAMPIONSHIP WINNERS OVER TIME ===
        st.markdown("### 🏆 Championship Winners by Year")
        
        # Get winners (P1) by year
        winners_df = df_filtered[df_filtered['positionOrder'] == 1].groupby('year').agg({
            'forename': 'first',
            'surname': 'first',
            'code': 'first',
            'points': 'sum'
        }).reset_index()
        winners_df['Driver'] = winners_df['forename'] + ' ' + winners_df['surname']
        
        fig_winners = px.bar(
            winners_df,
            x='year',
            y='points',
            color='Driver',
            text='code',
            labels={'points': 'Total Points', 'year': 'Year'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_winners.update_traces(textposition='outside')
        fig_winners.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            height=400
        )
        st.plotly_chart(fig_winners, use_container_width=True)
    
    with col_right:
        # === TOP PERFORMING DRIVERS ===
        st.markdown("### 🥇 Top Performing Drivers")
        
        driver_stats = df_filtered.groupby(['driverId', 'forename', 'surname', 'code']).agg({
            'points': 'sum',
            'positionOrder': lambda x: (x == 1).sum(),  # Count wins
            'raceId': 'count'  # Total races
        }).reset_index()
        driver_stats.columns = ['driverId', 'forename', 'surname', 'code', 'total_points', 'wins', 'races']
        driver_stats['Driver'] = driver_stats['forename'] + ' ' + driver_stats['surname']
        driver_stats = driver_stats.sort_values('total_points', ascending=False).head(10)
        
        fig_drivers = go.Figure()
        fig_drivers.add_trace(go.Bar(
            y=driver_stats['Driver'],
            x=driver_stats['total_points'],
            orientation='h',
            text=driver_stats['wins'].apply(lambda x: f"{x} wins"),
            textposition='auto',
            marker=dict(
                color=driver_stats['total_points'],
                colorscale='Viridis',
                showscale=False
            )
        ))
        fig_drivers.update_layout(
            xaxis_title="Total Points",
            yaxis_title="",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_drivers, use_container_width=True)
    
    st.divider()
    
    # === PERFORMANCE TRENDS OVER TIME ===
    st.markdown("### 📈 Performance Trends Over Time")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Speed Evolution", "⏱️ Lap Times", "🏢 Team Performance"])
    
    with tab1:
        # Average fastest lap speed by year
        # st.dataframe(df_filtered)
        df_speed = df_filtered.copy()
        df_speed['fastestLapSpeed'] = pd.to_numeric(df_speed['fastestLapSpeed'], errors='coerce').dropna()
        speed_trend = df_speed.groupby('year').agg({
            'fastestLapSpeed': ['mean', 'max', 'min']
        }).reset_index()
        speed_trend.columns = ['year', 'avg_speed', 'max_speed', 'min_speed']
        
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(
            x=speed_trend['year'],
            y=speed_trend['avg_speed'],
            mode='lines+markers',
            name='Average',
            line=dict(color='#FF6B6B', width=3)
        ))
        fig_speed.add_trace(go.Scatter(
            x=speed_trend['year'],
            y=speed_trend['max_speed'],
            mode='lines',
            name='Maximum',
            line=dict(color='#4ECDC4', dash='dash')
        ))
        fig_speed.add_trace(go.Scatter(
            x=speed_trend['year'],
            y=speed_trend['min_speed'],
            mode='lines',
            name='Minimum',
            line=dict(color='#95E1D3', dash='dash')
        ))
        fig_speed.update_layout(
            xaxis_title="Year",
            yaxis_title="Speed (km/h)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_speed, use_container_width=True)
        
        st.info("📊 This shows how car speeds have evolved over the years, reflecting technological advancements and regulation changes.")
    
    with tab2:
        # Convert lap times to seconds for analysis
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str) or time_str == '':
                    return None
                parts = time_str.split(':')
                if len(parts) == 2:
                    mins, secs = parts
                    return float(mins) * 60 + float(secs)
                return float(time_str)
            except:
                return None
        
        df_filtered['lap_seconds'] = df_filtered['fastestLapTime'].apply(time_to_seconds)
        
        # Get average lap time by year and circuit
        lap_time_trend = df_filtered.groupby(['year', 'location']).agg({
            'lap_seconds': 'mean'
        }).reset_index()
        
        # Get top 5 most common circuits
        top_circuits = df_filtered['location'].value_counts().head(5).index.tolist()
        lap_time_trend_filtered = lap_time_trend[lap_time_trend['location'].isin(top_circuits)]
        
        fig_laps = px.line(
            lap_time_trend_filtered,
            x='year',
            y='lap_seconds',
            color='location',
            markers=True,
            labels={'lap_seconds': 'Average Lap Time (seconds)', 'year': 'Year', 'location': 'Circuit'}
        )
        fig_laps.update_layout(
            hovermode='x unified',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )
        st.plotly_chart(fig_laps, use_container_width=True)
        
        st.info("⏱️ Track lap times across different circuits to see how performance improves year over year.")
    
    with tab3:
        # Team (constructor) performance over time
        team_performance = df_filtered.groupby(['year', 'name']).agg({
            'points': 'sum',
            'positionOrder': lambda x: (x <= 3).sum()  # Podiums
        }).reset_index()
        team_performance.columns = ['year', 'team', 'points', 'podiums']
        
        # Get top 5 teams by total points
        top_teams = team_performance.groupby('team')['points'].sum().sort_values(ascending=False).head(5).index
        team_performance_filtered = team_performance[team_performance['team'].isin(top_teams)]
        
        fig_teams = px.area(
            team_performance_filtered,
            x='year',
            y='points',
            color='team',
            labels={'points': 'Total Points', 'year': 'Year', 'team': 'Team'}
        )
        fig_teams.update_layout(
            hovermode='x unified',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )
        st.plotly_chart(fig_teams, use_container_width=True)
        
        st.info("🏢 Compare how different constructor teams have performed over the selected time period.")
    
    st.divider()
    
    # === RECENT SEASON SUMMARY ===
    st.markdown(f"### 🏁 {max_year} Season Highlights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most recent season stats
        latest_season = df[df['year'] == max_year]
        
        # Top 5 drivers in latest season
        st.markdown("#### Top 5 Drivers")
        top_drivers_latest = latest_season.groupby(['forename', 'surname', 'code']).agg({
            'points': 'sum',
            'positionOrder': lambda x: (x == 1).sum()
        }).reset_index()
        top_drivers_latest.columns = ['forename', 'surname', 'code', 'points', 'wins']
        top_drivers_latest['Driver'] = top_drivers_latest['forename'] + ' ' + top_drivers_latest['surname']
        top_drivers_latest = top_drivers_latest.sort_values('points', ascending=False).head(5)
        
        for idx, row in top_drivers_latest.iterrows():
            with st.container(border=True, vertical_alignment='top'):
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.markdown(f"**{row['Driver']}** ({row['code']})")
                with col_b:
                    st.metric("Points", f"{row['points']:.0f}")
                with col_c:
                    st.metric("Wins", f"{row['wins']:.0f}")
    
    with col2:
        # Most recent races
        st.markdown("#### Recent Races")
        recent_races = df[df['year'] == max_year].groupby(['raceId', 'name_race', 'location']).size().reset_index()
        recent_races = recent_races.sort_values('raceId', ascending=False).head(5)
        for idx, row in recent_races.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['name_race']}**")
                st.caption(f"📍 {row['location']}")
    
    st.divider()
    st.caption(f"📊 Data spans from {min_year} to {max_year} | Total of {len(df):,} race results")