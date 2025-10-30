import streamlit as st
import pandas as pd
from utils.loadDatasets import load_merged_dataset
import plotly.express as px

# ============================================================
# Load dataset once and cache
# ============================================================
@st.cache_data
def get_overall_data():
    return load_merged_dataset()

# ============================================================
# Driver Stats Tab
# ============================================================
def driverStatsTab():
    st.header("👨‍✈️ Driver Statistics")

    # Load dataset
    df_overall = get_overall_data()

    # Clean names
    df_overall['forename'] = df_overall['forename'].fillna('')
    df_overall['surname'] = df_overall['surname'].fillna('')
    df_overall['Driver'] = (df_overall['forename'] + " " + df_overall['surname']).str.strip()

    # Remove invalid rows
    df_overall = df_overall[df_overall['code'].notna() & (df_overall['Driver'] != '')]

    # Deduplicate by latest year
    df_drivers_unique = df_overall.sort_values('year', ascending=False) \
                                  .drop_duplicates(subset=['code'])

    # Mapping helpers
    driver_code_to_name = dict(zip(df_drivers_unique['code'], df_drivers_unique['Driver']))
    driver_name_to_code = dict(zip(df_drivers_unique['Driver'], df_drivers_unique['code']))
    driver_name_to_url  = dict(zip(df_drivers_unique['Driver'], df_drivers_unique['url']))

    # --- Driver selection ---
    driver_names = df_drivers_unique['Driver'].tolist()
    if "selected_driver" not in st.session_state or st.session_state.selected_driver not in driver_names:
        st.session_state.selected_driver = driver_names[0]

    selected_driver_name = st.selectbox(
        "Select Driver",
        driver_names,
        index=driver_names.index(st.session_state.selected_driver)
    )
    selected_driver_code = driver_name_to_code[selected_driver_name]
    st.session_state.selected_driver = selected_driver_name

    # --- Filter driver ---
    df_driver = df_overall[df_overall['code'] == selected_driver_code]
    if df_driver.empty:
        st.warning("No data found for this driver.")
        return

    # --- Compute stats ---
    total_races = len(df_driver)
    total_points = df_driver['points'].sum()
    total_wins = (df_driver['positionOrder'] == 1).sum()
    total_podiums = (df_driver['positionOrder'] <= 3).sum()
    total_dnfs = (df_driver['statusId'] != 1).sum() if 'statusId' in df_driver.columns else 0
    constructors = df_driver['name'].dropna().unique()
    nationality = df_driver.iloc[0]['nationality']
    driver_url = driver_name_to_url.get(selected_driver_name, None)

    # ============================================================
    # Layout
    # ============================================================
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader(f"🏎️ {selected_driver_name} ({selected_driver_code})")

        # Use columns for a clean 2x3 grid layout
        st.markdown("---")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col4, stat_col5, stat_col6 = st.columns(3)

        stat_col1.metric("Total Races", total_races)
        stat_col2.metric("Total Points", int(total_points))
        stat_col3.metric("Total Wins", total_wins)

        stat_col4.metric("Podiums", total_podiums)
        stat_col5.metric("DNFs", total_dnfs)
        stat_col6.metric("Nationality", nationality)

        st.markdown("---")
        if len(constructors) > 0:
            st.markdown(
                f"**Constructors Driven For:**<br>"
                f"<span style='color:#00BFFF'>{', '.join(constructors)}</span>",
                unsafe_allow_html=True
            )

    with col_right:
        st.markdown("### 📖 Wikipedia")
        if driver_url:
            st.components.v1.iframe(driver_url, height=400, scrolling=True)
        else:
            st.info("No URL available for this driver.")

    # --- NEW: Performance Trends & Season Summary ---
    st.divider()
    st.markdown("### 📊 Performance Trends & Season Summary")

    # Prepare per-season metrics
    df_season = df_driver.groupby('year').agg(
        points=('points', 'sum'),
        wins=('positionOrder', lambda x: (x == 1).sum()),
        podiums=('positionOrder', lambda x: x.le(3).sum()),
        races=('raceId', 'nunique')
    ).reset_index().sort_values('year')

    # Total races per year
    fig_season = px.line(
        df_season, x='year', y='points', markers=True, 
        labels={'points': 'Points', 'year': 'Season'},
        title=f"{selected_driver_name} – Points Per Season"
    )
    st.plotly_chart(fig_season, use_container_width=True)

    # Wins & podiums stacked bar
    fig_wins = px.bar(
        df_season, x='year', y=['wins', 'podiums'], 
        labels={'value': 'Count', 'year': 'Season', 'variable': 'Metric'},
        title=f"{selected_driver_name} – Wins & Podiums Per Season"
    )
    st.plotly_chart(fig_wins, use_container_width=True)

