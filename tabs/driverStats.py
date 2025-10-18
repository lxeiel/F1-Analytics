import streamlit as st
import pandas as pd
from utils.loadDatasets import load_merged_dateset

# ============================================================
# Load dataset once and cache
# ============================================================
@st.cache_data
def get_overall_data():
    return load_merged_dateset()

# ============================================================
# Driver Stats Tab
# ============================================================
def driverStatsTab():
    st.header("👨‍✈️ Driver Statistics")

    # Load dataset
    df_overall = get_overall_data()

    # Fill missing names to avoid issues
    df_overall['forename'] = df_overall['forename'].fillna('')
    df_overall['surname'] = df_overall['surname'].fillna('')
    df_overall['Driver'] = (df_overall['forename'] + " " + df_overall['surname']).str.strip()

    # Remove rows without driver code or empty names
    df_overall = df_overall[df_overall['code'].notna() & (df_overall['Driver'] != '')]

    # Deduplicate by code, keep the most recent entry (latest year)
    df_drivers_unique = df_overall.sort_values('year', ascending=False) \
                                  .drop_duplicates(subset=['code'])

    # Mappings
    driver_code_to_name = dict(zip(df_drivers_unique['code'], df_drivers_unique['Driver']))
    driver_name_to_code = dict(zip(df_drivers_unique['Driver'], df_drivers_unique['code']))
    driver_name_to_url  = dict(zip(df_drivers_unique['Driver'], df_drivers_unique['url']))

    # --- Driver selection (unsorted) ---
    driver_names = df_drivers_unique['Driver'].tolist()  # preserve dataset order
    if "selected_driver" not in st.session_state or st.session_state.selected_driver not in driver_names:
        st.session_state.selected_driver = driver_names[0]

    selected_driver_name = st.selectbox(
        "Select Driver",
        driver_names,
        index=driver_names.index(st.session_state.selected_driver)
    )
    selected_driver_code = driver_name_to_code[selected_driver_name]
    st.session_state.selected_driver = selected_driver_name

    # Filter dataset by driver code
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
    # fastest_lap_time = df_driver['fastestLapTime'].min() if 'fastestLapTime' in df_driver.columns else None
    # fastest_lap_speed = df_driver['fastestLapSpeed'].max() if 'fastestLapSpeed' in df_driver.columns else None
    constructors = df_driver['name'].dropna().unique()
    driver_url = driver_name_to_url.get(selected_driver_name, None)

    # --- Display stats and URL side by side ---
    col_left, col_right = st.columns([2, 1])  # left column wider
    with col_left:
        st.subheader(f"Stats for {selected_driver_name} ({selected_driver_code})")
        st.write(f"**Nationality:** {df_driver.iloc[0]['nationality']}")
        st.write(f"**Total Races:** {total_races}")
        st.write(f"**Total Points:** {total_points}")
        st.write(f"**Total Wins:** {total_wins}")
        st.write(f"**Total Podiums:** {total_podiums}")
        st.write(f"**Total DNFs:** {total_dnfs}")
        # if fastest_lap_time:
        #     st.write(f"**Best Fastest Lap Time:** {fastest_lap_time}")
        # if fastest_lap_speed:
        #     st.write(f"**Max Fastest Lap Speed:** {fastest_lap_speed} km/h")
        if len(constructors) > 0:
            st.write(f"**Constructors Driven For:** {', '.join(constructors)}")

    with col_right:
        st.markdown("### 📖 Wikipedia")
        if driver_url:
            st.components.v1.iframe(driver_url, height=400, scrolling=True)
        else:
            st.write("No URL available")
