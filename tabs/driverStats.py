import streamlit as st
import pandas as pd
from utils.loadDatasets import load_merged_dataset
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def get_overall_data():
    return load_merged_dataset()

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

    # Driver selection
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

    # Filter driver
    df_driver = df_overall[df_overall['code'] == selected_driver_code]
    if df_driver.empty:
        st.warning("No data found for this driver.")
        return

    # Compute stats
    total_races = len(df_driver)
    total_points = df_driver['points'].sum()
    total_wins = (df_driver['positionOrder'] == 1).sum()
    total_podiums = (df_driver['positionOrder'] <= 3).sum()
    total_dnfs = (df_driver['statusId'] != 1).sum() if 'statusId' in df_driver.columns else 0
    constructors = df_driver['name'].dropna().unique()
    nationality = df_driver.iloc[0]['nationality']
    driver_url = driver_name_to_url.get(selected_driver_name, None)

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

    st.markdown("### 📚 Story-driven Deep Dives")
    dd1, dd2, dd3 = st.tabs([
        "Circuit Sweet Spots",
        "Race Craft",
        "Teammate Duel",
    ])

    # Circuit Sweet Spots
    with dd1:
        st.caption("Where does this driver thrive? Points by circuit across career.")
        circuit_col = 'name_circuits' if 'name_circuits' in df_driver.columns else ('circuitRef' if 'circuitRef' in df_driver.columns else 'location')
        drv_circ = df_driver.groupby([circuit_col, 'year'], as_index=False)['points'].sum()
        if drv_circ.empty:
            st.info("No circuit data available for this driver.")
        else:
            pivot = drv_circ.pivot(index=circuit_col, columns='year', values='points').fillna(0)
            # sort circuits by total points
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
            fig_hm = px.imshow(
                pivot,
                labels=dict(x="Year", y="Circuit", color="Points"),
                color_continuous_scale='Magma'
            )
            fig_hm.update_layout(height=520, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_hm, use_container_width=True)

    # Race Craft
    with dd2:
        st.caption("How starting position translates to finishing bucket for this driver.")
        # bucketize
        dfc = df_driver.copy()
        dfc['grid'] = pd.to_numeric(dfc.get('grid'), errors='coerce')
        dfc['positionOrder'] = pd.to_numeric(dfc['positionOrder'], errors='coerce')
        dfc = dfc.dropna(subset=['grid', 'positionOrder'])
        def g_bucket(g):
            if g == 1:
                return 'Start: P1'
            if 2 <= g <= 5:
                return 'Start: P2-5'
            if 6 <= g <= 10:
                return 'Start: P6-10'
            return 'Start: P11+'
        def f_bucket(p):
            if p == 1:
                return 'Finish: Win'
            if 2 <= p <= 3:
                return 'Finish: Podium'
            if 4 <= p <= 10:
                return 'Finish: Points'
            return 'Finish: P11+'
        dfc['gb'] = dfc['grid'].apply(g_bucket)
        dfc['fb'] = dfc['positionOrder'].apply(f_bucket)
        flow = dfc.groupby(['gb', 'fb'], as_index=False).size().rename(columns={'size': 'count'})
        labels = sorted(dfc['gb'].unique().tolist()) + sorted(dfc['fb'].unique().tolist())
        idx = {l:i for i,l in enumerate(labels)}
        src, tgt, val = [], [], []
        for _, r in flow.iterrows():
            src.append(idx[r['gb']])
            tgt.append(idx[r['fb']])
            val.append(int(r['count']))
        sankey = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=15),
            link=dict(source=src, target=tgt, value=val)
        )])
        sankey.update_layout(height=520, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(sankey, use_container_width=True)

    # Teammate Duel
    with dd3:
        st.caption("Season-by-season points edge vs teammates (same constructor).")
        d = df_overall.copy()
        d['Driver'] = (d['forename'].fillna('') + ' ' + d['surname'].fillna('')).str.strip()
        # per season, constructor, driver points
        per = d.groupby(['year', 'name', 'Driver'], as_index=False)['points'].sum()
        # for selected driver
        my = per[per['Driver'] == selected_driver_name]
        if my.empty:
            st.info("No season points available.")
        else:
            # teammate avg points per same year & team (exclude driver)
            tm = per.merge(my[['year','name']].drop_duplicates(), on=['year','name'], how='inner')
            def teammate_avg(g):
                dr = selected_driver_name
                vals = g.loc[g['Driver'] != dr, 'points']
                return vals.mean() if not vals.empty else None
            tavg = tm.groupby(['year','name'], as_index=False).apply(lambda g: pd.Series({'tm_avg': teammate_avg(g)}))
            me = my.groupby(['year','name'], as_index=False)['points'].sum().rename(columns={'points':'me_pts'})
            combined = me.merge(tavg, on=['year','name'], how='left')
            combined['diff'] = combined['me_pts'] - combined['tm_avg']
            # choose representative team if multiple (e.g., mid-season switch): sum by year
            year_sum = combined.groupby('year', as_index=False).agg({'diff':'sum','me_pts':'sum','tm_avg':'mean'})
            year_sum = year_sum.sort_values('year')
            fig_duel = px.bar(year_sum, x='year', y='diff', labels={'diff': 'Points vs Teammate (Δ)'})
            fig_duel.add_hline(y=0, line_dash='dash', line_color='#888')
            fig_duel.update_layout(height=420, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_duel, use_container_width=True)

