import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loadDatasets import load_merged_dataset


# ============================================================
#  DATA LOADING HELPERS
# ============================================================

@st.cache_data
def get_overall_data() -> pd.DataFrame:
    df = load_merged_dataset()
    # Ensure numeric types for key columns
    for col in ['points', 'positionOrder', 'grid', 'year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Build Driver display name if not present
    if 'Driver' not in df.columns:
        df['forename'] = df.get('forename', '').fillna('')
        df['surname'] = df.get('surname', '').fillna('')
        df['Driver'] = (df['forename'].astype(str) + ' ' + df['surname'].astype(str)).str.strip()
    return df


@st.cache_data
def get_year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    years = pd.to_numeric(df['year'], errors='coerce').dropna()
    return int(years.min()), int(years.max())


@st.cache_data
def summarize_driver_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['points'] = pd.to_numeric(d['points'], errors='coerce')
    d['positionOrder'] = pd.to_numeric(d['positionOrder'], errors='coerce')
    d['grid'] = pd.to_numeric(d.get('grid'), errors='coerce') if 'grid' in d.columns else pd.NA

    agg = (
        d.groupby('Driver').agg(
            total_points=('points', 'sum'),
            wins=('positionOrder', lambda s: (s == 1).sum()),
            podiums=('positionOrder', lambda s: (s <= 3).sum()),
            avg_finish=('positionOrder', 'mean'),
            top10=('positionOrder', lambda s: (s <= 10).sum()),
            poles=('grid', lambda s: (s == 1).sum()),
        )
        .reset_index()
    )

    agg['avg_finish'] = pd.to_numeric(agg['avg_finish'], errors='coerce').round(2)
    return agg.sort_values('total_points', ascending=False)


@st.cache_data
def points_by_season(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['points'] = pd.to_numeric(d['points'], errors='coerce')
    pts = (
        d.groupby(['year', 'Driver'])['points']
        .sum()
        .reset_index()
        .sort_values(['Driver', 'year'])
    )
    return pts


@st.cache_data
def standings_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-season driver standings rank based on total points."""
    pts = points_by_season(df)
    if pts.empty:
        return pts.assign(rank=pd.Series(dtype=float))
    pts['rank'] = pts.groupby('year')['points'].rank(ascending=False, method='dense')
    return pts


@st.cache_data
def finish_distribution(df: pd.DataFrame, as_percent: bool = False) -> pd.DataFrame:
    """Stacked finish distribution per driver: P1, P2-3, P4-10, P11+."""
    d = df.copy()
    d['positionOrder'] = pd.to_numeric(d['positionOrder'], errors='coerce')

    def bucket(pos):
        if pd.isna(pos):
            return 'P11+'
        if pos == 1:
            return 'P1'
        if 2 <= pos <= 3:
            return 'P2-3'
        if 4 <= pos <= 10:
            return 'P4-10'
        return 'P11+'

    d['bucket'] = d['positionOrder'].apply(bucket)
    counts = (
        d.groupby(['Driver', 'bucket'])
        .size()
        .reset_index(name='count')
    )
    bucket_order = ['P1', 'P2-3', 'P4-10', 'P11+']
    counts['bucket'] = pd.Categorical(counts['bucket'], categories=bucket_order, ordered=True)

    if not as_percent:
        return counts.sort_values(['Driver', 'bucket'])

    totals = counts.groupby('Driver')['count'].transform('sum')
    counts['percent'] = (counts['count'] / totals) * 100.0
    return counts.sort_values(['Driver', 'bucket'])


@st.cache_data
def yoy_points_change(df: pd.DataFrame) -> pd.DataFrame:
    pts = points_by_season(df)
    if pts.empty:
        return pts.assign(points_prev=pd.Series(dtype=float), yoy=pd.Series(dtype=float))
    pts['points_prev'] = pts.groupby('Driver')['points'].shift(1)
    pts['yoy'] = pts['points'] - pts['points_prev']
    return pts.dropna(subset=['points_prev'])


# ============================================================
#  MAIN DRIVER COMPARISON TAB
# ============================================================

def driverComparisonTab():
    st.header("👥 Driver Comparison")

    df = get_overall_data()
    y_min, y_max = get_year_bounds(df)

    st.markdown("#### 📅 Filter by Year Range")
    year_range = st.slider(
        "Select Years",
        min_value=y_min,
        max_value=y_max,
        value=(max(y_min, y_max - 5), y_max),
        help="Adjust to focus on specific seasons",
        label_visibility="collapsed",
        key="drivercomp_year_range",
    )
    st.caption(f"Showing data from {year_range[0]} to {year_range[1]}")

    df_range = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])].copy()

    # Driver multiselect (default: top 8 by points in range)
    driver_points = df_range.groupby('Driver')['points'].sum().sort_values(ascending=False)
    all_drivers = driver_points.index.tolist()
    default_drivers = all_drivers[:8]
    selected_drivers = st.multiselect(
        "🧑‍✈️ Select Drivers",
        options=all_drivers,
        default=default_drivers,
        help="Pick one or more drivers to compare",
    )

    if not selected_drivers:
        st.warning("Please select at least one driver to view stats.")
        return

    df_sel = df_range[df_range['Driver'].isin(selected_drivers)].copy()

    # --- Merge status locally ---
    status = pd.read_csv('Dataset/status.csv')
    if 'status' not in df_sel.columns:
        df_sel = df_sel.merge(status[['statusId', 'status']], on='statusId', how='left')

    # In-tab navigation
    sub_overview, sub_h2h, sub_adv = st.tabs(["Overview", "Head-to-Head", "Advanced"])

    # =====================
    # Overview
    # =====================
    stats = summarize_driver_stats(df_sel)
    with sub_overview:
        total_points = int(stats['total_points'].sum())
        total_wins = int(stats['wins'].sum())
        total_podiums = int(stats['podiums'].sum())
        avg_finish_overall = stats['avg_finish'].mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏁 Total Points (selected)", f"{total_points:,}")
        with col2:
            st.metric("🥇 Total Wins", f"{total_wins}")
        with col3:
            st.metric("🥉 Total Podiums", f"{total_podiums}")
        with col4:
            st.metric("📊 Avg Finish (drivers)", f"{avg_finish_overall:.2f}")

        st.divider()

    # =====================
    # Charts grid
    # =====================
    with sub_overview:
        left, right = st.columns(2)

        with left:
            st.markdown("### 📈 Points by Season")
            pts = points_by_season(df_sel)
            pts = pts[pts['Driver'].isin(selected_drivers)]
            fig_pts = px.line(
                pts,
                x='year', y='points', color='Driver', markers=True,
                labels={'points': 'Total Points', 'year': 'Year', 'Driver': 'Driver'},
            )
            fig_pts.update_layout(
                height=420,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                margin=dict(t=60, b=60)
            )
            st.plotly_chart(fig_pts, use_container_width=True)
            st.caption("Season totals by driver. Hover to compare drivers per year.")

        with right:
            st.markdown("### 🏆 Wins and Podiums")
            stats_ordered = stats.set_index('Driver').loc[selected_drivers].reset_index()
            fig_wp = go.Figure()
            fig_wp.add_trace(go.Bar(
                x=stats_ordered['wins'],
                y=stats_ordered['Driver'],
                name='Wins',
                orientation='h',
                marker_color='#FF6B6B',
                text=stats_ordered['wins'], textposition='auto'
            ))
            fig_wp.add_trace(go.Bar(
                x=stats_ordered['podiums'],
                y=stats_ordered['Driver'],
                name='Podiums',
                orientation='h',
                marker_color='#4ECDC4',
                text=stats_ordered['podiums'], textposition='auto'
            ))
            fig_wp.update_layout(
                barmode='group', height=420, yaxis={'categoryorder': 'total ascending'},
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                margin=dict(t=60, b=40)
            )
            st.plotly_chart(fig_wp, use_container_width=True)
            st.caption("Side-by-side wins and podiums for the selected drivers.")

        st.divider()

    # =====================
    # Standings rank + Finish distribution
    # =====================
    with sub_overview:
        r1, r2 = st.columns(2)

        with r1:
            st.markdown("### 🧭 Driver Standings Rank by Season")
            ranks = standings_by_season(df_sel)
            ranks = ranks[ranks['Driver'].isin(selected_drivers)]
            fig_rank = px.line(
                ranks, x='year', y='rank', color='Driver', markers=True,
                labels={'rank': 'Rank (1 is best)', 'year': 'Year', 'Driver': 'Driver'},
            )
            fig_rank.update_yaxes(autorange='reversed', dtick=1)
            fig_rank.update_layout(
                height=420, hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                margin=dict(t=60, b=60)
            )
            st.plotly_chart(fig_rank, use_container_width=True)
            st.caption("Per-season rank computed from points (1 = champion).")

        with r2:
            hdr_col, toggle_col = st.columns([0.7, 0.3], vertical_alignment="center")
            with hdr_col:
                st.markdown("### 📊 Race Finish Distribution")
            with toggle_col:
                as_percent = st.checkbox("Show as % of driver finishes", value=False, help="Normalize counts per driver")
            dist = finish_distribution(df_sel[df_sel['Driver'].isin(selected_drivers)], as_percent=as_percent)
            if as_percent:
                fig_dist = px.bar(
                    dist, x='Driver', y='percent', color='bucket', barmode='stack',
                    category_orders={'bucket': ['P1', 'P2-3', 'P4-10', 'P11+']},
                    labels={'percent': 'Finishes (%)', 'Driver': '', 'bucket': 'Result'}
                )
                fig_dist.update_yaxes(range=[0, 100], ticksuffix='%')
            else:
                fig_dist = px.bar(
                    dist, x='Driver', y='count', color='bucket', barmode='stack',
                    category_orders={'bucket': ['P1', 'P2-3', 'P4-10', 'P11+']},
                    labels={'count': 'Finishes', 'Driver': '', 'bucket': 'Result'}
                )
            fig_dist.update_layout(
                height=420,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                margin=dict(t=60, b=60)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("Distribution of finishing positions. Toggle to see normalized percentages per driver.")

        st.divider()

    # =====================
    # NEW: Year-over-Year improvements
    # =====================
    with sub_overview:
        st.markdown("### 📈 Year-over-Year Points Change")
        yoy = yoy_points_change(df_sel)
        yoy = yoy[yoy['Driver'].isin(selected_drivers)]
        if yoy.empty:
            st.info("Not enough seasons in the selected range to compute year-over-year changes.")
        else:
            best = yoy.loc[yoy['yoy'].idxmax()]
            st.metric("Most Improved (YoY in range)", f"{best['Driver']} +{int(best['yoy'])} pts", f"{int(best['year'])}")

            fig_yoy = px.bar(
                yoy, x='year', y='yoy', color='Driver', barmode='group',
                labels={'yoy': 'YoY Points Δ', 'year': 'Year', 'Driver': 'Driver'},
            )
            fig_yoy.update_layout(
                height=420,
                legend=dict(orientation='h', yanchor='bottom', y=1.03, xanchor='left', x=0),
                margin=dict(t=70, b=70)
            )
            st.plotly_chart(fig_yoy, use_container_width=True)
            st.caption("Change in total points compared to the previous season (per driver).")

        st.divider()

    # =====================
    # Head-to-head comparison (optional)
    # =====================
    with sub_h2h:
        st.markdown("### ⚔️ Head-to-Head Comparison")
        if len(selected_drivers) < 2:
            st.info("Select at least two drivers above to compare head-to-head.")
        else:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                d_a = st.selectbox("Driver A", options=selected_drivers, index=0, key="h2h_driver_a")
            with comp_col2:
                d_b = st.selectbox("Driver B", options=[d for d in selected_drivers if d != d_a], index=0, key="h2h_driver_b")

            a_stats = stats[stats['Driver'] == d_a].iloc[0]
            b_stats = stats[stats['Driver'] == d_b].iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(f"{d_a} Points", f"{int(a_stats['total_points'])}")
                st.metric(f"{d_a} Avg Finish", f"{a_stats['avg_finish']:.2f}")
            with m2:
                st.metric(f"{d_a} Wins", f"{int(a_stats['wins'])}")
                st.metric(f"{d_a} Podiums", f"{int(a_stats['podiums'])}")
            with m3:
                st.metric(f"{d_b} Points", f"{int(b_stats['total_points'])}")
                st.metric(f"{d_b} Avg Finish", f"{b_stats['avg_finish']:.2f}")
            with m4:
                st.metric(f"{d_b} Wins", f"{int(b_stats['wins'])}")
                st.metric(f"{d_b} Podiums", f"{int(b_stats['podiums'])}")

            comp_df = pd.DataFrame([
                {'driver': d_a, 'Points': a_stats['total_points'], 'Wins': a_stats['wins'], 'Podiums': a_stats['podiums']},
                {'driver': d_b, 'Points': b_stats['total_points'], 'Wins': b_stats['wins'], 'Podiums': b_stats['podiums']},
            ])

            fig_comp = px.bar(
                comp_df.melt(id_vars='driver', var_name='Metric', value_name='Value'),
                x='Metric', y='Value', color='driver', barmode='group'
            )
            fig_comp.update_layout(height=420, legend=dict(orientation='h', yanchor='bottom', y=-0.25))
            st.plotly_chart(fig_comp, use_container_width=True)

            st.divider()

            # Circuit Head-to-Head
            st.caption("Per-circuit points difference between the two drivers across the selected years.")
            circuit_col = 'name_circuits' if 'name_circuits' in df_range.columns else ('circuitRef' if 'circuitRef' in df_range.columns else 'circuitId')
            h2h_df = df_range[df_range['Driver'].isin([d_a, d_b])].copy()
            if h2h_df.empty:
                st.info("No races for the selected drivers in this year range.")
            else:
                grp = (
                    h2h_df.groupby([circuit_col, 'Driver'], as_index=False)['points']
                    .sum()
                )
                piv = grp.pivot(index=circuit_col, columns='Driver', values='points').fillna(0)
                if d_a not in piv.columns:
                    piv[d_a] = 0.0
                if d_b not in piv.columns:
                    piv[d_b] = 0.0
                piv['diff'] = piv[d_a] - piv[d_b]
                plot_df = piv['diff'].reset_index().rename(columns={circuit_col: 'circuit'})
                plot_df['who'] = plot_df['diff'].apply(lambda d: d_a if d > 0 else (d_b if d < 0 else 'Tie'))
                plot_df['abs_diff'] = plot_df['diff'].abs()
                plot_df = plot_df.sort_values('abs_diff', ascending=False)

                total_circuits = len(plot_df)
                default_topn = 15 if total_circuits >= 15 else total_circuits
                topn = st.slider(
                    "Top N circuits by absolute diff",
                    min_value=5 if total_circuits >= 5 else total_circuits,
                    max_value=total_circuits,
                    value=default_topn,
                    key="h2h_div_topn_drivercomp",
                ) if total_circuits > 0 else 0

                if total_circuits == 0:
                    st.info("No circuit aggregation available.")
                else:
                    to_show = plot_df.head(topn) if topn and topn > 0 else plot_df
                    cmap = {d_a: '#1f77b4', d_b: '#ff7f0e', 'Tie': '#9e9e9e'}
                    fig_div = px.bar(
                        to_show,
                        x='diff', y='circuit', color='who', orientation='h',
                        color_discrete_map=cmap,
                        labels={'diff': f"Points difference ({d_a} – {d_b})", 'circuit': 'Circuit', 'who': ''}
                    )
                    fig_div.update_yaxes(categoryorder='array', categoryarray=list(to_show['circuit']))
                    fig_div.add_vline(x=0, line_width=1, line_dash='dash', line_color='#888')
                    bar_h = 28
                    fig_div.update_layout(height=max(420, 100 + bar_h * len(to_show)), legend=dict(orientation='h', yanchor='bottom', y=-0.25))
                    st.plotly_chart(fig_div, use_container_width=True)

    # =====================
    # Advanced deep dives (parallels & craft)
    # =====================
    with sub_adv:
        st.markdown("### 🎯 Story-driven Deep Dives")
        cat_score, cat_craft = st.tabs(["Scoreboard", "Race Craft"])

        with cat_score:
            st.caption("Who scored for each driver and when.")
            tm = df_range[df_range['Driver'].isin(selected_drivers)].copy()
            if tm.empty:
                st.info("No data in selection.")
            else:
                tre = (
                    tm.groupby(['Driver', 'year'], as_index=False)['points']
                    .sum()
                )
                fig_tm = px.treemap(
                    tre, path=['Driver', 'year'], values='points'
                )
                fig_tm.update_layout(height=600, margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig_tm, use_container_width=True)

        with cat_craft:
            st.caption("How starts translate into result buckets across drivers.")
            dv = df_sel.copy()
            if dv.empty:
                st.info("No data for selected drivers.")
            else:
                dv['grid'] = pd.to_numeric(dv.get('grid'), errors='coerce')
                dv['positionOrder'] = pd.to_numeric(dv['positionOrder'], errors='coerce')
                dv = dv.dropna(subset=['grid', 'positionOrder'])
                dv['grid_bucket'] = dv['grid'].apply(lambda v: 'P1' if v==1 else ('P2-5' if 2<=v<=5 else ('P6-10' if 6<=v<=10 else 'P11+')))
                dv['finish_bucket'] = dv['positionOrder'].apply(lambda x: 'P1' if x==1 else ('P2-3' if 2<=x<=3 else ('P4-10' if 4<=x<=10 else 'P11+')))
                pc = (
                    dv.groupby(['Driver', 'grid_bucket', 'finish_bucket'])
                    .size().reset_index(name='count')
                )
                if pc.empty:
                    st.info("Not enough data for parallel categories.")
                else:
                    dims = [
                        dict(label='Driver', values=pc['Driver']),
                        dict(label='Grid', values=pc['grid_bucket']),
                        dict(label='Finish', values=pc['finish_bucket']),
                    ]
                    parcats = go.Figure(data=[go.Parcats(
                        dimensions=dims,
                        counts=pc['count'],
                        line={'color': pc['count'], 'colorscale': 'Tealrose'},
                        bundlecolors=True,
                    )])
                    parcats.update_layout(height=520, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(parcats, use_container_width=True)

        # ============================================================
    
    # =====================
    # NEW: Avg Points vs Reliability (Finished Probability vs Avg Points)
    # =====================
    with sub_overview:
        st.markdown("### ⚡ Skill vs Reliability: Avg Points vs Finish Probability")

        if df_sel.empty or not selected_drivers:
            st.info("No data available for the selected drivers.")
        else:
            # Compute finished_binary
            df_sel['finished_binary'] = df_sel['status'].apply(
                lambda s: 1 if str(s) == "Finished" or str(s).startswith("+") else 0
            )

            # Aggregate per driver
            driver_stats = (
                df_sel.groupby('Driver', as_index=False)
                .agg(
                    avg_points=('points', 'mean'),
                    finish_prob=('finished_binary', 'mean')
                )
            )

            # Compute median points for skill threshold
            median_points = driver_stats['avg_points'].median()

            # Quadrant assignment using your new logic
            driver_stats['Quadrant'] = driver_stats.apply(
                lambda row: ('High Skill' if row['avg_points'] > median_points else 'Low Skill') +
                            ' / ' +
                            ('High Reliability' if row['finish_prob'] > 0.9 else 'Low Reliability'),
                axis=1
            )

            # Scatterplot
            fig_scatter = px.scatter(
                driver_stats,
                x='finish_prob',
                y='avg_points',
                color='Quadrant',
                text='Driver',
                labels={'finish_prob': 'Finish Probability', 'avg_points': 'Average Points'},
                color_discrete_map={
                    'High Skill / High Reliability': '#2ca02c',
                    'High Skill / Low Reliability': '#ff7f0e',
                    'Low Skill / High Reliability': '#1f77b4',
                    'Low Skill / Low Reliability': '#d62728'
                },
                size_max=18
            )

            fig_scatter.update_traces(
                textposition='top center',
                marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey'))
            )

            # Add median line for skill (y-axis)
            fig_scatter.add_hline(y=median_points, line_dash='dash', line_color='white')

            # Optional: x-axis line for reliability at 0.9
            fig_scatter.add_vline(x=0.9, line_dash='dash', line_color='white')

            # Fix x-axis from 0 to 1
            fig_scatter.update_xaxes(range=[0, 1])

            fig_scatter.update_layout(
                height=500,
                margin=dict(t=40, b=40),
                title="Driver Skill vs Reliability"
            )

            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(
                "Quadrants: Top-right = High Skill & High Reliability; "
                "Top-left = High Skill & Low Reliability; "
                "Bottom-right = Low Skill & High Reliability; "
                "Bottom-left = Low Skill & Low Reliability."
            )