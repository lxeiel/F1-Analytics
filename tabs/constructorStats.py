import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loadDatasets import load_merged_dataset


# ============================================================
#  DATA LOADING HELPERS
# ============================================================

@st.cache_data
def get_overall_data():
    return load_merged_dataset()


@st.cache_data
def get_year_bounds(df: pd.DataFrame):
    years = sorted(df['year'].dropna().unique())
    return (int(min(years)), int(max(years)))


@st.cache_data
def summarize_constructor_stats(df: pd.DataFrame):
    """Aggregate core constructor metrics for a given (filtered) dataframe.

    Returns a dataframe indexed by constructor name with:
    - total_points
    - wins (P1 results)
    - podiums (P1-3)
    - top10 (P1-10)
    - races (entries)
    - avg_finish (mean positionOrder over classified results)
    - poles (grid == 1)
    """
    d = df.copy()
    # Ensure numeric types
    d['positionOrder'] = pd.to_numeric(d['positionOrder'], errors='coerce')
    if 'grid' in d.columns:
        d['grid'] = pd.to_numeric(d['grid'], errors='coerce')
    else:
        d['grid'] = pd.NA
    d['points'] = pd.to_numeric(d['points'], errors='coerce')

    grp = d.groupby('name')
    agg = grp.agg(
        total_points=('points', 'sum'),
        wins=('positionOrder', lambda x: (x == 1).sum()),
        podiums=('positionOrder', lambda x: x.le(3).sum()),
        top10=('positionOrder', lambda x: x.le(10).sum()),
        races=('raceId', 'nunique'),
        avg_finish=('positionOrder', 'mean'),
        poles=('grid', lambda x: (x == 1).sum()),
    ).reset_index().rename(columns={'name': 'team'})

    # Clean avg_finish for display
    agg['avg_finish'] = agg['avg_finish'].round(2)
    return agg.sort_values('total_points', ascending=False)


@st.cache_data
def points_by_season(df: pd.DataFrame):
    d = df.copy()
    d['points'] = pd.to_numeric(d['points'], errors='coerce')
    pts = (
        d.groupby(['year', 'name'])['points']
        .sum()
        .reset_index()
        .rename(columns={'name': 'team'})
        .sort_values(['team', 'year'])
    )
    return pts


@st.cache_data
def standings_by_season(df: pd.DataFrame):
    """Compute per-season constructor standings rank based on total points."""
    pts = points_by_season(df)
    if pts.empty:
        return pts.assign(rank=pd.Series(dtype=float))
    pts['rank'] = pts.groupby('year')['points'].rank(ascending=False, method='dense')
    return pts


@st.cache_data
def finish_distribution(df: pd.DataFrame, as_percent: bool = False):
    """Stacked finish distribution per team: P1, P2-3, P4-10, P11+.

    If as_percent=True, returns percentage per team bucket (0-100).
    """
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
        d.groupby(['name', 'bucket'])
        .size()
        .reset_index(name='count')
        .rename(columns={'name': 'team'})
    )
    bucket_order = ['P1', 'P2-3', 'P4-10', 'P11+']
    counts['bucket'] = pd.Categorical(counts['bucket'], categories=bucket_order, ordered=True)

    if not as_percent:
        return counts.sort_values(['team', 'bucket'])

    totals = counts.groupby('team')['count'].transform('sum')
    counts['percent'] = (counts['count'] / totals) * 100.0
    return counts.sort_values(['team', 'bucket'])


@st.cache_data
def yoy_points_change(df: pd.DataFrame):
    """YoY points delta by team and season."""
    pts = points_by_season(df)
    if pts.empty:
        return pts.assign(points_prev=pd.Series(dtype=float), yoy=pd.Series(dtype=float))
    pts['points_prev'] = pts.groupby('team')['points'].shift(1)
    pts['yoy'] = pts['points'] - pts['points_prev']
    return pts.dropna(subset=['points_prev'])


# ============================================================
#  MAIN CONSTRUCTORS TAB
# ============================================================
def constructorStatsTab():
    st.header("🏢 Constructor Statistics")

    # Load and bound data
    df = get_overall_data()
    y_min, y_max = get_year_bounds(df)

    # Sidebar-like top filters (kept in main to match other tabs)
    st.markdown("#### 📅 Filter by Year Range")
    year_range = st.slider(
        "Select Years",
        min_value=y_min,
        max_value=y_max,
        value=(max(y_min, y_max - 5), y_max),
        help="Adjust to focus on specific seasons",
        label_visibility="collapsed",
    )
    st.caption(f"Showing data from {year_range[0]} to {year_range[1]}")

    df_range = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])].copy()

    # Team multiselect (default: top 5 by points in range)
    team_points = df_range.groupby('name')['points'].sum().sort_values(ascending=False)
    all_teams = team_points.index.tolist()
    default_teams = all_teams[:5]
    selected_teams = st.multiselect(
        "🏁 Select Teams",
        options=all_teams,
        default=default_teams,
        help="Pick one or more constructors to analyze",
    )

    if not selected_teams:
        st.warning("Please select at least one team to view stats.")
        return

    df_sel = df_range[df_range['name'].isin(selected_teams)].copy()

    # =====================
    # Key metrics row
    # =====================
    stats = summarize_constructor_stats(df_sel)
    total_points = int(stats['total_points'].sum())
    total_wins = int(stats['wins'].sum())
    total_podiums = int(stats['podiums'].sum())
    avg_finish_overall = stats['avg_finish'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Total Points", f"{total_points:,}")
    with col2:
        st.metric("🥇 Total Wins", f"{total_wins}")
    with col3:
        st.metric("🥉 Total Podiums", f"{total_podiums}")
    with col4:
        st.metric("📊 Avg Finish (teams)", f"{avg_finish_overall:.2f}")

    st.divider()

    # =====================
    # Charts grid
    # =====================
    left, right = st.columns(2)

    with left:
        st.markdown("### 📈 Points by Season")
        pts = points_by_season(df_sel)
        # Keep only selected teams
        pts = pts[pts['team'].isin(selected_teams)]
        fig_pts = px.line(
            pts,
            x='year', y='points', color='team', markers=True,
            labels={'points': 'Total Points', 'year': 'Year', 'team': 'Team'},
        )
        fig_pts.update_layout(
            height=420,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            margin=dict(t=60, b=60)
        )
        st.plotly_chart(fig_pts, use_container_width=True)
        st.caption("Season totals by constructor. Hover to compare teams per year.")

    with right:
        st.markdown("### 🏁 Wins and Podiums")
        stats_ordered = stats.set_index('team').loc[selected_teams].reset_index()
        fig_wp = go.Figure()
        fig_wp.add_trace(go.Bar(
            x=stats_ordered['wins'],
            y=stats_ordered['team'],
            name='Wins',
            orientation='h',
            marker_color='#FF6B6B',
            text=stats_ordered['wins'], textposition='auto'
        ))
        fig_wp.add_trace(go.Bar(
            x=stats_ordered['podiums'],
            y=stats_ordered['team'],
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
        st.caption("Side-by-side wins and podiums for the selected teams.")

    st.divider()

    # Avg finish + Top10 share
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📉 Average Finishing Position")
        fig_avg = px.bar(
            stats.sort_values('avg_finish'),
            x='team', y='avg_finish', color='team',
            labels={'avg_finish': 'Avg Position (lower is better)', 'team': ''},
        )
        fig_avg.update_layout(showlegend=False, height=420, margin=dict(t=40, b=60))
        st.plotly_chart(fig_avg, use_container_width=True)
        st.caption("Lower is better: average finishing positions across the selected period.")

    with c2:
        st.markdown("### 🔟 Top 10 Finishes")
        fig_t10 = px.bar(
            stats.sort_values('top10', ascending=False),
            x='team', y='top10', color='team',
            labels={'top10': 'Top 10 Finishes', 'team': ''},
        )
        fig_t10.update_layout(showlegend=False, height=420, margin=dict(t=40, b=60))
        st.plotly_chart(fig_t10, use_container_width=True)
        st.caption("Count of P10 or better finishes across the selected period.")

    st.divider()

    # =====================
    # NEW: Standings rank + Finish distribution
    # =====================
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("### 🧭 Constructor Standings Rank by Season")
        ranks = standings_by_season(df_sel)
        ranks = ranks[ranks['team'].isin(selected_teams)]
        fig_rank = px.line(
            ranks, x='year', y='rank', color='team', markers=True,
            labels={'rank': 'Rank (1 is best)', 'year': 'Year', 'team': 'Team'}
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
        # Title and toggle on the same row to avoid adding extra vertical space
        hdr_col, toggle_col = st.columns([0.7, 0.3], vertical_alignment="center")
        with hdr_col:
            st.markdown("### 📊 Race Finish Distribution")
        with toggle_col:
            as_percent = st.toggle("Show as % of team finishes", value=False, help="Normalize counts per team")
        dist = finish_distribution(df_sel[df_sel['name'].isin(selected_teams)], as_percent=as_percent)
        if as_percent:
            fig_dist = px.bar(
                dist, x='team', y='percent', color='bucket', barmode='stack',
                category_orders={'bucket': ['P1', 'P2-3', 'P4-10', 'P11+']},
                labels={'percent': 'Finishes (%)', 'team': '', 'bucket': 'Result'}
            )
            fig_dist.update_yaxes(range=[0, 100], ticksuffix='%')
        else:
            fig_dist = px.bar(
                dist, x='team', y='count', color='bucket', barmode='stack',
                category_orders={'bucket': ['P1', 'P2-3', 'P4-10', 'P11+']},
                labels={'count': 'Finishes', 'team': '', 'bucket': 'Result'}
            )
        fig_dist.update_layout(
            height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            margin=dict(t=60, b=60)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("Distribution of finishing positions. Toggle to see normalized percentages per team.")

    st.divider()

    # =====================
    # NEW: Year-over-Year improvements
    # =====================
    st.markdown("### 📈 Year-over-Year Points Change")
    yoy = yoy_points_change(df_sel)
    yoy = yoy[yoy['team'].isin(selected_teams)]
    if yoy.empty:
        st.info("Not enough seasons in the selected range to compute year-over-year changes.")
    else:
        best = yoy.loc[yoy['yoy'].idxmax()]
        st.metric("Most Improved (YoY in range)", f"{best['team']} +{int(best['yoy'])} pts", f"{int(best['year'])}")

        fig_yoy = px.bar(
            yoy, x='year', y='yoy', color='team', barmode='group',
            labels={'yoy': 'YoY Points Δ', 'year': 'Year', 'team': 'Team'}
        )
        # Place legend above the plot area to avoid overlapping tick labels
        fig_yoy.update_layout(
            height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.03, xanchor='left', x=0),
            margin=dict(t=70, b=70)
        )
        st.plotly_chart(fig_yoy, use_container_width=True)
        st.caption("Change in total points compared to the previous season (per team).")

    st.divider()

    # =====================
    # Head-to-head comparison (optional)
    # =====================
    st.markdown("### ⚔️ Head-to-Head Comparison")
    if len(selected_teams) < 2:
        st.info("Select at least two teams above to compare head-to-head.")
        return

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        team_a = st.selectbox("Team A", options=selected_teams, index=0)
    with comp_col2:
        team_b = st.selectbox("Team B", options=[t for t in selected_teams if t != team_a], index=0)

    a_stats = stats[stats['team'] == team_a].iloc[0]
    b_stats = stats[stats['team'] == team_b].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(f"{team_a} Points", f"{int(a_stats['total_points'])}")
        st.metric(f"{team_a} Avg Finish", f"{a_stats['avg_finish']:.2f}")
    with m2:
        st.metric(f"{team_a} Wins", f"{int(a_stats['wins'])}")
        st.metric(f"{team_a} Podiums", f"{int(a_stats['podiums'])}")
    with m3:
        st.metric(f"{team_b} Points", f"{int(b_stats['total_points'])}")
        st.metric(f"{team_b} Avg Finish", f"{b_stats['avg_finish']:.2f}")
    with m4:
        st.metric(f"{team_b} Wins", f"{int(b_stats['wins'])}")
        st.metric(f"{team_b} Podiums", f"{int(b_stats['podiums'])}")

    # Simple comparative bar
    comp_df = pd.DataFrame([
        {'team': team_a, 'Points': a_stats['total_points'], 'Wins': a_stats['wins'], 'Podiums': a_stats['podiums']},
        {'team': team_b, 'Points': b_stats['total_points'], 'Wins': b_stats['wins'], 'Podiums': b_stats['podiums']},
    ])

    fig_comp = px.bar(
        comp_df.melt(id_vars='team', var_name='Metric', value_name='Value'),
        x='Metric', y='Value', color='team', barmode='group'
    )
    fig_comp.update_layout(height=420, legend=dict(orientation='h', yanchor='bottom', y=-0.25))
    st.plotly_chart(fig_comp, use_container_width=True)
