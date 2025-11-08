import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loadDatasets import load_merged_dataset
from utils.teamColors import get_team_color_map, hex_to_rgba, TEAM_COLORS


def lighten_hex_color(hex_color: str, factor: float = 0.4) -> str:
    """Lighten a HEX color by blending it towards white."""
    if not hex_color:
        return '#aaaaaa'
    color = hex_color.lstrip('#')
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return '#aaaaaa'
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


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
    return df


@st.cache_data
def get_year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    years = pd.to_numeric(df['year'], errors='coerce').dropna()
    return int(years.min()), int(years.max())


@st.cache_data
def summarize_constructor_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['points'] = pd.to_numeric(d['points'], errors='coerce')
    d['positionOrder'] = pd.to_numeric(d['positionOrder'], errors='coerce')
    d['grid'] = pd.to_numeric(d['grid'], errors='coerce') if 'grid' in d.columns else pd.NA

    agg = (
        d.groupby('name').agg(
            total_points=('points', 'sum'),
            wins=('positionOrder', lambda s: (s == 1).sum()),
            podiums=('positionOrder', lambda s: (s <= 3).sum()),
            avg_finish=('positionOrder', 'mean'),
            top10=('positionOrder', lambda s: (s <= 10).sum()),
            poles=('grid', lambda s: (s == 1).sum()),
        )
        .reset_index()
        .rename(columns={'name': 'team'})
    )

    agg['avg_finish'] = pd.to_numeric(agg['avg_finish'], errors='coerce').round(2)
    return agg.sort_values('total_points', ascending=False)


@st.cache_data
def points_by_season(df: pd.DataFrame) -> pd.DataFrame:
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
def standings_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-season constructor standings rank based on total points."""
    pts = points_by_season(df)
    if pts.empty:
        return pts.assign(rank=pd.Series(dtype=float))
    pts['rank'] = pts.groupby('year')['points'].rank(ascending=False, method='dense')
    return pts


@st.cache_data
def finish_distribution(df: pd.DataFrame, as_percent: bool = False) -> pd.DataFrame:
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
def yoy_points_change(df: pd.DataFrame) -> pd.DataFrame:
    """YoY points delta by team and season."""
    pts = points_by_season(df)
    if pts.empty:
        return pts.assign(points_prev=pd.Series(dtype=float), yoy=pd.Series(dtype=float))
    pts['points_prev'] = pts.groupby('team')['points'].shift(1)
    pts['yoy'] = pts['points'] - pts['points_prev']
    return pts.dropna(subset=['points_prev'])


# ============================================================
#  ADVANCED CHART HELPERS
# ============================================================

@st.cache_data
def grid_bucketize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    def _b(v):
        if pd.isna(v):
            return 'Unknown'
        if v == 1:
            return 'P1'
        if 2 <= v <= 5:
            return 'P2-5'
        if 6 <= v <= 10:
            return 'P6-10'
        if 11 <= v <= 20:
            return 'P11-20'
        return '>20'
    return s.apply(_b)


@st.cache_data
def prepare_parcats_counts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['grid_bucket'] = grid_bucketize(d.get('grid'))
    d['positionOrder'] = pd.to_numeric(d['positionOrder'], errors='coerce')
    d['finish_bucket'] = d['positionOrder'].apply(lambda x: 'P1' if x == 1 else 'P2-3' if 2 <= x <= 3 else 'P4-10' if 4 <= x <= 10 else 'P11+')
    out = (
        d.groupby(['name', 'grid_bucket', 'finish_bucket'])
        .size().reset_index(name='count')
        .rename(columns={'name': 'team'})
    )
    return out


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
        key="constructor_year_range",
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
    team_colors = get_team_color_map(selected_teams)

    # In-tab navigation (replace global sidebar)
    sub_overview, sub_h2h, sub_adv = st.tabs(["Overview", "Head-to-Head", "Advanced"])

    # =====================
    # Key metrics row (Overview)
    # =====================
    stats = summarize_constructor_stats(df_sel)
    with sub_overview:
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
    with sub_overview:
        left, right = st.columns(2)

        with left:
            st.markdown("### 📈 Points by Season")
            pts = points_by_season(df_sel)
            pts = pts[pts['team'].isin(selected_teams)]
            fig_pts = px.line(
                pts,
                x='year', y='points', color='team', markers=True,
                labels={'points': 'Total Points', 'year': 'Year', 'team': 'Team'},
                color_discrete_map=team_colors,
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
    with sub_overview:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 📉 Average Finishing Position")
            fig_avg = px.bar(
                stats.sort_values('avg_finish'),
                x='team', y='avg_finish', color='team', color_discrete_map=team_colors,
                labels={'avg_finish': 'Avg Position (lower is better)', 'team': ''},
            )
            fig_avg.update_layout(showlegend=False, height=420, margin=dict(t=40, b=60))
            st.plotly_chart(fig_avg, use_container_width=True)
            st.caption("Lower is better: average finishing positions across the selected period.")

        with c2:
            st.markdown("### 🔟 Top 10 Finishes")
            fig_t10 = px.bar(
                stats.sort_values('top10', ascending=False),
                x='team', y='top10', color='team', color_discrete_map=team_colors,
                labels={'top10': 'Top 10 Finishes', 'team': ''},
            )
            fig_t10.update_layout(showlegend=False, height=420, margin=dict(t=40, b=60))
            st.plotly_chart(fig_t10, use_container_width=True)
            st.caption("Count of P10 or better finishes across the selected period.")

        st.divider()

    # =====================
    # NEW: Standings rank + Finish distribution
    # =====================
    with sub_overview:
        r1, r2 = st.columns(2)

        with r1:
            st.markdown("### 🧭 Constructor Standings Rank by Season")
            ranks = standings_by_season(df_sel)
            ranks = ranks[ranks['team'].isin(selected_teams)]
            fig_rank = px.line(
                ranks, x='year', y='rank', color='team', markers=True,
                labels={'rank': 'Rank (1 is best)', 'year': 'Year', 'team': 'Team'},
                color_discrete_map=team_colors,
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
                as_percent = st.checkbox("Show as % of team finishes", value=False, help="Normalize counts per team")
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
    with sub_overview:
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
                labels={'yoy': 'YoY Points Δ', 'year': 'Year', 'team': 'Team'},
                color_discrete_map=team_colors,
            )
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
    with sub_h2h:
        st.markdown("### ⚔️ Head-to-Head Comparison")
        if len(selected_teams) < 2:
            st.info("Select at least two teams above to compare head-to-head.")
        else:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                team_a = st.selectbox("Team A", options=selected_teams, index=0, key="h2h_team_a")
            with comp_col2:
                team_b = st.selectbox("Team B", options=[t for t in selected_teams if t != team_a], index=0, key="h2h_team_b")

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

            comp_df = pd.DataFrame([
                {'team': team_a, 'Points': a_stats['total_points'], 'Wins': a_stats['wins'], 'Podiums': a_stats['podiums']},
                {'team': team_b, 'Points': b_stats['total_points'], 'Wins': b_stats['wins'], 'Podiums': b_stats['podiums']},
            ])

            fig_comp = px.bar(
                comp_df.melt(id_vars='team', var_name='Metric', value_name='Value'),
                x='Metric', y='Value', color='team', barmode='group', color_discrete_map=team_colors
            )
            fig_comp.update_layout(height=420, legend=dict(orientation='h', yanchor='bottom', y=-0.25))
            st.plotly_chart(fig_comp, use_container_width=True)

            st.divider()

            # --- Diverging: Circuit Head-to-Head (moved from Advanced) ---
            st.caption("Per-circuit points difference between the two teams across the selected years.")
            # choose circuit label column
            circuit_col = 'name_circuits' if 'name_circuits' in df_range.columns else ('circuitRef' if 'circuitRef' in df_range.columns else 'circuitId')
            h2h_df = df_range[df_range['name'].isin([team_a, team_b])].copy()
            if h2h_df.empty:
                st.info("No races for the selected teams in this year range.")
            else:
                grp = (
                    h2h_df.groupby([circuit_col, 'name'], as_index=False)['points']
                    .sum()
                )
                piv = grp.pivot(index=circuit_col, columns='name', values='points').fillna(0)
                # ensure both columns exist
                if team_a not in piv.columns:
                    piv[team_a] = 0.0
                if team_b not in piv.columns:
                    piv[team_b] = 0.0
                piv['diff'] = piv[team_a] - piv[team_b]
                plot_df = piv['diff'].reset_index().rename(columns={circuit_col: 'circuit'})
                plot_df['who'] = plot_df['diff'].apply(lambda d: team_a if d > 0 else (team_b if d < 0 else 'Tie'))
                plot_df['abs_diff'] = plot_df['diff'].abs()
                plot_df = plot_df.sort_values('abs_diff', ascending=False)

                total_circuits = len(plot_df)
                default_topn = 15 if total_circuits >= 15 else total_circuits
                topn = st.slider("Top N circuits by absolute diff", min_value=5 if total_circuits >= 5 else total_circuits, max_value=total_circuits, value=default_topn, key="h2h_div_topn_constructor") if total_circuits > 0 else 0

                if total_circuits == 0:
                    st.info("No circuit aggregation available.")
                else:
                    to_show = plot_df.head(topn) if topn and topn > 0 else plot_df
                    # Build color map including a neutral for ties
                    cmap = {**team_colors, 'Tie': '#9e9e9e'}
                    fig_div = px.bar(
                        to_show,
                        x='diff', y='circuit', color='who', orientation='h',
                        color_discrete_map=cmap,
                        labels={'diff': f"Points difference ({team_a} – {team_b})", 'circuit': 'Circuit', 'who': ''}
                    )
                    # order bars from largest to smallest at the top
                    fig_div.update_yaxes(categoryorder='array', categoryarray=list(to_show['circuit']))
                    # zero line
                    fig_div.add_vline(x=0, line_width=1, line_dash='dash', line_color='#888')
                    # dynamic height
                    bar_h = 28
                    fig_div.update_layout(height=max(420, 100 + bar_h * len(to_show)), legend=dict(orientation='h', yanchor='bottom', y=-0.25))
                    st.plotly_chart(fig_div, use_container_width=True)

    # =====================
    # 🎯 Story-driven Deep Dives
    # =====================
    with sub_adv:
        st.markdown("### 🎯 Story-driven Deep Dives")
        cat_score, cat_craft, cat_battle, cat_momentum, cat_identity = st.tabs([
            "Scoreboard",
            "Race Craft",
            "Battlefields",
            "Momentum",
            "Identity",
        ])

        # ========== Scoreboard: Where points come from ==========
        with cat_score:
            sb_tab1, sb_tab2 = st.tabs([
                "Sunburst: Points Origins",
                "Treemap: Team → Driver → Year",
            ])

            with sb_tab1:
                st.caption("How team points are distributed across seasons.")
                pts_sb = points_by_season(df_sel)
                pts_sb = pts_sb[pts_sb['team'].isin(selected_teams)]
                if pts_sb.empty:
                    st.info("No points in the selected range.")
                else:
                    # Sort by team and year to display years chronologically
                    pts_sb = pts_sb.sort_values(['team', 'year']).copy()
                    available_teams = pts_sb['team'].unique().tolist()
                    team_order = [team for team in selected_teams if team in available_teams]
                    if not team_order:
                        team_order = available_teams

                    labels = []
                    parents = []
                    ids = []
                    values = []
                    colors = []

                    for team in team_order:
                        team_points = pts_sb[pts_sb['team'] == team]
                        if team_points.empty:
                            continue
                        team_total = float(team_points['points'].sum())
                        labels.append(team)
                        parents.append("")
                        ids.append(team)
                        values.append(team_total)
                        colors.append(team_colors.get(team, '#888888'))

                        for _, row in team_points.iterrows():
                            year_label = str(int(row['year'])) if pd.notna(row['year']) else "Unknown"
                            labels.append(year_label)
                            parents.append(team)
                            ids.append(f"{team}-{year_label}")
                            values.append(float(row['points']))
                            base_color = team_colors.get(team, '#888888')
                            colors.append(lighten_hex_color(base_color, factor=0.55))

                    if not labels:
                        st.info("No points in the selected range.")
                    else:
                        sun = go.Figure(go.Sunburst(
                            ids=ids,
                            labels=labels,
                            parents=parents,
                            values=values,
                            branchvalues="total",
                            sort=False,
                            marker=dict(colors=colors)
                        ))
                        sun.update_layout(height=520, margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(sun, use_container_width=True)

            with sb_tab2:
                st.caption("Who scored for each team and when.")
                tm = df_range[df_range['name'].isin(selected_teams)].copy()
                if tm.empty:
                    st.info("No data in selection.")
                else:
                    tm['driver_name'] = tm.get('driver', tm.get('surname', tm.get('code', 'Driver')))
                    tre = (
                        tm.groupby(['name', 'driver_name', 'year'], as_index=False)['points']
                        .sum()
                        .rename(columns={'name': 'team'})
                    )
                    fig_tm = px.treemap(
                        tre, path=['team', 'driver_name', 'year'], values='points', color='team', color_discrete_map=team_colors
                    )
                    fig_tm.update_layout(height=600, margin=dict(t=20, b=10, l=10, r=10))
                    st.plotly_chart(fig_tm, use_container_width=True)

        # ========== Race Craft: From grid to chequered ==========
        with cat_craft:
            rc_tab1, rc_tab2, rc_tab3, rc_tab4 = st.tabs([
                "Result Pathways (Sankey)",
                "Grid→Finish Paths",
                "Finish Distribution",
                "Grid vs Finish",
            ])

            with rc_tab1:
                st.caption("How starts translate into result buckets.")
                dist_counts = finish_distribution(df_sel[df_sel['name'].isin(selected_teams)], as_percent=False)
                buckets = ['P1', 'P2-3', 'P4-10', 'P11+']
                teams = selected_teams
                labels = teams + buckets
                label_to_idx = {lab: i for i, lab in enumerate(labels)}
                sources, targets, values = [], [], []
                for _, row in dist_counts.iterrows():
                    team = row['team']
                    bucket = row['bucket']
                    if team in label_to_idx and bucket in label_to_idx:
                        sources.append(label_to_idx[team])
                        targets.append(label_to_idx[bucket])
                        values.append(int(row['count']))
                
                # Build node colors: team colors for teams, neutral gray for result buckets
                node_colors = []
                for label in labels:
                    if label in team_colors:
                        node_colors.append(team_colors[label])
                    else:
                        node_colors.append('#9e9e9e')  # Gray for result buckets
                
                # Build link colors based on source team color with transparency
                link_colors = []
                for src_idx in sources:
                    src_label = labels[src_idx]
                    if src_label in team_colors:
                        link_colors.append(hex_to_rgba(team_colors[src_label], 0.4))
                    else:
                        link_colors.append('rgba(158,158,158,0.4)')
                
                sankey = go.Figure(
                    data=[go.Sankey(
                        arrangement='snap',
                        node=dict(pad=15, thickness=15, label=labels, color=node_colors),
                        link=dict(source=sources, target=targets, value=values, color=link_colors)
                    )]
                )
                sankey.update_layout(height=520, margin=dict(t=40, b=40))
                st.plotly_chart(sankey, use_container_width=True)

            with rc_tab2:
                st.caption("Typical pathways from start position to finishing bucket.")
                pc = prepare_parcats_counts(df_sel[df_sel['name'].isin(selected_teams)])
                if pc.empty:
                    st.info("Not enough data for parallel categories.")
                else:
                    dims = [
                        dict(label='Team', values=pc['team']),
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

            with rc_tab3:
                st.caption("Spread of final positions by team (lower is better).")
                dv = df_sel[df_sel['name'].isin(selected_teams)].copy()
                if dv.empty:
                    st.info("No finish data available.")
                else:
                    dv['positionOrder'] = pd.to_numeric(dv['positionOrder'], errors='coerce')
                    dv = dv.dropna(subset=['positionOrder'])
                    dv = dv.rename(columns={'name': 'team'})
                    v = px.violin(dv, x='team', y='positionOrder', color='team', box=True, points='outliers', color_discrete_map=team_colors)
                    v.update_yaxes(autorange='reversed', title='Finishing Position')
                    v.update_xaxes(title='')
                    v.update_layout(showlegend=False, height=560, margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(v, use_container_width=True)

            with rc_tab4:
                st.caption("How starting position correlates with the finish.")
                gvf = df_sel[df_sel['name'].isin(selected_teams)].copy()
                if gvf.empty:
                    st.info("No data for selected teams.")
                else:
                    gvf['grid'] = pd.to_numeric(gvf.get('grid'), errors='coerce')
                    gvf['positionOrder'] = pd.to_numeric(gvf['positionOrder'], errors='coerce')
                    gvf = gvf.dropna(subset=['grid', 'positionOrder'])
                    gvf = gvf.rename(columns={'name': 'team'})
                    sc = px.scatter(
                        gvf, x='grid', y='positionOrder', color='team',
                        labels={'grid': 'Start Grid', 'positionOrder': 'Finish'},
                        opacity=0.6, color_discrete_map=team_colors
                    )
                    sc.update_yaxes(autorange='reversed', dtick=1)
                    
                    import numpy as np
                    # Add regression lines for each team
                    for tm in gvf['team'].unique():
                        sub = gvf[gvf['team'] == tm]
                        if len(sub) >= 2:
                            x = pd.to_numeric(sub['grid'], errors='coerce')
                            y = pd.to_numeric(sub['positionOrder'], errors='coerce')
                            # Drop NaN and get matching indices
                            valid_mask = x.notna() & y.notna()
                            x = x[valid_mask]
                            y = y[valid_mask]
                            
                            # Need at least 2 valid points and variance in x
                            if len(x) >= 2 and x.std() > 0:
                                try:
                                    # Use polyfit with error handling
                                    m, c = np.polyfit(x, y, 1)
                                    xr = np.array([x.min(), x.max()])
                                    yr = m * xr + c
                                    sc.add_trace(go.Scatter(
                                        x=xr, y=yr, mode='lines', name=f"{tm} fit",
                                        line=dict(color=team_colors.get(tm), width=2),
                                        showlegend=False
                                    ))
                                except (np.linalg.LinAlgError, ValueError):
                                    # Skip regression line if numerical issues occur
                                    continue
                    
                    sc.update_layout(height=560, margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(sc, use_container_width=True)

        # ========== Battlefields: Circuits that matter ==========
        with cat_battle:
            st.caption("Which circuits reward which teams (sum of points across selected years).")
            circuit_col = 'name_circuits' if 'name_circuits' in df_sel.columns else ('circuitRef' if 'circuitRef' in df_sel.columns else 'circuitId')
            df_hm = (
                df_sel[df_sel['name'].isin(selected_teams)]
                .groupby([circuit_col, 'name'], as_index=False)['points']
                .sum()
                .rename(columns={'name': 'team'})
            )
            if df_hm.empty:
                st.info("No circuit data available in this selection.")
            else:
                pivot = df_hm.pivot(index=circuit_col, columns='team', values='points').fillna(0)
                # Ensure every selected team appears even if they scored zero here
                if selected_teams:
                    pivot = pivot.reindex(columns=selected_teams, fill_value=0)
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                fig_hm = px.imshow(
                    pivot,
                    labels=dict(x="Team", y="Circuit", color="Points"),
                    color_continuous_scale='Inferno'
                )
                fig_hm.update_xaxes(
                    tickmode='array',
                    tickvals=pivot.columns.tolist(),
                    ticktext=pivot.columns.tolist(),
                    tickangle=45
                )
                fig_hm.update_layout(height=580, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_hm, use_container_width=True)

        # ========== Momentum: Season dynamics ==========
        with cat_momentum:
            mo_tab1, mo_tab2 = st.tabs([
                "Bar Race: Cumulative Points",
                "Driver Contributions",
            ])

            with mo_tab1:
                st.caption("How championship momentum builds round by round.")
                years_avail = sorted(df_range['year'].dropna().unique())
                if not years_avail:
                    st.info("No years available in current selection.")
                else:
                    yr_pick = st.selectbox("Year", options=years_avail, index=len(years_avail)-1, key="bar_race_year")
                    yr_df = df[(df['year'] == yr_pick)].copy()
                    if yr_df.empty:
                        st.info("No data for that year.")
                    else:
                        if 'round' not in yr_df.columns:
                            st.info("Round information is missing in dataset for bar race.")
                        else:
                            team_race = (
                                yr_df.groupby(['round', 'name'], as_index=False)['points']
                                .sum()
                                .sort_values(['round'])
                                .rename(columns={'name': 'team'})
                            )
                            team_race = team_race[team_race['team'].isin(selected_teams)]
                            if team_race.empty:
                                st.info("No races for the selected teams in that year.")
                            else:
                                team_race['cum_points'] = team_race.groupby('team')['points'].cumsum()
                                anim_df = team_race.copy()
                                anim_df['round'] = pd.to_numeric(anim_df['round'], errors='coerce')
                                anim_df = anim_df.dropna(subset=['round'])
                                max_total = float(anim_df['cum_points'].max()) if not anim_df.empty else 0
                                br = px.bar(
                                    anim_df, x='cum_points', y='team', color='team', orientation='h',
                                    animation_frame='round', range_x=[0, max_total * 1.1 if max_total > 0 else 1.0],
                                    labels={'cum_points': 'Cumulative Points', 'team': ''},
                                    color_discrete_map=team_colors
                                )
                                br.update_layout(height=600, showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
                                st.plotly_chart(br, use_container_width=True)

            with mo_tab2:
                st.caption("Which drivers power the team's points each season.")
                if not selected_teams:
                    st.info("Select at least one team.")
                else:
                    team_pick = st.selectbox("Select Team", options=selected_teams, index=0, key="driver_contrib_team")
                    dc = df_range[df_range['name'] == team_pick].copy()
                    if dc.empty:
                        st.info("No data for the selected team/year range.")
                    else:
                        dc['driver_name'] = dc.get('driver', dc.get('surname', dc.get('code', 'Driver')))
                        per = (
                            dc.groupby(['year', 'driver_name'], as_index=False)['points']
                            .sum()
                            .sort_values(['year', 'points'], ascending=[True, False])
                        )
                        area = px.area(
                            per, x='year', y='points', color='driver_name',
                            line_group='driver_name',
                            labels={'points': 'Points', 'year': 'Year', 'driver_name': 'Driver'}
                        )
                        area.update_layout(height=560, legend=dict(orientation='h', yanchor='bottom', y=-0.25))
                        st.plotly_chart(area, use_container_width=True)

        # ========== Identity: Team makeup ==========
        with cat_identity:
            id_tab1, id_tab2 = st.tabs([
                "Team Profile (Radar)",
                "Lineup Stability",
            ])

            with id_tab1:
                st.caption("The team's strengths across key metrics (normalized to 0–100).")
                st_stats = stats.set_index('team').loc[selected_teams].reset_index()
                if st_stats.empty:
                    st.info("No stats available for radar.")
                else:
                    metrics = {
                        'Points': st_stats['total_points'],
                        'Wins': st_stats['wins'],
                        'Podiums': st_stats['podiums'],
                        'Top10': st_stats['top10'],
                        'Poles': st_stats['poles'],
                    }
                    def norm(series):
                        s = pd.to_numeric(series, errors='coerce').fillna(0)
                        mn, mx = s.min(), s.max()
                        if mx == mn:
                            return pd.Series([100.0] * len(s), index=s.index)
                        return (s - mn) / (mx - mn) * 100.0
                    af = pd.to_numeric(st_stats['avg_finish'], errors='coerce')
                    af_mn, af_mx = af.min(), af.max()
                    if af_mx == af_mn:
                        af_score = pd.Series([100.0] * len(af), index=af.index)
                    else:
                        af_score = (af_mx - af) / (af_mx - af_mn) * 100.0
                    categories = list(metrics.keys()) + ['AvgFinishScore']
                    normed = {k: norm(v) for k, v in metrics.items()}
                    normed['AvgFinishScore'] = af_score
                    radar_fig = go.Figure()
                    for i, row in st_stats.iterrows():
                        r_vals = [
                            float(normed['Points'].iloc[i]),
                            float(normed['Wins'].iloc[i]),
                            float(normed['Podiums'].iloc[i]),
                            float(normed['Top10'].iloc[i]),
                            float(normed['Poles'].iloc[i]),
                            float(normed['AvgFinishScore'].iloc[i]),
                        ]
                        col = team_colors.get(row['team'])
                        radar_fig.add_trace(go.Scatterpolar(
                            r=r_vals,
                            theta=categories,
                            fill='toself',
                            name=row['team'],
                            line=dict(color=col, width=2),
                            fillcolor=hex_to_rgba(col, 0.25)
                        ))
                    radar_fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=True,
                        height=520,
                        margin=dict(t=10, b=10, l=10, r=10)
                    )
                    st.plotly_chart(radar_fig, use_container_width=True)

            with id_tab2:
                st.caption("How stable each team's driver lineup is year to year (Jaccard similarity).")
                d = df_range[df_range['name'].isin(selected_teams)].copy()
                if d.empty:
                    st.info("No data in selection.")
                else:
                    d['driver_name'] = d.get('driver', d.get('surname', d.get('code', 'Driver')))
                    years_sorted = sorted(d['year'].dropna().unique())
                    rows = []
                    for team in selected_teams:
                        team_df = d[d['name'] == team]
                        if team_df.empty:
                            continue
                        prev_set = None
                        for y in years_sorted:
                            cur = set(team_df[team_df['year'] == y]['driver_name'].unique().tolist())
                            if prev_set is None or len(prev_set) == 0:
                                stability = None
                            else:
                                union = len(prev_set.union(cur))
                                inter = len(prev_set.intersection(cur))
                                stability = (inter / union) if union > 0 else None
                            if stability is not None:
                                rows.append({'team': team, 'year': y, 'stability': stability})
                            prev_set = cur
                    if not rows:
                        st.info("Not enough consecutive years for stability.")
                    else:
                        st_df = pd.DataFrame(rows)
                        fig_idx = px.line(
                            st_df, x='year', y='stability', color='team', markers=True,
                            labels={'stability': 'Lineup Stability (0–1)', 'year': 'Year'},
                            color_discrete_map=team_colors
                        )
                        fig_idx.update_yaxes(range=[0, 1])
                        fig_idx.update_layout(height=520, margin=dict(t=30, b=40, l=10, r=10))
                        st.plotly_chart(fig_idx, use_container_width=True)

        
