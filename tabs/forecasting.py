import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import sys

import plotly.express as px
# interactivity helper (removed): clicks are not used anymore

from utils.loadDatasets import load_merged_dataset
from utils.loadDatasets2025 import load_merged_dataset_2025

# ============================================================
#  CONSTANTS / UTILITIES
# ============================================================

# Team colours for 2025 (base team names ONLY)
TEAM_COLORS = {
    "Red Bull": "#0600EF",          # Red Bull dark blue
    "Ferrari": "#DC0000",           # Ferrari red
    "Mercedes": "#00D2BE",          # Mercedes teal
    "McLaren": "#FF8700",           # McLaren papaya
    "Aston Martin": "#006F62",      # Aston Martin green
    "Williams": "#005AFF",          # Williams blue
    "Alpine": "#FD4BC7",            # Alpine/BWT pink-ish
    "Haas": "#B6BABD",              # Haas grey
    "Kick Sauber": "#52E252",       # Stake/Kick neon green vibe
    "Racing Bulls": "#2B4562",      # RB navy
}

# Map noisy 2025 strings (with engine/branding) to base team keys above
def normalize_team(team: str) -> str:
    if not isinstance(team, str):
        return ""
    t = team.lower()
    # Order matters (match specific team names before engine suppliers)
    if "mclaren" in t: return "McLaren"
    if "williams" in t: return "Williams"
    if "aston martin" in t: return "Aston Martin"
    if "alpine" in t: return "Alpine"
    if "haas" in t: return "Haas"
    if "kick sauber" in t or "sauber" in t or "stake" in t: return "Kick Sauber"
    if "racing bulls" in t or "alphatauri" in t or "toro rosso" in t: return "Racing Bulls"
    if "red bull" in t: return "Red Bull"
    if "ferrari" in t: return "Ferrari"
    if "mercedes" in t: return "Mercedes"
    return team  # fall back to raw

def team_color(team_like: str) -> str:
    return TEAM_COLORS.get(normalize_team(team_like), "#888888")

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.array(x, dtype=float)
    temperature = max(temperature, 1e-6)
    z = (x - np.max(x)) / temperature
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)

# 2025 grid with mid-season swaps (used to ensure eligible drivers per round)
def get_2025_grid(rounds: int = 24) -> pd.DataFrame:
    lineup = [
        # Alpine
        {"Team":"Alpine","Driver":"Jack Doohan","r1":1,"r2":6},
        {"Team":"Alpine","Driver":"Franco Colapinto","r1":7,"r2":24},
        {"Team":"Alpine","Driver":"Pierre Gasly","r1":1,"r2":24},
        # Aston Martin
        {"Team":"Aston Martin","Driver":"Fernando Alonso","r1":1,"r2":24},
        {"Team":"Aston Martin","Driver":"Lance Stroll","r1":1,"r2":24},
        # Ferrari
        {"Team":"Ferrari","Driver":"Charles Leclerc","r1":1,"r2":24},
        {"Team":"Ferrari","Driver":"Lewis Hamilton","r1":1,"r2":24},
        # Haas
        {"Team":"Haas","Driver":"Esteban Ocon","r1":1,"r2":24},
        {"Team":"Haas","Driver":"Oliver Bearman","r1":1,"r2":24},
        # Kick Sauber
        {"Team":"Kick Sauber","Driver":"Gabriel Bortoleto","r1":1,"r2":24},
        {"Team":"Kick Sauber","Driver":"Nico Hülkenberg","r1":1,"r2":24},
        # McLaren
        {"Team":"McLaren","Driver":"Lando Norris","r1":1,"r2":24},
        {"Team":"McLaren","Driver":"Oscar Piastri","r1":1,"r2":24},
        # Mercedes
        {"Team":"Mercedes","Driver":"Kimi Antonelli","r1":1,"r2":24},
        {"Team":"Mercedes","Driver":"George Russell","r1":1,"r2":24},
        # Racing Bulls
        {"Team":"Racing Bulls","Driver":"Isack Hadjar","r1":1,"r2":24},
        {"Team":"Racing Bulls","Driver":"Yuki Tsunoda","r1":1,"r2":2},
        {"Team":"Racing Bulls","Driver":"Liam Lawson","r1":3,"r2":24},
        # Red Bull
        {"Team":"Red Bull","Driver":"Max Verstappen","r1":1,"r2":24},
        {"Team":"Red Bull","Driver":"Liam Lawson","r1":1,"r2":2},
        {"Team":"Red Bull","Driver":"Yuki Tsunoda","r1":3,"r2":24},
        # Williams
        {"Team":"Williams","Driver":"Alexander Albon","r1":1,"r2":24},
        {"Team":"Williams","Driver":"Carlos Sainz Jr.","r1":1,"r2":24},
    ]
    rows = []
    for e in lineup:
        for r in range(e["r1"], e["r2"] + 1):
            if r <= rounds:
                rows.append({"round": r, "Driver": e["Driver"], "Team": e["Team"]})
    return pd.DataFrame(rows)

# ============================================================
#  HISTORICAL (1950–2024) AGGREGATION
# ============================================================

@st.cache_data(show_spinner=False)
def get_historical_df() -> pd.DataFrame:
    df = load_merged_dataset().copy()

    if "year" not in df.columns:
        if "date" in df.columns:
            df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        else:
            raise ValueError("Historical DF missing 'year'/'date'.")

    if "Driver" not in df.columns:
        if "forename" in df.columns and "surname" in df.columns:
            df["Driver"] = df["forename"].astype(str) + " " + df["surname"].astype(str)
        else:
            df["Driver"] = df.get("code", "DRV")

    if "name" not in df.columns:
        df["name"] = "Unknown Team"

    for c in ["points", "positionOrder"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

@st.cache_data(show_spinner=False)
def constructor_season_points(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["year", "constructorId", "name"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "season_points"})
    )
    out["season_points"] = out["season_points"].fillna(0)
    return out

@st.cache_data(show_spinner=False)
def compute_constructor_champions(season_totals: pd.DataFrame):
    idx = (
        season_totals.groupby("year")["season_points"]
        .idxmax()
        .dropna()
        .astype(int)
        .values
    )
    champs = season_totals.loc[idx, ["year","constructorId","name"]].sort_values("year")
    titles_by_constructor = (
        champs.groupby("constructorId")["year"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
    )
    return champs.reset_index(drop=True), titles_by_constructor

def global_gap_counts(titles_by_constructor: Dict[int, list]) -> pd.Series:
    gaps = []
    for years in titles_by_constructor.values():
        yrs = sorted(years)
        if len(yrs) >= 2:
            gaps.extend(np.diff(yrs))
    if not gaps:
        return pd.Series(dtype="int64", name="gap_years")
    return pd.Series(gaps, name="gap_years").value_counts().sort_index()

def build_smoothed_pmf(counts: pd.Series, k_max: int, laplace_alpha: float) -> pd.DataFrame:
    idx = pd.Index(range(1, k_max + 1), name="gap_years")
    counts = counts.reindex(idx, fill_value=0).astype(float)
    counts += float(laplace_alpha)
    pmf = counts / counts.sum()
    return pmf.reset_index(name="probability")

def team_gap_counts(title_years: list) -> pd.Series:
    if not title_years or len(title_years) < 2:
        return pd.Series(dtype="int64", name="gap_years")
    gaps = np.diff(sorted(title_years)).astype(int)
    return pd.Series(gaps, name="gap_years").value_counts().sort_index()

def pmf_probability_at_gap(k: int, team_pmf: Optional[pd.DataFrame], global_pmf: pd.DataFrame) -> float:
    if team_pmf is not None and not team_pmf.empty:
        r = team_pmf.loc[team_pmf["gap_years"] == int(k)]
        if not r.empty:
            return float(r["probability"].iloc[0])
    r = global_pmf.loc[global_pmf["gap_years"] == int(k)]
    return float(r["probability"].iloc[0]) if not r.empty else 0.0

@st.cache_data(show_spinner=False)
def recency_momentum(
    season_totals: pd.DataFrame,
    latest_year: int,
    window: int = 3,
    decay: float = 0.7,
) -> pd.DataFrame:
    if season_totals.empty:
        return pd.DataFrame(columns=["constructorId", "name", "momentum_score", "points_share"])

    start_year = max(latest_year - window + 1, int(season_totals["year"].min()))
    recent = season_totals[(season_totals["year"] >= start_year) & (season_totals["year"] <= latest_year)].copy()
    if recent.empty:
        return pd.DataFrame(columns=["constructorId", "name", "momentum_score", "points_share"])

    recent["weight"] = decay ** (latest_year - recent["year"])
    recent["w_points"] = recent["season_points"] * recent["weight"]

    agg = (
        recent.groupby(["constructorId", "name"], as_index=False)
        .agg(w_points=("w_points", "sum"))
    )

    total = agg["w_points"].sum()
    agg["points_share"] = (agg["w_points"] / total) if total > 0 else 0.0

    max_share = agg["points_share"].max() if not agg.empty else 1.0
    agg["momentum_score"] = (
        (agg["points_share"] / max_share).clip(0.0, 1.0)
        if max_share > 0
        else 0.0
    )

    return agg[["constructorId", "name", "momentum_score", "points_share"]]

# ---------------- Driver aggregates (historical) ----------------

@st.cache_data(show_spinner=False)
def driver_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Which constructor did each driver mainly score for in a season?
    assign = (
        df.groupby(["year", "driverId", "constructorId", "name"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "pts_for_team"})
    )
    idx = (
        assign.groupby(["year", "driverId"])["pts_for_team"]
        .idxmax()
        .dropna()
        .astype(int)
        .values
    )
    primary_team = assign.loc[idx, ["year", "driverId", "constructorId", "name"]].rename(columns={"name": "team_name"})
    # Driver totals per season
    base = (
        df.groupby(["year", "driverId", "Driver"], as_index=False)
        .agg(
            season_points=("points", "sum"),
            wins=("positionOrder", lambda s: np.sum(s == 1)),
            podiums=("positionOrder", lambda s: np.sum(s.isin([1, 2, 3]))),
            entries=("raceId", "nunique"),
        )
    )
    out = base.merge(primary_team, on=["year", "driverId"], how="left")
    out["season_points"] = out["season_points"].fillna(0.0)
    out["wins"] = out["wins"].fillna(0).astype(int)
    out["podiums"] = out["podiums"].fillna(0).astype(int)
    out["entries"] = out["entries"].fillna(0).astype(int)
    return out

def driver_recent_form(stats: pd.DataFrame, end_year: int, window: int = 3, decay: float = 0.7) -> pd.DataFrame:
    start = max(end_year - window + 1, int(stats["year"].min()))
    sub = stats[(stats["year"] >= start) & (stats["year"] <= end_year)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Driver","form_points_share","win_rate","podium_rate"])

    sub["weight"] = decay ** (end_year - sub["year"])
    sub["w_points"] = sub["season_points"] * sub["weight"]

    agg = (
        sub.groupby(["Driver"], as_index=False)
        .agg(w_points=("w_points","sum"),
             wins=("wins","sum"),
             podiums=("podiums","sum"),
             entries=("entries","sum"))
    )
    total = agg["w_points"].sum()
    agg["form_points_share"] = (agg["w_points"] / total) if total > 0 else 0.0
    agg["win_rate"] = (agg["wins"] / agg["entries"]).replace([np.inf, np.nan], 0.0)
    agg["podium_rate"] = (agg["podiums"] / agg["entries"]).replace([np.inf, np.nan], 0.0)

    return agg[["Driver","form_points_share","win_rate","podium_rate"]]

@st.cache_data(show_spinner=False)
def circuit_driver_form(df: pd.DataFrame, end_year: int, window: int = 4, decay: float = 0.7) -> pd.DataFrame:
    # defensive: some dumps don't have circuitId -> fallback to raceId
    sub = df.copy()
    if "circuitId" not in sub.columns:
        sub["circuitId"] = sub["raceId"]

    start = max(end_year - window + 1, int(sub["year"].min()))
    sub = sub[(sub["year"] >= start) & (sub["year"] <= end_year)].copy()
    sub["weight"] = decay ** (end_year - sub["year"])

    starts = sub.groupby(["Driver","circuitId"], as_index=False)["raceId"].nunique().rename(columns={"raceId":"starts"})
    wpts = sub.groupby(["Driver","circuitId"], as_index=False).agg(
        w_points=("points", lambda s: np.sum(s.fillna(0) * sub.loc[s.index, "weight"])),
        w_wins=("positionOrder", lambda s: np.sum((s == 1) * sub.loc[s.index, "weight"])),
        w_starts=("weight", "sum"),
    )
    merged = starts.merge(wpts, on=["Driver","circuitId"], how="right")
    merged["c_points_per_start"] = (merged["w_points"] / merged["starts"]).replace([np.inf, np.nan], 0.0)
    merged["c_wins_rate"] = (merged["w_wins"] / merged["w_starts"]).replace([np.inf, np.nan], 0.0)
    merged["c_score"] = 0.6 * merged["c_points_per_start"] + 0.4 * merged["c_wins_rate"]

    return merged[["Driver","circuitId","c_score"]]

# ---------------- Small helper: map a 2025 track label to a circuitId (best-effort) -------------
def map_track_to_circuit_id(track: str, df_hist: pd.DataFrame) -> Optional[int]:
    """Map a 2025 'Track' label (e.g., 'Australia', 'China') to a historical circuitId."""
    if not track:
        return None
    # try fuzzy match against name_race / location / country
    cols = [c for c in ["name_race", "location", "country"] if c in df_hist.columns]
    if not cols:
        return None
    sub = df_hist.copy()
    mask = False
    for c in cols:
        mask = mask | sub[c].astype(str).str.contains(track, case=False, na=False)
    hits = sub[mask]
    if hits.empty or "circuitId" not in hits.columns:
        return None
    try:
        return int(hits.groupby("circuitId").size().sort_values(ascending=False).index[0])
    except Exception:
        return None

# ============================================================
#  MAIN TAB
# ============================================================
def forecastingTab():
    st.header("📈 Forecasting — 2025 Season (uses 2025 results + historical)")
    # debug/sample visuals disabled in UI
    _debug_show_visuals = False

    # Load historical + 2025 season-to-date
    df_hist = get_historical_df()
    season_totals = constructor_season_points(df_hist)
    latest_year_in_data = int(season_totals["year"].max())
    base_year_for_momentum = latest_year_in_data

    df2025 = load_merged_dataset_2025()
    if df2025.empty:
        st.error("No 2025 CSVs found. Add them to Dataset/ and reload.")
        return

    # -----------------------------------------------------------------
    # SECTION 1. Constructor Comeback Forecast
    # -----------------------------------------------------------------
    champs, titles_by_constructor = compute_constructor_champions(season_totals)
    counts_global = global_gap_counts(titles_by_constructor)

    # Fixed Constructor Comeback settings (no UI controls)
    k_max = 25            # max gap horizon (years)
    laplace_alpha = 0.5   # Laplace smoothing alpha
    c_window = 3          # recency window (historical seasons)
    c_decay = 0.70        # recency decay per season
    beta_hist = 0.35      # momentum adjustment strength (historical)
    beta_2025 = 0.40      # momentum adjustment strength (2025 season)


    global_pmf = build_smoothed_pmf(counts_global, k_max=k_max, laplace_alpha=laplace_alpha)
    hist_momentum = recency_momentum(
        season_totals,
        latest_year=base_year_for_momentum,
        window=c_window,
        decay=c_decay
    )

    st.subheader("Historical Return-to-Title Probability (Constructors)")
    st.caption("Laplace-smoothed PMF = probability the *next* title arrives after a gap of k years, smoothed so rare/unseen k don't become 0.")
    if not global_pmf.empty:
        fig = px.bar(
            global_pmf,
            x="gap_years",
            y="probability",
            labels={"gap_years":"Years since last title", "probability":"Probability"},
            title="Laplace-smoothed PMF (all constructors)"
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap: gap counts per constructor (shows which teams had repeats at which gaps)
    if not global_pmf.empty or _debug_show_visuals:
        try:
            rows = []
            for cid, yrs in titles_by_constructor.items():
                if not yrs or len(yrs) < 2:
                    for i in range(1, k_max + 1):
                        rows.append({"constructorId": cid, "gap_years": i, "count": 0})
                else:
                    gaps = np.diff(sorted(yrs)).astype(int)
                    for i in range(1, k_max + 1):
                        rows.append({"constructorId": cid, "gap_years": i, "count": int((gaps == i).sum())})
            mat = pd.DataFrame(rows) if rows else pd.DataFrame()
            if mat.empty and _debug_show_visuals:
                # create small synthetic sample
                mat = pd.DataFrame([{"constructorId": 1, "gap_years": g, "count": int(np.random.poisson(1))} for g in range(1, 6)])
                names = {1: "Sample Team"}
            else:
                names = season_totals.drop_duplicates(subset=["constructorId"]).set_index("constructorId")["name"].to_dict()
            if not mat.empty:
                mat["name"] = mat["constructorId"].map(lambda x: names.get(x, str(x)))
                heat = mat.pivot(index="name", columns="gap_years", values="count").fillna(0)
                if not heat.empty:
                    # use team palette for the continuous scale to keep F1 colours
                    fig_h = px.imshow(
                        heat,
                        labels={"x":"Gap Years","y":"Constructor","color":"Count"},
                        aspect="auto",
                        color_continuous_scale=list(TEAM_COLORS.values()),
                        title="Constructor gap counts by gap length"
                    )
                    fig_h.update_xaxes(side="bottom")
                    st.plotly_chart(fig_h, use_container_width=True)
        except Exception:
            pass

    # 2025 constructor momentum (actual points scored so far)
    cons_2025 = (
        df2025[df2025["session"].isin(["Race","Sprint"])]
        .groupby(["Team"], as_index=False)["points"].sum()
        .rename(columns={"points":"points_2025"})
    )
    if not cons_2025.empty:
        cons_2025["share_2025"] = cons_2025["points_2025"] / cons_2025["points_2025"].sum()
        max_share = cons_2025["share_2025"].max()
        cons_2025["momentum_2025"] = (cons_2025["share_2025"] / max_share).clip(0,1)

    # map 2025 team names to historical constructorIds for comeback logic
    alias_map = {
        "Alpine": ["Renault"],
        "Aston Martin": ["Racing Point", "Force India"],
        "Ferrari": ["Ferrari"],
        "Haas": ["Haas"],
        "Kick Sauber": ["Alfa Romeo","Sauber"],
        "McLaren": ["McLaren"],
        "Mercedes": ["Mercedes"],
        "Racing Bulls": ["AlphaTauri","Toro Rosso"],
        "Red Bull": ["Red Bull"],
        "Williams": ["Williams"],
    }

    cand = season_totals[["year","constructorId","name"]].drop_duplicates()
    alias_rows = []
    for alias, prefs in alias_map.items():
        sub = cand[cand["year"] == base_year_for_momentum]
        row = None
        for p in prefs:
            m = sub[sub["name"].str.contains(p, case=False, regex=False)]
            if not m.empty:
                row = m.iloc[0]
                break
        if row is None:
            for p in prefs:
                m = cand[cand["name"].str.contains(p, case=False, regex=False)]
                if not m.empty:
                    row = m.iloc[0]
                    break
        alias_rows.append({
            "Team2025": alias,
            "constructorId": (int(row["constructorId"]) if row is not None else None)
        })
    alias_df = pd.DataFrame(alias_rows)

    def adjust_constructor_prob(team_alias: str, p_base: float) -> float:
        # historical momentum (0..1)
        row = alias_df[alias_df["Team2025"] == team_alias]
        m_hist = 0.0
        if not row.empty and pd.notna(row.iloc[0]["constructorId"]):
            cid = int(row.iloc[0]["constructorId"])
            mrow = hist_momentum[hist_momentum["constructorId"] == cid]
            if not mrow.empty:
                m_hist = float(mrow["momentum_score"].iloc[0])

        # 2025 in-season momentum (0..1)
        m_25 = 0.0
        if team_alias in set(cons_2025["Team"].apply(normalize_team)):
            # normalize both sides for safety
            norm_lookup = {normalize_team(t): v for t, v in cons_2025.set_index("Team")["momentum_2025"].items()}
            m_25 = float(norm_lookup.get(team_alias, 0.0))

        # combine both signals
        p = p_base * (1.0 + beta_hist  * (m_hist - 0.6))
        p = p      * (1.0 + beta_2025  * (m_25  - 0.6))
        return float(np.clip(p, 0.0, 1.0))

    # score each 2025 constructor's title chances
    rows_c = []
    for team_alias in alias_map.keys():
        row = alias_df[alias_df["Team2025"] == team_alias]
        title_years = []
        if not row.empty and pd.notna(row.iloc[0]["constructorId"]):
            cid = int(row.iloc[0]["constructorId"])
            title_years = titles_by_constructor.get(cid, [])
        p_base = 0.0
        if title_years:
            gap_k = 2025 - max(title_years)
            team_counts = team_gap_counts(title_years)
            team_pmf = (
                build_smoothed_pmf(team_counts, k_max=k_max, laplace_alpha=laplace_alpha)
                if not team_counts.empty else
                pd.DataFrame()
            )
            p_base = pmf_probability_at_gap(gap_k, team_pmf, global_pmf)
        p_adj = adjust_constructor_prob(team_alias, p_base)

        rows_c.append({
            "Constructor (2025)": team_alias,
            "Baseline P(Title 2025)": p_base,
            "Adjusted P(Title 2025)": p_adj
        })

    table_c = pd.DataFrame(rows_c).sort_values("Adjusted P(Title 2025)", ascending=False)
    if not table_c.empty:
        show_c = table_c.copy()
        show_c["Baseline P(Title 2025)"] = (show_c["Baseline P(Title 2025)"] * 100).map(lambda x: f"{x:.1f}%")
        show_c["Adjusted P(Title 2025)"] = (show_c["Adjusted P(Title 2025)"] * 100).map(lambda x: f"{x:.1f}%")
        st.dataframe(show_c, use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------
    # SECTION 2. Race Winner Prediction (Dropdown)
    # -----------------------------------------------------------------
    st.subheader("🏁 2025 Race Winner Prediction (Select a Round)")

    # ---------- Build 2025 calendar (robust) ----------
    # Coerce rounds to integers across ALL sessions we have
    df2025["round"] = pd.to_numeric(df2025["round"], errors="coerce").astype("Int64")

    # Finished rounds are those that have a Race result row
    finished_rounds = (
        df2025[df2025["session"] == "Race"]["round"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    finished_rounds = sorted(finished_rounds)

    # Season length from the 2025 grid definition (default 24)
    _grid_df = get_2025_grid()
    season_len = int(_grid_df["round"].max()) if not _grid_df.empty else 24
    all_rounds = list(range(1, season_len + 1))


    # Calendar for existing races we know names for (from any session)
    calendar_2025 = (
        df2025[["round", "race_name"]]
        .dropna(subset=["round"])
        .drop_duplicates()
        .astype({"round": int})
        .sort_values("round")
        .reset_index(drop=True)
    )

    # Build a full calendar frame [1..season_len]; fill names when we have them
    calendar_full = pd.DataFrame({"round": all_rounds}).merge(
        calendar_2025, on="round", how="left"
    )
    calendar_full["race_name"] = calendar_full["race_name"].fillna(
        calendar_full["round"].apply(lambda r: f"Round {r}")
    )

    # Remaining rounds = those NOT in finished_rounds
    remaining = calendar_full[~calendar_full["round"].isin(finished_rounds)].copy()

    if remaining.empty:
        st.info(
            "ℹ️ No ‘remaining’ rounds detected. "
            "This can happen if RaceResults already include all rounds or "
            "if the ‘round’ column wasn’t parsed correctly."
        )

    # Dropdown uses remaining (fallback to last finished if truly none)
    race_options = remaining["round"].tolist() or (finished_rounds[-1:] if finished_rounds else [1])
    selected_round = st.selectbox("Select upcoming round", race_options, index=0)

    # Selected race info — prefer Track/track column; else strip "Grand Prix" to get country-ish label
    race_row_25 = calendar_full.loc[calendar_full["round"] == selected_round]  
    sel_race_name = race_row_25["race_name"].iloc[0] if not race_row_25.empty else f"Round {selected_round}"

    # Try to get the 'Track' (country) name from  2025 CSVs
    track_col = "Track" if "Track" in df2025.columns else ("track" if "track" in df2025.columns else None)
    track_name = None
    if track_col is not None:
        track_vals = (
            df2025.loc[df2025["round"] == selected_round, track_col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        track_name = track_vals[0] if track_vals else None

    def _extract_country(name: str) -> Optional[str]:
        if not isinstance(name, str):
            return None
        # e.g. "Spanish Grand Prix" -> "Spanish"; "São Paulo Grand Prix" -> "São Paulo"
        if "Grand Prix" in name:
            return (
                name.replace("FORMULA 1", "")
                    .replace("Grand Prix", "")
                    .replace("™", "")
                    .strip(" -")
            )
        return name.strip()

    # Final display name for titles
    sel_display_name = track_name or _extract_country(sel_race_name) or f"Round {selected_round}"

    # Circuit id (unchanged logic)
    sel_circuit = map_track_to_circuit_id(sel_display_name, df_hist) or map_track_to_circuit_id(sel_race_name, df_hist)


    # And when you later need the total_rounds / remaining_rounds for WDC:
    total_rounds = season_len
    remaining_rounds = remaining["round"].tolist()

    # Fixed prediction settings (no UI controls)
    gamma = 0.45          # circuit vs overall blend
    w_team_2025 = 0.10    # team momentum weight (2025)
    w_hist = 0.60         # historical driver form weight
    w_2025 = 0.40         # 2025 driver form weight
    w_quali = 0.08        # qualifying boost if quali exists for that round
    temp = 1.0            # softmax temperature
    topK = 3              # number of bars to display
    st.caption("Bars are coloured by each driver's 2025 team.")


    # 2025 driver performance so far (Race + Sprint points)
    d2025_results = df2025[df2025["session"].isin(["Race","Sprint"])].copy()
    drv_2025 = (
        d2025_results.groupby(["Driver","Team"], as_index=False)["points"].sum()
        .rename(columns={"points":"points_2025"})
    )
    if not drv_2025.empty:
        drv_2025["share_2025"] = drv_2025["points_2025"] / drv_2025["points_2025"].sum()
        max_share_25 = drv_2025["share_2025"].max()
        drv_2025["form_2025"] = (drv_2025["share_2025"] / max_share_25).clip(0,1)
    else:
        drv_2025 = pd.DataFrame(columns=["Driver","Team","points_2025","share_2025","form_2025"])

    # Qualifying info for that (future) round if it already exists in 2025 data
    q_sel = df2025[
        (df2025["session"] == "Qualifying") &
        (df2025["round"] == selected_round)
    ].copy()
    if not q_sel.empty:
        q_sel["quali_boost"] = (q_sel["positionOrder"].max() + 1 - q_sel["positionOrder"]).astype(float)
        if q_sel["quali_boost"].max() > 0:
            q_sel["quali_boost"] = q_sel["quali_boost"] / q_sel["quali_boost"].max()
    else:
        q_sel = pd.DataFrame(columns=["Driver","quali_boost"])

    # Historical form (driver overall + circuit form)
    stats = driver_season_stats(df_hist)
    d_form_hist = driver_recent_form(stats, end_year=base_year_for_momentum, window=3, decay=0.75)
    c_form_hist = circuit_driver_form(df_hist, end_year=base_year_for_momentum, window=4, decay=0.7)

    if sel_circuit is not None:
        c_sel = c_form_hist[c_form_hist["circuitId"] == sel_circuit][["Driver","c_score"]].copy()
        if not c_sel.empty and c_sel["c_score"].max() > 0:
            c_sel["c_score"] = c_sel["c_score"] / c_sel["c_score"].max()
    else:
        c_sel = pd.DataFrame(columns=["Driver","c_score"])

    # Drivers actually eligible that round
    # Drivers actually eligible that round
    grid = get_2025_grid(rounds=season_len)
    elig = grid[grid["round"] == selected_round][["Driver","Team"]].copy()


    # Merge all scoring components
    merged = elig.merge(d_form_hist, on="Driver", how="left")
    merged = merged.merge(drv_2025[["Driver","form_2025"]], on="Driver", how="left")
    merged = merged.merge(c_sel, on="Driver", how="left")
    merged = merged.merge(q_sel[["Driver","quali_boost"]], on="Driver", how="left")

    merged["form_points_share"] = merged["form_points_share"].fillna(0.0)
    merged["win_rate"]          = merged["win_rate"].fillna(0.0)
    merged["podium_rate"]       = merged["podium_rate"].fillna(0.0)
    merged["form_2025"]         = merged["form_2025"].fillna(0.0)
    merged["c_score"]           = merged["c_score"].fillna(0.0)
    merged["quali_boost"]       = merged["quali_boost"].fillna(0.0)

    # historical driver composite
    merged["overall_hist"] = (
        0.65 * merged["form_points_share"] +
        0.25 * merged["win_rate"] +
        0.10 * merged["podium_rate"]
    )
    if merged["overall_hist"].max() > 0:
        merged["overall_hist"] = merged["overall_hist"] / merged["overall_hist"].max()

    # team momentum from 2025 so far (normalised by team label)
    if not cons_2025.empty:
        tmp = cons_2025.copy()
        tmp["Team_norm"] = tmp["Team"].apply(normalize_team)
        team_mom = tmp.set_index("Team_norm")["momentum_2025"].to_dict()
        merged["team_mom_2025"] = merged["Team"].apply(normalize_team).map(team_mom).fillna(0.0)
    else:
        merged["team_mom_2025"] = 0.0

    # normalize the weight sum
    weight_sum = max(w_hist + w_2025 + gamma + w_quali + w_team_2025, 1e-9)
    merged["score"] = (
        w_hist      * merged["overall_hist"] +
        w_2025      * merged["form_2025"] +
        gamma       * merged["c_score"] +
        w_quali     * merged["quali_boost"] +
        w_team_2025 * merged["team_mom_2025"]
    ) / weight_sum

    # convert to win probabilities
    merged["p_win"] = softmax(merged["score"].values, temperature=temp)

    # Top-K only
    topk = merged.sort_values("p_win", ascending=False).head(topK).copy()
    topk["p_win_pct"] = topk["p_win"] * 100.0

    # colour bars by Team using TEAM_COLORS via normalize_team (use plotly-express)
    topk = topk.copy()
    topk["Team_base"] = topk["Team"].apply(normalize_team)
    color_map = {t: team_color(t) for t in topk["Team_base"].unique()}
    fig_topk = px.bar(
        topk,
        x="Driver",
        y="p_win",
        color="Team_base",
        color_discrete_map=color_map,
        hover_data={"p_win":":.2f","Team_base":True},
        title=f"{sel_display_name} — Top-{topK} Win Probabilities",
    )
    fig_topk.update_layout(
        yaxis_title="Win Probability",
        xaxis_title="Driver",
        yaxis_tickformat=".0%",
        bargap=0.2,
        showlegend=False,
    )
    st.plotly_chart(fig_topk, use_container_width=True)
    # non-interactive Top-K (clicks disabled)

    st.dataframe(
        topk[["Driver","Team","p_win_pct"]].rename(columns={"p_win_pct":"Win Probability (%)"}),
        use_container_width=True,
        hide_index=True
    )

    st.caption("Bars are tinted by each driver's 2025 team colour. Quali boost is applied only if that round has qualifying data.")

    # --- Radar: component profile for Top-K drivers
    if (not topk.empty) or _debug_show_visuals:
        try:
            radar_metrics = ["overall_hist", "form_2025", "c_score", "quali_boost", "team_mom_2025"]
            radar_src = merged.set_index("Driver") if "Driver" in merged.columns else pd.DataFrame()
            # if user clicked a driver, show only that one
            drivers_to_plot = list(topk["Driver"]) if not topk.empty else radar_src.index.tolist()[:3]
            if _debug_show_visuals and not drivers_to_plot:
                # synthetic sample
                drivers_to_plot = ["Driver A","Driver B","Driver C"]
                radar_src = pd.DataFrame({m: np.random.rand(3) for m in radar_metrics}, index=drivers_to_plot)
            # build a long-form dataframe suitable for px.line_polar
            rlist = []
            for drv in drivers_to_plot:
                if drv not in radar_src.index:
                    continue
                row = radar_src.loc[drv]
                for m in radar_metrics:
                    val = float(row[m]) if m in radar_src.columns else 0.0
                    rlist.append({"Driver": drv, "metric": m, "value": val, "Team": row.get("Team", None) if hasattr(row, 'get') else (radar_src.at[drv, 'Team'] if 'Team' in radar_src.columns else None)})
            if rlist:
                df_rad = pd.DataFrame(rlist)
                # derive a base team column for each row (normalized team key)
                if "Team" in df_rad.columns:
                    df_rad["Team_base"] = df_rad["Team"].apply(lambda t: normalize_team(t) if pd.notna(t) else "Unknown")
                else:
                    df_rad["Team_base"] = "Unknown"

                # Build a mapping driver -> team-colour so Plotly colours each driver's trace by their team
                driver_list = df_rad["Driver"].unique().tolist()
                driver_color_map = {}
                for drv in driver_list:
                    try:
                        team_val = df_rad.loc[df_rad["Driver"] == drv, "Team_base"].iloc[0]
                    except Exception:
                        team_val = None
                    driver_color_map[drv] = team_color(team_val)

                # Use driver names for the colour scale (each driver gets their team's colour)
                fig_rad = px.line_polar(
                    df_rad,
                    r="value",
                    theta="metric",
                    color="Driver",
                    line_close=True,
                    color_discrete_map=driver_color_map,
                )
                fig_rad.update_traces(fill="toself")
                fig_rad.update_layout(title=f"Component profile — Top-{topK} drivers", polar=dict(radialaxis=dict(visible=True, range=[0,1])))
                st.plotly_chart(fig_rad, use_container_width=True)
        except Exception:
            pass

    st.divider()

    # -----------------------------------------------------------------
    # SECTION 3. WDC Expected Points Projection (Rest of 2025)
    # -----------------------------------------------------------------
    st.subheader("👑 2025 WDC — Expected Points (Actual so far + Forecast for remaining)")

    # Points already scored in 2025 (Race + Sprint)
    earned = (
        d2025_results.groupby(["Driver"], as_index=False)["points"].sum()
        .rename(columns={"points":"points_so_far"})
    )

    # Remaining rounds to forecast (from calendar_2025 built above)
    remaining_rounds = remaining["round"].tolist()
    points_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

    # expected points across remaining rounds
    grid_all_rounds = get_2025_grid(rounds=season_len)

    exp_points = {drv: 0.0 for drv in grid_all_rounds["Driver"].unique()}

    for rnd in remaining_rounds:
        # get track name for this round and map to a historical circuitId
        rr = calendar_2025[calendar_2025["round"] == rnd]
        track_name = rr["race_name"].iloc[0] if not rr.empty else None
        cir = map_track_to_circuit_id(track_name, df_hist) if track_name else None

        elig_r = grid_all_rounds[grid_all_rounds["round"] == rnd][["Driver","Team"]].copy()

        # circuit form for this round (optional)
        if cir is not None:
            cf = c_form_hist[c_form_hist["circuitId"] == cir][["Driver","c_score"]].copy()
            if not cf.empty and cf["c_score"].max() > 0:
                cf["c_score"] = cf["c_score"] / cf["c_score"].max()
        else:
            cf = pd.DataFrame(columns=["Driver","c_score"])

        merged_r = elig_r.merge(d_form_hist, on="Driver", how="left")
        merged_r = merged_r.merge(drv_2025[["Driver","form_2025"]], on="Driver", how="left")
        merged_r = merged_r.merge(cf, on="Driver", how="left")

        # No quali for future rounds by default
        merged_r["quali_boost"] = 0.0

        for col in ["form_points_share","win_rate","podium_rate","form_2025","c_score"]:
            merged_r[col] = merged_r[col].fillna(0.0)

        merged_r["overall_hist"] = (
            0.65 * merged_r["form_points_share"] +
            0.25 * merged_r["win_rate"] +
            0.10 * merged_r["podium_rate"]
        )
        if merged_r["overall_hist"].max() > 0:
            merged_r["overall_hist"] = merged_r["overall_hist"] / merged_r["overall_hist"].max()

        # team momentum (normalize team labels first)
        if not cons_2025.empty:
            tmp = cons_2025.copy()
            tmp["Team_norm"] = tmp["Team"].apply(normalize_team)
            team_mom2 = tmp.set_index("Team_norm")["momentum_2025"].to_dict()
            merged_r["team_mom_2025"] = merged_r["Team"].apply(normalize_team).map(team_mom2).fillna(0.0)
        else:
            merged_r["team_mom_2025"] = 0.0

        weight_sum_future = max(w_hist + w_2025 + gamma + w_quali + w_team_2025, 1e-9)
        merged_r["score"] = (
            w_hist      * merged_r["overall_hist"] +
            w_2025      * merged_r["form_2025"] +
            gamma       * merged_r["c_score"] +
            w_quali     * 0.0 +
            w_team_2025 * merged_r["team_mom_2025"]
        ) / weight_sum_future

        # Expected points via a simple probabilistic ranking (Plackett-Luce style)
        # We'll run a small Monte-Carlo to sample finishing orders proportionally to each
        # driver's score to produce more realistic position-point expectations.
        try:
            rng = np.random.default_rng(12345)
            n_sims = 400  # adjust for speed/accuracy tradeoff

            drivers = merged_r["Driver"].tolist()
            scores = merged_r["score"].to_numpy(dtype=float)
            # shift to non-negative and ensure non-zero mass
            if scores.size == 0:
                continue
            if scores.min() < 0:
                scores = scores - scores.min()
            if scores.sum() == 0:
                scores = np.ones_like(scores)

            # prepare accumulator
            round_points_acc = {drv: 0.0 for drv in drivers}

            for _ in range(n_sims):
                remaining = list(range(len(drivers)))
                # sample top-10 by iterative proportional sampling
                for pos, pts in points_map.items():
                    rem_scores = scores[remaining]
                    total = rem_scores.sum()
                    if total <= 0:
                        # fallback: uniform among remaining
                        probs = np.ones(len(remaining)) / len(remaining)
                    else:
                        probs = rem_scores / total
                    chosen = rng.choice(remaining, p=probs)
                    round_points_acc[drivers[chosen]] += pts
                    remaining.remove(chosen)

            # average over sims and add to exp_points
            for drv, pts_sum in round_points_acc.items():
                exp_points[drv] += pts_sum / float(n_sims)
        except Exception:
            # fallback to previous simple softmax method if MC fails
            for pos, pts in points_map.items():
                probs = softmax(merged_r["score"].values, temperature=1.0 + 0.15*(pos-1))
                for i, row in merged_r.iterrows():
                    exp_points[row["Driver"]] += pts * probs[i]

    # Round expected remaining points to whole numbers (user preference)
    rounded_exp = {drv: int(round(v)) for drv, v in exp_points.items()}

    # Combine with already scored
    wdc = pd.DataFrame({
        "Driver": list(rounded_exp.keys()),
        "Expected Points (remaining)": list(rounded_exp.values())
    })
    wdc = wdc.merge(earned, on="Driver", how="left").fillna({"points_so_far":0.0})
    wdc["Total Expected (2025)"] = wdc["points_so_far"] + wdc["Expected Points (remaining)"]

    # Only drivers actually in 2025 grid
    wdc = wdc[wdc["Driver"].isin(grid_all_rounds["Driver"].unique())]
    wdc = wdc.sort_values("Total Expected (2025)", ascending=False)

    # ---- Team colouring for bars (robust to noisy team strings) ----
    # Prefer the team a driver scored most 2025 points with; fallback = grid majority team
    drv_team_points = d2025_results.groupby(['Driver','Team'], as_index=False)['points'].sum()
    if not drv_team_points.empty:
        idx_primary = drv_team_points.groupby('Driver')['points'].idxmax()
        primary_team_pts = drv_team_points.loc[idx_primary, ['Driver','Team']]
    else:
        primary_team_pts = pd.DataFrame(columns=['Driver','Team'])

    grid_counts = grid_all_rounds.groupby(['Driver','Team']).size().reset_index(name='n')
    if not grid_counts.empty:
        idx_grid = grid_counts.groupby('Driver')['n'].idxmax()
        primary_grid = grid_counts.loc[idx_grid, ['Driver','Team']]
    else:
        primary_grid = pd.DataFrame(columns=['Driver','Team'])

    primary_team = primary_team_pts.merge(primary_grid, on='Driver', how='outer', suffixes=('', '_grid'))
    primary_team['Team'] = primary_team['Team'].fillna(primary_team['Team_grid'])
    primary_team = primary_team[['Driver','Team']]

    # Attach team & normalise to base key for colours
    wdc = wdc.merge(primary_team, on='Driver', how='left')
    wdc['Team_base'] = wdc['Team'].apply(normalize_team)

    # Plot top-15 with team colours (plotly-express)
    top15 = wdc.head(15).copy()
    top15['Team_base'] = top15['Team'].apply(normalize_team)
    color_map_wdc = {t: team_color(t) for t in top15['Team_base'].unique()}
    fig_wdc = px.bar(
        top15,
        x='Driver',
        y='Total Expected (2025)',
        color='Team_base',
        color_discrete_map=color_map_wdc,
        hover_data={'Total Expected (2025)':':.1f','Team_base':True},
        title='2025 WDC — Expected Points (Actual so far + Projected Remaining)'
    )
    fig_wdc.update_layout(xaxis_title='Driver', yaxis_title='Expected Points', bargap=0.2, showlegend=False)
    st.plotly_chart(fig_wdc, use_container_width=True)
    # non-interactive WDC chart

    # --- Violin: distribution of Total Expected per Team + Parallel Coordinates for top drivers
    if (not wdc.empty and 'Team_base' in wdc.columns) or _debug_show_visuals:
        try:
            vdf = wdc.copy()
            if vdf.empty and _debug_show_visuals:
                # synthetic sample
                vdf = pd.DataFrame({
                    'Driver': ['A','B','C','D','E'],
                    'Team_base': ['Team X','Team X','Team Y','Team Y','Team Z'],
                    'Total Expected (2025)': np.random.uniform(10, 300, 5)
                })
            # (non-interactive) show violin distribution for available teams
            if not vdf.empty:
                # map Team_base categories to exact team hex colours
                uniq_teams = sorted(vdf["Team_base"].dropna().unique().tolist())
                color_map = {t: team_color(t) for t in uniq_teams}
                fig_v = px.violin(
                    vdf,
                    x="Team_base",
                    y="Total Expected (2025)",
                    color="Team_base",
                    color_discrete_map=color_map,
                    box=True,
                    points="all",
                    title="Distribution of Total Expected (2025) by Team"
                )
                fig_v.update_layout(xaxis_title="Team", yaxis_title="Total Expected Points", showlegend=False)
                st.plotly_chart(fig_v, use_container_width=True)

            # Parallel coordinates for top-10 drivers to compare multi-dim signals
            top_n = wdc.head(10).copy()
            if top_n.empty and _debug_show_visuals:
                top_n = pd.DataFrame({
                    'Driver': ['A','B','C'],
                    'points_so_far': [50,120,80],
                    'Expected Points (remaining)': [30,40,20],
                    'Total Expected (2025)': [80,160,100]
                })
            # (non-interactive) show parallel coordinates for top drivers
            if not top_n.empty:
                pcs = top_n[["Driver","Team","points_so_far","Expected Points (remaining)","Total Expected (2025)"]].copy()
                pcs = pcs.rename(columns={"points_so_far":"Points So Far","Expected Points (remaining)":"Expected (Remaining)","Total Expected (2025)":"Total Expected","Team":"Team"})
                # map teams to integer codes so we can apply a discrete colour scale
                teams = pcs["Team"].fillna("Unknown").astype(str)
                uniq = teams.unique().tolist()
                team_to_code = {t: i for i, t in enumerate(uniq)}
                pcs["Team_code"] = teams.map(team_to_code)
                # build color scale matching team order
                color_list = [team_color(t) for t in uniq]
                # parallel_coordinates expects numeric columns only; include Team_code so we can colour by it
                plot_df = pcs[["Points So Far","Expected (Remaining)","Total Expected","Team_code"]].copy()
                fig_p = px.parallel_coordinates(
                    plot_df,
                    color="Team_code",
                    color_continuous_scale=color_list,
                    labels={"Points So Far":"Points So Far","Expected (Remaining)":"Expected (Remaining)","Total Expected":"Total Expected","Team_code":"Team"},
                    title="Parallel coordinates — top drivers (multi-dim comparison)"
                )
                st.plotly_chart(fig_p, use_container_width=True)
        except Exception:
            pass

    # Table (include Team for clarity)
    st.dataframe(
        wdc.rename(columns={
            "points_so_far":"Points So Far",
            "Expected Points (remaining)":"Expected (Remaining)"
        })[['Driver','Team','Points So Far','Expected (Remaining)','Total Expected (2025)']],
        use_container_width=True, hide_index=True
    )

    if not wdc.empty:
        champ = wdc.iloc[0]
        st.success(
            f"**Projected WDC (2025)**: {champ['Driver']} — total expected ≈ "
            f"**{champ['Total Expected (2025)']:.1f}**"
        )
