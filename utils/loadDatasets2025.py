# utils/loadDatasets2025.py
import pandas as pd
import streamlit as st

SEASON = 2025

# Exact file paths you provided
RACE_PATH   = "Dataset/F1_2025_RaceResults.csv"
QUALI_PATH  = "Dataset/F1_2025_QualifyingResults.csv"
SPRINT_PATH = "Dataset/F1_2025_SprintResults.csv"
SQ_PATH     = "Dataset/F1_2025_SprintQualifyingResults.csv"


# ---------------------------- helpers ----------------------------
def _read(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # strip any accidental whitespace from headers
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def _parse_int_nullable(s: pd.Series) -> pd.Series:
    # Position can be number, 'NC', 'DSQ' → NaN for non-numeric
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _parse_float(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float).fillna(fill)

def _clean_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for c in ["Track", "Driver", "Team"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "Laps" in df.columns:
        df["Laps"] = _parse_int_nullable(df["Laps"]).fillna(0)
    return df

def _round_map_from_tracks(race: pd.DataFrame, sprint: pd.DataFrame,
                           sq: pd.DataFrame, quali: pd.DataFrame) -> dict:
    order = []
    for block in (race, sprint, sq, quali):
        if block is not None and not block.empty and "Track" in block.columns:
            for t in block["Track"].dropna():
                if t not in order:
                    order.append(t)
    return {t: i + 1 for i, t in enumerate(order)}

def _finalize_block(df: pd.DataFrame, session: str, round_map: dict, has_points: bool) -> pd.DataFrame:
    """
    Return canonical columns used by forecasting:
    ['season','session','round','race_name','date','Driver','Team','positionOrder','points']
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "season","session","round","race_name","date","Driver","Team","positionOrder","points"
        ])

    df = _clean_fields(df)

    # Position → positionOrder (nullable int). 'NC'/'DSQ' -> <NA>
    if "Position" in df.columns:
        df["positionOrder"] = _parse_int_nullable(df["Position"])
    else:
        df["positionOrder"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # Points for Race/Sprint; 0 otherwise
    if has_points and "Points" in df.columns:
        df["points"] = _parse_float(df["Points"], fill=0.0)
    else:
        df["points"] = 0.0

    # Round & race name
    df["round"] = df["Track"].map(round_map).astype("Int64") if "Track" in df.columns else pd.Series(pd.NA, dtype="Int64")
    df["race_name"] = df["Track"] if "Track" in df.columns else "Unknown GP"

    # Fixed fields (no date in your CSVs)
    df["season"] = SEASON
    df["session"] = session
    df["date"] = pd.NaT

    keep = ["season","session","round","race_name","date","Driver","Team","positionOrder","points"]
    return df[keep]


# ----------------------------- public -----------------------------
@st.cache_data(show_spinner=False)
def load_merged_dataset_2025() -> pd.DataFrame:
    """
    Load the four 2025 CSVs (exact headers), make a tidy unified table.
    """
    race   = _read(RACE_PATH)[["Track","Position","No","Driver","Team","Starting Grid","Laps","Time/Retired","Points","Set Fastest Lap","Fastest Lap Time"]].copy() if not _read(RACE_PATH).empty else pd.DataFrame()
    sprint = _read(SPRINT_PATH)[["Track","Position","No","Driver","Team","Starting Grid","Laps","Time/Retired","Points"]].copy() if not _read(SPRINT_PATH).empty else pd.DataFrame()
    sq     = _read(SQ_PATH)[["Track","Position","No","Driver","Team","Q1","Q2","Q3","Laps"]].copy() if not _read(SQ_PATH).empty else pd.DataFrame()
    quali  = _read(QUALI_PATH)[["Track","Position","No","Driver","Team","Q1","Q2","Q3","Laps"]].copy() if not _read(QUALI_PATH).empty else pd.DataFrame()

    round_map = _round_map_from_tracks(race, sprint, sq, quali)

    blocks = []
    if not race.empty:
        blocks.append(_finalize_block(race,   "Race",              round_map, has_points=True))
    if not sprint.empty:
        blocks.append(_finalize_block(sprint, "Sprint",            round_map, has_points=True))
    if not sq.empty:
        blocks.append(_finalize_block(sq,     "Sprint Qualifying", round_map, has_points=False))
    if not quali.empty:
        blocks.append(_finalize_block(quali,  "Qualifying",        round_map, has_points=False))

    if not blocks:
        return pd.DataFrame(columns=[
            "season","session","round","race_name","date","Driver","Team","positionOrder","points"
        ])

    merged = pd.concat(blocks, ignore_index=True)

    # Sort: round asc → session priority → finishing order
    session_order = {"Qualifying":0, "Sprint Qualifying":1, "Sprint":2, "Race":3}
    merged["__s"] = merged["session"].map(session_order).fillna(9)
    merged = merged.sort_values(["round","__s","positionOrder"], na_position="last").drop(columns="__s").reset_index(drop=True)
    return merged


@st.cache_data(show_spinner=False)
def load_recent_races_results_2025(df_2025_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Latest 5 completed races (session='Race') with top-5 finishers,
    ordered by 'round' (no dates in the 2025 CSVs).
    """
    race_fin = df_2025_merged[df_2025_merged["session"] == "Race"].dropna(subset=["round"])
    if race_fin.empty:
        return pd.DataFrame(columns=["round","race_name","Driver","Team","positionOrder","points"])

    last_rounds = (
        race_fin[["round","race_name"]]
        .drop_duplicates()
        .sort_values("round", ascending=False)
        .head(5)["round"]
        .tolist()
    )
    out = (
        race_fin[race_fin["round"].isin(last_rounds)]
        .sort_values(["round","positionOrder"])
        .groupby("round")
        .head(5)[["round","race_name","Driver","Team","positionOrder","points"]]
        .reset_index(drop=True)
    )
    return out
