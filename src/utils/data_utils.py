import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "batting_team",
    "bowling_team",
    "venue",
    "innings_phase",
    "current_score",
    "wickets_remaining",
    "balls_remaining",
    "current_run_rate",
    "target",
    "required_run_rate",
    "last_30_runs",
    "pressure_index",
]

TARGET_COLUMN = "win"


def parse_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()

    text_cols = [
        "phase",
        "date",
        "venue",
        "batting_team",
        "bowling_team",
        "striker",
        "bowler",
        "wicket_type",
        "player_dismissed",
        "fielder",
    ]
    for col in text_cols:
        clean[col] = clean[col].astype(str).str.strip()

    for col in ["wicket_type", "player_dismissed", "fielder"]:
        clean[col] = clean[col].replace({"nan": np.nan, "NaN": np.nan, "": np.nan})

    numeric_cols = [
        "match_id",
        "season",
        "match_no",
        "innings",
        "over",
        "runs_of_bat",
        "extras",
        "wide",
        "legbyes",
        "byes",
        "noballs",
    ]
    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean["byes"] = clean["byes"].fillna(0)
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")

    clean["batting_team"] = clean["batting_team"].str.upper()
    clean["bowling_team"] = clean["bowling_team"].str.upper()

    clean["over_no"] = clean["over"].astype(int)
    clean["ball_no"] = ((clean["over"] - clean["over_no"]) * 10).round().astype(int)

    clean = clean.sort_values(["match_id", "innings", "over_no", "ball_no"]).reset_index(drop=True)
    return clean


def infer_match_winners(feat_df: pd.DataFrame) -> pd.DataFrame:
    innings_summary = (
        feat_df.groupby(["match_id", "innings", "batting_team"], as_index=False)["ball_runs"]
        .sum()
        .rename(columns={"ball_runs": "innings_total"})
    )

    score_map = innings_summary.pivot(index="match_id", columns="innings", values="innings_total")

    team_1 = innings_summary[innings_summary["innings"] == 1][["match_id", "batting_team"]].rename(
        columns={"batting_team": "team_1"}
    )
    team_2 = innings_summary[innings_summary["innings"] == 2][["match_id", "batting_team"]].rename(
        columns={"batting_team": "team_2"}
    )
    team_3 = innings_summary[innings_summary["innings"] == 3][["match_id", "batting_team"]].rename(
        columns={"batting_team": "team_3"}
    )
    team_4 = innings_summary[innings_summary["innings"] == 4][["match_id", "batting_team"]].rename(
        columns={"batting_team": "team_4"}
    )

    match_df = (
        score_map.reset_index()
        .merge(team_1, on="match_id", how="left")
        .merge(team_2, on="match_id", how="left")
        .merge(team_3, on="match_id", how="left")
        .merge(team_4, on="match_id", how="left")
    )

    def _winner(row):
        s1, s2 = row.get(1, np.nan), row.get(2, np.nan)
        if pd.isna(s1) or pd.isna(s2):
            return np.nan
        if s1 > s2:
            return row["team_1"]
        if s2 > s1:
            return row["team_2"]

        s3, s4 = row.get(3, np.nan), row.get(4, np.nan)
        if pd.notna(s3) and pd.notna(s4):
            if s3 > s4:
                return row["team_3"]
            if s4 > s3:
                return row["team_4"]
        return "TIE/NO_RESULT"

    match_df["winner"] = match_df.apply(_winner, axis=1)
    match_df["target"] = match_df[1] + 1
    return match_df[["match_id", "winner", "target"]]


def build_modeling_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    feat = clean_df.copy()
    feat["ball_runs"] = feat["runs_of_bat"] + feat["extras"]
    feat["wicket_event"] = feat["player_dismissed"].notna().astype(int)
    feat["legal_delivery"] = ((feat["wide"] == 0) & (feat["noballs"] == 0)).astype(int)

    innings_group = ["match_id", "innings"]
    feat["current_score"] = feat.groupby(innings_group)["ball_runs"].cumsum()
    feat["wickets_fallen"] = feat.groupby(innings_group)["wicket_event"].cumsum()
    feat["balls_bowled"] = feat.groupby(innings_group)["legal_delivery"].cumsum()

    feat["balls_remaining"] = (120 - feat["balls_bowled"]).clip(lower=0)
    feat["wickets_remaining"] = (10 - feat["wickets_fallen"]).clip(lower=0)

    feat["current_run_rate"] = np.where(
        feat["balls_bowled"] > 0,
        feat["current_score"] * 6 / feat["balls_bowled"],
        0,
    )

    feat["last_30_runs"] = (
        feat.groupby(innings_group)["ball_runs"]
        .rolling(window=30, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    feat["innings_phase"] = np.select(
        [feat["balls_bowled"] <= 36, feat["balls_bowled"] <= 90],
        ["Powerplay", "Middle Overs"],
        default="Death Overs",
    )

    winner_df = infer_match_winners(feat)
    feat = feat.merge(winner_df, on="match_id", how="left")

    feat = feat[(feat["innings"] == 2) & feat["winner"].notna()].copy()
    feat = feat[feat["winner"] != "TIE/NO_RESULT"].copy()

    feat["runs_required"] = (feat["target"] - feat["current_score"]).clip(lower=0)
    feat["required_run_rate"] = np.where(
        feat["balls_remaining"] > 0,
        feat["runs_required"] * 6 / feat["balls_remaining"],
        0,
    )

    feat["pressure_index"] = (feat["required_run_rate"] - feat["current_run_rate"]).clip(
        lower=-10, upper=20
    )

    # Keep only live chase states to avoid terminal-state leakage.
    feat = feat[
        (feat["balls_remaining"] > 0)
        & (feat["wickets_remaining"] > 0)
        & (feat["runs_required"] > 0)
    ].copy()

    feat[TARGET_COLUMN] = (feat["batting_team"] == feat["winner"]).astype(int)

    model_df = feat[FEATURE_COLUMNS + [TARGET_COLUMN, "match_id"]].dropna().copy()
    return model_df
