import numpy as np
import pandas as pd


FIRST_INNINGS_FEATURES = [
    "batting_team",
    "bowling_team",
    "venue",
    "phase",
    "current_score",
    "current_wickets",
    "wickets_remaining",
    "balls_remaining",
    "current_run_rate",
    "last_30_runs",
    "pressure_index",
]

SECOND_INNINGS_FEATURES = [
    "batting_team",
    "bowling_team",
    "venue",
    "phase",
    "current_score",
    "current_wickets",
    "wickets_remaining",
    "balls_remaining",
    "target",
    "runs_required",
    "current_run_rate",
    "required_run_rate",
    "last_30_runs",
    "pressure_index",
]

SECOND_INNINGS_SCORE_FEATURES = SECOND_INNINGS_FEATURES.copy()


def clean_cricket_dataset(df: pd.DataFrame) -> pd.DataFrame:
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


def _phase_from_balls(balls_bowled: pd.Series) -> pd.Series:
    return np.select(
        [balls_bowled <= 36, balls_bowled <= 90],
        ["Powerplay", "Middle Overs"],
        default="Death Overs",
    )


def _infer_match_context(ball_df: pd.DataFrame) -> pd.DataFrame:
    innings_summary = (
        ball_df.groupby(["match_id", "innings", "batting_team", "bowling_team", "venue"], as_index=False)
        .agg(total_score=("ball_runs", "sum"), total_wickets=("wicket_event", "sum"))
    )

    score_map = innings_summary.pivot(index="match_id", columns="innings", values="total_score")

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

    match_context = (
        score_map.reset_index()
        .merge(team_1, on="match_id", how="left")
        .merge(team_2, on="match_id", how="left")
        .merge(team_3, on="match_id", how="left")
        .merge(team_4, on="match_id", how="left")
    )

    def winner_from_row(row):
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

    match_context["winner"] = match_context.apply(winner_from_row, axis=1)
    match_context["target"] = match_context[1] + 1
    match_context["first_innings_total"] = match_context[1]
    match_context["second_innings_total"] = match_context.get(2, np.nan)
    return match_context[["match_id", "winner", "target", "first_innings_total", "second_innings_total"]]


def build_connected_feature_frames(clean_df: pd.DataFrame):
    ball_df = clean_df.copy()

    ball_df["ball_runs"] = ball_df["runs_of_bat"] + ball_df["extras"]
    ball_df["wicket_event"] = ball_df["player_dismissed"].notna().astype(int)
    ball_df["legal_delivery"] = ((ball_df["wide"] == 0) & (ball_df["noballs"] == 0)).astype(int)

    grp = ["match_id", "innings"]
    ball_df["current_score"] = ball_df.groupby(grp)["ball_runs"].cumsum()
    ball_df["current_wickets"] = ball_df.groupby(grp)["wicket_event"].cumsum()
    ball_df["balls_bowled"] = ball_df.groupby(grp)["legal_delivery"].cumsum()

    ball_df["balls_remaining"] = (120 - ball_df["balls_bowled"]).clip(lower=0)
    ball_df["wickets_remaining"] = (10 - ball_df["current_wickets"]).clip(lower=0)
    ball_df["current_run_rate"] = np.where(
        ball_df["balls_bowled"] > 0,
        ball_df["current_score"] * 6 / ball_df["balls_bowled"],
        0,
    )
    # Backward-compatible alias used by some visualization/runtime paths.
    ball_df["run_rate"] = ball_df["current_run_rate"]

    ball_df["last_30_runs"] = (
        ball_df.groupby(grp)["ball_runs"]
        .rolling(window=30, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    ball_df["phase"] = _phase_from_balls(ball_df["balls_bowled"])

    match_context = _infer_match_context(ball_df)
    ball_df = ball_df.merge(match_context, on="match_id", how="left")

    ball_df["runs_required"] = np.where(
        ball_df["innings"] == 2,
        (ball_df["target"] - ball_df["current_score"]).clip(lower=0),
        np.nan,
    )

    ball_df["required_run_rate"] = np.where(
        (ball_df["innings"] == 2) & (ball_df["balls_remaining"] > 0),
        ball_df["runs_required"] * 6 / ball_df["balls_remaining"],
        np.nan,
    )

    first_pressure = (8.0 - ball_df["current_run_rate"]) + (ball_df["current_wickets"] * 0.35)
    second_pressure = (ball_df["required_run_rate"] - ball_df["current_run_rate"]) + (ball_df["current_wickets"] * 0.5)
    ball_df["pressure_index"] = np.where(ball_df["innings"] == 2, second_pressure, first_pressure)
    ball_df["pressure_index"] = ball_df["pressure_index"].clip(lower=-5.0, upper=25.0)

    # First innings regression dataset.
    first_df = ball_df[(ball_df["innings"] == 1) & (ball_df["balls_bowled"] > 0)].copy()
    first_df["projected_total_target"] = first_df["first_innings_total"]
    first_df = first_df[FIRST_INNINGS_FEATURES + ["projected_total_target", "match_id"]].dropna()

    # Second innings classification dataset.
    second_df = ball_df[(ball_df["innings"] == 2) & (ball_df["winner"].notna())].copy()
    second_df = second_df[second_df["winner"] != "TIE/NO_RESULT"]
    # Keep only live in-play states to avoid label leakage from terminal states.
    second_df = second_df[
        (second_df["balls_remaining"] > 0)
        & (second_df["wickets_remaining"] > 0)
        & (second_df["runs_required"] > 0)
    ]
    second_df["win"] = (second_df["batting_team"] == second_df["winner"]).astype(int)
    second_df = second_df[SECOND_INNINGS_FEATURES + ["win", "match_id"]].dropna()

    # Second innings regression dataset for projected chase total.
    second_score_df = ball_df[(ball_df["innings"] == 2) & ball_df["second_innings_total"].notna()].copy()
    second_score_df = second_score_df[
        (second_score_df["balls_bowled"] > 0)
        & (second_score_df["balls_remaining"] > 0)
        & (second_score_df["wickets_remaining"] > 0)
        & (second_score_df["runs_required"] > 0)
    ]
    second_score_df["second_innings_total_target"] = second_score_df["second_innings_total"]
    second_score_df = second_score_df[
        SECOND_INNINGS_SCORE_FEATURES + ["second_innings_total_target", "match_id"]
    ].dropna()

    return ball_df, first_df, second_df, second_score_df
