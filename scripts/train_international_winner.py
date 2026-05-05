import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COUNTRY_LOGOS = {
    "IND": "https://flagcdn.com/w80/in.png",
    "AUS": "https://flagcdn.com/w80/au.png",
    "ENG": "https://flagcdn.com/w80/gb.png",
    "NZ": "https://flagcdn.com/w80/nz.png",
    "PAK": "https://flagcdn.com/w80/pk.png",
    "SA": "https://flagcdn.com/w80/za.png",
    "RSA": "https://flagcdn.com/w80/za.png",
    "WI": "https://flagcdn.com/w80/jm.png",
    "SL": "https://flagcdn.com/w80/lk.png",
    "AFG": "https://flagcdn.com/w80/af.png",
    "BAN": "https://flagcdn.com/w80/bd.png",
    "USA": "https://flagcdn.com/w80/us.png",
    "IRE": "https://flagcdn.com/w80/ie.png",
    "SCO": "https://flagcdn.com/w80/gb-sct.png",
    "NED": "https://flagcdn.com/w80/nl.png",
    "NAM": "https://flagcdn.com/w80/na.png",
    "UAE": "https://flagcdn.com/w80/ae.png",
    "OMAN": "https://flagcdn.com/w80/om.png",
    "CAN": "https://flagcdn.com/w80/ca.png",
    "NEP": "https://flagcdn.com/w80/np.png",
    "PNG": "https://flagcdn.com/w80/pg.png",
    "UGA": "https://flagcdn.com/w80/ug.png",
    "ZIM": "https://flagcdn.com/w80/zw.png",
}

TEAM_COLORS = {
    "IND": "#1d4ed8",
    "AUS": "#f59e0b",
    "ENG": "#ef4444",
    "NZ": "#111827",
    "PAK": "#16a34a",
    "SA": "#15803d",
    "RSA": "#15803d",
    "WI": "#7c3aed",
    "SL": "#2563eb",
    "AFG": "#f97316",
    "BAN": "#16a34a",
    "USA": "#dc2626",
    "IRE": "#22c55e",
    "SCO": "#0ea5e9",
    "NED": "#f97316",
    "NAM": "#06b6d4",
    "UAE": "#0f766e",
    "OMAN": "#b91c1c",
    "CAN": "#dc2626",
    "NEP": "#1d4ed8",
    "PNG": "#f59e0b",
    "UGA": "#15803d",
    "ZIM": "#84cc16",
}

TEAM_ALIASES = {
    "SOUTH AFRICA": "SA",
    "SOUTHAFRICA": "SA",
    "SRI LANKA": "SL",
    "WEST INDIES": "WI",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    "UNITED ARAB EMIRATES": "UAE",
    "UNITED KINGDOM": "ENG",
    "ENGLAND": "ENG",
    "AUSTRALIA": "AUS",
    "INDIA": "IND",
    "PAKISTAN": "PAK",
    "NEW ZEALAND": "NZ",
    "BANGLADESH": "BAN",
    "AFGHANISTAN": "AFG",
    "SCOTLAND": "SCO",
    "NETHERLANDS": "NED",
    "NAMIBIA": "NAM",
    "OMAN": "OMAN",
    "CANADA": "CAN",
    "NEPAL": "NEP",
    "PAPUA NEW GUINEA": "PNG",
    "UGANDA": "UGA",
    "ZIMBABWE": "ZIM",
}


def _canonical_team(name: str) -> str:
    if name is None:
        return ""
    key = str(name).strip().upper()
    return TEAM_ALIASES.get(key, key)


def _encode_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_analytics_chart(df: pd.DataFrame, title: str) -> str:
    fig, axes = plt.subplots(4, 3, figsize=(28, 24), facecolor="white")
    fig.suptitle(title, fontsize=32, fontweight="bold", y=0.98)

    df["batting_team_c"] = df["batting_team"].apply(_canonical_team)
    df["bowling_team_c"] = df["bowling_team"].apply(_canonical_team)

    # Plot 1: Total runs by team
    if "runs_of_bat" in df.columns:
        runs = df.groupby("batting_team_c")["runs_of_bat"].sum().sort_values(ascending=False).head(12)
        axes[0, 0].bar(runs.index, runs.values, color="#2563eb")
        axes[0, 0].set_title("Total Runs Scored by Team", fontsize=16)
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(alpha=0.2)

    # Plot 2: Wickets by team
    if "player_dismissed" in df.columns:
        wickets = df.dropna(subset=["player_dismissed"]).groupby("bowling_team_c").size().sort_values(ascending=False).head(12)
        axes[0, 1].bar(wickets.index, wickets.values, color="#f97316")
        axes[0, 1].set_title("Total Wickets Taken by Team", fontsize=16)
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(alpha=0.2)

    # Plot 3: Average runs per over by year
    if "tournament_year" in df.columns and "runs_of_bat" in df.columns:
        df["over"] = pd.to_numeric(df["over"], errors="coerce")
        per_over = df.groupby(["tournament_year", "match_id", "innings", "over"])["runs_of_bat"].sum().reset_index()
        avg_rr = per_over.groupby("tournament_year")["runs_of_bat"].mean()
        axes[0, 2].plot(avg_rr.index, avg_rr.values, marker="D", color="#0ea5e9", linewidth=2)
        axes[0, 2].set_title("Average Runs per Over by Year", fontsize=16)
        axes[0, 2].grid(alpha=0.2)

    # Plot 4: Boundaries by year
    if "runs_of_bat" in df.columns and "tournament_year" in df.columns:
        fours = df[df["runs_of_bat"] == 4].groupby("tournament_year").size()
        sixes = df[df["runs_of_bat"] == 6].groupby("tournament_year").size()
        axes[1, 0].plot(fours.index, fours.values, marker="o", label="Fours", color="#22c55e", linewidth=2)
        axes[1, 0].plot(sixes.index, sixes.values, marker="s", label="Sixes", color="#facc15", linewidth=2)
        axes[1, 0].set_title("Boundaries by Year", fontsize=16)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.2)

    # Plot 5: Extras by year
    if "extras" in df.columns and "tournament_year" in df.columns:
        extras = df.groupby("tournament_year")["extras"].sum()
        axes[1, 1].bar(extras.index, extras.values, color="#a855f7")
        axes[1, 1].set_title("Extras Conceded by Year", fontsize=16)
        axes[1, 1].grid(alpha=0.2)

    # Plot 6: Wickets by year
    if "player_dismissed" in df.columns and "tournament_year" in df.columns:
        wk = df.dropna(subset=["player_dismissed"]).groupby("tournament_year").size()
        axes[1, 2].bar(wk.index, wk.values, color="#ef4444")
        axes[1, 2].set_title("Wickets by Year", fontsize=16)
        axes[1, 2].grid(alpha=0.2)

    # Plot 7: Top batters
    if "striker" in df.columns and "runs_of_bat" in df.columns:
        top_batters = df.groupby("striker")["runs_of_bat"].sum().sort_values(ascending=False).head(10)
        axes[2, 0].barh(top_batters.index[::-1], top_batters.values[::-1], color="#38bdf8")
        axes[2, 0].set_title("Top Run Scorers", fontsize=16)
        axes[2, 0].grid(alpha=0.2)

    # Plot 8: Top bowlers
    if "bowler" in df.columns and "player_dismissed" in df.columns:
        top_bowlers = df.dropna(subset=["player_dismissed"]).groupby("bowler").size().sort_values(ascending=False).head(10)
        axes[2, 1].barh(top_bowlers.index[::-1], top_bowlers.values[::-1], color="#f472b6")
        axes[2, 1].set_title("Top Wicket Takers", fontsize=16)
        axes[2, 1].grid(alpha=0.2)

    # Plot 9: Dismissal types
    if "wicket_type" in df.columns:
        dismissals = df["wicket_type"].dropna().value_counts().head(8)
        axes[2, 2].pie(dismissals.values, labels=dismissals.index, autopct="%1.1f%%", startangle=90)
        axes[2, 2].set_title("Dismissal Types", fontsize=16)

    # Plot 10: Matches by venue
    if "venue" in df.columns:
        venues = df.drop_duplicates(["match_id", "venue"]).groupby("venue").size().sort_values(ascending=False).head(10)
        axes[3, 0].barh(venues.index[::-1], venues.values[::-1], color="#10b981")
        axes[3, 0].set_title("Top Venues by Matches", fontsize=16)
        axes[3, 0].grid(alpha=0.2)

    # Plot 11: Matches by year
    if "tournament_year" in df.columns:
        matches = df.drop_duplicates("match_id").groupby("tournament_year").size()
        axes[3, 1].plot(matches.index, matches.values, marker="o", color="#0ea5e9")
        axes[3, 1].set_title("Matches by Year", fontsize=16)
        axes[3, 1].grid(alpha=0.2)

    # Plot 12: Team run rate (runs per ball)
    if "runs_of_bat" in df.columns:
        team_runs = df.groupby("batting_team_c")["runs_of_bat"].sum()
        team_balls = df.groupby("batting_team_c").size()
        rr = (team_runs / team_balls).sort_values(ascending=False).head(12)
        axes[3, 2].bar(rr.index, rr.values, color="#6366f1")
        axes[3, 2].set_title("Runs per Ball by Team", fontsize=16)
        axes[3, 2].tick_params(axis="x", rotation=45)

    for ax in axes.flat:
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    b64 = _encode_png(fig)
    plt.close(fig)
    return b64


def run_training():
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path("artifacts/combined_worldcup_deliveries.csv")
    if not source_path.exists():
        raise FileNotFoundError("Missing artifacts/combined_worldcup_deliveries.csv. Run train.py first.")

    df = pd.read_csv(source_path, low_memory=False)
    df["batting_team"] = df["batting_team"].astype(str).str.strip()
    df["bowling_team"] = df["bowling_team"].astype(str).str.strip()

    analytics_b64 = generate_analytics_chart(df, "International Dataset (2016-2024) - Analytics Overview")

    teams = sorted(set(_canonical_team(t) for t in df["batting_team"].unique().tolist()))
    teams = [t for t in teams if t]

    runs_scored = df.groupby("batting_team")["runs_of_bat"].sum()
    wickets_taken = df.dropna(subset=["player_dismissed"]).groupby("bowling_team").size()

    results_raw = []
    for team in teams:
        runs = float(runs_scored.get(team, 0))
        wkts = float(wickets_taken.get(team, 0))
        strength = runs + (wkts * 20.0)
        results_raw.append({"team": team, "strength": strength})

    df_pred = pd.DataFrame(results_raw).sort_values("strength", ascending=False)
    df_pred = df_pred.head(12)

    temp = df_pred["strength"].max() * 0.4 if not df_pred.empty else 1.0
    df_pred["prob_raw"] = np.exp(df_pred["strength"] / temp)
    df_pred["probability"] = np.round((df_pred["prob_raw"] / df_pred["prob_raw"].sum()) * 100, 2)

    results = []
    for _, row in df_pred.iterrows():
        team = row["team"]
        results.append(
            {
                "team": team,
                "probability": row["probability"],
                "color": TEAM_COLORS.get(team, "#64748b"),
                "logo": COUNTRY_LOGOS.get(team, ""),
            }
        )

    fig_prob, ax_prob = plt.subplots(figsize=(10, 6), facecolor="#111827")
    ax_prob.set_facecolor("#111827")
    bars = ax_prob.bar([r["team"] for r in results], [r["probability"] for r in results], color=[r["color"] for r in results])
    ax_prob.set_title("International 2026 Historical Dominance Prediction", color="white", fontsize=16, pad=20)
    ax_prob.set_ylabel("Winning Probability (%)", color="white", fontsize=12)
    ax_prob.spines["top"].set_visible(False)
    ax_prob.spines["right"].set_visible(False)
    ax_prob.spines["left"].set_color("#374151")
    ax_prob.spines["bottom"].set_color("#374151")
    ax_prob.tick_params(colors="white")

    for bar, r in zip(bars, results):
        height = bar.get_height()
        ax_prob.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{r['probability']}%",
            ha="center",
            va="bottom",
            color="white",
            fontweight="bold",
        )

    prob_b64 = _encode_png(fig_prob)
    plt.close(fig_prob)

    out_payload = {
        "next_season": 2026,
        "predictions": results,
        "chart_base64": prob_b64,
        "analytics_chart_base64": analytics_b64,
    }

    with open(output_dir / "tournament_winner_prediction.json", "w") as f:
        json.dump(out_payload, f, indent=2)

    print("Generated international predictions successfully!")


if __name__ == "__main__":
    run_training()
