from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def generate_broadcast_charts(sim_df, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Score progression.
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=sim_df, x="balls_bowled", y="current_score", hue="innings", linewidth=2.5, ax=ax)
    ax.set_title("Score Progression")
    ax.set_xlabel("Balls Bowled")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(output_dir / "score_progression.png", dpi=150)
    plt.close(fig)

    # 2) Win probability graph (2nd innings).
    chase_df = sim_df[sim_df["innings"] == 2].copy()
    if not chase_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=chase_df, x="balls_bowled", y="win_probability", linewidth=2.5, color="#d62728", ax=ax)
        ax.set_title("Win Probability (Chasing Team)")
        ax.set_xlabel("Balls Bowled")
        ax.set_ylabel("Win Probability %")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(output_dir / "win_probability.png", dpi=150)
        plt.close(fig)

        # 3) Run rate vs required run rate.
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=chase_df, x="balls_bowled", y="run_rate", label="Current Run Rate", ax=ax)
        sns.lineplot(data=chase_df, x="balls_bowled", y="required_run_rate", label="Required Run Rate", ax=ax)
        ax.set_title("Run Rate vs Required Run Rate")
        ax.set_xlabel("Balls Bowled")
        ax.set_ylabel("Runs Per Over")
        fig.tight_layout()
        fig.savefig(output_dir / "run_rate_vs_required_rate.png", dpi=150)
        plt.close(fig)

    # 4) Wickets timeline.
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=sim_df, x="balls_bowled", y="current_wickets", hue="innings", linewidth=2.5, ax=ax)
    ax.set_title("Wickets Timeline")
    ax.set_xlabel("Balls Bowled")
    ax.set_ylabel("Wickets Fallen")
    fig.tight_layout()
    fig.savefig(output_dir / "wickets_timeline.png", dpi=150)
    plt.close(fig)

    # 5) Momentum graph (runs per over).
    over_runs = sim_df.groupby(["innings", "over_no"], as_index=False)["ball_runs"].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=over_runs, x="over_no", y="ball_runs", hue="innings", marker="o", ax=ax)
    ax.set_title("Match Momentum (Runs per Over)")
    ax.set_xlabel("Over")
    ax.set_ylabel("Runs in Over")
    fig.tight_layout()
    fig.savefig(output_dir / "momentum_runs_per_over.png", dpi=150)
    plt.close(fig)
