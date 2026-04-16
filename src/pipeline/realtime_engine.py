from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.config import DualDataIngestionConfig, DualModelTrainerConfig
from src.exception import ProjectException
from src.utils.common import load_object
from src.utils.dual_data_utils import (
    FIRST_INNINGS_FEATURES,
    SECOND_INNINGS_FEATURES,
    SECOND_INNINGS_SCORE_FEATURES,
)


@dataclass
class FirstInningsState:
    batting_team: str
    bowling_team: str
    venue: str
    phase: str
    current_score: float
    current_wickets: float
    balls_bowled: float
    balls_remaining: float
    run_rate: float
    last_30_runs: float
    pressure_index: float


@dataclass
class SecondInningsState:
    batting_team: str
    bowling_team: str
    venue: str
    phase: str
    current_score: float
    current_wickets: float
    wickets_remaining: float
    balls_bowled: float
    balls_remaining: float
    run_rate: float
    target: float
    runs_required: float
    required_run_rate: float
    last_30_runs: float
    pressure_index: float


class RealtimeEngine:
    MIN_SAMPLES_VENUE_MATCHUP = 120
    MIN_SAMPLES_VENUE = 260
    MIN_SAMPLES_MATCHUP = 180

    def __init__(
        self,
        first_model_path=None,
        second_model_path=None,
        second_score_model_path=None,
        full_context_path=None,
    ):
        model_cfg = DualModelTrainerConfig()
        ingestion_cfg = DualDataIngestionConfig()

        first_model_path = first_model_path or model_cfg.first_best_model_path
        second_model_path = second_model_path or model_cfg.second_best_model_path
        second_score_model_path = second_score_model_path or model_cfg.second_score_best_model_path
        full_context_path = full_context_path or ingestion_cfg.full_context_path

        self.first_model = load_object(first_model_path)
        self.second_model = load_object(second_model_path)
        self.second_score_model = load_object(second_score_model_path)
        self.full_context_df = pd.read_csv(full_context_path, low_memory=False)
        self._rng = np.random.default_rng(42)
        self._run_outcome_tables, self._run_outcome_fallback = self._build_run_outcome_tables()

    @staticmethod
    def _phase_from_balls(balls_bowled: int) -> str:
        if balls_bowled <= 36:
            return "Powerplay"
        if balls_bowled <= 90:
            return "Middle Overs"
        return "Death Overs"

    @staticmethod
    def _wicket_bucket(wickets_remaining: int) -> str:
        if wickets_remaining <= 2:
            return "low"
        if wickets_remaining <= 5:
            return "mid"
        return "high"

    @staticmethod
    def _sigmoid_pct(x: float, scale: float = 1.0) -> float:
        return float(100.0 / (1.0 + np.exp(-(x / max(scale, 1e-6)))))

    @staticmethod
    def _normalize_team(value: str) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_venue(value: str) -> str:
        return str(value or "").strip().lower()

    def _make_outcome_table(self, sub_df: pd.DataFrame, fallback_df: pd.DataFrame):
        sample = sub_df if len(sub_df) > 0 else fallback_df
        run_counts = sample["ball_runs"].value_counts(normalize=True).sort_index()
        run_values = run_counts.index.to_numpy()
        run_probs = run_counts.to_numpy()
        wicket_prob = float(sample["wicket_event"].mean())
        return {
            "run_values": run_values,
            "run_probs": run_probs,
            "wicket_prob": min(max(wicket_prob, 0.001), 0.30),
            "sample_size": int(len(sample)),
        }

    def _build_run_outcome_tables(self):
        df = self.full_context_df.copy()
        if "wicket_event" not in df.columns:
            df["wicket_event"] = df["player_dismissed"].notna().astype(int)
        if "phase" not in df.columns:
            df["phase"] = np.select(
                [df["balls_bowled"] <= 36, df["balls_bowled"] <= 90],
                ["Powerplay", "Middle Overs"],
                default="Death Overs",
            )

        if "batting_team" not in df.columns:
            df["batting_team"] = ""
        if "bowling_team" not in df.columns:
            df["bowling_team"] = ""
        if "venue" not in df.columns:
            df["venue"] = ""

        df["batting_team_norm"] = df["batting_team"].map(self._normalize_team)
        df["bowling_team_norm"] = df["bowling_team"].map(self._normalize_team)
        df["venue_norm"] = df["venue"].map(self._normalize_venue)
        df["wicket_bucket"] = df["wickets_remaining"].astype(int).apply(self._wicket_bucket)
        df["ball_runs"] = df["ball_runs"].clip(lower=0, upper=8).astype(int)

        tables = {}
        fallback_tables = {}
        for phase in ["Powerplay", "Middle Overs", "Death Overs"]:
            for bucket in ["low", "mid", "high"]:
                sub = df[(df["phase"] == phase) & (df["wicket_bucket"] == bucket)]
                if sub.empty:
                    sub = df[df["phase"] == phase]
                if sub.empty:
                    sub = df

                fallback_tables[(phase, bucket)] = self._make_outcome_table(sub, sub)

                grouped_vm = sub.groupby(["venue_norm", "batting_team_norm", "bowling_team_norm"], dropna=False)
                for (venue_key, bat_key, bowl_key), grp in grouped_vm:
                    if len(grp) >= self.MIN_SAMPLES_VENUE_MATCHUP:
                        tables[(phase, bucket, venue_key, bat_key, bowl_key)] = self._make_outcome_table(grp, sub)

                grouped_v = sub.groupby(["venue_norm"], dropna=False)
                for (venue_key,), grp in grouped_v:
                    if len(grp) >= self.MIN_SAMPLES_VENUE:
                        tables[(phase, bucket, venue_key, "", "")] = self._make_outcome_table(grp, sub)

                grouped_m = sub.groupby(["batting_team_norm", "bowling_team_norm"], dropna=False)
                for (bat_key, bowl_key), grp in grouped_m:
                    if len(grp) >= self.MIN_SAMPLES_MATCHUP:
                        tables[(phase, bucket, "", bat_key, bowl_key)] = self._make_outcome_table(grp, sub)

        return tables, fallback_tables

    def _get_run_outcome_table(self, phase: str, bucket: str, venue: str, batting_team: str, bowling_team: str):
        venue_key = self._normalize_venue(venue)
        bat_key = self._normalize_team(batting_team)
        bowl_key = self._normalize_team(bowling_team)

        direct_key = (phase, bucket, venue_key, bat_key, bowl_key)
        venue_key_only = (phase, bucket, venue_key, "", "")
        matchup_key = (phase, bucket, "", bat_key, bowl_key)

        return (
            self._run_outcome_tables.get(direct_key)
            or self._run_outcome_tables.get(venue_key_only)
            or self._run_outcome_tables.get(matchup_key)
            or self._run_outcome_fallback[(phase, bucket)]
        )

    def simulate_first_innings_total(self, state: FirstInningsState, n_sims: int = 2000):
        totals = []
        for _ in range(n_sims):
            score = float(state.current_score)
            wickets_remaining = int(max(0, 10 - state.current_wickets))
            balls_bowled = int(state.balls_bowled)
            balls_remaining = int(state.balls_remaining)

            for _ball in range(balls_remaining):
                if wickets_remaining <= 0:
                    break

                phase = self._phase_from_balls(balls_bowled)
                bucket = self._wicket_bucket(wickets_remaining)
                table = self._get_run_outcome_table(
                    phase,
                    bucket,
                    state.venue,
                    state.batting_team,
                    state.bowling_team,
                )

                run_scored = int(self._rng.choice(table["run_values"], p=table["run_probs"]))
                is_wicket = bool(self._rng.random() < table["wicket_prob"])

                score += run_scored
                balls_bowled += 1
                if is_wicket:
                    wickets_remaining -= 1

            totals.append(score)

        totals_arr = np.array(totals)
        return {
            "mean_total": float(np.mean(totals_arr)),
            "p10": float(np.percentile(totals_arr, 10)),
            "p90": float(np.percentile(totals_arr, 90)),
        }

    def simulate_second_innings(self, state: SecondInningsState, n_sims: int = 2500):
        wins = 0
        totals = []

        for _ in range(n_sims):
            score = float(state.current_score)
            target = float(state.target)
            wickets_remaining = int(state.wickets_remaining)
            balls_bowled = int(state.balls_bowled)
            balls_remaining = int(state.balls_remaining)

            won = score >= target
            for _ball in range(balls_remaining):
                if won or wickets_remaining <= 0:
                    break

                phase = self._phase_from_balls(balls_bowled)
                bucket = self._wicket_bucket(wickets_remaining)
                table = self._get_run_outcome_table(
                    phase,
                    bucket,
                    state.venue,
                    state.batting_team,
                    state.bowling_team,
                )

                run_scored = int(self._rng.choice(table["run_values"], p=table["run_probs"]))
                is_wicket = bool(self._rng.random() < table["wicket_prob"])

                score += run_scored
                balls_bowled += 1
                if score >= target:
                    won = True
                    break
                if is_wicket:
                    wickets_remaining -= 1

            wins += int(won)
            totals.append(score)

        totals_arr = np.array(totals)
        return {
            "win_probability": float((wins / n_sims) * 100.0),
            "projected_total_mean": float(np.mean(totals_arr)),
            "projected_total_p10": float(np.percentile(totals_arr, 10)),
            "projected_total_p90": float(np.percentile(totals_arr, 90)),
        }

    def predict_first_innings(self, state: FirstInningsState) -> float:
        try:
            df = pd.DataFrame(
                [
                    {
                        "batting_team": state.batting_team,
                        "bowling_team": state.bowling_team,
                        "venue": state.venue,
                        "phase": state.phase,
                        "current_score": state.current_score,
                        "current_wickets": state.current_wickets,
                        "balls_bowled": state.balls_bowled,
                        "balls_remaining": state.balls_remaining,
                        "current_run_rate": state.run_rate,
                        "last_30_runs": state.last_30_runs,
                        "pressure_index": state.pressure_index,
                        "wickets_remaining": float(max(0.0, 10.0 - state.current_wickets)),
                    }
                ]
            )
            val = float(self.first_model.predict(df[FIRST_INNINGS_FEATURES])[0])
            return max(state.current_score, round(val, 2))
        except Exception as exc:
            raise ProjectException(exc, context="RealtimeEngine.predict_first_innings") from exc

    def predict_second_innings(self, state: SecondInningsState) -> float:
        try:
            df = pd.DataFrame(
                [
                    {
                        "batting_team": state.batting_team,
                        "bowling_team": state.bowling_team,
                        "venue": state.venue,
                        "phase": state.phase,
                        "current_score": state.current_score,
                        "current_wickets": state.current_wickets,
                        "wickets_remaining": state.wickets_remaining,
                        "balls_bowled": state.balls_bowled,
                        "balls_remaining": state.balls_remaining,
                        "current_run_rate": state.run_rate,
                        "target": state.target,
                        "runs_required": state.runs_required,
                        "required_run_rate": state.required_run_rate,
                        "last_30_runs": state.last_30_runs,
                        "pressure_index": state.pressure_index,
                    }
                ]
            )
            val = float(self.second_model.predict_proba(df[SECOND_INNINGS_FEATURES])[0, 1] * 100)
            return max(0.0, min(100.0, round(val, 2)))
        except Exception as exc:
            raise ProjectException(exc, context="RealtimeEngine.predict_second_innings") from exc

    def predict_second_innings_total(self, state: SecondInningsState) -> float:
        try:
            df = pd.DataFrame(
                [
                    {
                        "batting_team": state.batting_team,
                        "bowling_team": state.bowling_team,
                        "venue": state.venue,
                        "phase": state.phase,
                        "current_score": state.current_score,
                        "current_wickets": state.current_wickets,
                        "wickets_remaining": state.wickets_remaining,
                        "balls_bowled": state.balls_bowled,
                        "balls_remaining": state.balls_remaining,
                        "current_run_rate": state.run_rate,
                        "target": state.target,
                        "runs_required": state.runs_required,
                        "required_run_rate": state.required_run_rate,
                        "last_30_runs": state.last_30_runs,
                        "pressure_index": state.pressure_index,
                    }
                ]
            )
            val = float(self.second_score_model.predict(df[SECOND_INNINGS_SCORE_FEATURES])[0])
            return max(state.current_score, round(val, 2))
        except Exception as exc:
            raise ProjectException(exc, context="RealtimeEngine.predict_second_innings_total") from exc

    def predict_first_innings_bundle(self, state: FirstInningsState, n_sims: int = 2200):
        model_total = self.predict_first_innings(state)
        sim = self.simulate_first_innings_total(state, n_sims=n_sims)

        progress = min(max(float(state.balls_bowled) / 120.0, 0.0), 1.0)
        model_weight = 0.55 + (0.20 * progress)
        sim_weight = 1.0 - model_weight
        blended_total = (model_weight * model_total) + (sim_weight * sim["mean_total"])

        return {
            "projected_total": float(round(max(state.current_score, blended_total), 2)),
            "projection_low": float(sim["p10"]),
            "projection_high": float(sim["p90"]),
            "model_total": float(model_total),
            "sim_total": float(sim["mean_total"]),
            "blend_weights": {"model": float(model_weight), "simulation": float(sim_weight)},
        }

    def predict_second_innings_bundle(self, state: SecondInningsState, n_sims: int = 2800):
        model_win = self.predict_second_innings(state)
        projected_total_model = self.predict_second_innings_total(state)
        sim = self.simulate_second_innings(state, n_sims=n_sims)

        margin_from_projection = projected_total_model - float(state.target)
        projection_win = self._sigmoid_pct(margin_from_projection, scale=9.0)

        rr_delta = float(state.run_rate - state.required_run_rate)
        control_signal = 50.0 + (rr_delta * 8.5) + ((float(state.wickets_remaining) - 5.0) * 1.8)
        if float(state.balls_remaining) <= 36:
            control_signal += rr_delta * 3.0
        control_signal = max(1.0, min(99.0, control_signal))

        progress = min(max(float(state.balls_bowled) / 120.0, 0.0), 1.0)
        w_sim = 0.48 + (0.10 * progress)
        w_model = 0.30 - (0.05 * progress)
        w_projection = 0.14
        w_control = 1.0 - (w_sim + w_model + w_projection)

        chasing_win_probability = (
            (w_model * model_win)
            + (w_projection * projection_win)
            + (w_sim * sim["win_probability"])
            + (w_control * control_signal)
        )

        projected_chase_total = (0.72 * projected_total_model) + (0.28 * sim["projected_total_mean"])
        projected_chase_total = max(
            sim["projected_total_p10"],
            min(sim["projected_total_p90"], projected_chase_total),
        )

        sim_margin = sim["projected_total_mean"] - float(state.target)
        if sim_margin <= -20:
            chasing_win_probability = min(chasing_win_probability, 35.0)
        elif sim_margin <= -10:
            chasing_win_probability = min(chasing_win_probability, 50.0)
        elif sim_margin >= 20:
            chasing_win_probability = max(chasing_win_probability, 70.0)
        elif sim_margin >= 10:
            chasing_win_probability = max(chasing_win_probability, 58.0)

        if float(state.runs_required) <= 1 and float(state.balls_remaining) >= 1 and float(state.wickets_remaining) >= 1:
            chasing_win_probability = max(chasing_win_probability, 97.0)
        elif float(state.runs_required) <= 3 and float(state.balls_remaining) >= 3 and float(state.wickets_remaining) >= 1:
            chasing_win_probability = max(chasing_win_probability, 92.0)
        elif float(state.runs_required) <= 6 and float(state.balls_remaining) >= 6 and float(state.wickets_remaining) >= 2:
            chasing_win_probability = max(chasing_win_probability, 85.0)
        elif float(state.required_run_rate) <= 4 and float(state.wickets_remaining) >= 5 and float(state.balls_remaining) >= 18:
            chasing_win_probability = max(chasing_win_probability, 70.0)

        if float(state.current_score) >= float(state.target):
            chasing_win_probability = 100.0
        elif float(state.current_wickets) >= 10.0:
            chasing_win_probability = 0.0

        chasing_win_probability = max(0.0, min(100.0, float(chasing_win_probability)))
        return {
            "win_probability": float(round(chasing_win_probability, 2)),
            "projected_total": float(round(max(float(state.current_score), projected_chase_total), 2)),
            "projection_low": float(sim["projected_total_p10"]),
            "projection_high": float(sim["projected_total_p90"]),
            "components": {
                "model_win": float(model_win),
                "projection_win": float(projection_win),
                "simulation_win": float(sim["win_probability"]),
                "control_signal": float(control_signal),
            },
            "blend_weights": {
                "model": float(w_model),
                "projection": float(w_projection),
                "simulation": float(w_sim),
                "control": float(w_control),
            },
        }

    def simulate_match_ball_by_ball(self, match_id: int) -> pd.DataFrame:
        try:
            match_df = self.full_context_df[self.full_context_df["match_id"] == match_id].copy()
            if match_df.empty:
                raise ValueError(f"No match found for match_id={match_id}")

            match_df = match_df.sort_values(["innings", "over_no", "ball_no"]).reset_index(drop=True)

            first_mask = match_df["innings"] == 1
            second_mask = match_df["innings"] == 2

            match_df["projected_score"] = pd.NA
            match_df["win_probability"] = pd.NA

            if first_mask.any():
                first_input = match_df.loc[first_mask, FIRST_INNINGS_FEATURES].copy()
                first_valid_idx = first_input.dropna().index
                if len(first_valid_idx) > 0:
                    first_pred = self.first_model.predict(match_df.loc[first_valid_idx, FIRST_INNINGS_FEATURES])
                    match_df.loc[first_valid_idx, "projected_score"] = first_pred

                match_df.loc[first_mask, "projected_score"] = match_df.loc[
                    first_mask, ["projected_score", "current_score"]
                ].max(axis=1)

            if second_mask.any():
                second_input = match_df.loc[second_mask, SECOND_INNINGS_FEATURES].copy()
                second_valid_idx = second_input.dropna().index
                if len(second_valid_idx) > 0:
                    match_df.loc[second_valid_idx, "win_probability"] = (
                        self.second_model.predict_proba(match_df.loc[second_valid_idx, SECOND_INNINGS_FEATURES])[:, 1] * 100
                    )

                second_score_input = match_df.loc[second_mask, SECOND_INNINGS_SCORE_FEATURES].copy()
                second_score_valid_idx = second_score_input.dropna().index
                if len(second_score_valid_idx) > 0:
                    second_score_pred = self.second_score_model.predict(
                        match_df.loc[second_score_valid_idx, SECOND_INNINGS_SCORE_FEATURES]
                    )
                    match_df.loc[second_score_valid_idx, "projected_score"] = second_score_pred

                match_df.loc[second_mask, "projected_score"] = match_df.loc[
                    second_mask, ["projected_score", "current_score"]
                ].max(axis=1)

            return match_df
        except Exception as exc:
            raise ProjectException(exc, context="RealtimeEngine.simulate_match_ball_by_ball") from exc
