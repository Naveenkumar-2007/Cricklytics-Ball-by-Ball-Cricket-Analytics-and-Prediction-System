import pandas as pd
from flask import Flask, render_template, request, session
import numpy as np
from typing import Dict, List
import io
import base64
import os
import json
import hashlib
from pathlib import Path

from src.logger import get_logger
from src.pipeline.realtime_engine import FirstInningsState, RealtimeEngine, SecondInningsState

logger = get_logger(__name__)
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "cricket-analytics-local-secret")

COMPETITION_CONFIGS = {
    "international": {
        "label": "International",
        "artifact_root": Path("artifacts"),
        "train_command": "python train_full_pipeline.py",
    },
    "ipl": {
        "label": "IPL",
        "artifact_root": Path("artifacts") / "ipl",
        "train_command": "python train_ipl.py",
    },
}

RUNTIME_CACHE: Dict[str, Dict[str, object]] = {}
CHART_DATA_CACHE: Dict[str, Dict[str, object]] = {}
CHART_IMAGE_CACHE: Dict[str, Dict[str, str]] = {}


PREFERRED_TEAM_ORDER = [
    "CSK",
    "MI",
    "RCB",
    "KKR",
    "DC",
    "SRH",
    "RR",
    "PBKS",
    "GT",
    "LSG",
    "IND",
    "AUS",
    "ENG",
    "NZ",
    "PAK",
    "SA",
    "RSA",
    "WI",
    "SL",
    "AFG",
    "BAN",
    "USA",
    "IRE",
    "SCO",
    "NED",
    "NAM",
    "UAE",
    "OMAN",
    "CAN",
    "NEP",
    "HK",
    "PNG",
    "UGA",
    "ZIM",
]

TEAM_ALIASES = {
    "CHENNAI SUPER KINGS": "CSK",
    "MUMBAI INDIANS": "MI",
    "ROYAL CHALLENGERS BANGALORE": "RCB",
    "ROYAL CHALLENGERS BENGALURU": "RCB",
    "KOLKATA KNIGHT RIDERS": "KKR",
    "SUNRISERS HYDERABAD": "SRH",
    "DELHI CAPITALS": "DC",
    "DELHI DAREDEVILS": "DC",
    "RAJASTHAN ROYALS": "RR",
    "PUNJAB KINGS": "PBKS",
    "KINGS XI PUNJAB": "PBKS",
    "GUJARAT TITANS": "GT",
    "LUCKNOW SUPER GIANTS": "LSG",
    "DECCAN CHARGERS": "SRH",
    "GUJARAT LIONS": "GT",
    "KOCHI TUSKERS KERALA": "PBKS",
    "PUNE WARRIORS": "MI",
    "RISING PUNE SUPERGIANT": "CSK",
    "RISING PUNE SUPERGIANTS": "CSK",
}

CURRENT_IPL_CANONICAL = ["RCB", "MI", "KKR", "DC", "SRH", "RR", "PBKS", "GT", "LSG", "CSK"]
CURRENT_IPL_DISPLAY_NAME = {
    "CSK": "CHENNAI SUPER KINGS",
    "MI": "MUMBAI INDIANS",
    "RCB": "ROYAL CHALLENGERS BENGALURU",
    "KKR": "KOLKATA KNIGHT RIDERS",
    "DC": "DELHI CAPITALS",
    "SRH": "SUNRISERS HYDERABAD",
    "RR": "RAJASTHAN ROYALS",
    "PBKS": "PUNJAB KINGS",
    "GT": "GUJARAT TITANS",
    "LSG": "LUCKNOW SUPER GIANTS",
}


def _canonical_team_key(team_name: str) -> str:
    key = str(team_name or "").upper().strip()
    return TEAM_ALIASES.get(key, key)


def _order_teams(team_values):
    unique = sorted(set(str(t).upper() for t in team_values if str(t).strip()))
    order_index = {team: idx for idx, team in enumerate(PREFERRED_TEAM_ORDER)}

    def _sort_key(team_name: str):
        canonical = _canonical_team_key(team_name)
        return (order_index.get(canonical, 999), team_name)

    return sorted(unique, key=_sort_key)


def _filter_current_ipl_teams(teams: List[str]) -> List[str]:
    by_canonical: Dict[str, List[str]] = {}
    for team_name in teams:
        by_canonical.setdefault(_canonical_team_key(team_name), []).append(team_name)

    selected = []
    for canonical in CURRENT_IPL_CANONICAL:
        candidates = by_canonical.get(canonical, [])
        if not candidates:
            continue
        preferred_name = CURRENT_IPL_DISPLAY_NAME.get(canonical, "")
        chosen = preferred_name if preferred_name in candidates else sorted(candidates)[0]
        selected.append(chosen)
    return selected

def _competition_paths(competition: str) -> Dict[str, Path]:
    cfg = COMPETITION_CONFIGS[competition]
    artifact_root = cfg["artifact_root"]
    dual_root = artifact_root / "dual"
    return {
        "first_model_path": dual_root / "first_innings_model.pkl",
        "second_model_path": dual_root / "second_innings_model.pkl",
        "second_score_model_path": dual_root / "second_innings_score_model.pkl",
        "full_context_path": dual_root / "full_context.csv",
    }


def _artifacts_ready(paths: Dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def _default_teams(competition: str, teams: List[str]) -> Dict[str, str]:
    preferred_pairs = {
        "international": ("IND", "PAK"),
        "ipl": ("RCB", "MI"),
    }
    first_pref, second_pref = preferred_pairs.get(competition, ("", ""))

    canonical_to_name = {_canonical_team_key(t): t for t in teams}

    if first_pref in canonical_to_name and second_pref in canonical_to_name:
        first_bat = canonical_to_name[first_pref]
        first_bowl = canonical_to_name[second_pref]
    elif len(teams) >= 2:
        first_bat, first_bowl = teams[0], teams[1]
    elif len(teams) == 1:
        first_bat, first_bowl = teams[0], teams[0]
    else:
        first_bat, first_bowl = "", ""

    second_bat = first_bowl
    second_bowl = first_bat

    return {
        "first_batting_team": first_bat,
        "first_bowling_team": first_bowl,
        "second_batting_team": second_bat,
        "second_bowling_team": second_bowl,
    }


def _runtime_for_competition(competition: str) -> Dict[str, object]:
    if competition in RUNTIME_CACHE:
        return RUNTIME_CACHE[competition]

    paths = _competition_paths(competition)
    ready = _artifacts_ready(paths)

    load_error = None
    if ready:
        try:
            engine = RealtimeEngine(
                first_model_path=paths["first_model_path"],
                second_model_path=paths["second_model_path"],
                second_score_model_path=paths["second_score_model_path"],
                full_context_path=paths["full_context_path"],
            )
            full_df = engine.full_context_df.copy()
            teams = _order_teams(pd.concat([full_df["batting_team"], full_df["bowling_team"]], ignore_index=True))
            if competition == "ipl":
                teams = _filter_current_ipl_teams(teams)
            venues = sorted(full_df["venue"].dropna().astype(str).str.strip().unique().tolist())
        except Exception as exc:
            logger.exception("Failed to load runtime artifacts for competition=%s", competition)
            engine = None
            teams = []
            venues = []
            load_error = (
                f"{COMPETITION_CONFIGS[competition]['label']} artifacts could not be loaded: {exc}. "
                "This usually means environment version mismatch. "
                "Use the project .venv and install requirements.txt."
            )
    else:
        engine = None
        teams = []
        venues = []

    runtime = {
        "engine": engine,
        "teams": teams,
        "venues": venues,
        "artifacts_ready": ready,
        "load_error": load_error,
        "paths": paths,
    }
    RUNTIME_CACHE[competition] = runtime
    return runtime


def _default_form_values(competition: str, teams: List[str], venues: List[str]):
    team_defaults = _default_teams(competition, teams)
    return {
        "first_batting_team": team_defaults["first_batting_team"],
        "first_bowling_team": team_defaults["first_bowling_team"],
        "first_venue": venues[0] if venues else "",
        "first_current_score": "",
        "first_wickets_lost": "",
        "first_overs": "",
        "first_last_30_runs": "",
        "second_batting_team": team_defaults["second_batting_team"],
        "second_bowling_team": team_defaults["second_bowling_team"],
        "second_venue": venues[0] if venues else "",
        "second_target": "",
        "second_current_score": "",
        "second_wickets_lost": "",
        "second_overs": "",
        "second_last_30_runs": "",
    }


TEAM_COLORS = {
    "AFG": {"primary": "#1E3A8A", "secondary": "#FF0000", "dark": "#1E3A8A", "light": "#FF0000"},
    "AUS": {"primary": "#FFD700", "secondary": "#006400", "dark": "#FFD700", "light": "#006400"},
    "BAN": {"primary": "#006A4E", "secondary": "#F42A41", "dark": "#006A4E", "light": "#F42A41"},
    "CAN": {"primary": "#FF0000", "secondary": "#FFFFFF", "dark": "#FF0000", "light": "#FFFFFF"},
    "ENG": {"primary": "#00247D", "secondary": "#FFFFFF", "dark": "#00247D", "light": "#FFFFFF"},
    "HK": {"primary": "#DE2910", "secondary": "#FFFFFF", "dark": "#DE2910", "light": "#FFFFFF"},
    "IND": {"primary": "#1E90FF", "secondary": "#FF9933", "dark": "#1E90FF", "light": "#FF9933"},
    "IRE": {"primary": "#169B62", "secondary": "#FFFFFF", "dark": "#169B62", "light": "#FFFFFF"},
    "NAM": {"primary": "#0033A0", "secondary": "#D21034", "dark": "#0033A0", "light": "#D21034"},
    "NED": {"primary": "#FF6600", "secondary": "#000080", "dark": "#FF6600", "light": "#000080"},
    "NEP": {"primary": "#DC143C", "secondary": "#003893", "dark": "#DC143C", "light": "#003893"},
    "NZ": {"primary": "#000000", "secondary": "#FFFFFF", "dark": "#000000", "light": "#FFFFFF"},
    "OMAN": {"primary": "#D40000", "secondary": "#006400", "dark": "#D40000", "light": "#006400"},
    "PAK": {"primary": "#006600", "secondary": "#FFFFFF", "dark": "#006600", "light": "#FFFFFF"},
    "PNG": {"primary": "#000000", "secondary": "#FFD700", "dark": "#000000", "light": "#FFD700"},
    "RSA": {"primary": "#006400", "secondary": "#FFD700", "dark": "#006400", "light": "#FFD700"},
    "SA": {"primary": "#006400", "secondary": "#FFD700", "dark": "#006400", "light": "#FFD700"},
    "SCO": {"primary": "#800080", "secondary": "#FFFFFF", "dark": "#800080", "light": "#FFFFFF"},
    "SL": {"primary": "#0033A0", "secondary": "#FFD700", "dark": "#0033A0", "light": "#FFD700"},
    "UAE": {"primary": "#FF0000", "secondary": "#00732F", "dark": "#FF0000", "light": "#00732F"},
    "UGA": {"primary": "#000000", "secondary": "#FFD700", "dark": "#000000", "light": "#FFD700"},
    "USA": {"primary": "#B22234", "secondary": "#3C3B6E", "dark": "#B22234", "light": "#3C3B6E"},
    "WI": {"primary": "#800000", "secondary": "#FFD700", "dark": "#800000", "light": "#FFD700"},
    "ZIM": {"primary": "#FF0000", "secondary": "#006400", "dark": "#FF0000", "light": "#006400"},
    "CSK": {"primary": "#F9CD05", "secondary": "#1E3A8A", "dark": "#C89E00", "light": "#FFF2A8"},
    "MI": {"primary": "#004BA0", "secondary": "#D4AF37", "dark": "#003472", "light": "#A8C7EA"},
    "RCB": {"primary": "#EC1C24", "secondary": "#1A1A1A", "dark": "#A80F0F", "light": "#F6B3B3"},
    "KKR": {"primary": "#3A225D", "secondary": "#B49A57", "dark": "#2A1744", "light": "#C9BAE3"},
    "SRH": {"primary": "#F26522", "secondary": "#000000", "dark": "#B24A17", "light": "#F7C1A2"},
    "DC": {"primary": "#004C93", "secondary": "#EF1B23", "dark": "#00386D", "light": "#A7C8EA"},
    "RR": {"primary": "#FF1493", "secondary": "#003A8C", "dark": "#B71465", "light": "#F8B4D8"},
    "GT": {"primary": "#1C2C5B", "secondary": "#8CC8FF", "dark": "#142040", "light": "#B6CDF7"},
    "LSG": {"primary": "#A6D8F5", "secondary": "#F78F1E", "dark": "#5AA8D2", "light": "#E5F5FD"},
    "PBKS": {"primary": "#D71920", "secondary": "#C5C5C5", "dark": "#A41217", "light": "#F5B7BA"},
}

TEAM_LOGOS = {
    "IND": "https://flagcdn.com/w80/in.png",
    "PAK": "https://flagcdn.com/w80/pk.png",
    "AUS": "https://flagcdn.com/w80/au.png",
    "ENG": "https://flagcdn.com/w80/gb.png",
    "NZ": "https://flagcdn.com/w80/nz.png",
    "RSA": "https://flagcdn.com/w80/za.png",
    "SA": "https://flagcdn.com/w80/za.png",
    "SL": "https://flagcdn.com/w80/lk.png",
    "WI": "https://flagcdn.com/w80/ag.png",
    "BAN": "https://flagcdn.com/w80/bd.png",
    "AFG": "https://flagcdn.com/w80/af.png",
    "NED": "https://flagcdn.com/w80/nl.png",
    "NEP": "https://flagcdn.com/w80/np.png",
    "CAN": "https://flagcdn.com/w80/ca.png",
    "IRE": "https://flagcdn.com/w80/ie.png",
    "SCO": "https://flagcdn.com/w80/gb.png",
    "USA": "https://flagcdn.com/w80/us.png",
    "UGA": "https://flagcdn.com/w80/ug.png",
    "OMAN": "https://flagcdn.com/w80/om.png",
    "PNG": "https://flagcdn.com/w80/pg.png",
    "NAM": "https://flagcdn.com/w80/na.png",
    "HK": "https://flagcdn.com/w80/hk.png",
    "UAE": "https://flagcdn.com/w80/ae.png",
    "ZIM": "https://flagcdn.com/w80/zw.png",
    "CSK": "https://documents.iplt20.com/ipl/CSK/logos/Logooutline/CSKoutline.png",
    "MI": "https://documents.iplt20.com/ipl/MI/Logos/Logooutline/MIoutline.png",
    "RCB": "https://documents.iplt20.com/ipl/RCB/Logos/Logooutline/RCBoutline.png",
    "KKR": "https://documents.iplt20.com/ipl/KKR/Logos/Logooutline/KKRoutline.png",
    "SRH": "https://documents.iplt20.com/ipl/SRH/Logos/Logooutline/SRHoutline.png",
    "DC": "https://documents.iplt20.com/ipl/DC/Logos/LogoOutline/DCoutline.png",
    "RR": "https://documents.iplt20.com/ipl/RR/Logos/Logooutline/RRoutline.png",
    "GT": "https://documents.iplt20.com/ipl/GT/Logos/Logooutline/GToutline.png",
    "LSG": "https://documents.iplt20.com/ipl/LSG/Logos/Logooutline/LSGoutline.png",
    "PBKS": "https://documents.iplt20.com/ipl/PBKS/Logos/Logooutline/PBKSoutline.png",
}

IPL_OFFICIAL_LOGO_URL = "/static/images/brand/ipl-official.png"
INTERNATIONAL_OFFICIAL_LOGO_URL = "/static/images/brand/icc-t20wc-2026-trim.png"
MAIN_WEBSITE_LOGO_URL = "/static/images/brand/cricklytics-main-logo.jpeg"


def _team_logo_url(team_code: str) -> str:
    canonical = _canonical_team_key(team_code)
    return TEAM_LOGOS.get(canonical, TEAM_LOGOS.get(str(team_code or "").upper(), ""))


def _team_palette(team_code: str) -> Dict[str, str]:
    canonical = _canonical_team_key(team_code)
    return TEAM_COLORS.get(canonical, TEAM_COLORS.get(team_code, {"primary": "#1e3a8a", "secondary": "#f97316", "dark": "#1d2f6f", "light": "#1e40af"}))


def parse_overs_to_balls(overs_text: str) -> int:
    overs_text = overs_text.strip()
    if "." in overs_text:
        over_part, ball_part = overs_text.split(".", 1)
    else:
        over_part, ball_part = overs_text, "0"

    overs_completed = int(over_part)
    balls_in_current_over = int(ball_part) if ball_part else 0

    if overs_completed < 0 or overs_completed > 20:
        raise ValueError("Overs must be between 0 and 20.")
    if balls_in_current_over < 0 or balls_in_current_over > 5:
        raise ValueError("In overs format x.y, y must be between 0 and 5.")

    balls_bowled = (overs_completed * 6) + balls_in_current_over
    if balls_bowled > 120:
        raise ValueError("Overs cannot exceed 20.0.")

    return balls_bowled


def get_phase_from_balls(balls_bowled: int) -> str:
    if balls_bowled <= 36:
        return "Powerplay"
    if balls_bowled <= 90:
        return "Middle Overs"
    return "Death Overs"


def _rr_projection_scenarios(current_rr: float) -> List[Dict[str, float]]:
    rr = float(current_rr or 0.0)
    if rr <= 0:
        rr = 7.0

    candidates = [rr - 0.25, rr, rr + 0.25, rr + 0.75]
    cleaned = []
    for value in candidates:
        bounded = max(3.0, min(14.0, value))
        rounded = round(bounded, 2)
        if rounded not in cleaned:
            cleaned.append(rounded)

    while len(cleaned) < 4:
        next_rr = round(min(14.0, cleaned[-1] + 0.5), 2)
        if next_rr in cleaned:
            break
        cleaned.append(next_rr)

    scenarios = []
    current_key = round(rr, 2)
    for rr_value in sorted(cleaned):
        scenarios.append(
            {
                "rr": rr_value,
                "is_current": abs(rr_value - current_key) < 0.01,
                "score_6": int(round(rr_value * 6)),
                "score_20": int(round(rr_value * 20)),
            }
        )
    return scenarios


def _second_rr_projection_scenarios(
    current_rr: float,
    current_score: float,
    balls_remaining: int,
) -> List[Dict[str, float]]:
    base_scenarios = _rr_projection_scenarios(current_rr)
    score_now = float(current_score or 0.0)
    balls_rem = max(0, int(balls_remaining or 0))
    next_6_balls = min(36, balls_rem)

    scenarios = []
    for row in base_scenarios:
        rr_value = float(row["rr"])
        score_after_6 = score_now + (rr_value * next_6_balls / 6.0)
        score_at_20 = score_now + (rr_value * balls_rem / 6.0)
        scenarios.append(
            {
                "rr": rr_value,
                "is_current": bool(row.get("is_current")),
                "score_6": int(round(score_after_6)),
                "score_20": int(round(score_at_20)),
            }
        )
    return scenarios


def _safe_overs_to_float(overs_text: str, fallback: float = 0.0) -> float:
    try:
        balls = parse_overs_to_balls(str(overs_text))
        return max(0.0, min(20.0, balls / 6.0))
    except Exception:
        return fallback


def _interpolate_at_over(curve: List[float], over_float: float) -> float:
    over_float = max(0.0, min(20.0, over_float))
    lo = int(np.floor(over_float))
    hi = int(np.ceil(over_float))
    if lo == hi:
        return float(curve[lo])
    weight = over_float - lo
    return float((1 - weight) * curve[lo] + (weight * curve[hi]))


def _historical_curves(
    df: pd.DataFrame,
    innings_no: int,
    batting_team: str,
    bowling_team: str,
    venue: str,
) -> Dict[str, List[float]]:
    base = df[df["innings"] == innings_no].copy()
    filt = base[
        (base["batting_team"] == batting_team)
        & (base["bowling_team"] == bowling_team)
        & (base["venue"] == venue)
    ]

    if filt["match_id"].nunique() < 3:
        filt = base[
            (base["batting_team"] == batting_team)
            & (base["bowling_team"] == bowling_team)
        ]
    if filt["match_id"].nunique() < 3:
        filt = base[base["batting_team"] == batting_team]
    if filt["match_id"].nunique() < 3:
        filt = base

    over_df = (
        filt.groupby(["match_id", "over_no"], as_index=False)
        .agg(
            over_runs=("ball_runs", "sum"),
            over_wickets=("wicket_event", "sum"),
        )
        .sort_values(["match_id", "over_no"])
    )

    run_curves: List[List[float]] = []
    wicket_curves: List[List[float]] = []

    for match_id, grp in over_df.groupby("match_id"):
        _ = match_id
        runs_by_over = {int(r.over_no): float(r.over_runs) for r in grp.itertuples()}
        wkts_by_over = {int(r.over_no): float(r.over_wickets) for r in grp.itertuples()}

        run_curve = [0.0]
        wkt_curve = [0.0]
        cum_runs = 0.0
        cum_wkts = 0.0

        for ov in range(1, 21):
            cum_runs += runs_by_over.get(ov, 0.0)
            cum_wkts += wkts_by_over.get(ov, 0.0)
            run_curve.append(cum_runs)
            wkt_curve.append(min(10.0, cum_wkts))

        run_curves.append(run_curve)
        wicket_curves.append(wkt_curve)

    if not run_curves:
        return {
            "runs": [0.0] * 21,
            "wickets": [0.0] * 21,
        }

    avg_runs = np.mean(np.asarray(run_curves), axis=0).tolist()
    avg_wkts = np.mean(np.asarray(wicket_curves), axis=0).tolist()
    return {
        "runs": [float(x) for x in avg_runs],
        "wickets": [float(x) for x in avg_wkts],
    }


def _anchor_curve_to_state(
    base_curve: List[float],
    current_over: float,
    current_value: float,
    final_value: float,
) -> List[float]:
    current_over = max(0.0, min(20.0, current_over))
    current_value = max(0.0, float(current_value))
    final_value = max(current_value, float(final_value))

    base_at_current = max(1e-6, _interpolate_at_over(base_curve, current_over))
    base_final = max(base_curve[-1], base_at_current + 1e-6)

    out: List[float] = []
    for ov in range(21):
        if ov <= current_over:
            scaled = base_curve[ov] * (current_value / base_at_current)
            out.append(max(0.0, scaled))
        else:
            progress = (base_curve[ov] - base_at_current) / max(1e-6, (base_final - base_at_current))
            val = current_value + (progress * (final_value - current_value))
            out.append(max(current_value, val))

    out[0] = 0.0
    out[-1] = final_value

    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]

    return [float(x) for x in out]


def _build_chart_data(engine, form_values, first_prediction, second_prediction):
    if engine is None:
        return None

    df = engine.full_context_df

    first_batting = str(form_values.get("first_batting_team", ""))
    first_bowling = str(form_values.get("first_bowling_team", ""))
    venue = str(form_values.get("first_venue", ""))
    second_batting = str(form_values.get("second_batting_team", ""))
    second_bowling = str(form_values.get("second_bowling_team", ""))

    first_current_score = float(form_values.get("first_current_score", 0) or 0)
    first_current_wkts = float(form_values.get("first_wickets_lost", 0) or 0)
    second_current_score = float(form_values.get("second_current_score", 0) or 0)
    second_current_wkts = float(form_values.get("second_wickets_lost", 0) or 0)

    first_over = _safe_overs_to_float(str(form_values.get("first_overs", "0.0")), fallback=0.0)
    second_over = _safe_overs_to_float(str(form_values.get("second_overs", "0.0")), fallback=0.0)
    second_started = bool(second_over > 0.0 or second_current_score > 0.0 or second_current_wkts > 0.0)

    first_final = float(first_prediction["projected_total"]) if first_prediction else max(first_current_score, 140.0)
    if second_prediction:
        second_target = float(second_prediction["live_target"])
        second_final = float(second_prediction["projected_chase_total"])
        chase_win_current = float(second_prediction["chasing_win_probability"])
    else:
        second_target = float(form_values.get("second_target", 180) or 180)
        if second_started:
            second_final = max(second_current_score, min(second_target + 5, 200.0))
        else:
            second_final = second_current_score
        if first_prediction:
            chase_win_current = float(first_prediction.get("chasing_win_at_start", 50.0))
        else:
            chase_win_current = 50.0

    first_hist = _historical_curves(df, 1, first_batting, first_bowling, venue)
    second_hist = _historical_curves(df, 2, second_batting, second_bowling, venue)

    first_score_curve = _anchor_curve_to_state(first_hist["runs"], first_over, first_current_score, first_final)
    second_score_curve = _anchor_curve_to_state(second_hist["runs"], second_over, second_current_score, max(second_current_score, second_final))

    first_wkt_final = max(first_current_wkts, _interpolate_at_over(first_hist["wickets"], 20.0))
    second_wkt_final = max(second_current_wkts, _interpolate_at_over(second_hist["wickets"], 20.0))
    first_wkt_curve = _anchor_curve_to_state(first_hist["wickets"], first_over, first_current_wkts, min(10.0, first_wkt_final))
    second_wkt_curve = _anchor_curve_to_state(second_hist["wickets"], second_over, second_current_wkts, min(10.0, second_wkt_final))

    overs = list(range(21))
    target_curve = [second_target for _ in overs]

    rr_curve = [0.0]
    req_rr_curve = [0.0]
    for ov in range(1, 21):
        score = second_score_curve[ov]
        rr_curve.append(float(score / ov))
        if ov >= 20:
            req_rr_curve.append(0.0)
        else:
            req_runs = max(0.0, second_target - score)
            req_rr_curve.append(float(req_runs / (20 - ov)))

    win_prob = []
    for ov in overs:
        score = second_score_curve[ov]
        wkts = second_wkt_curve[ov]
        if ov >= 20:
            win_prob.append(100.0 if score >= second_target else 0.0)
            continue
        runs_req = max(0.0, second_target - score)
        balls_rem = max(1.0, (20 - ov) * 6)
        req_rr = (runs_req * 6) / balls_rem
        crr = score / max(0.5, ov)
        wkts_rem = 10.0 - wkts
        signal = 50.0 + ((crr - req_rr) * 8.0) + ((wkts_rem - 5.0) * 3.0)
        p = float(100.0 / (1.0 + np.exp(-(signal - 50.0) / 8.0)))
        win_prob.append(max(0.0, min(100.0, p)))

    first_over_idx = int(max(0, min(20, np.floor(first_over))))
    anchor_over = int(max(0, min(20, np.floor(second_over))))
    delta = chase_win_current - win_prob[anchor_over]
    smoothed = []
    for i, p in enumerate(win_prob):
        weight = 0.3 + (0.7 * (i / max(1, anchor_over))) if i <= anchor_over else 1.0
        smoothed.append(max(0.0, min(100.0, p + (delta * weight))))
    win_prob = smoothed

    return {
        "teams": {
            "first_batting": first_batting,
            "first_bowling": first_bowling,
            "second_batting": second_batting,
            "second_bowling": second_bowling,
        },
        "state": {
            "first_over": round(first_over, 2),
            "first_over_idx": first_over_idx,
            "second_over": round(second_over, 2),
            "second_over_idx": anchor_over,
            "chasing_wp_now": round(chase_win_current, 2),
            "second_started": second_started,
        },
        "summary": {
            "first": {
                "batting_team": first_batting,
                "bowling_team": first_bowling,
                "score_now": round(first_current_score, 1),
                "wickets_now": round(first_current_wkts, 1),
                "overs_now": round(first_over, 2),
                "projected_score": round(first_final, 1),
            },
            "second": {
                "batting_team": second_batting,
                "bowling_team": second_bowling,
                "score_now": round(second_current_score, 1),
                "wickets_now": round(second_current_wkts, 1),
                "overs_now": round(second_over, 2),
                "projected_score": round(second_final, 1) if (second_started or second_prediction) else None,
                "target": round(second_target, 1),
            },
        },
        "overs": overs,
        "score_progression": {
            "first": [round(x, 2) for x in first_score_curve],
            "second": [round(x, 2) for x in second_score_curve],
            "target": [round(x, 2) for x in target_curve],
        },
        "run_rate": {
            "current_rr": [round(x, 3) for x in rr_curve],
            "required_rr": [round(x, 3) for x in req_rr_curve],
        },
        "win_probability": {
            "chasing": [round(x, 2) for x in win_prob],
            "defending": [round(100.0 - x, 2) for x in win_prob],
        },
        "wickets": {
            "first": [round(x, 2) for x in first_wkt_curve],
            "second": [round(x, 2) for x in second_wkt_curve],
        },
    }


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _chart_cache_key(payload: Dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _build_chart_images(chart_data):
    if not chart_data:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    overs = chart_data["overs"]
    t = chart_data["teams"]
    score = chart_data["score_progression"]
    rr = chart_data["run_rate"]
    wp = chart_data["win_probability"]
    wk = chart_data["wickets"]
    state = chart_data["state"]

    c_first_bat = _team_palette(t["first_batting"])
    c_first_bowl = _team_palette(t["first_bowling"])
    c_second_bat = _team_palette(t["second_batting"])
    c_second_bowl = _team_palette(t["second_bowling"])

    images = {}

    first_idx = int(state["first_over_idx"])
    second_idx = int(state["second_over_idx"])

    first_score = np.asarray(score["first"], dtype=float)
    second_score = np.asarray(score["second"], dtype=float)
    first_wk = np.asarray(wk["first"], dtype=float)
    second_wk = np.asarray(wk["second"], dtype=float)

    def _shadow_line(ax, x, y, color, lw=2.6, label=None, style="-"):
        ax.plot(x, y, color="#111827", linewidth=lw + 1.8, alpha=0.15, linestyle=style)
        ax.plot(x, y, color=color, linewidth=lw, linestyle=style, label=label)

    def _legend_team(team_name: str) -> str:
        canonical = _canonical_team_key(team_name)
        if canonical and canonical != team_name:
            return canonical
        if len(str(team_name)) <= 14:
            return str(team_name)
        return str(team_name)[:12] + ".."

    def _inside_legend(ax, handles=None, labels=None, ncol=1):
        legend_kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (0.01, 0.99),
            "borderaxespad": 0.2,
            "fontsize": 7,
            "frameon": True,
            "framealpha": 0.9,
            "facecolor": "#ffffff",
            "edgecolor": "#cbd5e1",
            "ncol": ncol,
            "columnspacing": 0.7,
            "handlelength": 1.8,
            "labelspacing": 0.35,
        }
        if handles is not None and labels is not None:
            ax.legend(handles, labels, **legend_kwargs)
        else:
            ax.legend(**legend_kwargs)

    def _segmented(curve, split_idx):
        split_idx = max(0, min(len(curve) - 1, int(split_idx)))
        actual = curve.copy()
        proj = curve.copy()
        for i in range(len(curve)):
            if i > split_idx:
                actual[i] = np.nan
            if i <= split_idx:
                proj[i] = np.nan
        return actual, proj

    def _wicket_fall_overs(cum_wk_curve, upto_idx):
        upto_idx = max(0, min(len(cum_wk_curve) - 1, int(upto_idx)))
        wk_now = int(max(0, min(10, np.floor(float(cum_wk_curve[upto_idx]) + 1e-9))))
        falls = []
        if wk_now <= 0:
            return falls

        reached = 0
        for ov in range(1, upto_idx + 1):
            wk_prev = int(np.floor(float(cum_wk_curve[ov - 1]) + 1e-9))
            wk_curr = int(np.floor(float(cum_wk_curve[ov]) + 1e-9))
            if wk_curr > wk_prev:
                for _ in range(wk_curr - wk_prev):
                    reached += 1
                    if reached <= wk_now:
                        falls.append(ov)
        return falls

    first_actual, first_proj = _segmented(first_score, first_idx)
    second_actual, second_proj = _segmented(second_score, second_idx)

    # First innings dedicated chart panel.
    fig0, ax0 = plt.subplots(figsize=(8.5, 3.3))
    ax0.axvspan(0, 6, color=c_first_bat["light"], alpha=0.28)
    ax0.axvspan(15, 20, color=c_first_bat["light"], alpha=0.28)
    first_bat_short = _legend_team(t["first_batting"])
    first_bowl_short = _legend_team(t["first_bowling"])
    second_bat_short = _legend_team(t["second_batting"])
    second_bowl_short = _legend_team(t["second_bowling"])

    _shadow_line(ax0, overs, first_actual, c_first_bat["primary"], lw=2.7, label=f"{first_bat_short} Score Act")
    _shadow_line(ax0, overs, first_proj, c_first_bat["secondary"], lw=2.2, label=f"{first_bat_short} Score Proj", style="--")
    ax0.set_xlim(0, 20)
    ax0.set_xlabel("Overs")
    ax0.set_ylabel("Runs")
    ax0.grid(alpha=0.2)
    ax0.set_title("First Innings Analytics: Score + Wickets")
    ax0b = ax0.twinx()
    first_w_a, first_w_p = _segmented(first_wk, first_idx)
    ax0b.plot(overs, first_w_a, color=c_first_bowl["dark"], linewidth=2.1, label=f"{first_bat_short} Wkts Act")
    ax0b.plot(overs, first_w_p, color=c_first_bowl["secondary"], linewidth=1.9, linestyle="--", label=f"{first_bat_short} Wkts Proj")
    ax0b.set_ylim(0, 10)
    ax0b.set_ylabel("Wickets Lost")
    h0, l0 = ax0.get_legend_handles_labels()
    h0b, l0b = ax0b.get_legend_handles_labels()
    _inside_legend(ax0, h0 + h0b, l0 + l0b, ncol=2)
    images["first_innings_panel"] = _fig_to_base64(fig0)
    plt.close(fig0)

    # Second innings dedicated chart panel.
    fig00, ax00 = plt.subplots(figsize=(8.5, 3.3))
    ax00.axvspan(0, 6, color=c_second_bat["light"], alpha=0.24)
    ax00.axvspan(15, 20, color=c_second_bowl["light"], alpha=0.24)
    _shadow_line(ax00, overs, second_actual, c_second_bat["primary"], lw=2.7, label=f"{second_bat_short} Score Act")
    _shadow_line(ax00, overs, second_proj, c_second_bat["secondary"], lw=2.2, label=f"{second_bat_short} Score Proj", style="--")
    ax00.plot(overs, score["target"], color="#f59e0b", linewidth=1.8, linestyle=":", label="Target")
    ax00.set_xlim(0, 20)
    ax00.set_xlabel("Overs")
    ax00.set_ylabel("Runs")
    ax00.grid(alpha=0.2)
    ax00.set_title("Second Innings Analytics: Chase + Pressure")

    rr_current = np.asarray(rr["current_rr"], dtype=float)
    rr_required = np.asarray(rr["required_rr"], dtype=float)
    rr_a, rr_p = _segmented(rr_current, second_idx)
    req_a, req_p = _segmented(rr_required, second_idx)
    ax00b = ax00.twinx()
    ax00b.plot(overs, rr_a, color=c_second_bat["dark"], linewidth=2.0, label=f"{second_bat_short} RR")
    ax00b.plot(overs, req_a, color=c_second_bowl["primary"], linewidth=2.0, label=f"{second_bowl_short} Req RR")
    ax00b.plot(overs, rr_p, color=c_second_bat["secondary"], linewidth=1.6, linestyle="--", label=f"{second_bat_short} RR Proj")
    ax00b.plot(overs, req_p, color=c_second_bowl["secondary"], linewidth=1.6, linestyle="--", label=f"{second_bowl_short} Req Proj")
    ax00b.set_ylabel("Run Rate")
    h00, l00 = ax00.get_legend_handles_labels()
    h00b, l00b = ax00b.get_legend_handles_labels()
    _inside_legend(ax00, h00 + h00b, l00 + l00b, ncol=2)
    images["second_innings_panel"] = _fig_to_base64(fig00)
    plt.close(fig00)

    fig1, ax1 = plt.subplots(figsize=(8.5, 3.6))
    ax1.axvspan(0, 6, color=c_first_bat["light"], alpha=0.45)
    ax1.axvspan(6, 15, color="#f8fafc", alpha=0.85)
    ax1.axvspan(15, 20, color=c_second_bat["light"], alpha=0.45)

    _shadow_line(ax1, overs, first_actual, c_first_bat["primary"], lw=2.6, label=f"{first_bat_short} Act")
    _shadow_line(ax1, overs, first_proj, c_first_bat["secondary"], lw=2.3, label=f"{first_bat_short} Proj", style="--")
    _shadow_line(ax1, overs, second_actual, c_second_bat["primary"], lw=2.6, label=f"{second_bat_short} Act")
    _shadow_line(ax1, overs, second_proj, c_second_bat["secondary"], lw=2.3, label=f"{second_bat_short} Proj", style="--")
    ax1.plot(overs, score["target"], color="#f59e0b", linewidth=1.8, linestyle=":", label="Target")

    wicket_overs = _wicket_fall_overs(second_wk, second_idx)
    wicket_scores = [float(second_score[o]) for o in wicket_overs]
    if wicket_overs and wicket_scores:
        ax1.scatter(wicket_overs, wicket_scores, color="#b91c1c", s=26, zorder=5)

    ax1.set_title("Score Progression (Real Match Context)")
    ax1.set_xlabel("Overs")
    ax1.set_ylabel("Runs")
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, 20)
    _inside_legend(ax1, ncol=2)
    images["score_progression"] = _fig_to_base64(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.5, 3.6))
    rr_a, rr_p = _segmented(rr_current, second_idx)
    req_a, req_p = _segmented(rr_required, second_idx)

    _shadow_line(ax2, overs, rr_a, c_second_bat["primary"], lw=2.4, label=f"{second_bat_short} RR")
    _shadow_line(ax2, overs, rr_p, c_second_bat["secondary"], lw=2.1, label=f"{second_bat_short} RR Proj", style="--")
    _shadow_line(ax2, overs, req_a, c_second_bowl["primary"], lw=2.4, label=f"{second_bowl_short} Req RR")
    _shadow_line(ax2, overs, req_p, c_second_bowl["secondary"], lw=2.1, label=f"{second_bowl_short} Req Proj", style="--")

    ax2.set_title("Run Rate vs Required Run Rate")
    ax2.set_xlabel("Overs")
    ax2.set_ylabel("Run Rate")
    ax2.grid(alpha=0.25)
    ax2.set_xlim(0, 20)
    _inside_legend(ax2, ncol=2)
    images["run_rate"] = _fig_to_base64(fig2)
    plt.close(fig2)

    chase_now = float(state["chasing_wp_now"])
    defend_now = max(0.0, 100.0 - chase_now)
    fig3, ax3 = plt.subplots(figsize=(8.5, 2.5))
    ax3.set_title("Win Probability Graph (Most Advanced)", fontsize=12, pad=10)

    # 3D-style base/shadow strip.
    ax3.barh([-0.03], [100], color="#111827", alpha=0.15, height=0.38)
    ax3.barh([0], [chase_now], color=c_second_bat["primary"], height=0.34, label=t["second_batting"])
    ax3.barh([0], [defend_now], left=[chase_now], color=c_second_bowl["primary"], height=0.34, label=t["second_bowling"])

    ax3.text(0, 0.43, f"{t['second_batting']}\n{chase_now:.1f}%", va="bottom", ha="left", fontsize=10, fontweight="bold", color=c_second_bat["dark"])
    ax3.text(100, 0.43, f"{t['second_bowling']}\n{defend_now:.1f}%", va="bottom", ha="right", fontsize=10, fontweight="bold", color=c_second_bowl["dark"])
    ax3.text(50, 0.72, "WIN PROBABILITY", va="bottom", ha="center", fontsize=11, fontweight="bold", color="#111827")

    ax3.set_xlim(0, 100)
    ax3.set_ylim(-0.6, 1.05)
    ax3.set_yticks([])
    ax3.set_xticks([])
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.legend(loc="lower center", ncol=2, fontsize=8, frameon=False)
    images["win_probability"] = _fig_to_base64(fig3)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(8.5, 3.8))
    # Broadcast-like phase shading for worm-style timeline.
    ax4.axvspan(0, 6, color=c_first_bat["light"], alpha=0.55)
    ax4.axvspan(6, 15, color="#f8fafc", alpha=0.95)
    ax4.axvspan(15, 20, color=c_second_bat["light"], alpha=0.55)

    first_s_a, first_s_p = _segmented(first_score, first_idx)
    second_s_a, second_s_p = _segmented(second_score, second_idx)
    _shadow_line(ax4, overs, first_s_a, c_first_bat["primary"], lw=2.8, label=f"{first_bat_short} Act")
    _shadow_line(ax4, overs, first_s_p, c_first_bat["secondary"], lw=2.3, label=f"{first_bat_short} Proj", style="--")
    _shadow_line(ax4, overs, second_s_a, c_second_bat["primary"], lw=2.8, label=f"{second_bat_short} Act")
    _shadow_line(ax4, overs, second_s_p, c_second_bat["secondary"], lw=2.3, label=f"{second_bat_short} Proj", style="--")

    # Wicket markers shown on score worm at wicket-fall overs.
    first_wk_overs = _wicket_fall_overs(first_wk, first_idx)
    second_wk_overs = _wicket_fall_overs(second_wk, second_idx)

    for ov in first_wk_overs:
        if ov > first_idx:
            continue
        y = float(first_score[ov])
        ax4.scatter([ov], [y], s=130, color=c_first_bat["primary"], edgecolors=c_first_bat["light"], linewidths=1.4, zorder=6)
        ax4.text(ov, y, "W", color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=7)

    for ov in second_wk_overs:
        if ov > second_idx:
            continue
        y = float(second_score[ov])
        ax4.scatter([ov], [y], s=130, color=c_second_bat["primary"], edgecolors=c_second_bat["light"], linewidths=1.4, zorder=6)
        ax4.text(ov, y, "W", color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=7)

    ax4.set_title("Wickets Timeline (Worm Style)")
    ax4.set_xlabel("Overs")
    ax4.set_ylabel("Runs")
    ax4.set_xlim(0, 20)
    ax4.grid(alpha=0.2)
    ax4.text(3, max(max(first_score), max(second_score)) * 0.03, "Powerplay", fontsize=9, color="#334155", ha="center")
    ax4.text(10.5, max(max(first_score), max(second_score)) * 0.03, "Middle Overs", fontsize=9, color="#334155", ha="center")
    ax4.text(17.5, max(max(first_score), max(second_score)) * 0.03, "Death Overs", fontsize=9, color="#334155", ha="center")
    _inside_legend(ax4, ncol=2)
    images["wickets"] = _fig_to_base64(fig4)
    plt.close(fig4)

    return images


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    competition = request.values.get("competition", session.get("competition", "international")).strip().lower()
    if competition not in COMPETITION_CONFIGS:
        competition = "international"

    previous_competition = session.get("competition")
    switched_competition = previous_competition is not None and previous_competition != competition
    session["competition"] = competition

    runtime = _runtime_for_competition(competition)
    engine = runtime["engine"]
    teams = runtime["teams"]
    venues = runtime["venues"]
    artifacts_ready = runtime["artifacts_ready"]
    runtime_load_error = runtime.get("load_error")

    first_prediction = session.get("first_prediction")
    second_prediction = session.get("second_prediction")
    debug_values = None
    error = runtime_load_error
    mode = request.form.get("mode", "second")
    did_submit = request.method == "POST"

    # Fresh page load should start from clean defaults unless user has just submitted.
    if request.method == "GET" or switched_competition:
        session.pop("first_prediction", None)
        session.pop("second_prediction", None)
        session.pop("form_values", None)
        first_prediction = None
        second_prediction = None

    form_values = _default_form_values(competition, teams, venues)
    saved_form_values = session.get("form_values")
    if isinstance(saved_form_values, dict) and not switched_competition:
        form_values.update(saved_form_values)
    chart_data = None
    chart_images = None

    if request.method == "POST":
        try:
            # Keep submitted values in form after post-back.
            for key, value in request.form.items():
                form_values[key] = value

            if engine is None:
                raise ValueError(
                    f"{COMPETITION_CONFIGS[competition]['label']} artifacts not found. "
                    f"Run: {COMPETITION_CONFIGS[competition]['train_command']}"
                )

            if mode == "first":
                batting_team = request.form["first_batting_team"].strip().upper()
                bowling_team = request.form["first_bowling_team"].strip().upper()
                venue = request.form["first_venue"].strip()

                current_score = int(request.form["first_current_score"])
                wickets_lost = int(request.form["first_wickets_lost"])
                overs_text = request.form["first_overs"]
                last_30_runs = int(request.form["first_last_30_runs"])

                if batting_team not in teams or bowling_team not in teams:
                    raise ValueError("Please select valid teams from the dropdown.")
                if batting_team == bowling_team:
                    raise ValueError("First innings: batting and bowling teams must be different.")
                if venue not in venues:
                    raise ValueError("Please select a valid venue.")

                if current_score < 0:
                    raise ValueError("Current score cannot be negative.")
                if wickets_lost < 0 or wickets_lost > 10:
                    raise ValueError("Wickets lost must be between 0 and 10.")
                if last_30_runs < 0:
                    raise ValueError("Last 5 overs runs cannot be negative.")

                balls_bowled = parse_overs_to_balls(overs_text)
                balls_remaining = 120 - balls_bowled
                run_rate = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0.0
                pressure_index = max(-5.0, min(20.0, (8 - run_rate) + (wickets_lost * 0.35)))

                state = FirstInningsState(
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    venue=venue,
                    phase=get_phase_from_balls(balls_bowled),
                    current_score=float(current_score),
                    current_wickets=float(wickets_lost),
                    balls_bowled=float(balls_bowled),
                    balls_remaining=float(balls_remaining),
                    run_rate=float(run_rate),
                    last_30_runs=float(last_30_runs),
                    pressure_index=float(pressure_index),
                )
                if balls_bowled >= 120 or wickets_lost >= 10:
                    # Innings complete: use actual final, not projection.
                    projected_total = float(current_score)
                    sim_first = {
                        "p10": float(current_score),
                        "p90": float(current_score),
                        "mean_total": float(current_score),
                    }
                else:
                    first_bundle = engine.predict_first_innings_bundle(state, n_sims=2200)
                    projected_total = float(first_bundle["projected_total"])
                    sim_first = {
                        "p10": float(first_bundle["projection_low"]),
                        "p90": float(first_bundle["projection_high"]),
                        "mean_total": float(first_bundle["sim_total"]),
                    }

                # Connected projection for second innings at chase start.
                projected_target = int(round(projected_total)) + 1
                start_required_rr = projected_target / 20
                chase_start_state = SecondInningsState(
                    batting_team=bowling_team,
                    bowling_team=batting_team,
                    venue=venue,
                    phase="Powerplay",
                    current_score=0.0,
                    current_wickets=0.0,
                    wickets_remaining=10.0,
                    balls_bowled=0.0,
                    balls_remaining=120.0,
                    run_rate=0.0,
                    target=float(projected_target),
                    runs_required=float(projected_target),
                    required_run_rate=float(start_required_rr),
                    last_30_runs=0.0,
                    pressure_index=float(start_required_rr),
                )
                chasing_win_at_start = engine.predict_second_innings(chase_start_state)
                defending_win_at_start = 100 - chasing_win_at_start

                first_prediction = {
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "projected_chasing_team": bowling_team,
                    "projected_defending_team": batting_team,
                    "projected_total": round(projected_total),
                    "projected_target": projected_target,
                    "chasing_win_at_start": round(chasing_win_at_start, 2),
                    "defending_win_at_start": round(defending_win_at_start, 2),
                    "projection_low": round(sim_first["p10"]),
                    "projection_high": round(sim_first["p90"]),
                    "current_rr": round(run_rate, 2),
                    "rr_scenarios": _rr_projection_scenarios(run_rate),
                }
                prediction = first_prediction
                session["first_prediction"] = first_prediction
                # Fresh first-innings run invalidates previous chase prediction context.
                second_prediction = None
                session.pop("second_prediction", None)

                # Carry first innings output into second innings setup.
                form_values["second_batting_team"] = bowling_team
                form_values["second_bowling_team"] = batting_team
                form_values["second_venue"] = venue
                form_values["second_target"] = projected_target

                # If first innings is complete, initialize chase defaults.
                if balls_bowled >= 120 or wickets_lost >= 10:
                    form_values["second_current_score"] = 0
                    form_values["second_wickets_lost"] = 0
                    form_values["second_overs"] = "0.0"
                    form_values["second_last_30_runs"] = 0

                debug_values = {
                    "balls_bowled": balls_bowled,
                    "balls_remaining": balls_remaining,
                    "current_run_rate": round(run_rate, 2),
                    "pressure_index": round(pressure_index, 2),
                    "innings_phase": get_phase_from_balls(balls_bowled),
                }

            elif mode == "second":
                batting_team = request.form["second_batting_team"].strip().upper()
                bowling_team = request.form["second_bowling_team"].strip().upper()
                venue = request.form["second_venue"].strip()
                context_first_batting = request.form.get("context_first_batting_team", "").strip().upper()
                context_first_bowling = request.form.get("context_first_bowling_team", "").strip().upper()
                context_first_venue = request.form.get("context_first_venue", "").strip()

                current_score = int(request.form["second_current_score"])
                wickets_lost = int(request.form["second_wickets_lost"])
                target = int(request.form["second_target"])
                overs_text = request.form["second_overs"]
                last_30_runs = int(request.form["second_last_30_runs"])

                if batting_team not in teams or bowling_team not in teams:
                    raise ValueError("Please select valid teams from the dropdown.")
                if batting_team == bowling_team:
                    raise ValueError("Second innings: chasing and defending teams must be different.")
                if venue not in venues:
                    raise ValueError("Please select a valid venue.")

                if context_first_batting and context_first_bowling:
                    if batting_team != context_first_bowling or bowling_team != context_first_batting:
                        raise ValueError(
                            "Second innings must be reversed from first innings: "
                            "chasing team = first innings bowling team, "
                            "defending team = first innings batting team."
                        )
                if context_first_venue and venue != context_first_venue:
                    raise ValueError("Second innings venue must be same as first innings venue.")

                if current_score < 0:
                    raise ValueError("Current score cannot be negative.")
                if wickets_lost < 0 or wickets_lost > 10:
                    raise ValueError("Wickets lost must be between 0 and 10.")
                if target < 1:
                    raise ValueError("Target must be at least 1.")
                if last_30_runs < 0:
                    raise ValueError("Last 5 overs runs cannot be negative.")

                balls_bowled = parse_overs_to_balls(overs_text)
                balls_remaining = 120 - balls_bowled
                wickets_remaining = 10 - wickets_lost

                run_rate = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0.0
                runs_required = max(target - current_score, 0)
                required_rr = (runs_required * 6) / balls_remaining if balls_remaining > 0 else 0.0
                pressure_index = max(-10.0, min(20.0, required_rr - run_rate))

                if balls_remaining == 0:
                    chasing_win_probability = 100.0 if current_score >= target else 0.0
                    projected_chase_total = float(current_score)
                    model_win_probability = float(chasing_win_probability)
                    projection_win_probability = float(chasing_win_probability)
                    sim_second = {
                        "win_probability": float(chasing_win_probability),
                        "projected_total_p10": float(current_score),
                        "projected_total_p90": float(current_score),
                    }
                else:
                    state = SecondInningsState(
                        batting_team=batting_team,
                        bowling_team=bowling_team,
                        venue=venue,
                        phase=get_phase_from_balls(balls_bowled),
                        current_score=float(current_score),
                        current_wickets=float(wickets_lost),
                        wickets_remaining=float(wickets_remaining),
                        balls_bowled=float(balls_bowled),
                        balls_remaining=float(balls_remaining),
                        run_rate=float(run_rate),
                        target=float(target),
                        runs_required=float(runs_required),
                        required_run_rate=float(required_rr),
                        last_30_runs=float(last_30_runs),
                        pressure_index=float(pressure_index),
                    )
                    second_bundle = engine.predict_second_innings_bundle(state, n_sims=2800)
                    model_win_probability = float(second_bundle["components"]["model_win"])
                    projection_win_probability = float(second_bundle["components"]["projection_win"])
                    chasing_win_probability = float(second_bundle["win_probability"])
                    projected_chase_total = float(second_bundle["projected_total"])
                    sim_second = {
                        "win_probability": float(second_bundle["components"]["simulation_win"]),
                        "projected_total_p10": float(second_bundle["projection_low"]),
                        "projected_total_p90": float(second_bundle["projection_high"]),
                    }
                if balls_remaining == 0:
                    projected_chase_total = current_score

                if current_score >= target:
                    projected_chase_total = max(projected_chase_total, current_score)
                elif wickets_lost >= 10:
                    projected_chase_total = current_score

                second_prediction = {
                    "chasing_team": batting_team,
                    "defending_team": bowling_team,
                    "chasing_win_probability": round(chasing_win_probability, 2),
                    "defending_win_probability": round(100.0 - chasing_win_probability, 2),
                    "live_target": target,
                    "projected_chase_total": round(projected_chase_total),
                    "projection_low": round(sim_second["projected_total_p10"]) if balls_remaining > 0 else round(projected_chase_total),
                    "projection_high": round(sim_second["projected_total_p90"]) if balls_remaining > 0 else round(projected_chase_total),
                    "current_rr": round(run_rate, 2),
                    "rr_scenarios": _second_rr_projection_scenarios(
                        current_rr=run_rate,
                        current_score=current_score,
                        balls_remaining=balls_remaining,
                    ),
                }
                prediction = second_prediction
                session["second_prediction"] = second_prediction

                debug_values = {
                    "balls_bowled": balls_bowled,
                    "balls_remaining": balls_remaining,
                    "wickets_remaining": wickets_remaining,
                    "current_run_rate": round(run_rate, 2),
                    "required_run_rate": round(required_rr, 2),
                    "pressure_index": round(pressure_index, 2),
                    "innings_phase": get_phase_from_balls(balls_bowled),
                    "model_win_probability": round(model_win_probability, 2) if balls_remaining > 0 else round(chasing_win_probability, 2),
                    "projection_win_probability": round(projection_win_probability, 2) if balls_remaining > 0 else round(chasing_win_probability, 2),
                    "simulation_win_probability": round(sim_second["win_probability"], 2) if balls_remaining > 0 else round(chasing_win_probability, 2),
                }
            else:
                raise ValueError("Unknown mode")

        except (ValueError, TypeError) as exc:
            logger.warning("Validation error: %s", exc)
            error = str(exc)
        except Exception as exc:
            logger.exception("Unexpected prediction error")
            error = f"Prediction failed: {exc}"

    session["form_values"] = form_values

    show_analytics = bool(first_prediction or second_prediction)
    chart_data = None
    chart_images = None

    if show_analytics:
        chart_payload = {
            "competition": competition,
            "form_values": form_values,
            "first_prediction": first_prediction,
            "second_prediction": second_prediction,
        }
        data_key = _chart_cache_key(chart_payload)

        chart_data = CHART_DATA_CACHE.get(data_key)
        if chart_data is None:
            chart_data = _build_chart_data(engine, form_values, first_prediction, second_prediction)
            CHART_DATA_CACHE[data_key] = chart_data
            # Keep cache bounded to avoid unbounded memory growth in long sessions.
            if len(CHART_DATA_CACHE) > 64:
                CHART_DATA_CACHE.pop(next(iter(CHART_DATA_CACHE)))

        chart_images = CHART_IMAGE_CACHE.get(data_key)
        if chart_images is None:
            chart_images = _build_chart_images(chart_data)
            if chart_images:
                CHART_IMAGE_CACHE[data_key] = chart_images
                if len(CHART_IMAGE_CACHE) > 32:
                    CHART_IMAGE_CACHE.pop(next(iter(CHART_IMAGE_CACHE)))

    return render_template(
        "index.html",
        teams=teams,
        team_logos={t: _team_logo_url(t) for t in teams},
        team_colors={t: _team_palette(t) for t in teams},
        venues=venues,
        prediction=prediction,
        first_prediction=first_prediction,
        second_prediction=second_prediction,
        debug_values=debug_values,
        error=error,
        mode=mode,
        artifacts_ready=artifacts_ready,
        form_values=form_values,
        chart_data=chart_data,
        chart_images=chart_images,
        show_analytics=show_analytics,
        did_submit=did_submit,
        competition=competition,
        competition_label=COMPETITION_CONFIGS[competition]["label"],
        competition_options=COMPETITION_CONFIGS,
        train_command=COMPETITION_CONFIGS[competition]["train_command"],
        main_logo_url=MAIN_WEBSITE_LOGO_URL,
        ipl_logo_url=IPL_OFFICIAL_LOGO_URL,
        international_logo_url=INTERNATIONAL_OFFICIAL_LOGO_URL,
    )


if __name__ == "__main__":
    # Default to fast local serving; enable debugger explicitly with FLASK_DEBUG=1.
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)
