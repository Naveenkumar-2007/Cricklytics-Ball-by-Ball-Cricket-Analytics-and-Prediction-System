import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project training in a DVC stage.")
    parser.add_argument("--competition", choices=["international", "ipl"], required=True)
    parser.add_argument("--profile", choices=["fast", "balanced", "full"], default="balanced")
    parser.add_argument("--max-tuning-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    train_script = "train.py" if args.competition == "international" else "train_ipl.py"

    env = os.environ.copy()
    env["CRICKET_TRAIN_PROFILE"] = args.profile
    if args.max_tuning_rows > 0:
        env["CRICKET_MAX_TUNING_ROWS"] = str(args.max_tuning_rows)
    else:
        env.pop("CRICKET_MAX_TUNING_ROWS", None)

    python_exe = VENV_PYTHON if VENV_PYTHON.exists() else Path(os.environ.get("PYTHON", "python"))
    cmd = [str(python_exe), train_script]

    print(
        f"[DVC] competition={args.competition} profile={args.profile} "
        f"max_tuning_rows={args.max_tuning_rows}"
    )
    print("[DVC] Running:", " ".join(cmd))

    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
