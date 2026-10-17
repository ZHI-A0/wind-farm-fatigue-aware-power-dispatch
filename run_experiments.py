"""Fit load models and run wind-farm power-dispatch experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
CASES = {
    "wf1_temporal": (1, 1, 1000), "wf2_temporal": (2, 2, 1000),
    "wf1_to_wf2": (1, 2, 0), "wf2_to_wf1": (2, 1, 0),
}


def write_manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("scripts/config.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", default="", help="Comma-separated strategy names; defaults to the config")
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    args = parser.parse_args()
    selected_output = args.out_root or Path("outputs") / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output = selected_output if selected_output.is_absolute() else ROOT / selected_output
    config = args.config if args.config.is_absolute() else ROOT / args.config
    if not config.is_file():
        parser.error(f"Configuration file not found: {config}")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output directory is not empty: {output}; use a new --out-root")
    if output == ROOT or output in ROOT.parents:
        parser.error("Output must be a new experiment directory")
    cases = list(CASES) if args.case == "all" else [args.case]
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    manifest = {
        "status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "seed": args.seed,
        "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "PyYAML"]},
        "thread_limits": {key: env[key] for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
        "cases": cases, "completed": [],
    }
    manifest_path = output / "experiment_manifest.json"
    write_manifest(manifest_path, manifest)
    try:
        for case in cases:
            source, target, start = CASES[case]
            arguments = ["--config", str(config), "--data-npz", f"data/data_wf{target}.npz",
                         "--eval-start", str(start),
                         "--seed", str(args.seed), "--skip-snapshot", "--out-root", str(output), "--run-name", case]
            if args.strategies:
                arguments += ["--strategies", args.strategies]
            if source != target:
                arguments += ["--fit-data-npz", f"data/data_wf{source}.npz"]
            print(f"[START] {case}", flush=True)
            started = time.monotonic()
            with (output / "logs" / f"{case}.log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-B", "scripts/main.py", *arguments],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
            elapsed = time.monotonic() - started
            manifest["completed"].append({"case": case, "elapsed_seconds": elapsed, "arguments": arguments})
            write_manifest(manifest_path, manifest)
            print(f"[DONE] {case}: {elapsed:.2f} s", flush=True)
        manifest["status"] = "COMPLETED"
    except Exception:
        manifest["status"] = "FAILED"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
    print(f"[DONE] Experiments completed. Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
