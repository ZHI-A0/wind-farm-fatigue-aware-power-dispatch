"""Verify the bundled data or regenerate NPZ arrays from the original MAT file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_from_mat(path: Path) -> dict[str, dict[str, np.ndarray]]:
    farms = loadmat(path, simplify_cells=True)["data_TS_WF"]
    result = {}
    for number in [1, 2]:
        turbines = farms[f"WF_{number}"]["WT"]
        channel = lambda group, column: np.column_stack(
            [turbine[group][:, column] for turbine in turbines]
        ).astype(np.float64)
        power = channel("outputs", 2) / 1e6
        arrays = {
            "v": channel("inputs", 1), "pref": channel("inputs", 0) / 1e6,
            "pout": power.copy(), "p_out": power.copy(),
            "tshaft": channel("outputs", 0), "ft": channel("outputs", 1),
            "pitch": channel("states", 0), "omega_r": channel("states", 1),
            "omega_f": channel("states", 2),
        }
        arrays["p_grid"] = arrays["pref"].sum(axis=1)
        result[f"wf{number}"] = arrays
    return result


def verify_dataset(*, check_npz: bool = True) -> dict:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    report = {"file_hashes": {}, "raw_to_npz": {}}
    for name, expected in manifest["files"].items():
        if not check_npz and name != "raw/wind_farms.mat":
            continue
        path = ROOT / "data" / name
        if not path.is_file():
            raise FileNotFoundError(f"Bundled dataset file missing: data/{name}")
        observed = sha256(path)
        if observed != expected["sha256"]:
            raise ValueError(f"Dataset SHA-256 mismatch: data/{name}")
        report["file_hashes"][name] = observed
    converted = arrays_from_mat(ROOT / "data/raw/wind_farms.mat")
    for farm, arrays in converted.items():
        for key, array in arrays.items():
            shape = (2000,) if key == "p_grid" else (2000, 100)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid source array: {farm}/{key}")
        if not check_npz:
            continue
        with np.load(ROOT / f"data/data_{farm}.npz", allow_pickle=False) as stored:
            if set(stored.files) != set(arrays):
                raise ValueError(f"Unexpected NPZ channels: {farm}")
            matches = {}
            for key, array in arrays.items():
                matches[key] = bool(np.array_equal(array, stored[key]))
            if not all(matches.values()):
                raise ValueError(f"MAT/NPZ numerical mismatch: {farm}: {matches}")
        report["raw_to_npz"][farm] = matches
    report["passed"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Check SHA-256 and every MAT/NPZ element (default)")
    action.add_argument("--rebuild", action="store_true", help="Write regenerated NPZ files to a new directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/rebuilt"))
    args = parser.parse_args()
    report = verify_dataset(check_npz=not args.rebuild)
    if args.rebuild:
        output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        output = output.resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Refusing to overwrite a nonempty directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        for farm, arrays in arrays_from_mat(ROOT / "data/raw/wind_farms.mat").items():
            np.savez_compressed(output / f"data_{farm}.npz", **arrays)
        (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[PASS] Regenerated both NPZ datasets in {output}")
    else:
        print("[PASS] All three data hashes and all channels of both farms match the original MAT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
