"""Interactive H₂ yield prediction from manual user input."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = PROJECT_ROOT / "data" / "models" / "h2_yield_pipeline.joblib"

VALID_FM_RATIOS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

FEATURE_PROMPTS = {
    "substrate_mc": "Substrate moisture content (%)",
    "substrate_ts": "Substrate total solids (%)",
    "substrate_vs": "Substrate volatile solids (%)",
    "substrate_fs": "Substrate fixed solids (%)",
    "inoculum_mc": "Inoculum moisture content (%)",
    "inoculum_ts": "Inoculum total solids (%)",
    "inoculum_vs": "Inoculum volatile solids (%)",
    "inoculum_fs": "Inoculum fixed solids (%)",
    "scod": "SCOD (mg/L)",
    "tcod": "TCOD (mg/L)",
    "scod_tcod_ratio": "SCOD/TCOD ratio",
    "trs": "TRS",
    "lignin": "Lignin (mg/L)",
    "hmf": "HMF (mg/L)",
    "fm_ratio": "Food-to-microorganism (F/M) ratio",
}

EXAMPLE_VALUES = {
    "substrate_mc": 5.26,
    "substrate_ts": 94.74,
    "substrate_vs": 89.41,
    "substrate_fs": 10.59,
    "inoculum_mc": 89.92,
    "inoculum_ts": 10.08,
    "inoculum_vs": 90.30,
    "inoculum_fs": 9.70,
    "scod": 6000.0,
    "tcod": 11400.0,
    "scod_tcod_ratio": 0.53,
    "trs": 1.0,
    "lignin": 25.0,
    "hmf": 45.0,
    "fm_ratio": 1.0,
}


def load_pipeline() -> dict:
    if not PIPELINE_PATH.exists():
        print(f"Error: model file not found at\n  {PIPELINE_PATH}")
        print(
            "\nRun the notebook (Section 38) or:\n"
            "  py -3 data/train_and_save_pipeline.py"
        )
        sys.exit(1)
    return joblib.load(PIPELINE_PATH)


def read_float(prompt: str, default: float | None = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()

        if raw == "" and default is not None:
            return default

        try:
            return float(raw)
        except ValueError:
            print("  Please enter a valid number.")


def collect_inputs(feature_columns: list[str]) -> dict[str, float]:
    print("\nEnter process values (press Enter to use the example default).\n")

    values: dict[str, float] = {}
    for feature in feature_columns:
        label = FEATURE_PROMPTS.get(feature, feature)
        default = EXAMPLE_VALUES.get(feature)
        value = read_float(label, default)
        values[feature] = value

    fm_ratio = values["fm_ratio"]
    if fm_ratio not in VALID_FM_RATIOS:
        print(
            f"\nNote: F/M ratio {fm_ratio} is outside the six levels used in "
            f"training {VALID_FM_RATIOS}. Predictions may be less reliable."
        )

    return values


def predict(pipeline: dict, values: dict[str, float]) -> float:
    feature_columns = pipeline["feature_columns"]
    row = pd.DataFrame([values])[feature_columns]
    return float(pipeline["model"].predict(row)[0])


def print_result(pipeline: dict, values: dict[str, float], prediction: float) -> None:
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Model              : {pipeline['model_name']}")
    print(f"F/M ratio          : {values['fm_ratio']}")
    print(f"Predicted H₂ yield : {prediction:.2f}")
    print("=" * 60)


def main() -> None:
    print("=" * 60)
    print("H₂ YIELD PREDICTOR")
    print("=" * 60)
    print("Enter fermentation/process parameters to predict hydrogen yield.")

    pipeline = load_pipeline()
    feature_columns = pipeline["feature_columns"]

    while True:
        values = collect_inputs(feature_columns)
        prediction = predict(pipeline, values)
        print_result(pipeline, values, prediction)

        again = input("\nPredict again? (y/n) [y]: ").strip().lower()
        if again in ("n", "no"):
            print("Done.")
            break


if __name__ == "__main__":
    main()
