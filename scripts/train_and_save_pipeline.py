"""Train the tuned Extra Trees model and save the H2 yield pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "h2_data.csv"
REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "reference_experimental_data.csv"
)
MODELS_DIR = PROJECT_ROOT / "data" / "models"
PIPELINE_PATH = MODELS_DIR / "h2_yield_pipeline.joblib"

TARGET = "h2_yield"
GROUP_COLUMN = "source_id"

BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 2,
    "max_features": 1.0,
}


def build_cv_splits(groups_train, source_level_data, train_source_set):
    training_source_data = source_level_data[
        source_level_data[GROUP_COLUMN].isin(train_source_set)
    ].reset_index(drop=True)

    source_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_splits = []
    for _, (source_train_idx, source_valid_idx) in enumerate(
        source_cv.split(
            training_source_data[GROUP_COLUMN],
            training_source_data["fm_ratio"].astype(str),
        ),
        start=1,
    ):
        fold_train_sources = set(
            training_source_data.iloc[source_train_idx][GROUP_COLUMN]
        )
        fold_valid_sources = set(
            training_source_data.iloc[source_valid_idx][GROUP_COLUMN]
        )
        row_train_idx = np.flatnonzero(
            groups_train.isin(fold_train_sources).to_numpy()
        )
        row_valid_idx = np.flatnonzero(
            groups_train.isin(fold_valid_sources).to_numpy()
        )
        cv_splits.append((row_train_idx, row_valid_idx))

    return cv_splits


def main():
    df = pd.read_csv(DATA_PATH)
    feature_columns = [
        column
        for column in df.columns
        if column not in [TARGET, GROUP_COLUMN]
    ]

    X = df[feature_columns].copy()
    y = df[TARGET].copy()
    groups = df[GROUP_COLUMN].copy()

    source_level_data = (
        df[[GROUP_COLUMN, "fm_ratio"]].drop_duplicates().reset_index(drop=True)
    )

    train_sources, _ = train_test_split(
        source_level_data[GROUP_COLUMN],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=source_level_data["fm_ratio"],
    )
    train_source_set = set(train_sources)
    train_mask = groups.isin(train_source_set)

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    groups_train = groups.loc[train_mask].copy()

    cv_splits = build_cv_splits(
        groups_train, source_level_data, train_source_set
    )

    model = ExtraTreesRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **BEST_PARAMS,
    )

    cv_results = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv_splits,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    experimental_df = pd.read_csv(REFERENCE_PATH)
    experimental_df["source_id"] = [
        f"EXP_{index + 1:03d}"
        for index in range(len(experimental_df))
    ]

    test_source_set = set(groups.loc[~train_mask].unique())
    original_holdout_df = (
        experimental_df[experimental_df["source_id"].isin(test_source_set)]
        .sort_values("source_id")
        .reset_index(drop=True)
    )

    X_original_holdout = original_holdout_df[feature_columns]
    y_original_holdout = original_holdout_df[TARGET].to_numpy()
    original_predictions = model.predict(X_original_holdout)

    pipeline = {
        "model": model,
        "model_name": "Tuned Extra Trees",
        "feature_columns": feature_columns,
        "target": TARGET,
        "group_column": GROUP_COLUMN,
        "random_state": RANDOM_STATE,
        "hyperparameters": BEST_PARAMS,
        "training_observations": len(X_train),
        "training_source_groups": int(groups_train.nunique()),
        "group_cv_mean_r2": float(np.mean(cv_results["test_r2"])),
        "group_cv_mean_mae": float(-np.mean(cv_results["test_mae"])),
        "group_cv_mean_rmse": float(-np.mean(cv_results["test_rmse"])),
        "original_holdout_r2": float(
            r2_score(y_original_holdout, original_predictions)
        ),
        "original_holdout_mae": float(
            mean_absolute_error(y_original_holdout, original_predictions)
        ),
        "original_holdout_rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_original_holdout, original_predictions
                )
            )
        ),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)

    print("Pipeline saved:", PIPELINE_PATH)
    print("Group CV R┬▓  :", round(pipeline["group_cv_mean_r2"], 4))
    print("Original R┬▓    :", round(pipeline["original_holdout_r2"], 4))
    print("Original MAE   :", round(pipeline["original_holdout_mae"], 4))


if __name__ == "__main__":
    main()
