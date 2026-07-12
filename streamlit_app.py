"""H₂ yield predictor — deployable Streamlit app."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

H2_YIELD_UNIT = "mL H₂/g VS"
VALID_FM_RATIOS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

PRIMARY_DEFAULTS = {
    "substrate_mc": 5.26,
    "substrate_vs": 89.41,
    "inoculum_mc": 89.92,
    "inoculum_vs": 90.30,
    "scod": 6000.0,
    "tcod": 11400.0,
    "trs": 1.0,
    "lignin": 25.0,
    "hmf": 45.0,
    "fm_ratio": 1.0,
}


def find_pipeline_path() -> Path:
    candidates = [
        Path("data/models/h2_yield_pipeline.joblib"),
        Path("models/h2_yield_pipeline.joblib"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Model file not found. Expected data/models/h2_yield_pipeline.joblib"
    )


@st.cache_resource
def load_pipeline():
    return joblib.load(find_pipeline_path())


def derive_model_features(primary: dict) -> dict:
    if primary["tcod"] == 0:
        raise ValueError("TCOD must be greater than zero.")

    return {
        "substrate_mc": primary["substrate_mc"],
        "substrate_ts": 100.0 - primary["substrate_mc"],
        "substrate_vs": primary["substrate_vs"],
        "substrate_fs": 100.0 - primary["substrate_vs"],
        "inoculum_mc": primary["inoculum_mc"],
        "inoculum_ts": 100.0 - primary["inoculum_mc"],
        "inoculum_vs": primary["inoculum_vs"],
        "inoculum_fs": 100.0 - primary["inoculum_vs"],
        "scod": primary["scod"],
        "tcod": primary["tcod"],
        "scod_tcod_ratio": primary["scod"] / primary["tcod"],
        "trs": primary["trs"],
        "lignin": primary["lignin"],
        "hmf": primary["hmf"],
        "fm_ratio": primary["fm_ratio"],
    }


def main():
    st.set_page_config(
        page_title="H₂ Yield Predictor",
        page_icon="⚗️",
        layout="centered",
    )

    st.title("Hydrogen Yield Predictor")
    st.caption(
        "Predict H₂ yield from dark fermentation process parameters "
        f"(output unit: **{H2_YIELD_UNIT}**)."
    )

    pipeline = load_pipeline()
    model = pipeline["model"]
    feature_columns = pipeline["feature_columns"]

    with st.sidebar:
        st.header("About the model")
        st.write(f"**Model:** {pipeline.get('model_name', 'Extra Trees')}")
        if "group_cv_mean_r2" in pipeline:
            st.metric("Group CV R²", f"{pipeline['group_cv_mean_r2']:.3f}")
        if "original_holdout_r2" in pipeline:
            st.metric("Original holdout R²", f"{pipeline['original_holdout_r2']:.3f}")
        st.markdown(
            "Enter **10 primary inputs**. TS, FS, and SCOD/TCOD ratio "
            "are calculated automatically."
        )

    st.subheader("Primary inputs")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Substrate & inoculum**")
        substrate_mc = st.number_input(
            "Substrate MC (%)", value=PRIMARY_DEFAULTS["substrate_mc"], min_value=0.0, max_value=100.0
        )
        substrate_vs = st.number_input(
            "Substrate VS (%)", value=PRIMARY_DEFAULTS["substrate_vs"], min_value=0.0, max_value=100.0
        )
        inoculum_mc = st.number_input(
            "Inoculum MC (%)", value=PRIMARY_DEFAULTS["inoculum_mc"], min_value=0.0, max_value=100.0
        )
        inoculum_vs = st.number_input(
            "Inoculum VS (%)", value=PRIMARY_DEFAULTS["inoculum_vs"], min_value=0.0, max_value=100.0
        )
        fm_ratio = st.selectbox(
            "F/M ratio (dimensionless)",
            options=VALID_FM_RATIOS,
            index=VALID_FM_RATIOS.index(PRIMARY_DEFAULTS["fm_ratio"]),
        )

    with col2:
        st.markdown("**Process chemistry**")
        scod = st.number_input("SCOD (mg/L)", value=PRIMARY_DEFAULTS["scod"], min_value=0.0)
        tcod = st.number_input("TCOD (mg/L)", value=PRIMARY_DEFAULTS["tcod"], min_value=0.0)
        trs = st.number_input("TRS (mg/L)", value=PRIMARY_DEFAULTS["trs"], min_value=0.0)
        lignin = st.number_input("Lignin (%)", value=PRIMARY_DEFAULTS["lignin"], min_value=0.0)
        hmf = st.number_input("HMF / 5-HMF (mg/L)", value=PRIMARY_DEFAULTS["hmf"], min_value=0.0)

    primary = {
        "substrate_mc": substrate_mc,
        "substrate_vs": substrate_vs,
        "inoculum_mc": inoculum_mc,
        "inoculum_vs": inoculum_vs,
        "scod": scod,
        "tcod": tcod,
        "trs": trs,
        "lignin": lignin,
        "hmf": hmf,
        "fm_ratio": float(fm_ratio),
    }

    if st.button("Predict H₂ yield", type="primary", use_container_width=True):
        try:
            features = derive_model_features(primary)
        except ValueError as exc:
            st.error(str(exc))
            return

        X_new = pd.DataFrame([features])[feature_columns]
        predicted = float(model.predict(X_new)[0])

        st.subheader("Derived values")
        d1, d2 = st.columns(2)
        with d1:
            st.write(f"Substrate TS: **{features['substrate_ts']:.2f} %**")
            st.write(f"Substrate FS: **{features['substrate_fs']:.2f} %**")
            st.write(f"Inoculum TS: **{features['inoculum_ts']:.2f} %**")
        with d2:
            st.write(f"Inoculum FS: **{features['inoculum_fs']:.2f} %**")
            st.write(
                f"SCOD/TCOD ratio: **{features['scod_tcod_ratio']:.4f}** (dimensionless)"
            )

        st.subheader("Prediction")
        st.metric("Predicted H₂ yield", f"{predicted:.2f} {H2_YIELD_UNIT}")


if __name__ == "__main__":
    main()
