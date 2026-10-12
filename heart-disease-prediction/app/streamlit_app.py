"""Streamlit UI.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import config
from src.predict import load_bundle, predict_one

st.set_page_config(page_title="Heart Disease Risk", page_icon="🫀", layout="wide")
st.title("🫀 Heart Disease Risk Screening")
st.caption("Screening-support demo on the public UCI Cleveland dataset. Not a medical diagnosis.")

if not config.MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

bundle = load_bundle()
tab_predict, tab_model = st.tabs(["Risk prediction", "Model performance"])

with tab_predict:
    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 100, 55)
        sex = c1.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female")
        cp = c1.selectbox("Chest pain type (0-3)", [0, 1, 2, 3])
        trestbps = c2.number_input("Resting blood pressure (mm Hg)", 70, 250, 130)
        chol = c2.number_input("Cholesterol (mg/dl)", 80, 700, 240)
        fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                           format_func=lambda x: "Yes" if x else "No")
        restecg = c3.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = c3.number_input("Max heart rate achieved", 50, 230, 150)
        exang = c3.selectbox("Exercise-induced angina", [0, 1],
                             format_func=lambda x: "Yes" if x else "No")
        c4, c5, c6 = st.columns(3)
        oldpeak = c4.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = c5.selectbox("ST slope (0-2)", [0, 1, 2], index=1)
        ca = c5.selectbox("Major vessels coloured (0-4)", [0, 1, 2, 3, 4])
        thal = c6.selectbox("Thalassemia result (0-3)", [0, 1, 2, 3], index=2)
        go = st.form_submit_button("Predict risk", type="primary")

    if go:
        res = predict_one(dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                               restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
                               slope=slope, ca=ca, thal=thal))
        colour = {"low": "green", "elevated": "orange", "high": "red"}[res["risk_level"]]
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated risk", f"{res['probability']:.0%}")
        m2.markdown(f"**Risk level:** :{colour}[{res['risk_level'].upper()}]")
        m3.metric("Decision threshold", f"{res['threshold']:.2f}")
        st.progress(min(res["probability"], 1.0))
        if res["prediction"]:
            st.warning("Above the screening threshold: suggest clinical follow-up.")
        else:
            st.success("Below the screening threshold.")

with tab_model:
    st.subheader(f"Selected model: {bundle['model_name']}")
    mt = bundle["metrics"]
    cols = st.columns(5)
    for col, key in zip(cols, ["recall", "precision", "f1", "roc_auc", "pr_auc"]):
        col.metric(key.replace("_", " ").upper(), f"{mt[key]:.2f}")
    st.caption("Metrics are on a small held-out test set (61 patients), so treat them as indicative.")
    comp = config.REPORTS_DIR / "model_comparison.csv"
    if comp.exists():
        st.dataframe(pd.read_csv(comp).drop(columns=["best_params"]).round(3), hide_index=True)
    left, right = st.columns(2)
    for path, col, cap in [(config.REPORTS_DIR / "roc_curve.png", left, "ROC curve"),
                           (config.REPORTS_DIR / "confusion_matrix.png", right, "Confusion matrix"),
                           (config.REPORTS_DIR / "shap_summary.png", left, "SHAP feature impact"),
                           (config.REPORTS_DIR / "pr_curve.png", right, "Precision-recall curve")]:
        if path.exists():
            col.image(str(path), caption=cap)
