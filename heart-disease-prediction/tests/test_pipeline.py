import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src import config
from src.data_preprocessing import clean, load_raw, split
from src.feature_engineering import add_features, build_pipeline

SAMPLE = dict(age=58, sex=1, cp=0, trestbps=140, chol=260, fbs=0, restecg=1,
              thalach=120, exang=1, oldpeak=2.0, slope=1, ca=2, thal=3)


def test_clean_removes_duplicates():
    df = load_raw()
    assert len(clean(pd.concat([df, df.iloc[[0]]]))) <= len(df)


def test_split_is_stratified():
    X_tr, X_te, y_tr, y_te = split(clean(load_raw()))
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
    assert list(X_tr.columns) == config.FEATURES


def test_engineered_feature():
    out = add_features(pd.DataFrame([SAMPLE]))
    assert out.loc[0, "hr_reserve_pct"] == pytest.approx(120 / (220 - 58))


def test_pipeline_fits_and_predicts():
    X_tr, X_te, y_tr, _ = split(clean(load_raw()))
    pipe = build_pipeline(LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_api():
    from app.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", json=SAMPLE)
    assert r.status_code == 200 and 0 <= r.json()["probability"] <= 1
    assert c.post("/predict", json={**SAMPLE, "age": 5}).status_code == 422


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_streamlit_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Estimated risk" in m.label for m in at.metric)
