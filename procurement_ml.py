"""Illustrative supplier delay-risk model for a learning project.

Synthetic data only; not validated for real-world procurement decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "unit_price_krw",
    "promised_lead_days",
    "past_on_time_rate_pct",
    "past_defect_rate_pct",
    "capacity_utilization_pct",
    "international_shipping",
    "order_quantity",
]
TARGET = "delayed"
KOREAN_NAMES = {
    "supplier": "협력사",
    "unit_price_krw": "단가(원)",
    "promised_lead_days": "약속 납기(일)",
    "past_on_time_rate_pct": "과거 납기 준수율(%)",
    "past_defect_rate_pct": "과거 불량률(%)",
    "capacity_utilization_pct": "설비 가동률(%)",
    "international_shipping": "해외 운송(1=예)",
    "order_quantity": "주문 수량(개)",
    "delayed": "실제 납기 지연(1=예)",
}


def generate_demo_history(size: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Generate explicitly synthetic pre-order features & simulated delay labels."""
    rng = np.random.default_rng(seed)
    lead = rng.integers(5, 31, size)
    past_rate = np.clip(rng.beta(10.0, 2.2, size) * 100.0, 55, 99.9)
    defects = np.clip(rng.gamma(1.5, 1.25, size), 0.1, 8)
    utilization = rng.integers(45, 101, size)
    international = rng.binomial(1, 0.40, size)
    quantities = rng.integers(100, 3100, size)
    price = rng.integers(12000, 82000, size)
    # Synthetic assumptions. This equation represents no real suppliers.
    logit = (
        -1.9
        + 0.065 * (lead - 14)
        + 0.060 * (utilization - 75)
        - 0.061 * (past_rate - 85)
        + 0.16 * (defects - 1.5)
        + 0.55 * international
        + 0.00017 * (quantities - 1200)
    )
    event_probability = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
    delayed = rng.binomial(1, event_probability, size)
    return pd.DataFrame(
        {
            "unit_price_krw": price,
            "promised_lead_days": lead,
            "past_on_time_rate_pct": np.round(past_rate, 2),
            "past_defect_rate_pct": np.round(defects, 2),
            "capacity_utilization_pct": utilization,
            "international_shipping": international,
            "order_quantity": quantities,
            "delayed": delayed,
        }
    )


def validate_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate uploaded histories and reject ambiguous or invalid inputs."""
    absent = [col for col in FEATURES + [TARGET] if col not in raw.columns]
    if absent:
        raise ValueError("필수 열이 없습니다: " + ", ".join(absent))
    clean = raw[FEATURES + [TARGET]].copy()
    for name in FEATURES + [TARGET]:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
    if clean.isnull().any().any():
        raise ValueError("빈칸 또는 숫자가 아닌 값이 있습니다. 데이터를 확인하세요.")
    for name in ("past_on_time_rate_pct", "past_defect_rate_pct", "capacity_utilization_pct"):
        if not clean[name].between(0, 100).all():
            raise ValueError(name + "은(는) 0~100 범위여야 합니다.")
    if (clean["unit_price_krw"] <= 0).any() or (clean["order_quantity"] <= 0).any() or (clean["promised_lead_days"] <= 0).any():
        raise ValueError("단가, 주문수량, 약속 납기는 0보다 커야 합니다.")
    if not clean["international_shipping"].isin([0, 1]).all():
        raise ValueError("international_shipping 값은 0 또는 1이어야 합니다.")
    if not clean[TARGET].isin([0, 1]).all():
        raise ValueError("delayed 값은 0 또는 1이어야 합니다.")
    counts = clean[TARGET].value_counts()
    if len(clean) < 100 or len(counts) != 2 or counts.min() < 15:
        raise ValueError("학습 데이터 100건 이상, 지연/정상 각 15건 이상이 필요합니다.")
    return clean


def train_model(history: pd.DataFrame, seed: int = 42):
    """Train on a held-out test split, never evaluate on the training rows."""
    history = validate_history(history)
    x_train, x_test, y_train, y_test = train_test_split(
        history[FEATURES], history[TARGET],
        test_size=0.25, random_state=seed, stratify=history[TARGET],
    )
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=9,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    guesses = (probabilities >= 0.5).astype(int)
    metrics = {
        "test_rows": len(y_test),
        "auc": float(roc_auc_score(y_test, probabilities)),
        "recall": float(recall_score(y_test, guesses, zero_division=0)),
        "precision": float(precision_score(y_test, guesses, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, guesses)),
        "delay_rate": float(history[TARGET].mean()),
    }
    return model, metrics


def example_suppliers() -> pd.DataFrame:
    """Fictional firms for a hands-on demonstration."""
    return pd.DataFrame(
        [
            ["A 협력사", 26000, 12, 92, 0.8, 68, 0, 1000],
            ["B 협력사", 24000, 19, 76, 3.3, 96, 1, 1000],
            ["C 협력사", 29500, 10, 96, 0.5, 70, 0, 1000],
        ],
        columns=["supplier"] + FEATURES,
    )


def score_suppliers(model, candidates: pd.DataFrame, risk_weight_pct: int = 65) -> pd.DataFrame:
    """Predict simulated risk and rank by clearly disclosed teaching-only weights."""
    absent = [col for col in ["supplier"] + FEATURES if col not in candidates]
    if absent:
        raise ValueError("협력사 입력에 필요한 열이 없습니다: " + ", ".join(absent))
    df = candidates.copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("협력사를 최소 1곳 입력해야 합니다.")
    if df["supplier"].astype(str).str.strip().eq("").any():
        raise ValueError("협력사 이름을 입력하세요.")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[FEATURES].isna().any().any():
        raise ValueError("숫자 열에 빈칸 또는 잘못된 입력이 있습니다.")
    if (df["unit_price_krw"] <= 0).any() or (df["promised_lead_days"] <= 0).any() or (df["order_quantity"] <= 0).any():
        raise ValueError("단가, 납기, 수량은 0보다 커야 합니다.")
    for col in ("past_on_time_rate_pct", "past_defect_rate_pct", "capacity_utilization_pct"):
        if not df[col].between(0, 100).all():
            raise ValueError("비율 입력은 0~100이어야 합니다.")
    if not df["international_shipping"].isin([0, 1]).all():
        raise ValueError("해외 운송은 0 또는 1을 입력하세요.")
    df["예측 지연 위험(%)"] = model.predict_proba(df[FEATURES])[:, 1] * 100
    prices = df["unit_price_krw"]
    if prices.max() == prices.min():
        normalized_cost = np.zeros(len(df))
    else:
        normalized_cost = (prices - prices.min()) / (prices.max() - prices.min()) * 100
    df["비교점수(낮을수록 유리)"] = (
        df["예측 지연 위험(%)"] * risk_weight_pct / 100
        + normalized_cost * (100 - risk_weight_pct) / 100
    )
    return df.sort_values("비교점수(낮을수록 유리)").reset_index(drop=True)
