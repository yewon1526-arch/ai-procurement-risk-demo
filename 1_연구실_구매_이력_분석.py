"""Research lab purchasing analysis page for a Streamlit portfolio.

Upload the separately supplied simulated biosensor lab CSV to run it.
Do not portray generated purchasing records as University of Arizona originals.
"""
from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="연구실 구매 이력 분석", page_icon="🧪", layout="wide")
st.title("🧪 바이오센서 연구실 구매 이력 분석")
st.caption("SALSA 종이 미세유체칩 · IMCD Kidney-on-a-Chip · 스마트폰 측정 시스템")
st.warning(
    "이 페이지는 학습용 프로토타입입니다. 제공된 예시 CSV의 모든 발주 기록·업체·가격은 "
    "실제 연구실 기록이 아닌 합성 모의 데이터입니다. 분석 결과와 머신러닝 위험도는 "
    "실제 연구실 또는 공급업체의 실적을 의미하지 않습니다."
)

REQUIRED = [
    "PO_ID", "Project", "Item_Code", "Item_Name", "Specification", "Supplier_ID",
    "Supplier_Name", "Order_Date", "Promised_Date", "Receipt_Date", "Quantity",
    "Unit_Price_USD", "Total_Cost_USD", "Planned_Lead_Days", "Delay_Days",
    "Delivery_Status", "Quality_Result",
]
NUMERIC = ["Quantity", "Unit_Price_USD", "Total_Cost_USD", "Planned_Lead_Days"]
FEATURE_NUM = ["Quantity", "Unit_Price_USD", "Planned_Lead_Days"]
FEATURE_CAT = ["Project", "Category", "Supplier_ID", "Criticality", "Order_Channel"]


def read_data(blob: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(blob), encoding="utf-8-sig")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("누락된 필수 열: " + ", ".join(missing))
    if df.empty:
        raise ValueError("CSV에 발주 이력이 없습니다.")
    for name in NUMERIC:
        df[name] = pd.to_numeric(df[name], errors="coerce")
        if df[name].isna().any():
            raise ValueError(f"'{name}' 열에 빈칸 또는 숫자가 아닌 값이 있습니다.")
    if (df[["Quantity", "Unit_Price_USD", "Planned_Lead_Days"]] <= 0).any().any():
        raise ValueError("수량, 단가 및 약속 납기는 0보다 커야 합니다.")
    df["Delay_Days"] = pd.to_numeric(df["Delay_Days"], errors="coerce")
    for name in ["Order_Date", "Promised_Date", "Receipt_Date"]:
        df[name] = pd.to_datetime(df[name], errors="coerce")
    if df["Order_Date"].isna().any() or df["Promised_Date"].isna().any():
        raise ValueError("Order_Date 또는 Promised_Date에 잘못된 날짜가 있습니다.")
    if "Category" not in df:
        df["Category"] = "Unknown"
    if "Criticality" not in df:
        df["Criticality"] = "Unknown"
    if "Order_Channel" not in df:
        df["Order_Channel"] = "Unknown"
    df["Delivery_Status"] = df["Delivery_Status"].str.upper().str.strip()
    df["Quality_Result"] = df["Quality_Result"].str.upper().str.strip()
    df["Is_Delayed"] = np.where(
        df["Delivery_Status"].isin(["ON_TIME", "LATE"]),
        (df["Delivery_Status"] == "LATE").astype(int),
        np.nan,
    )
    df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
    return df.sort_values(["Order_Date", "PO_ID"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_uploaded(blob: bytes) -> pd.DataFrame:
    return read_data(blob)


@st.cache_resource(show_spinner="모의 발주 이력으로 분류 모델을 학습 중입니다...")
def fit_model(blob: bytes):
    """Chronological holdout; no actual receipts or post-order variables used as features."""
    df = read_data(blob)
    complete = df[df["Is_Delayed"].notna()].sort_values(["Order_Date", "PO_ID"]).copy()
    if len(complete) < 100 or complete["Is_Delayed"].sum() < 15:
        return None
    boundary = int(len(complete) * 0.75)
    train, test = complete.iloc[:boundary], complete.iloc[boundary:]
    if train["Is_Delayed"].nunique() < 2 or test["Is_Delayed"].nunique() < 2:
        return None
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), FEATURE_NUM),
        ("category", OneHotEncoder(handle_unknown="ignore"), FEATURE_CAT),
    ])
    clf = Pipeline([
        ("prep", prep),
        ("forest", RandomForestClassifier(
            n_estimators=160, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        )),
    ])
    clf.fit(train[FEATURE_NUM + FEATURE_CAT], train["Is_Delayed"].astype(int))
    probs = clf.predict_proba(test[FEATURE_NUM + FEATURE_CAT])[:, 1]
    guesses = (probs >= 0.5).astype(int)
    return clf, {
        "test_rows": len(test),
        "auc": float(roc_auc_score(test["Is_Delayed"], probs)),
        "recall": float(recall_score(test["Is_Delayed"], guesses, zero_division=0)),
        "precision": float(precision_score(test["Is_Delayed"], guesses, zero_division=0)),
        "holdout_start": test["Order_Date"].min().strftime("%Y-%m-%d"),
    }


uploaded = st.file_uploader("구매 이력 CSV 업로드", type=["csv"], help="이전에 만든 UA_Biosensor_Lab_Purchasing_SIMULATED_Upload.csv를 선택하세요.")
if uploaded is None:
    st.info("위에서 CSV를 업로드하면 165건의 모의 발주 이력 대시보드와 공급 리스크 분석이 나타납니다.")
    st.markdown(
        "**지원 항목:** 구매액·월별 지출·납기 준수율·품질 상태·공급업체별 비교·"
        "동일 품목 대체 구매 이력·Random Forest 모델의 시간순 테스트 결과"
    )
    st.stop()

raw = uploaded.getvalue()
try:
    df = load_uploaded(raw)
except (ValueError, UnicodeError, pd.errors.ParserError) as e:
    st.error(f"파일을 분석할 수 없습니다: {e}")
    st.stop()

if "Data_Type" in df.columns and df["Data_Type"].eq("SYNTHETIC_DEMO").all():
    st.info("확인: 업로드된 파일은 합성 모의 구매 이력으로 표기되어 있습니다.")

st.sidebar.subheader("연구실 구매 이력 필터")
projects = st.sidebar.multiselect("프로젝트", sorted(df["Project"].dropna().unique()), default=sorted(df["Project"].dropna().unique()))
filtered = df[df["Project"].isin(projects)].copy()
if filtered.empty:
    st.warning("표시할 프로젝트를 하나 이상 선택하세요.")
    st.stop()

st.subheader("1. 구매 현황")
completed = filtered[filtered["Is_Delayed"].notna()]
late = int((completed["Is_Delayed"] == 1).sum())
on_time = int((completed["Is_Delayed"] == 0).sum())
pending = len(filtered) - len(completed)
rate = on_time / len(completed) if len(completed) else float("nan")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("발주 기록", f"{len(filtered):,}건")
c2.metric("총 주문 금액", f"${filtered['Total_Cost_USD'].sum():,.0f}")
c3.metric("정시 납품률", f"{rate:.1%}" if len(completed) else "—")
c4.metric("납기 지연", f"{late}건")
c5.metric("미입고/미확인", f"{pending}건")
st.caption("정시 납품률은 실제 수령 상태가 ON_TIME 또는 LATE로 확인된 이력만 계산합니다. 금액은 시연 데이터의 USD 기준입니다.")

x, y = st.columns(2)
with x:
    st.markdown("**월별 발주 금액 (USD)**")
    monthly = filtered.groupby("Month", as_index=False)["Total_Cost_USD"].sum()
    st.bar_chart(monthly.set_index("Month"))
with y:
    st.markdown("**프로젝트별 발주 금액 (USD)**")
    per_project = filtered.groupby("Project")["Total_Cost_USD"].sum().sort_values(ascending=False)
    st.bar_chart(per_project)

st.subheader("2. 공급업체별 납기·품질 비교")
rows = []
for (sid, name), group in filtered.groupby(["Supplier_ID", "Supplier_Name"], dropna=False):
    done = group[group["Is_Delayed"].notna()]
    rows.append({
        "업체 ID": sid,
        "공급업체": name,
        "발주 건수": int(len(group)),
        "총 발주금액 ($)": round(float(group["Total_Cost_USD"].sum()), 2),
        "정시 납품률 (%)": round(100 * (done["Is_Delayed"] == 0).mean(), 1) if len(done) else None,
        "평균 약속 납기 (일)": round(float(group["Planned_Lead_Days"].mean()), 1),
        "평균 지연일 (일)": round(float(done["Delay_Days"].mean()), 1) if len(done) else None,
        "품질 보류/반려 (건)": int(group["Quality_Result"].isin(["HOLD", "REJECTED"]).sum()),
    })
supplier_summary = pd.DataFrame(rows).sort_values(["발주 건수", "공급업체"], ascending=[False, True])
st.dataframe(supplier_summary, hide_index=True, use_container_width=True)
st.caption("업체별 발주 품목이 다를 수 있으므로 단가와 평균 납기를 업체 간 절대 성능 지표로 비교하지 않습니다.")

st.subheader("3. 동일 품목 대체 소싱 시나리오")
st.markdown("발주 경험이 있는 **동일 품목코드**의 공급업체를 비교합니다. 이력에 없는 업체의 견적은 생성하지 않습니다.")
item_options = (filtered[["Item_Code", "Item_Name"]].drop_duplicates()
                .sort_values(["Item_Name", "Item_Code"]))
items = {f"{r.Item_Name} [{r.Item_Code}]": r.Item_Code for r in item_options.itertuples(index=False)}
selection = st.selectbox("검토할 구매 품목", list(items))
item_code = items[selection]
item_df = filtered[filtered["Item_Code"] == item_code].copy()
item_rows = []
for (sid, sname), grp in item_df.groupby(["Supplier_ID", "Supplier_Name"]):
    known = grp[grp["Is_Delayed"].notna()]
    item_rows.append({
        "업체 ID": sid,
        "공급업체": sname,
        "해당 품목 과거 발주": len(grp),
        "평균 단가 ($)": round(float(grp["Unit_Price_USD"].mean()), 2),
        "평균 약속 납기 (일)": round(float(grp["Planned_Lead_Days"].mean()), 1),
        "해당 품목 정시 납품률 (%)": round(float((known["Is_Delayed"] == 0).mean()) * 100, 1) if len(known) else None,
    })
st.dataframe(pd.DataFrame(item_rows), hide_index=True, use_container_width=True)
backup_ids = sorted(set(str(a) for a in item_df.get("Backup_Supplier_ID", pd.Series(dtype=str)).dropna().unique() if str(a).strip()))
if backup_ids:
    st.caption("원본 파일에 적힌 대체 업체 후보 ID: " + ", ".join(backup_ids) + " (실제 견적·품질 승인 여부 미확인)")
if len(item_rows) == 1:
    st.info("이 품목은 기록상 구매 실적이 있는 업체가 1곳입니다. 대체 소싱을 확정하려면 별도 견적 및 규격·품질 검토가 필요합니다.")
else:
    st.success("2곳 이상의 발주 이력이 있어 해당 품목의 과거 가격과 납기를 비교할 수 있습니다.")

st.subheader("4. 머신러닝 납기 지연 분류 실험")
st.markdown(
    "Scikit-learn **Random Forest**를 사용합니다. 발주 전 알 수 있는 "
    "가격·수량·약속 납기·품목분류·업체 등의 정보만 사용하고, "
    "실제 입고일이나 지연일수는 모델 입력에서 제외했습니다."
)
try:
    trained = fit_model(raw)
except (ValueError, RuntimeError) as e:
    trained = None
    st.warning(f"모델 학습을 수행하지 못했습니다: {e}")
if trained is None:
    st.info("모델을 학습하기에 데이터가 부족하거나 테스트 구간에 정상·지연 사례가 모두 존재하지 않습니다.")
else:
    model, metrics = trained
    a, b, c, d = st.columns(4)
    a.metric("시간순 테스트 ROC-AUC", f"{metrics['auc']:.3f}")
    b.metric("지연 탐지율", f"{metrics['recall']:.1%}")
    c.metric("정밀도", f"{metrics['precision']:.1%}")
    d.metric("테스트 발주 건수", str(metrics["test_rows"]))
    st.caption(
        f"가상 이력의 앞 75%로 학습하고 이후 25%({metrics['holdout_start']} 이후)로 평가했습니다. "
        "소규모 모의 자료이므로 실제 미래 납기 성능으로 일반화할 수 없습니다."
    )
    examples = filtered[filtered["Is_Delayed"].notna()].copy()
    po_id = st.selectbox("위험 점수를 확인할 발주 사례", examples["PO_ID"].tolist(), key="po_risk")
    example = examples.loc[examples["PO_ID"] == po_id].iloc[[0]].copy()
    pred = float(model.predict_proba(example[FEATURE_NUM + FEATURE_CAT])[0, 1])
    st.metric("모델의 납기 지연 분류 점수", f"{pred:.1%}")
    st.caption("기존 발주 조건을 다시 입력한 참고용 모의 분류 점수입니다. 실제 납기 확률이나 원인·인과 효과로 해석하면 안 됩니다.")

st.subheader("5. 발주 상세 내역 및 파일 내보내기")
cols = ["PO_ID", "Project", "Item_Name", "Specification", "Supplier_Name", "Order_Date", "Promised_Date", "Receipt_Date", "Quantity", "Unit_Price_USD", "Total_Cost_USD", "Delivery_Status", "Quality_Result"]
view = filtered[cols].copy()
for c in ["Order_Date", "Promised_Date", "Receipt_Date"]:
    view[c] = view[c].dt.strftime("%Y-%m-%d").fillna("")
st.dataframe(view, hide_index=True, use_container_width=True)
st.download_button("현재 필터된 발주 내역 CSV 다운로드", data=view.to_csv(index=False).encode("utf-8-sig"), file_name="lab_procurement_filtered.csv", mime="text/csv")
st.caption("사용자 업로드 CSV는 이 Streamlit 페이지에서 분석하며, 이 코드에는 외부 데이터베이스 저장 기능이 없습니다.")
