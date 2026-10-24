"""Streamlit prototype: automotive parts procurement and simulated delivery risk."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from procurement_ml import (
    FEATURES,
    KOREAN_NAMES,
    example_suppliers,
    generate_demo_history,
    score_suppliers,
    train_model,
)

st.set_page_config(page_title="AI 구매 리스크 분석", page_icon="🚗", layout="wide")
st.title("🚗 AI 기반 자동차 부품 공급 리스크 예측")
st.caption("자동차 구매 직무 포트폴리오 | 개인 제작 학습용 시제품")
st.warning(
    "⚠️ 이 서비스의 기본 학습 데이터·협력사·결과는 모두 가상입니다. "
    "예측 확률은 실제 회사의 공급 리스크를 의미하지 않으며, 구매 결정을 자동으로 내릴 수 없습니다."
)

@st.cache_data(show_spinner=False)
def demo_data():
    return generate_demo_history()

@st.cache_resource(show_spinner="머신러닝 모델을 학습하는 중입니다...")
def model_for_demo():
    return train_model(demo_data())

st.sidebar.header("① 학습 데이터 설정")
mode = st.sidebar.radio("데이터 원본", ["가상 데이터(처음에는 이것 선택)", "내 CSV 업로드"], index=0)
source = "가상 데이터"
if mode.startswith("가상"):
    history = demo_data()
    model, metrics = model_for_demo()
else:
    st.sidebar.markdown("실제 발주 이력이라면 **개인정보·업체명·계약 정보는 제거**하세요.")
    template = demo_data().head(10).copy()
    st.sidebar.download_button(
        "CSV 열 이름 예시 내려받기", data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="training_columns_example.csv", mime="text/csv",
    )
    uploaded = st.sidebar.file_uploader("발주 이력 CSV 파일", type=["csv"])
    if uploaded is None:
        st.info("왼쪽 메뉴에서 CSV 파일을 올리거나 '가상 데이터'로 바꿔주세요.")
        st.stop()
    try:
        history = pd.read_csv(uploaded)
        model, metrics = train_model(history)
        source = "직접 업로드한 데이터"
    except (ValueError, KeyError, UnicodeError, pd.errors.ParserError) as exc:
        st.error("학습 데이터 오류: " + str(exc))
        st.stop()

st.sidebar.divider()
st.sidebar.header("② 의사결정 가중치")
risk_weight = st.sidebar.slider("납기 위험 중요도 (%)", 0, 100, 65, 5)
st.sidebar.caption(f"비교점수 = 위험도 {risk_weight}% + 정규화 단가 {100-risk_weight}%")

st.subheader("1. 데이터와 모델 검증")
st.write(f"데이터 원본: **{source}**  |  학습 및 평가에 사용된 전체 발주 기록: **{len(history):,}건**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("테스트 ROC-AUC", f"{metrics['auc']:.3f}")
col2.metric("지연 탐지율(Recall)", f"{metrics['recall']:.1%}")
col3.metric("정밀도(Precision)", f"{metrics['precision']:.1%}")
col4.metric("테스트 기록 수", f"{metrics['test_rows']:,}")
st.caption(
    "학습·평가는 다른 행으로 분리했습니다(훈련 75%, 평가 25%). "
    "데모에서 수치는 가상 데이터의 성능일 뿐 실제 납기 예측 성능이 아닙니다. "
    "실무 검증에서는 날짜 기준 검증과 공급업체별 데이터 누출 점검도 필요합니다."
)
with st.expander("학습 데이터와 변수 중요도 확인"):
    st.dataframe(history.head(20).rename(columns=KOREAN_NAMES), use_container_width=True)
    st.markdown("**Random Forest 모델 변수 중요도(설명용)**")
    imp = pd.Series(model.feature_importances_, index=[KOREAN_NAMES[c] for c in FEATURES]).sort_values(ascending=False)
    st.bar_chart(imp)
    st.caption("변수 중요도는 인과관계나 가격 인하 효과를 증명하지 않습니다.")

st.divider()
st.subheader("2. 협력사 후보 조건 입력")
st.write("아래 셀을 클릭해서 값을 수정해보세요. 표의 A·B·C 협력사는 모두 가상입니다.")

with st.expander("입력항목 설명", expanded=False):
    st.markdown(
        "- **과거 납기 준수율:** 발주 전에 관측된 과거 이력의 비율 (%)\n"
        "- **과거 불량률:** 발주 전에 관측된 품질 이력 (%)\n"
        "- **설비 가동률:** 협력사의 설비 사용 비율 (%)\n"
        "- **약속 납기:** 이번 주문의 계약상 예정 리드타임(일)\n"
        "- **해외 운송:** 0=국내 / 1=해외 운송 포함\n"
        "- **주문 수량:** 이번 주문 수량(개)"
    )

candidates = example_suppliers()
candidates_editable = candidates.rename(columns=KOREAN_NAMES)
user_edited = st.data_editor(
    candidates_editable,
    hide_index=True,
    use_container_width=True,
    num_rows="dynamic",
    key="suppliers_editor",
)

if st.button("🔎 AI 위험도 분석하기", type="primary", use_container_width=True):
    st.session_state["analyze"] = True

if st.session_state.get("analyze", True):
    try:
        result = score_suppliers(
            model,
            user_edited.rename(columns={v: k for k, v in KOREAN_NAMES.items()}),
            risk_weight_pct=risk_weight,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.divider()
    st.subheader("3. AI 예측 결과")
    show = result[["supplier", "unit_price_krw", "past_on_time_rate_pct", "예측 지연 위험(%)", "비교점수(낮을수록 유리)"]].copy()
    show = show.rename(columns={
        "supplier": "협력사", "unit_price_krw": "단가(원)",
        "past_on_time_rate_pct": "과거 납기 준수율(%)"
    })
    show["예측 지연 위험(%)"] = show["예측 지연 위험(%)"].round(1)
    show["비교점수(낮을수록 유리)"] = show["비교점수(낮을수록 유리)"].round(1)
    st.dataframe(show, hide_index=True, use_container_width=True)
    plot_data = result.set_index("supplier")[["예측 지연 위험(%)"]]
    st.bar_chart(plot_data)

    winner = result.iloc[0]
    st.info(
        f"📌 **우선 검토 후보: {winner['supplier']}** — "
        f"현재 설정한 비용·납기 위험 가중치에 따른 비교점수가 가장 낮습니다. "
        "실제 선정에는 부품 적합성, 품질 인증, 생산능력 검증 등이 추가로 필요합니다."
    )
    st.download_button(
        "분석 결과 CSV로 저장",
        data=show.to_csv(index=False).encode("utf-8-sig"),
        file_name="supplier_risk_analysis_demo.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("4. 가상 개선 시나리오")
    chosen = st.selectbox("가동률 개선 효과를 실험할 협력사", result["supplier"].tolist())
    line = result.loc[result["supplier"] == chosen].iloc[0]
    adjustment = st.slider("설비 가동률 감소(퍼센트포인트)", 0, 25, 10, 5)
    changed = result[result["supplier"] == chosen][["supplier"] + FEATURES].copy()
    changed["capacity_utilization_pct"] = (changed["capacity_utilization_pct"] - adjustment).clip(lower=0)
    changed_prob = model.predict_proba(changed[FEATURES])[:, 1][0] * 100
    ca, cb = st.columns(2)
    ca.metric("원래 예측 위험", f"{line['예측 지연 위험(%)']:.1f}%")
    cb.metric("가동률 개선 가정 후", f"{changed_prob:.1f}%", delta=f"{changed_prob - line['예측 지연 위험(%)']:+.1f}%p")
    st.caption("이 분석은 입력값을 변경한 시뮬레이션이며, 가동률 변화가 실제 지연 확률을 이만큼 개선한다는 인과관계는 아닙니다.")

st.divider()
st.markdown("**프로젝트 설명**: Python · Random Forest (Scikit-learn) · Streamlit / 인위적으로 생성한 발주 이력으로 시연하는 개인 학습용 프로토타입")
st.caption("개인정보, 실제 거래처명, 대외비 구매 정보 및 회사 내부 데이터를 업로드하지 마세요.")
