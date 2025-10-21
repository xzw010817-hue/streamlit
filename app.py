import streamlit as st, pandas as pd, numpy as np, joblib

st.set_page_config(page_title="지능형 신용평가(Logit)", page_icon=“6688”, layout="wide")
st.title("지능형 신용평가 모형 (Logistic Regression)")

pipe = joblib.load("pipeline.joblib")

NUM = ["loan_amnt","term","int_rate","annual_inc","dti","emp_length","revol_util",
       "open_acc","total_acc","inq_last_6mths","delinq_2yrs","revol_bal","pub_rec"]
CAT = ["home_ownership","purpose","grade","sub_grade","verification_status","addr_state"]

st.subheader(" 개별 예측")
cols = st.columns(3); vals={}
for i,c in enumerate(NUM):
    with cols[i%3]:
        vals[c] = st.number_input(c, value=10000.0 if c=="loan_amnt" else 0.0)
for i,c in enumerate(CAT):
    with cols[i%3]:
        vals[c] = st.text_input(c, "A1" if c=="sub_grade" else "RENT" if c=="home_ownership" else "CA" if c=="addr_state" else "Verified")

if st.button("예측"):
    x = pd.DataFrame([vals])
    for c in NUM+CAT:
        if c not in x: x[c]=np.nan
    p_bad = float(pipe.predict_proba(x)[:,1][0])
    st.metric("부실 확률(p_bad)", f"{p_bad:.3f}")
    st.write("판정:", "부실(1)" if p_bad>=0.5 else "건전(0)")

st.subheader("CSV 배치 예측")
up = st.file_uploader("학습 스키마와 동일한 CSV 업로드", type="csv")
if up is not None:
    df = pd.read_csv(up)
    for c in NUM+CAT:
        if c not in df: df[c]=np.nan
    prob = pipe.predict_proba(df[NUM+CAT])[:,1]
    out = df.copy(); out["p_bad"]=prob; out["pred"]=(prob>=0.5).astype(int)
    st.dataframe(out.head(20))
    st.download_button("결과 다운로드", out.to_csv(index=False), "predictions.csv", "text/csv")
