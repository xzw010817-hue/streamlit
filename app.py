# ---------------------------------------------------------------
# LendingClub Intelligent Credit Scoring (15k/15k) — Logistic
# Streamlit app | Korean report-ready | Multi-file merge
# ---------------------------------------------------------------

# Emergency hotfix: ensure scikit-learn stack exists (for some hosts)
try:
    from sklearn.linear_model import LogisticRegression  # probe
except Exception:
    import sys, subprocess
    pkgs = [
        "scikit-learn==1.5.1",
        "numpy>=1.26,<3.0",
        "scipy>=1.10",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "joblib>=1.3"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
    from sklearn.linear_model import LogisticRegression

import io, re, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)

st.set_page_config(page_title="지능형 신용평가 (15k/15k) — Logistic", layout="wide")
st.title("지능형 신용평가 (15k/15k) — Logistic Baseline")
st.caption("여러 분기의 LendingClub 데이터를 병합하여 Good 15,000 / Bad 15,000 균형 샘플로 로지스틱 회귀를 학습합니다.")

BAD = {
    'Charged Off','Default','Late (31-120 days)','Late (16-30 days)',
    'In Grace Period','Does not meet the credit policy. Status: Charged Off'
}
GOOD = {
    'Fully Paid','Does not meet the credit policy. Status: Fully Paid'
}
CAND = [
    'loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length',
    'home_ownership','annual_inc','verification_status','purpose','addr_state',
    'dti','delinq_2yrs','fico_range_low','fico_range_high','inq_last_6mths',
    'open_acc','pub_rec','revol_bal','revol_util','total_acc','application_type'
]
CATS = ['term','grade','sub_grade','home_ownership','verification_status','purpose','addr_state','application_type']

def pct_to_float(x):
    try:
        return float(str(x).replace('%',''))/100.0
    except:
        return np.nan

def emp_to_years(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    if s == '10+ years': return 10
    if s == '< 1 year': return 0
    m = re.search(r'(\d+)', s)
    return float(m.group(1)) if m else np.nan

def read_any_csv_or_zip(uf):
    "Accept CSV or ZIP(contains CSV). Be forgiving with headers and comments."
    name = uf.name
    raw = uf.read()
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            inner = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not inner: return None
            data = zf.read(inner[0])
            fileobj = io.BytesIO(data)
    else:
        fileobj = io.BytesIO(raw)

    # try multiple read strategies (LC files sometimes have a comment row 1)
    for kw in [
        dict(low_memory=False),
        dict(low_memory=False, skiprows=1),
        dict(low_memory=False, engine="python"),
        dict(low_memory=False, engine="python", skiprows=1),
        dict(low_memory=False, engine="python", on_bad_lines="skip"),
    ]:
        fileobj.seek(0)
        try:
            return pd.read_csv(fileobj, **kw)
        except Exception:
            continue
    return None

def normalize_accepted(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if 'loan_status' not in df.columns:
        return pd.DataFrame()
    df['loan_status'] = df['loan_status'].astype(str).str.strip()
    df = df[df['loan_status'].isin(BAD|GOOD)].copy()
    if df.empty: return df
    df['target'] = np.where(df['loan_status'].isin(BAD), 1, 0)

    avail = [c for c in CAND if c in df.columns]
    work = df[avail + ['target']].copy()

    if 'int_rate' in work.columns:
        work['int_rate'] = work['int_rate'].apply(pct_to_float)
    if 'revol_util' in work.columns:
        work['revol_util'] = work['revol_util'].apply(pct_to_float)
    if 'emp_length' in work.columns:
        work['emp_length'] = work['emp_length'].apply(emp_to_years)

    num_cols = [c for c in avail if c not in CATS]
    if num_cols:
        work = work.dropna(subset=num_cols, how='any')
    return work

def normalize_rejects(df):
    lower = {c.lower().strip(): c for c in df.columns}
    def gc(name): return name if name in df.columns else lower.get(name.lower().strip())
    out = pd.DataFrame()

    c = gc("Amount Requested")
    if c is not None: out["loan_amnt"] = pd.to_numeric(df[c], errors="coerce")

    c = gc("Employment Length")
    if c is not None: out["emp_length"] = df[c].apply(emp_to_years)

    c = gc("State")
    if c is not None: out["addr_state"] = df[c].astype(str).str.strip()

    c = gc("Debt-To-Income Ratio")
    if c is not None:
        try:
            out["dti"] = df[c].astype(str).str.replace('%','', regex=False).astype(float)
        except:
            out["dti"] = pd.to_numeric(df[c], errors="coerce")

    c = gc("FICO Range")
    if c is not None:
        lohi = df[c].astype(str).str.extract(r'(\d{2,3})\s*-\s*(\d{2,3})')
        if lohi.shape[1] == 2:
            out["fico_range_low"]  = pd.to_numeric(lohi[0], errors="coerce")
            out["fico_range_high"] = pd.to_numeric(lohi[1], errors="coerce")

    c = gc("Loan Title")
    if c is not None:
        out["purpose"] = df[c].astype(str).str.strip().replace("", "Unknown")

    out["application_type"] = "INDIVIDUAL"
    return out

def build_pipeline(X):
    from sklearn.linear_model import LogisticRegression
    cat = [c for c in X.columns if c in CATS]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    return pipe, num, cat

t1, t2, t3 = st.tabs(["① 데이터 업로드·학습", "② 리포트 & 해석", "③ RejectStats 점수화"])

with t1:
    st.header("데이터 업로드 → 정규화 → 15k/15k 샘플 → 로지스틱 학습")
    files = st.file_uploader("승인 데이터(LoanStats) CSV/ZIP 업로드 — 여러 파일 병합", type=["csv","zip"], accept_multiple_files=True)
    seed = st.number_input("Random Seed", min_value=1, value=3033, step=1)
    run = st.button("학습 실행 (15k/15k)")

    if run and files:
        dfs = []
        for f in files:
            raw = read_any_csv_or_zip(f)
            if raw is not None and len(raw):
                norm = normalize_accepted(raw)
                if len(norm): dfs.append(norm)
        if not dfs:
            st.error("정상 행이 있는 데이터가 없습니다.")
        else:
            data = pd.concat(dfs, ignore_index=True).drop_duplicates()
            good = data[data['target']==0]; bad = data[data['target']==1]
            st.write(f"정규화 후 표본: Good={len(good):,}, Bad={len(bad):,}")
            g_n = min(15000, len(good)); b_n = min(15000, len(bad))
            g = good.sample(n=g_n, random_state=int(seed), replace=(g_n>len(good)))
            b = bad.sample(n=b_n,  random_state=int(seed), replace=(b_n>len(bad)))
            df_bal = pd.concat([g,b], ignore_index=True)

            X = df_bal.drop(columns=['target']); y = df_bal['target'].astype(int)
            pipe, num_cols, cat_cols = build_pipeline(X)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=int(seed))
            pipe.fit(Xtr, ytr)

            proba = pipe.predict_proba(Xte)[:,1]
            pred  = (proba>=0.5).astype(int)
            auroc = roc_auc_score(yte, proba)
            auprc = average_precision_score(yte, proba)
            cm = confusion_matrix(yte, pred)
            rep = classification_report(yte, pred, digits=4)

            st.success(f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
            st.text("분류 리포트:\n" + rep)

            pr_p, pr_r, _ = precision_recall_curve(yte, proba)
            fpr, tpr, _ = roc_curve(yte, proba)

            fig1 = plt.figure(); plt.plot(pr_r, pr_p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve")
            st.pyplot(fig1)

            fig2 = plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
            st.pyplot(fig2)

            # coefficients
            try:
                feat_names = list(pipe.named_steps["prep"].get_feature_names_out())
            except Exception:
                feat_names = list(X.columns)
            coef = pipe.named_steps["clf"].coef_.ravel()
            imp = (pd.DataFrame({"feature": feat_names, "coef": coef, "abs_coef": np.abs(coef)})
                     .sort_values("abs_coef", ascending=False))
            st.subheader("상위 계수 중요도")
            st.dataframe(imp.head(40))

            # download trained model
            import joblib
            buf = io.BytesIO(); joblib.dump(pipe, buf)
            st.download_button("모형 다운로드 (.joblib)", data=buf.getvalue(), file_name="logit_pipeline_15k15k.joblib")

            st.session_state["model"] = pipe

with t2:
    st.header("보고서(요약)")
    st.markdown("""
- 표본: **건전 15,000 / 부실 15,000** (보유량 부족 시 자동 축소)
- 알고리즘: **Logistic Regression (L2)**
- 전처리: 수치 **표준화**, 범주형 **원-핫 인코딩**
- 분할: Stratified **8:2**
- 지표: AUROC, AUPRC, 혼동행렬, PR/ROC 곡선, 계수 중요도
""")

with t3:
    st.header("RejectStats (거절신청) 점수화")
    rej_files = st.file_uploader("RejectStats CSV/ZIP 업로드 (A/B/D 등)", type=["csv","zip"], accept_multiple_files=True)
    cutoff = st.slider("임계값(cut-off): 낮을수록 엄격", min_value=0.05, max_value=0.95, value=0.7, step=0.01)
    go = st.button("점수화 실행")

    if go and rej_files:
        if "model" not in st.session_state:
            st.error("먼저 ① 탭에서 모델을 학습하세요.")
        else:
            pipe = st.session_state["model"]
            pre = pipe.named_steps["prep"]
            exp_num, exp_cat = [], []
            for name, trans, cols in pre.transformers_:
                if name == "num": exp_num = list(cols)
                elif name == "cat": exp_cat = list(cols)

            parts = []
            for f in rej_files:
                raw = read_any_csv_or_zip(f)
                if raw is None or not len(raw): continue
                norm = normalize_rejects(raw)
                if not len(norm): continue
                for c in exp_num:
                    if c not in norm.columns: norm[c] = 0.0
                for c in exp_cat:
                    if c not in norm.columns: norm[c] = "Unknown"
                keep = list(dict.fromkeys(exp_num + exp_cat))
                ali = norm[keep].copy()

                proba = pipe.predict_proba(ali)[:,1]
                ali["pd_hat"] = proba
                ali["approve_flag"] = (ali["pd_hat"] < cutoff).astype(int)
                parts.append(ali)

            if not parts:
                st.error("정상적으로 파싱된 RejectStats 데이터가 없습니다.")
            else:
                scored = pd.concat(parts, ignore_index=True)
                st.write("합산 표본 수:", len(scored))
                st.write("승인율(임계값 기준):", float(scored["approve_flag"].mean()))
                q = scored["pd_hat"].quantile([0.1,0.5,0.9]).to_dict()
                st.write("PD 분위수:", {k: float(v) for k,v in q.items()})
                st.dataframe(scored.head(100))

                csv_bytes = scored.to_csv(index=False).encode("utf-8")
                st.download_button("RejectStats 점수 결과 다운로드 (CSV)", data=csv_bytes, file_name="rejects_scored.csv")
