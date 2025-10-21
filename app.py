import io, re, zipfile
import numpy as np, pandas as pd, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, precision_recall_curve, roc_curve
import joblib, matplotlib.pyplot as plt

st.set_page_config(page_title="Credit 15k/15k — Logistic", layout="wide")
st.title("지능형 신용평가 (15k/15k) — Logistic")

BAD = {'Charged Off','Default','Late (31-120 days)','Late (16-30 days)','In Grace Period','Does not meet the credit policy. Status: Charged Off'}
GOOD = {'Fully Paid','Does not meet the credit policy. Status: Fully Paid'}
CAND = ['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership','annual_inc','verification_status','purpose','addr_state','dti','delinq_2yrs','fico_range_low','fico_range_high','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','application_type']
CAT  = ['term','grade','sub_grade','home_ownership','verification_status','purpose','addr_state','application_type']

def pct_to_float(x):
    try: return float(str(x).replace('%',''))/100.0
    except: return np.nan
def emp_to_years(s):
    if pd.isna(s): return np.nan
    s=str(s)
    if s.startswith('10+'): return 10
    if s.startswith('< 1'): return 0
    m=re.search(r'(\d+)', s); return float(m.group(1)) if m else np.nan

def read_any(f):
    name=f.name.lower(); data=f.read()
    if name.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            inner=[n for n in z.namelist() if n.lower().endswith('.csv')]
            if not inner: return None
            raw=z.read(inner[0])
        for kw in [dict(low_memory=False), dict(low_memory=False, skiprows=1), dict(low_memory=False, engine='python'), dict(low_memory=False, engine='python', skiprows=1), dict(low_memory=False, engine='python', on_bad_lines='skip')]:
            try: return pd.read_csv(io.BytesIO(raw), **kw)
            except: pass
        return None
    else:
        for kw in [dict(low_memory=False), dict(low_memory=False, skiprows=1), dict(low_memory=False, engine='python'), dict(low_memory=False, engine='python', skiprows=1), dict(low_memory=False, engine='python', on_bad_lines='skip')]:
            try:
                f.seek(0); return pd.read_csv(f, **kw)
            except: pass
        return None

def normalize_accepted(df):
    df=df.copy(); df.columns=[c.strip().lower().replace(' ','_') for c in df.columns]
    if 'loan_status' not in df.columns: return pd.DataFrame()
    df=df[df['loan_status'].isin(BAD.union(GOOD))].copy()
    if df.empty: return df
    df['target']=np.where(df['loan_status'].isin(BAD),1,0)
    use=[c for c in CAND if c in df.columns]+['target']; d=df[use].copy()
    if 'int_rate' in d: d['int_rate']=d['int_rate'].apply(pct_to_float)
    if 'revol_util' in d: d['revol_util']=d['revol_util'].apply(pct_to_float)
    if 'emp_length' in d: d['emp_length']=d['emp_length'].apply(emp_to_years)
    num=[c for c in use if c not in CAT+['target']]
    if num: d=d.dropna(subset=num, how='any')
    return d

t1,t2=st.tabs(["① 학습","② 리포트"])
with t1:
    files=st.file_uploader("승인 CSV/ZIP 업로드(복수)", type=['csv','zip'], accept_multiple_files=True)
    seed=st.number_input("Seed",1,99999,3033)
    if st.button("학습 (15k/15k)") and files:
        parts=[]
        for f in files:
            raw=read_any(f)
            if raw is None or not len(raw): continue
            norm=normalize_accepted(raw)
            if len(norm): parts.append(norm)
        if not parts: st.error("정상 데이터 없음")
        else:
            d=pd.concat(parts, ignore_index=True).drop_duplicates()
            g=d[d.target==0]; b=d[d.target==1]
            g=g.sample(n=min(15000,len(g)), random_state=int(seed), replace=len(g)<15000)
            b=b.sample(n=min(15000,len(b)), random_state=int(seed), replace=len(b)<15000)
            bal=pd.concat([g,b], ignore_index=True)
            X=bal.drop(columns=['target']); y=bal['target'].astype(int)
            cat=[c for c in X.columns if c in CAT]; num=[c for c in X.columns if c not in cat]
            pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
            pipe=Pipeline([('prep',pre),('clf',LogisticRegression(max_iter=500, solver='lbfgs'))])
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=int(seed))
            pipe.fit(Xtr,ytr)
            proba=pipe.predict_proba(Xte)[:,1]; pred=(proba>=0.5).astype(int)
            st.success(f"AUROC={roc_auc_score(yte,proba):.4f}  AUPRC={average_precision_score(yte,proba):.4f}")
            st.write("혼동행렬:", confusion_matrix(yte,pred).tolist())
            st.text(classification_report(yte,pred,digits=4))
            pr_p,pr_r,_=precision_recall_curve(yte,proba); fpr,tpr,_=roc_curve(yte,proba)
            fig1=plt.figure(); plt.plot(pr_r,pr_p); st.pyplot(fig1)
            fig2=plt.figure(); plt.plot(fpr,tpr); st.pyplot(fig2)
            import io as _io; buf=_io.BytesIO(); joblib.dump(pipe,buf)
            st.download_button("모형 다운로드(.joblib)", data=buf.getvalue(), file_name="logit_pipeline_15k15k.joblib")
with t2:
    st.markdown("변수정의→탐색→시각화→전처리→분할(8:2)→Logit / Cut-off 탐색·PSI 권장")
