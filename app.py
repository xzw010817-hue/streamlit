# ---- emergency hotfix: auto-install sklearn if missing ----
try:
    from sklearn.linear_model import LogisticRegression  # just to test availability
except ModuleNotFoundError:
    import sys, subprocess
    pkgs = [
        "scikit-learn==1.5.1",
        "numpy>=1.26,<3.0",
        "scipy>=1.10",
        "pandas>=2.0",
        "joblib>=1.3",
        "matplotlib>=3.7",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
    # re-import after installation
    from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title("LendingClub 대출 데이터 기반 신용평가 모델")

# 한글 표시를 위해 폰트 설정 (필요시)
st.set_option('deprecation.showfileUploaderEncoding', False)

# 사용자로부터 승인 대출 데이터 파일 업로드
uploaded_files = st.file_uploader("✅ 승인된 대출 데이터 CSV 파일을 업로드하세요 (여러 파일 선택 가능)", type=['csv'], accept_multiple_files=True)

# 세션 상태 초기화
if 'model_pipeline' not in st.session_state:
    st.session_state.model_pipeline = None

if uploaded_files:
    # Train Model 버튼
    if st.button("📊 모델 훈련 시작"):
        # 여러 CSV 파일 병합
        data_list = []
        for uploaded_file in uploaded_files:
            # 파일의 첫 줄 확인 (LendingClub 데이터의 불필요한 헤더)
            first_line = uploaded_file.readline().decode('utf-8', errors='ignore')
            if first_line.startswith("Notes offered by"):
                # 첫 줄 건너뛰고 데이터 읽기
                df = pd.read_csv(uploaded_file, skiprows=1, low_memory=False)
            else:
                # 첫 줄부터 읽기 (정상 CSV 헤더)
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, low_memory=False)
            data_list.append(df)
        df_accept = pd.concat(data_list, ignore_index=True)
        
        # 필요한 컬럼만 선택 (모델 입력 변수 + loan_status)
        features = ["loan_amnt","term","int_rate","installment","grade","sub_grade","emp_length",
                    "home_ownership","annual_inc","verification_status","purpose","addr_state","dti",
                    "delinq_2yrs","fico_range_low","fico_range_high","inq_last_6mths","open_acc",
                    "pub_rec","revol_bal","revol_util","total_acc","application_type","loan_status"]
        df_accept = df_accept.loc[:, [col for col in features if col in df_accept.columns]]
        
        # 대상 loan_status 필터링 (Good/Bad)
        good_statuses = ["Fully Paid", "Does not meet the credit policy. Status: Fully Paid"]
        bad_statuses = ["Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)", 
                        "In Grace Period", "Does not meet the credit policy. Status: Charged Off"]
        df_accept = df_accept[df_accept['loan_status'].isin(good_statuses + bad_statuses)].copy()
        # 레이블 생성: Bad=1, Good=0
        df_accept['target'] = df_accept['loan_status'].isin(bad_statuses).astype(int)
        
        # Good/Bad 각각 15,000건 샘플링 (혹시 부족하면 가능한 만큼만 사용)
        good_df = df_accept[df_accept['target'] == 0]
        bad_df = df_accept[df_accept['target'] == 1]
        n_sample = 15000
        if len(good_df) < n_sample:
            n_sample = len(good_df)
        if len(bad_df) < n_sample:
            n_sample = min(n_sample, len(bad_df))
        if len(good_df) > n_sample:
            good_df = good_df.sample(n=n_sample, random_state=42)
        if len(bad_df) > n_sample:
            bad_df = bad_df.sample(n=n_sample, random_state=42)
        df_sample = pd.concat([good_df, bad_df], ignore_index=True)
        
        # 전처리: emp_length 변환
        emp_mapping = {"10+ years": 10, "10 years": 10, "9 years": 9, "8 years": 8, "7 years": 7,
                       "6 years": 6, "5 years": 5, "4 years": 4, "3 years": 3, "2 years": 2,
                       "1 year": 1, "< 1 year": 0, "n/a": 0}
        if 'emp_length' in df_sample.columns:
            df_sample['emp_length'] = df_sample['emp_length'].map(emp_mapping)
        # 전처리: int_rate, revol_util % 제거
        if 'int_rate' in df_sample.columns:
            df_sample['int_rate'] = df_sample['int_rate'].astype(str).str.rstrip('%').astype('float')
        if 'revol_util' in df_sample.columns:
            df_sample['revol_util'] = df_sample['revol_util'].astype(str).str.rstrip('%')
            df_sample['revol_util'] = df_sample['revol_util'].replace('', np.nan).astype('float')
        
        # 결측치 제거 (수치형)
        numeric_cols = ["loan_amnt","int_rate","installment","emp_length","annual_inc","dti",
                        "delinq_2yrs","fico_range_low","fico_range_high","inq_last_6mths",
                        "open_acc","pub_rec","revol_bal","revol_util","total_acc"]
        exist_numeric = [col for col in numeric_cols if col in df_sample.columns]
        df_sample = df_sample.dropna(subset=exist_numeric)
        
        # 입력 X, 레이블 y 분리
        X = df_sample.drop(columns=['loan_status','target'])
        y = df_sample['target']
        
        # 학습/테스트 분할 (stratify=y로 계층샘플링)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # 범주형/수치형 컬럼 구분
        categorical_features = ["term","grade","sub_grade","home_ownership","verification_status",
                                 "purpose","addr_state","application_type"]
        numerical_features = ["loan_amnt","int_rate","installment","emp_length","annual_inc","dti",
                               "delinq_2yrs","fico_range_low","fico_range_high","inq_last_6mths",
                               "open_acc","pub_rec","revol_bal","revol_util","total_acc"]
        categorical_features = [col for col in categorical_features if col in X_train.columns]
        numerical_features = [col for col in numerical_features if col in X_train.columns]
        
        # 컬럼 변환기 정의
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        # 로지스틱 회귀 모델 파이프라인
        model = LogisticRegression(penalty='l2', max_iter=500, solver='lbfgs')
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('classifier', model)
        ])
        # 모델 훈련
        pipeline.fit(X_train, y_train)
        
        # 테스트 성능 평가
        from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
        y_pred_prob = pipeline.predict_proba(X_test)[:,1]
        y_pred = pipeline.predict(X_test)
        auroc = roc_auc_score(y_test, y_pred_prob)
        auprc = average_precision_score(y_test, y_pred_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        # 결과 화면 표시
        st.subheader("📈 모델 테스트 성능")
        st.write(f"- **테스트 AUROC**: {auroc:.3f}")
        st.write(f"- **테스트 AUPRC**: {auprc:.3f}")
        # 혼동 행렬 표시
        cm_df = pd.DataFrame(cm, index=["실제 Good(0)","실제 Bad(1)"], columns=["예측 Good(0)","예측 Bad(1)"])
        st.write("**혼동 행렬**:")
        st.table(cm_df)
        
        # 회귀 계수 기반 상위 특징 출력
        # 특성 이름 추출
        feature_names_num = numerical_features  # 그대로
        # 범주형 더미 특성 이름 추출
        onehot: OneHotEncoder = pipeline.named_steps['preprocess'].named_transformers_['cat']
        if onehot is not None:
            cat_cols_expanded = onehot.get_feature_names_out(categorical_features)
            feature_names = list(feature_names_num) + list(cat_cols_expanded)
        else:
            feature_names = feature_names_num
        coefs = pipeline.named_steps['classifier'].coef_[0]
        # 가장 큰 양의/음의 계수 정렬
        coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
        coef_df['abs_coef'] = coef_df['coef'].abs()
        coef_df = coef_df.sort_values('coef', ascending=False)
        top_pos = coef_df.head(5)
        top_neg = coef_df.sort_values('coef').head(5)
        st.write("**🔑 부실 확률을 높이는 주요 변수 (양의 계수 Top5):**")
        for i, row in top_pos.iterrows():
            st.write(f"- {row['feature']} (계수=+{row['coef']:.2f})")
        st.write("**🔑 부실 확률을 낮추는 주요 변수 (음의 계수 Top5):**")
        for i, row in top_neg.iterrows():
            st.write(f"- {row['feature']} (계수={row['coef']:.2f})")
        
        # 학습된 파이프라인 및 필요한 정보 저장
        st.session_state.model_pipeline = pipeline
        # 학습 데이터에서 Grade별 평균 이자율 계산 (Reject 데이터 처리에 활용)
        if 'grade' in X_train.columns and 'int_rate' in X_train.columns:
            grade_int_rate_map = X_train.groupby('grade')['int_rate'].mean().to_dict()
        else:
            grade_int_rate_map = {}
        st.session_state.grade_int_rate_map = grade_int_rate_map
        
        st.success("✅ 모델 훈련이 완료되었습니다. 아래에 새로운 신청서에 대한 평가를 진행할 수 있습니다.")
        
# Reject 데이터 평가 섹션 (모델 훈련 후에만 활성화)
if st.session_state.model_pipeline is not None:
    st.subheader("🔍 신규 거절 신청서 평가")
    reject_file = st.file_uploader("💼 거절된 신청 데이터 파일을 업로드하세요 (RejectStats CSV 또는 ZIP)", type=['csv','zip'])
    if reject_file:
        if st.button("🚀 부실 확률 예측하기"):
            # RejectStats 파일 읽기 (ZIP인 경우 내부 CSV 추출)
            import io, zipfile
            if reject_file.name.endswith('.zip'):
                # ZIP 파일에서 CSV 추출
                bytes_data = reject_file.read()
                z = zipfile.ZipFile(io.BytesIO(bytes_data))
                csv_name = z.namelist()[0]  # 첫 번째 CSV 파일명
                df_reject = pd.read_csv(z.open(csv_name), skiprows=1, low_memory=False)
            else:
                # CSV 파일 직접 읽기
                first_line = reject_file.readline().decode('utf-8', errors='ignore')
                if first_line.startswith("Notes offered by"):
                    df_reject = pd.read_csv(reject_file, skiprows=1, low_memory=False)
                else:
                    reject_file.seek(0)
                    df_reject = pd.read_csv(reject_file, low_memory=False)
            
            # 컬럼명 표준화 (RejectStats의 컬럼을 모델 입력 컬럼으로 맵핑)
            # RejectStats 데이터 예시 컬럼: Amount Requested, Application Date, Loan Title, Risk_Score, Debt-To-Income Ratio, Zip Code, State, Employment Length, Policy Code
            # 필요한 모델 입력 컬럼에 대응:
            df_new = pd.DataFrame()
            # 1) loan_amnt
            if 'Amount Requested' in df_reject.columns:
                df_new['loan_amnt'] = df_reject['Amount Requested']
            # 2) term (신청서에는 없으므로 대출금액 기준 가정: 15,000초과이면 60개월, 이하면 36개월)
            if 'loan_amnt' in df_new.columns:
                df_new['term'] = np.where(df_new['loan_amnt'] > 15000, "60 months", "36 months")
            else:
                df_new['term'] = "36 months"
            # 3) int_rate (신청 대출의 예상 이자율: 학습 데이터의 등급별 평균으로 대체)
            df_new['grade'] = None
            df_new['sub_grade'] = None
            if 'Risk_Score' in df_reject.columns:
                # Risk_Score를 이용하여 등급 추정
                def assign_grade(score):
                    try:
                        s = float(score)
                    except:
                        return "Unknown"
                    if np.isnan(s):
                        return "Unknown"
                    if s >= 740:
                        return "A"
                    elif s >= 700:
                        return "B"
                    elif s >= 660:
                        return "C"
                    elif s >= 620:
                        return "D"
                    elif s >= 580:
                        return "E"
                    elif s >= 540:
                        return "F"
                    else:
                        return "G"
                df_new['grade'] = df_reject['Risk_Score'].apply(assign_grade)
            # 빈 Grade는 Unknown 처리
            df_new['grade'] = df_new['grade'].fillna("Unknown")
            # Sub_grade는 Grade기준 중간값으로 (Unknown은 Unknown으로)
            df_new['sub_grade'] = df_new['grade'].apply(lambda g: f"{g}3" if g not in ["Unknown"] else "Unknown")
            # 이자율 할당: 학습데이터 Grade별 평균 사용, 없으면 전체 평균
            default_int = np.mean(list(st.session_state.grade_int_rate_map.values())) if st.session_state.grade_int_rate_map else 10.0
            df_new['int_rate'] = df_new['grade'].apply(lambda g: st.session_state.grade_int_rate_map.get(g, default_int))
            # 4) installment (월 납부액 계산: 원리금 균등분할 공식)
            def calc_installment(principal, annual_rate, term_months):
                if annual_rate <= 0:
                    return principal / term_months
                r = annual_rate / 100 / 12
                n = term_months
                return principal * (r / (1 - (1+r)**(-n)))
            df_new['installment'] = df_new.apply(lambda row: calc_installment(row['loan_amnt'], row['int_rate'], 
                                                                             60 if row['term']=="60 months" else 36), axis=1)
            # 5) emp_length
            if 'Employment Length' in df_reject.columns:
                emp_map = {"10+ years": 10, "10 years": 10, "9 years": 9, "8 years": 8, "7 years": 7,
                           "6 years": 6, "5 years": 5, "4 years": 4, "3 years": 3, "2 years": 2,
                           "1 year": 1, "< 1 year": 0, "n/a": 0}
                df_new['emp_length'] = df_reject['Employment Length'].map(emp_map)
            else:
                df_new['emp_length'] = 0
            # 6) home_ownership (없으므로 기본값 RENT로 설정)
            df_new['home_ownership'] = "RENT"
            # 7) annual_inc (없으므로 추정 불가 - 평균값으로 대체하거나 0으로 설정)
            df_new['annual_inc'] = df_accept['annual_inc'].mean() if 'annual_inc' in df_accept.columns else 0
            # 8) verification_status (없으므로 Not Verified로)
            df_new['verification_status'] = "Not Verified"
            # 9) purpose (Loan Title로 추론 또는 기타)
            if 'Loan Title' in df_reject.columns:
                def infer_purpose(title):
                    if not isinstance(title, str):
                        return "other"
                    t = title.lower()
                    if "debt" in t:
                        return "debt_consolidation"
                    if "credit" in t and "card" in t:
                        return "credit_card"
                    if "home improvement" in t or "home_improvement" in t:
                        return "home_improvement"
                    if "business" in t:
                        return "small_business"
                    if "wedding" in t:
                        return "wedding"
                    if "medical" in t:
                        return "medical"
                    if "vacation" in t:
                        return "vacation"
                    if "moving" in t:
                        return "moving"
                    if "house" in t:
                        return "house"
                    if "education" in t:
                        return "educational"
                    return "other"
                df_new['purpose'] = df_reject['Loan Title'].apply(infer_purpose)
            else:
                df_new['purpose'] = "other"
            # 10) addr_state
            if 'State' in df_reject.columns:
                df_new['addr_state'] = df_reject['State']
            else:
                df_new['addr_state'] = "NA"
            # 11) dti
            if 'Debt-To-Income Ratio' in df_reject.columns:
                df_new['dti'] = df_reject['Debt-To-Income Ratio'].astype(str).str.rstrip('%').replace('', '0').astype('float')
            else:
                df_new['dti'] = 0.0
            # 12) delinq_2yrs, inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util, total_acc
            # 거절 데이터에 신용보고서 정보가 없으므로 모두 0 또는 평균 추정치로 채움
            df_new['delinq_2yrs'] = 0
            df_new['inq_last_6mths'] = 0
            df_new['open_acc'] = 0
            df_new['pub_rec'] = 0
            df_new['revol_bal'] = 0
            df_new['revol_util'] = 50.0  # 평균 50%
            df_new['total_acc'] = 0
            # 13) application_type
            df_new['application_type'] = "Individual"
            
            # 결측 처리 (혹시 모를 NaN)
            df_new = df_new.fillna({
                'emp_length': 0, 'annual_inc': df_new['annual_inc'].mean(), 'revol_util': 50.0
            })
            
            # 모델 예측
            pipeline = st.session_state.model_pipeline
            pred_probs = pipeline.predict_proba(df_new)[:,1]  # 부실 확률 (Class 1 확률)
            pred_labels = (pred_probs >= 0.5).astype(int)
            
            df_new['Predicted_PD'] = pred_probs
            df_new['Decision'] = np.where(pred_labels==1, "거절", "승인")
            
            # 결과 요약 및 표시
            total = len(df_new)
            approved = (df_new['Decision'] == "승인").sum()
            rejected = (df_new['Decision'] == "거절").sum()
            st.write(f"전체 신청 건수: **{total}건**")
            st.write(f"▶ 모델 결과 **승인**: {approved}건, **거절**: {rejected}건")
            # 상위 10개 미리보기
            st.dataframe(df_new[['loan_amnt','term','grade','sub_grade','int_rate','dti','Predicted_PD','Decision']].head(10))
            
            # CSV 다운로드 버튼
            output_csv = df_new.copy()
            output_csv.index = range(1, len(output_csv)+1)
            csv_data = output_csv.to_csv(index=False).encode('utf-8')
            st.download_button("💾 평가 결과 CSV 다운로드", csv_data, file_name="Reject_predictions.csv")
            
            st.info("✔️ 신규 신청서에 대한 부실확률(PD) 및 승인여부 판단이 완료되었습니다.")
