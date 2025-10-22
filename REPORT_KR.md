# 지능형신용평가모형 보고서 (15k/15k, Logistic)

**작성일:** 2025-10-22

## 1. 개요
- 표본: Good 15,000 + Bad 15,000 (부족 시 가능한 최대치)
- 알고리즘: Logistic Regression (L2), max_iter=500
- 분할: Stratified 8:2
- 전처리: 수치 표준화, 범주형 원-핫 인코딩
- 산출물: AUROC/AUPRC/혼동행렬, PR/ROC 곡선, 계수 중요도

## 2. 타깃 정의
- Bad(1): Charged Off, Default, Late (31-120 days), Late (16-30 days), In Grace Period, Does not meet the credit policy. Status: Charged Off
- Good(0): Fully Paid, Does not meet the credit policy. Status: Fully Paid

## 3. 주요 변수
loan_amnt, term, int_rate, installment, grade, sub_grade, emp_length, home_ownership,
annual_inc, verification_status, purpose, addr_state, dti, delinq_2yrs, fico_range_low,
fico_range_high, inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util, total_acc, application_type

## 4. RejectStats 점수화
- 라벨이 없는 거절데이터에 대해 학습된 모델로 PD 추정
- cut-off 슬라이더로 승인율/위험 조정
- 결과 CSV 다운로드 제공

## 5. 배포 링크
- Streamlit: (배포 후 URL 기입)
