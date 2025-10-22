# Streamlit — 지능형 신용평가(Logit, 15k/15k)

여러 분기의 LendingClub 승인 데이터(LoanStats)를 병합하여 **Good 15,000 + Bad 15,000** 균형 샘플로
로지스틱 회귀(Logit) 모델을 학습하고, AUROC/AUPRC/혼동행렬/계수중요도/Reject 점수화 기능을 제공합니다.

## 빠른 시작
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포
1. 이 레포 루트에 `app.py`와 `requirements.txt`가 있어야 합니다.
2. Streamlit Cloud → New app → 이 레포 선택 → Main file: `app.py` → Deploy
3. 첫 배포에서 패키지 설치에 시간이 걸릴 수 있습니다.
   - 만약 `ModuleNotFoundError: sklearn` 발생 시, 이 레포는 `requirements.txt`로 해결되며,
     추가로 `app.py` 최상단의 **핫픽스**가 자동으로 설치를 시도합니다.

## 사용법
- **데이터 업로드**: `LoanStats*.csv` 또는 압축 `*.zip` 내부 CSV (복수 파일 병합 지원)
- **학습/검증**: Good/Bad 각 15,000 샘플(부족시 가능한 최대치) → 8:2 분할 → 로지스틱 회귀
- **지표**: AUROC, AUPRC, Confusion Matrix, PR/ROC 곡선, 계수 중요도
- **RejectStats 점수화**: 거절신청 CSV/ZIP 업로드 → PD 예측 → 승인여부 플래그 및 CSV 다운로드

## 트러블슈팅
- `requirements.txt`가 **루트**에 있는지 확인
- Python >= 3.9 권장 (Cloud에서 자동)
- 큰 CSV는 레포에 커밋하지 말고, 실행 시 업로드하세요
