import io
import sys
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

def _init_session_state():
    defaults = {
        "raw_df": None,
        "work_df": None, 
        "target": None,
        "positive_class": None,
        "features": [],
        "num_cols": [],
        "cat_cols": [],
        "preprocess": None,  # ColumnTransformer
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "model": None,
        "y_proba_test": None,
        "random_state": 42,
        "test_size": 0.2,
        "num_impute": "median",
        "cat_impute": "most_frequent",
        "use_scaler": True,
        "class_weight_balanced": False,
        "solver": "liblinear",
        "C": 1.0,
        "max_iter": 1000,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def detect_column_types(df: pd.DataFrame, target: str = None) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in num_cols:
        num_cols = [c for c in num_cols if c != target]
    cat_cols = [c for c in df.columns if c not in num_cols and c != target]
    return num_cols, cat_cols


def build_onehot(**kwargs):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)


def build_preprocess(num_cols: List[str], cat_cols: List[str], num_impute: str, cat_impute: str, use_scaler: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy=num_impute))]
    if use_scaler and len(num_cols) > 0:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(steps=num_steps)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=("most_frequent" if cat_impute == "most_frequent" else "constant"), fill_value="missing")),
            ("onehot", build_onehot()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return preprocess

def binarize_target(y: pd.Series, positive_class) -> pd.Series:
    return (y == positive_class).astype(int)

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, figsize=(4.0, 3.0), dpi=120) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_confusion(
    cm: np.ndarray,
    labels: List[str] = ["0", "1"],
    normalize: bool = True,
    show: str = "both",
    cmap: str = "Blues",
    figsize=(4.0, 3.0),
    dpi=140,
) -> plt.Figure:
    """Confusion matrix with better colors/labels; no f-strings to avoid unterminated issues."""
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    Z = cm_norm if normalize else cm.astype(float)
    vmax = 1.0 if normalize else (Z.max() if Z.size else 1.0)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    im = ax.imshow(Z, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # thin grid
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if show == "both":
                denom = row_sum[i, 0] if row_sum[i, 0] > 0 else 1
                pct = (cm[i, j] / denom) if denom else 0.0
                text = "{:,}\n({:.1f}%)".format(int(cm[i, j]), pct * 100.0)
            elif show == "count":
                text = "{:,}".format(int(cm[i, j]))
            else:  # "percent"
                denom = row_sum[i, 0] if row_sum[i, 0] > 0 else 1
                pct = (cm[i, j] / denom) if denom else 0.0
                text = "{:.1f}%".format(pct * 100.0)

            val = Z[i, j]
            color = "white" if val > (vmax * 0.5) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    return fig

def render_eval_tab():
    st.header("성능평가 및 시각화")
    if st.session_state.model is None or st.session_state.y_proba_test is None:
        st.warning("이전탭에서 모델을 학습하세요.")
        return

    st.subheader("빠른 재학습 (옵션)")
    newC = st.number_input(
        "C (규제 강도의 역수)",
        min_value=1e-4,
        max_value=1e4,
        value=float(st.session_state.C),
        step=0.1,
        format="%f",
        key="C_eval",
    )
    if st.button("이 값으로 재학습", key="retrain_eval"):
        class_weight = "balanced" if st.session_state.class_weight_balanced else None
        clf2 = LogisticRegression(
            C=float(newC),
            solver=st.session_state.solver,
            max_iter=st.session_state.max_iter,
            class_weight=class_weight,
        )
        model2 = Pipeline(steps=[("preprocess", st.session_state.preprocess), ("clf", clf2)])
        model2.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model2
        st.session_state.y_proba_test = model2.predict_proba(st.session_state.X_test)[:, 1]
        st.session_state.C = float(newC)
        st.success("재학습 완료")

    y_test = np.asarray(st.session_state.y_test)
    y_score = np.asarray(st.session_state.y_proba_test)

    threshold = st.slider(
        "예측 임계값 (Positive 판단 기준)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="threshold_eval",
    )
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_score)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1-Score", f"{f1:.4f}")
    c5.metric("ROC AUC", f"{auc:.4f}")

    # Plot size + layout
    size_choice = st.select_slider("플롯 크기", options=["S", "M", "L"], value="S", key="plot_size")
    size_map = {"S": (4.0, 3.0), "M": (5.2, 4.0), "L": (6.4, 4.8)}
    _figsize = size_map.get(size_choice, (4.0, 3.0))

    # Confusion Matrix display options
    cm_cmap = st.selectbox(
        "Colormap",
        ["Blues", "Greens", "Purples", "Oranges", "Reds", "viridis", "magma", "plasma", "cividis"],
        index=0,
        key="cm_cmap",
    )
    cm_normalize = st.checkbox("Normalize (row-wise %)", value=True, key="cm_norm")
    cm_show = st.selectbox("표시 값", ["both", "count", "percent"], index=0, key="cm_show")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("ROC Curve")
        fig1 = plot_roc(y_test, y_score, figsize=_figsize, dpi=140)
        st.pyplot(fig1, use_container_width=False)

    with colB:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig2 = plot_confusion(
            cm,
            labels=["0", "1"],
            normalize=cm_normalize,
            show=cm_show,
            cmap=cm_cmap,
            figsize=_figsize,
            dpi=140,
        )
        st.pyplot(fig2, use_container_width=False)

    st.subheader("예측 결과 다운로드")
    out = pd.DataFrame({
        "y_true": y_test,
        "y_score": y_score,
        "y_pred": y_pred,
    })
    csv = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSV 다운로드", data=csv, file_name="predictions.csv", mime="text/csv")

_init_session_state()
st.set_page_config(page_title="Binary Logit App", layout="wide")
st.title("Lending Club Logit 실습")

TAB1, TAB2, TAB3, TAB4 = st.tabs([
    "① 업로드/탐색",
    "② 변수 선택/전처리",
    "③ 모델 학습",
    "④ 성능평가/시각화",
])

with TAB1:
    st.header("데이터 업로드 및 탐색")
    file = st.file_uploader("CSV 파일 업로드", type=["csv"], accept_multiple_files=False)
    sample_n = st.number_input("미리보기 행 수", min_value=5, max_value=1000, value=20, step=5)

    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp949")

        st.session_state.raw_df = df.copy()
        st.session_state.work_df = df.copy()

        st.subheader("기본 정보")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("행 수", f"{len(df):,}")
        with c2:
            st.metric("열 수", f"{df.shape[1]:,}")
        with c3:
            st.metric("결측치 포함 열 수", df.isna().any().sum())

        st.subheader("샘플 미리보기")
        st.dataframe(df.head(int(sample_n)))

        st.subheader("변수별 결측치 개수")
        na_counts = df.isna().sum().sort_values(ascending=False)
        st.bar_chart(na_counts)

        st.subheader("수치형 변수 기술통계")
        num_desc = df.select_dtypes(include=[np.number]).describe().T
        st.dataframe(num_desc)

    else:
        st.warning("CSV를 업로드하세요.")

with TAB2:
    st.header("변수 선택 및 전처리 설정")
    if st.session_state.work_df is None:
        st.warning("이전 탭에서 먼저 CSV를 업로드하세요.")
    else:
        df = st.session_state.work_df
        cols = df.columns.tolist()

        c1, c2 = st.columns([2, 3])
        with c1:
            target = st.selectbox("Target (binary)", options=cols, index=0 if cols else None)
            st.session_state.target = target

            if target is not None:
                vc = df[target].value_counts(dropna=False)
                st.write("Target distribution:")
                st.dataframe(pd.DataFrame({"count": vc, "ratio": (vc / len(df)).round(4)}))

            unique_vals = sorted(df[target].dropna().unique().tolist()) if target else []
            if target is not None:
                if len(unique_vals) == 2:
                    pos_default = unique_vals[1]
                    positive = st.selectbox("양성(Positive) 클래스를 선택", options=unique_vals, index=1)
                    st.session_state.positive_class = positive
                else:
                    st.error("이진 분류만 지원합니다.")

        with c2:
            all_feature_candidates = [c for c in cols if c != target]
            features = st.multiselect("Select Feature(s) (X)", options=all_feature_candidates, default=all_feature_candidates)
            st.session_state.features = features

            num_cols, cat_cols = detect_column_types(df[features + [target]] if target else df, target=target)
            st.session_state.num_cols = num_cols
            st.session_state.cat_cols = cat_cols

            st.write("자동 타입 분류 결과")
            c21, c22 = st.columns(2)
            with c21:
                st.write("수치형:")
                st.code(num_cols if len(num_cols) else "(없음)")
            with c22:
                st.write("범주형:")
                st.code(cat_cols if len(cat_cols) else "(없음)")

        st.divider()
        st.subheader("결측치/스케일링 옵션")
        c3, c4, c5 = st.columns(3)
        with c3:
            num_impute = st.selectbox("수치형 결측치 대체", options=["mean", "median"], index=1)
        with c4:
            cat_impute = st.selectbox("범주형 결측치 대체", options=["most_frequent", "constant"], index=0)
        with c5:
            use_scaler = st.checkbox("수치형 StandardScaler 적용", value=True)

        st.session_state.num_impute = num_impute
        st.session_state.cat_impute = cat_impute
        st.session_state.use_scaler = use_scaler

        st.divider()
        st.subheader("데이터 분할")
        c6, c7 = st.columns(2)
        with c6:
            test_size = st.slider("테스트 비율", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        with c7:
            random_state = st.number_input("Random State", min_value=0, max_value=10_000, value=42, step=1)
        st.session_state.test_size = float(test_size)
        st.session_state.random_state = int(random_state)

        if st.button("전처리 적용 및 데이터 분할 실행", type="primary"):
            if target is None or len(st.session_state.features) == 0:
                st.error("타깃과 특징 변수를 설정하세요.")
            elif st.session_state.positive_class is None and len(df[target].dropna().unique()) == 2:
                st.error("양성 클래스를 선택하세요.")
            else:
                # 타깃 이진화
                y = df[target]
                if len(y.dropna().unique()) == 2:
                    y = binarize_target(y, st.session_state.positive_class)
                else:
                    st.stop()

                X = df[features].copy()
                num_cols, cat_cols = st.session_state.num_cols, st.session_state.cat_cols

                preprocess = build_preprocess(
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    num_impute=st.session_state.num_impute,
                    cat_impute=st.session_state.cat_impute,
                    use_scaler=st.session_state.use_scaler,
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=st.session_state.test_size, random_state=st.session_state.random_state, stratify=y
                )

                st.session_state.preprocess = preprocess
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                st.success("전처리기 생성 및 데이터 분할 완료")

        if st.session_state.X_train is not None:
            st.write("학습/테스트 크기:")
            st.code({
                "X_train": st.session_state.X_train.shape,
                "X_test": st.session_state.X_test.shape,
                "y_train": st.session_state.y_train.shape,
                "y_test": st.session_state.y_test.shape,
            })

with TAB3:
    st.header("모델 학습 (Logistic Regression)")
    if st.session_state.preprocess is None or st.session_state.X_train is None:
        st.warning("이전탭에서 전처리를 완료하세요.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            C = st.number_input(
                "C (규제 강도의 역수)",
                min_value=1e-4,
                max_value=1e4,
                value=float(st.session_state.get("C", 1.0)),
                step=0.1,
                format="%f",
            )
        with c2:
            solver = st.selectbox(
                "Solver",
                options=["liblinear", "lbfgs", "saga"],
                index=["liblinear", "lbfgs", "saga"].index(st.session_state.get("solver", "liblinear")),
            )
        allowed_map = {
            "liblinear": ["l2", "l1"],
            "lbfgs": ["l2", "none"],
            "saga": ["l2", "l1", "elasticnet", "none"],
        }
        allowed_penalties = allowed_map.get(solver, ["l2"])
        current_penalty = st.session_state.get("penalty", "l2")
        if current_penalty not in allowed_penalties:
            current_penalty = allowed_penalties[0]
        with c3:
            penalty = st.selectbox(
                "Penalty",
                options=allowed_penalties,
                index=allowed_penalties.index(current_penalty),
            )
        with c4:
            class_weight_balanced = st.checkbox(
                "class_weight='balanced' 사용",
                value=bool(st.session_state.get("class_weight_balanced", False)),
            )

        c5, c6, _, _ = st.columns(4)
        with c5:
            max_iter = st.number_input(
                "Max Iter",
                min_value=100,
                max_value=10000,
                value=int(st.session_state.get("max_iter", 1000)),
                step=100,
            )
        l1_ratio_val = None
        with c6:
            if penalty == "elasticnet":
                l1_ratio_val = st.slider(
                    "l1_ratio (ElasticNet)",
                    0.0,
                    1.0,
                    float(st.session_state.get("l1_ratio", 0.5)),
                    0.05,
                )

        st.session_state.C = float(C)
        st.session_state.solver = solver
        st.session_state.penalty = penalty
        st.session_state.max_iter = int(max_iter)
        st.session_state.class_weight_balanced = bool(class_weight_balanced)
        st.session_state.l1_ratio = float(l1_ratio_val) if l1_ratio_val is not None else st.session_state.get("l1_ratio")

        with st.expander("모델 옵션 설명 / What do these mean?", expanded=False):
            st.markdown(
                """
**모델 옵션 설명**

### 1) C (규제 강도의 역수)
- 역할: 모델 복잡도 제어. 작은 값 -> 규제 강함(단순/과적합 방지), 큰 값 -> 규제 약함(복잡/과적합 위험).

### 2) Solver (학습 알고리즘)
- 역할: 계수를 찾는 계산 방법
  - lbfgs: 빠르고 안정적. L2/none 지원. 기본 선택.
  - liblinear: 소규모 데이터. L1/L2 지원. 이진 분류에 안정적.
  - saga: 대규모/희소(원-핫) 데이터. L1/L2/ElasticNet/none 지원.
- 규제 호환성:
  - liblinear -> L1, L2
  - lbfgs     -> L2, none
  - saga      -> L1, L2, elasticnet, none

### 3) Penalty (규제 방식)
- L2: 계수를 부드럽게 줄임(안정적, 기본값).
- L1: 불필요한 계수를 0으로 만들어 변수 선택 효과.
- ElasticNet: L1+L2 혼합. l1_ratio로 비율 조절(0=L2, 1=L1).
- none: 규제 없음(권장되지 않음; 과적합 위험).

### 4) Max Iter (최대 반복)
- 역할: 최적화 반복 상한. 수렴 경고가 나오면 값 증가.
- 권장: 1000 정도로 시작.

### 5) Class Weight (클래스 가중치)
- 목적: 불균형 데이터에서 소수 클래스 중요도 확대.
- 옵션: None(기본) / 'balanced'(클래스 빈도 역수 기반 자동 가중).
- 불균형이 크면 'balanced' 사용 고려.


                """
            )

        if st.button("학습 실행", type="primary"):
            class_weight = "balanced" if st.session_state.class_weight_balanced else None
            # penalty/solver 조합에 맞춰 인자 구성
            lr_kwargs = dict(
                C=st.session_state.C,
                solver=st.session_state.solver,
                max_iter=st.session_state.max_iter,
                class_weight=class_weight,
                penalty=st.session_state.penalty,
            )
            if st.session_state.penalty == "elasticnet":
                lr_kwargs["l1_ratio"] = st.session_state.l1_ratio if st.session_state.l1_ratio is not None else 0.5

            clf = LogisticRegression(**lr_kwargs)

            model = Pipeline(steps=[
                ("preprocess", st.session_state.preprocess),
                ("clf", clf),
            ])

            model.fit(st.session_state.X_train, st.session_state.y_train)
            y_proba_test = model.predict_proba(st.session_state.X_test)[:, 1]
            st.session_state.model = model
            st.session_state.y_proba_test = y_proba_test
            st.success("모델 학습 완료")

with TAB4:
    render_eval_tab()
