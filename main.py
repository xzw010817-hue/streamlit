import streamlit as st

st.set_page_config(page_title="과제1: Streamlit 더하기 앱", page_icon="➕", layout="centered")

st.title("과제1: 숫자 더하기 & 선택된 수까지의 합 구하기 앱")
st.caption("요구사항: number_input을 사용해 A, B를 받고, selectbox로 A 또는 B를 선택해 1부터 선택된 수까지의 합을 계산한다")

# --- 입력 영역 ---
col1, col2 = st.columns(2)
with col1:
    A = st.number_input("A 입력", min_value=0, step=1, value=0, format="%d", help="정수만 입력한다")
with col2:
    B = st.number_input("B 입력", min_value=0, step=1, value=0, format="%d", help="정수만 입력한다")

st.divider()

# --- 두 수의 합 ---
sum_ab = int(A) + int(B)
st.subheader("1) A + B")
st.metric(label="A + B 결과", value=sum_ab)

st.divider()

# --- 선택박스: A 또는 B 중 선택 ---
st.subheader("2) 선택된 숫자까지의 모든 수 합 (1부터 n까지)")
choice = st.selectbox("숫자 선택", ("A", "B"), index=0)

n = int(A) if choice == "A" else int(B)

def triangular(n: int) -> int:
    """1부터 n까지의 합을 O(1) 공식으로 계산한다.
    n이 0이면 0을 반환한다.
    """
    return n * (n + 1) // 2

if n < 0:
    st.error("음수는 허용하지 않는다. 0 이상의 정수를 입력하라")
else:
    s = triangular(n)
    st.write(f"선택된 숫자: **{choice} = {n}**")
    st.success(f"1부터 {n}까지의 합: **{s}**")

st.info("참고: 본 앱은 number_input을 사용하여 정수만 받도록 구성했다. text_input을 사용할 경우 정수 변환 및 예외 처리가 필요하다")
