import streamlit as st
import sys
import os

st.title("🔍 환경 진단 도구")
st.write("각 항목을 확인하여 문제를 진단합니다.")

st.divider()

# Python 버전 확인
st.write("### 1️⃣ Python 버전")
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
st.success(f"✅ Python {python_version}")

# 패키지 확인
st.write("### 2️⃣ 필수 패키지 확인")

packages_to_check = {
    'streamlit': 'Streamlit (웹 UI)',
    'pandas': 'Pandas (데이터처리)',
    'numpy': 'NumPy (수치계산)',
    'sklearn': 'Scikit-learn (머신러닝)',
    'plotly': 'Plotly (시각화)',
    'konlpy': 'KoNLPy (한국어처리)',
    'nltk': 'NLTK (자연어처리)',
}

for package, description in packages_to_check.items():
    try:
        __import__(package)
        st.write(f"✅ {description}")
    except ImportError:
        st.error(f"❌ {description} - 설치 필요")
        st.code(f"pip install {package}", language="bash")

st.divider()

# 로컬 모듈 확인
st.write("### 3️⃣ 로컬 모듈 확인")

local_modules = [
    'utils.text_processor',
    'utils.clustering',
    'utils.sentiment_analyzer',
    'utils.summarizer',
]

for module in local_modules:
    try:
        __import__(module)
        st.write(f"✅ {module}")
    except ImportError as e:
        st.error(f"❌ {module} - {str(e)}")

st.divider()

st.info("""
### ✅ 모든 항목이 초록색이면 문제 없습니다!

❌ 빨간색 항목이 있으면 위의 해결명령을 실행하세요.
""")
