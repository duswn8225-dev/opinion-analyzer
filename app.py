import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json
from datetime import datetime

# 로컬 모듈 임포트
from utils.text_processor import TextProcessor
from utils.clustering import OpinionClustering
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.summarizer import Summarizer

# 페이지 설정
st.set_page_config(
    page_title="주관식 의견 분석 도구",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 추가
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# 제목
st.title("📊 주관식 의견 분석 도구")
st.markdown("**CSV/Excel 업로드 → 웹에서 바로 분석**")

# ==================== 사이드바 ====================
with st.sidebar:
    st.header("⚙️ 설정")
    
    analysis_type = st.radio(
        "분석 방식",
        ["간단 분석 (전체 의견)", "상세 분석 (필터링)"],
        help="간단 분석: 전체 의견을 종합적으로 분석\n상세 분석: 과정/강사/교과목/문항별로 세부 분석"
    )
    
    n_clusters = st.slider(
        "의견 그룹 수",
        min_value=2,
        max_value=10,
        value=5,
        help="비슷한 의견들을 몇 개의 그룹으로 나눌지 결정합니다"
    )
    
    min_group_size = st.slider(
        "최소 그룹 크기",
        min_value=1,
        max_value=10,
        value=2,
        help="최소 몇 개 이상의 의견이 있어야 그룹으로 표시될지 결정합니다"
    )
    
    st.divider()
    st.markdown("### 📋 파일 양식")
    st.info("""
    **필수 컬럼:**
    - `id`: 응답자 ID
    - `question`: 질문
    - `answer`: 주관식 답변
    
    **선택 컬럼** (상세 분석 시):
    - `course`: 과정명
    - `lecture_date`: 강의일자
    - `instructor`: 강사명
    - `subject`: 교과목명
    - `question_num`: 문항번호
    """)

# ==================== 메인 영역 ====================
st.header("1️⃣ 파일 업로드")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일 선택",
        type=['csv', 'xlsx', 'xls'],
        help="준비된 CSV/Excel 파일을 선택하세요"
    )

with col2:
    if st.button("📥 샘플 데이터 로드", use_container_width=True):
        uploaded_file = "sample_data.csv"

# ==================== 파일 처리 ====================
if uploaded_file is not None:
    try:
        # 파일 읽기
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
        else:
            if uploaded_file.name.endswith('csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        
        # 데이터 검증
        required_cols = ['id', 'question', 'answer']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ 필수 컬럼이 없습니다: {', '.join(missing_cols)}")
            st.stop()
        
        # 데이터 정보 표시
        st.success(f"✅ 파일 로드 완료! ({len(df)}개 행)")
        
        with st.expander("📋 데이터 미리보기"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # ==================== 필터링 ====================
        st.header("2️⃣ 분석 필터 설정")
        
        filter_cols = []
        filter_values = {}
        
        if analysis_type == "상세 분석 (필터링)":
            # 선택 가능한 필터 컬럼
            optional_cols = ['course', 'lecture_date', 'instructor', 'subject', 'question_num']
            available_filters = [col for col in optional_cols if col in df.columns]
            
            if available_filters:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'course' in df.columns:
                        courses = df['course'].unique().tolist()
                        selected_courses = st.multiselect(
                            "🎓 과정 선택",
                            courses,
                            default=courses
                        )
                        if selected_courses:
                            filter_values['course'] = selected_courses
                
                with col2:
                    if 'instructor' in df.columns:
                        instructors = df['instructor'].unique().tolist()
                        selected_instructors = st.multiselect(
                            "👨‍🏫 강사명 선택",
                            instructors,
                            default=instructors
                        )
                        if selected_instructors:
                            filter_values['instructor'] = selected_instructors
                
                with col3:
                    if 'subject' in df.columns:
                        subjects = df['subject'].unique().tolist()
                        selected_subjects = st.multiselect(
                            "📚 교과목 선택",
                            subjects,
                            default=subjects
                        )
                        if selected_subjects:
                            filter_values['subject'] = selected_subjects
                
                col4, col5 = st.columns(2)
                
                with col4:
                    if 'question_num' in df.columns:
                        questions = sorted(df['question_num'].unique().tolist())
                        selected_questions = st.multiselect(
                            "❓ 문항 선택",
                            questions,
                            default=questions
                        )
                        if selected_questions:
                            filter_values['question_num'] = selected_questions
                
                with col5:
                    if 'lecture_date' in df.columns:
                        st.write("")  # 여백
        
        # 필터 적용
        filtered_df = df.copy()
        for col, values in filter_values.items():
            filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        st.info(f"분석 대상: **{len(filtered_df)}개 의견** (전체: {len(df)}개)")
        
        if len(filtered_df) == 0:
            st.error("❌ 필터 조건에 맞는 데이터가 없습니다")
            st.stop()
        
        # ==================== 분석 시작 ====================
        if st.button("🚀 분석 시작", use_container_width=True, type="primary"):
            with st.spinner("분석 진행 중... 잠시만 기다려주세요"):
                # 초기화
                text_processor = TextProcessor()
                sentiment_analyzer = SentimentAnalyzer()
                summarizer = Summarizer()
                
                answers = filtered_df['answer'].tolist()
                
                # 감정분석
                sentiments = sentiment_analyzer.analyze_batch(answers)
                filtered_df['sentiment'] = sentiments
                
                # 의견 클러스터링
                clustering = OpinionClustering(n_clusters=n_clusters)
                cluster_labels = clustering.fit(answers)
                filtered_df['cluster'] = cluster_labels
                
                # 결과 저장
                st.session_state.analysis_result = {
                    'df': filtered_df,
                    'clustering': clustering,
                    'sentiment_analyzer': sentiment_analyzer,
                    'summarizer': summarizer,
                    'answers': answers
                }
                
                st.success("✅ 분석 완료!")
        
        # ==================== 결과 표시 ====================
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            filtered_df = result['df']
            clustering = result['clustering']
            sentiment_analyzer = result['sentiment_analyzer']
            summarizer = result['summarizer']
            
            st.header("3️⃣ 분석 결과")
            
            # ====== 탭 분류 ======
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 감정 분석",
                "🔄 의견 그룹핑",
                "⭐ 상세 분석",
                "📥 결과 내보내기"
            ])
            
            # ====== 탭1: 감정 분석 ======
            with tab1:
                st.subheader("감정 분석 결과")
                
                distribution, distribution_pct = sentiment_analyzer.get_sentiment_distribution(result['answers'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    positive_count = distribution.get('positive', 0)
                    st.metric("😊 긍정", positive_count, f"{distribution_pct['positive']:.1f}%")
                
                with col2:
                    neutral_count = distribution.get('neutral', 0)
                    st.metric("😐 중립", neutral_count, f"{distribution_pct['neutral']:.1f}%")
                
                with col3:
                    negative_count = distribution.get('negative', 0)
                    st.metric("😞 부정", negative_count, f"{distribution_pct['negative']:.1f}%")
                
                # 감정 분포 차트
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['😊 긍정', '😐 중립', '😞 부정'],
                        values=[
                            distribution_pct['positive'],
                            distribution_pct['neutral'],
                            distribution_pct['negative']
                        ],
                        marker=dict(colors=['#4CAF50', '#FFC107', '#F44336']),
                        textinfo='label+percent+value',
                        hovertemplate='<b>%{label}</b><br>%{value}개 (%{percent})<extra></extra>'
                    )
                ])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # 감정별 의견 샘플
                st.subheader("감정별 의견 샘플")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 😊 긍정 의견")
                    positive_samples = filtered_df[filtered_df['sentiment'] == 'positive']['answer'].head(3).tolist()
                    for sample in positive_samples:
                        st.write(f"✓ {sample}")
                
                with col2:
                    st.markdown("### 😐 중립 의견")
                    neutral_samples = filtered_df[filtered_df['sentiment'] == 'neutral']['answer'].head(3).tolist()
                    for sample in neutral_samples:
                        st.write(f"- {sample}")
                
                with col3:
                    st.markdown("### 😞 부정 의견")
                    negative_samples = filtered_df[filtered_df['sentiment'] == 'negative']['answer'].head(3).tolist()
                    for sample in negative_samples:
                        st.write(f"✗ {sample}")
            
            # ====== 탭2: 의견 그룹핑 ======
            with tab2:
                st.subheader("의견 그룹핑 결과")
                
                clusters = clustering.get_clusters(result['answers'])
                keywords_dict = clustering.get_cluster_keywords(n_keywords=5)
                
                # 클러스터 크기 분석
                cluster_sizes = [(cluster_id, len(opinions)) for cluster_id, opinions in clusters.items()]
                cluster_sizes.sort(key=lambda x: x[1], reverse=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"그룹 {cid}" for cid, _ in cluster_sizes],
                        y=[size for _, size in cluster_sizes],
                        marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(cluster_sizes)])
                    )
                ])
                fig.update_layout(
                    title="각 그룹의 의견 개수",
                    xaxis_title="의견 그룹",
                    yaxis_title="개수",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 그룹 정보
                st.subheader("그룹별 상세 정보")
                
                for cluster_id, size in cluster_sizes:
                    if size >= min_group_size:
                        with st.expander(f"📌 그룹 {cluster_id} (의견 {size}개)"):
                            # 주요 키워드
                            keywords = keywords_dict.get(cluster_id, [])
                            st.markdown(f"**🔑 주요 키워드:** {', '.join(keywords)}")
                            
                            # 대표 의견
                            opinions = clusters[cluster_id]
                            representative = summarizer.get_representative_opinion(opinions)
                            st.markdown(f"**⭐ 대표 의견:** {representative}")
                            
                            # 그룹 요약
                            summary = summarizer.summarize_cluster(opinions)
                            st.markdown(f"**📝 요약:** {summary}")
                            
                            # 감정 분포
                            cluster_sentiments = filtered_df[filtered_df['cluster'] == cluster_id]['sentiment'].value_counts()
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                pos = cluster_sentiments.get('positive', 0)
                                st.metric("😊 긍정", pos)
                            with col2:
                                neu = cluster_sentiments.get('neutral', 0)
                                st.metric("😐 중립", neu)
                            with col3:
                                neg = cluster_sentiments.get('negative', 0)
                                st.metric("😞 부정", neg)
                            
                            # 의견 목록
                            st.markdown("**💬 전체 의견:**")
                            for i, opinion in enumerate(opinions[:5], 1):
                                st.write(f"{i}. {opinion}")
                            
                            if len(opinions) > 5:
                                st.write(f"... 및 {len(opinions) - 5}개 더")
            
            # ====== 탭3: 상세 분석 ======
            with tab3:
                st.subheader("상세 분석")
                
                analysis_option = st.radio(
                    "분석 대상 선택",
                    ["전체", "과정별", "강사별", "교과목별", "문항별"],
                    horizontal=True
                )
                
                if analysis_option == "전체":
                    st.markdown("### 전체 의견 분석")
                    st.write(f"**총 의견 수:** {len(filtered_df)}개")
                    
                    sentiment_dist = filtered_df['sentiment'].value_counts()
                    cluster_dist = filtered_df['cluster'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**감정별 분포:**")
                        st.bar_chart(sentiment_dist)
                    
                    with col2:
                        st.markdown("**그룹별 분포:**")
                        st.bar_chart(cluster_dist)
                
                elif analysis_option == "과정별" and 'course' in filtered_df.columns:
                    st.markdown("### 과정별 분석")
                    
                    courses = filtered_df['course'].unique()
                    selected_course = st.selectbox("분석할 과정 선택", courses)
                    
                    course_df = filtered_df[filtered_df['course'] == selected_course]
                    st.write(f"**과정명:** {selected_course}")
                    st.write(f"**의견 수:** {len(course_df)}개")
                    
                    sentiment_dist = course_df['sentiment'].value_counts()
                    st.bar_chart(sentiment_dist)
                
                elif analysis_option == "강사별" and 'instructor' in filtered_df.columns:
                    st.markdown("### 강사별 분석")
                    
                    instructors = filtered_df['instructor'].unique()
                    selected_instructor = st.selectbox("분석할 강사 선택", instructors)
                    
                    inst_df = filtered_df[filtered_df['instructor'] == selected_instructor]
                    st.write(f"**강사명:** {selected_instructor}")
                    st.write(f"**의견 수:** {len(inst_df)}개")
                    
                    sentiment_dist = inst_df['sentiment'].value_counts()
                    st.bar_chart(sentiment_dist)
                
                elif analysis_option == "교과목별" and 'subject' in filtered_df.columns:
                    st.markdown("### 교과목별 분석")
                    
                    subjects = filtered_df['subject'].unique()
                    selected_subject = st.selectbox("분석할 교과목 선택", subjects)
                    
                    subj_df = filtered_df[filtered_df['subject'] == selected_subject]
                    st.write(f"**교과목명:** {selected_subject}")
                    st.write(f"**의견 수:** {len(subj_df)}개")
                    
                    sentiment_dist = subj_df['sentiment'].value_counts()
                    st.bar_chart(sentiment_dist)
                
                elif analysis_option == "문항별" and 'question_num' in filtered_df.columns:
                    st.markdown("### 문항별 분석")
                    
                    questions = sorted(filtered_df['question_num'].unique())
                    selected_question = st.selectbox("분석할 문항 선택", questions)
                    
                    q_df = filtered_df[filtered_df['question_num'] == selected_question]
                    st.write(f"**문항번호:** {selected_question}")
                    st.write(f"**의견 수:** {len(q_df)}개")
                    
                    sentiment_dist = q_df['sentiment'].value_counts()
                    st.bar_chart(sentiment_dist)
            
            # ====== 탭4: 결과 내보내기 ======
            with tab4:
                st.subheader("📥 결과 내보내기")
                
                export_format = st.radio(
                    "저장 형식 선택",
                    ["Excel", "JSON", "CSV"],
                    horizontal=True
                )
                
                if export_format == "Excel":
                    # Excel 생성
                    output = BytesIO()
                    
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # 시트 1: 원본 데이터 + 분석 결과
                        filtered_df.to_excel(writer, sheet_name='분석결과', index=False)
                        
                        # 시트 2: 그룹별 요약
                        summary_data = []
                        for cluster_id in sorted(clusters.keys()):
                            opinions = clusters[cluster_id]
                            if len(opinions) >= min_group_size:
                                summary_data.append({
                                    '그룹번호': cluster_id,
                                    '의견수': len(opinions),
                                    '주요키워드': ', '.join(keywords_dict.get(cluster_id, [])),
                                    '대표의견': summarizer.get_representative_opinion(opinions),
                                    '요약': summarizer.summarize_cluster(opinions)
                                })
                        
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='그룹요약', index=False)
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="📊 Excel로 다운로드",
                        data=output.getvalue(),
                        file_name=f"의견분석결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                elif export_format == "JSON":
                    # JSON 생성
                    json_data = {
                        'metadata': {
                            '분석일시': datetime.now().isoformat(),
                            '총의견수': len(filtered_df),
                            '그룹수': len(clusters)
                        },
                        '감정분석': {
                            '긍정': int(distribution.get('positive', 0)),
                            '중립': int(distribution.get('neutral', 0)),
                            '부정': int(distribution.get('negative', 0))
                        },
                        '그룹': []
                    }
                    
                    for cluster_id in sorted(clusters.keys()):
                        opinions = clusters[cluster_id]
                        if len(opinions) >= min_group_size:
                            json_data['그룹'].append({
                                '그룹번호': int(cluster_id),
                                '의견수': len(opinions),
                                '주요키워드': keywords_dict.get(cluster_id, []),
                                '대표의견': summarizer.get_representative_opinion(opinions),
                                '요약': summarizer.summarize_cluster(opinions),
                                '의견목록': opinions[:5]
                            })
                    
                    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                    
                    st.download_button(
                        label="📄 JSON으로 다운로드",
                        data=json_str,
                        file_name=f"의견분석결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                else:  # CSV
                    # CSV 생성
                    csv_data = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        label="📋 CSV로 다운로드",
                        data=csv_data,
                        file_name=f"의견분석결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.divider()
                st.markdown("### 📊 분석 요약")
                
                st.markdown(f"""
                **분석 통계:**
                - 📝 총 분석 의견: {len(filtered_df)}개
                - 📌 의견 그룹: {len([c for c in clusters if len(clusters[c]) >= min_group_size])}개
                - 😊 긍정: {distribution.get('positive', 0)}개 ({distribution_pct['positive']:.1f}%)
                - 😐 중립: {distribution.get('neutral', 0)}개 ({distribution_pct['neutral']:.1f}%)
                - 😞 부정: {distribution.get('negative', 0)}개 ({distribution_pct['negative']:.1f}%)
                
                **분석 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
    
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.write("파일 형식을 확인해주세요.")

else:
    # 초기 상태
    st.info("📁 위에서 파일을 업로드하거나 샘플 데이터를 로드해주세요.")
    
    st.markdown("""
    ### 🎯 사용 방법
    
    1. **파일 준비**
       - CSV 또는 Excel 파일 준비
       - 필수 컬럼: `id`, `question`, `answer`
    
    2. **파일 업로드**
       - 파일 선택 후 업로드
    
    3. **필터 설정** (선택사항)
       - 특정 과정, 강사, 교과목, 문항별로 분석 가능
    
    4. **분석 시작**
       - "분석 시작" 버튼 클릭
    
    5. **결과 확인**
       - 감정 분석, 의견 그룹핑, 상세 분석 확인
    
    6. **결과 내보내기**
       - Excel, JSON, CSV 형식으로 저장
    
    ### 📋 파일 양식 예시
    
    | id | question | answer |
    |----|----------|--------|
    | 1 | 강의가 도움이 되었나요? | 매우 도움이 되었습니다 |
    | 2 | 강의가 도움이 되었나요? | 실무 예제가 많아서 좋았어요 |
    | 3 | 강의가 도움이 되었나요? | 진행이 너무 빨라요 |
    """)
