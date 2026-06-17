from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Summarizer:
    """요약 및 대표문장 추출 클래스"""
    
    def __init__(self):
        pass
    
    def extract_top_sentences(self, texts, n_sentences=3):
        """상위 문장 추출"""
        if not texts or len(texts) == 0:
            return []
        
        if len(texts) <= n_sentences:
            return texts
        
        try:
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(max_features=50)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 각 문장의 TF-IDF 점수 평균 계산
            scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
            
            # 상위 n개 인덱스 추출
            top_indices = np.argsort(scores)[-n_sentences:][::-1]
            top_sentences = [texts[i] for i in sorted(top_indices)]
            
            return top_sentences
        except:
            # 벡터화 실패시 첫 n개 반환
            return texts[:n_sentences]
    
    def get_representative_opinion(self, texts):
        """대표 의견 추출"""
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # 가장 길고 내용이 풍부한 문장을 대표의견으로 선택
        # 길이와 단어 개수로 판단
        scored_texts = []
        for text in texts:
            words = text.split()
            score = len(text) + len(words) * 0.5
            scored_texts.append((score, text))
        
        scored_texts.sort(reverse=True)
        return scored_texts[0][1]
    
    def summarize_cluster(self, texts):
        """클러스터 요약"""
        if not texts:
            return ""
        
        # 모든 문장에서 명사 추출
        words = []
        for text in texts:
            words.extend(text.split())
        
        # 상위 단어들로 요약 생성
        from collections import Counter
        word_freq = Counter(words)
        top_words = [word for word, _ in word_freq.most_common(5)]
        
        # 요약 문장 생성
        summary = f"주요 내용: {', '.join(top_words)}"
        return summary
    
    def extract_keywords(self, texts, n_keywords=5):
        """주요 키워드 추출"""
        try:
            # 모든 문장 합치기
            combined_text = ' '.join(texts)
            
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(max_features=n_keywords)
            vectorizer.fit_transform([combined_text])
            
            keywords = vectorizer.get_feature_names_out().tolist()
            return keywords
        except:
            from collections import Counter
            words = []
            for text in texts:
                words.extend(text.split())
            word_freq = Counter(words)
            keywords = [word for word, _ in word_freq.most_common(n_keywords)]
            return keywords
