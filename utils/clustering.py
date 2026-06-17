import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processor import TextProcessor

class OpinionClustering:
    """의견 클러스터링 클래스"""
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
        self.kmeans = None
        self.processor = TextProcessor()
        self.tfidf_matrix = None
        self.cluster_labels = None
    
    def fit(self, texts):
        """클러스터 학습"""
        # TF-IDF 벡터화
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # 적절한 클러스터 개수 결정 (최소 1, 최대 입력된 텍스트 수의 절반)
        n_texts = len(texts)
        actual_clusters = min(self.n_clusters, max(1, n_texts // 2))
        
        # K-means 클러스터링
        self.kmeans = KMeans(n_clusters=actual_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.tfidf_matrix)
        
        return self.cluster_labels
    
    def get_clusters(self, texts):
        """클러스터 결과 반환"""
        if self.cluster_labels is None:
            return {}
        
        clusters = {}
        for idx, label in enumerate(self.cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[idx])
        
        return clusters
    
    def get_cluster_keywords(self, n_keywords=5):
        """각 클러스터의 주요 키워드 추출"""
        if self.kmeans is None:
            return {}
        
        keywords_dict = {}
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        for i in range(self.kmeans.n_clusters):
            # 각 클러스터 중심에서 가장 중요한 피처 추출
            center = self.kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-n_keywords:][::-1]
            keywords = feature_names[top_indices].tolist()
            keywords_dict[i] = keywords
        
        return keywords_dict
    
    def cluster_by_similarity(self, texts, similarity_threshold=0.6):
        """유사도 기반 클러스터링"""
        if len(texts) <= 1:
            return {0: texts}
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        for i in range(len(texts)):
            if i in assigned:
                continue
            
            cluster = [texts[i]]
            assigned.add(i)
            
            # 유사도 높은 텍스트들을 같은 클러스터로
            for j in range(i + 1, len(texts)):
                if j not in assigned and similarity_matrix[i][j] >= similarity_threshold:
                    cluster.append(texts[j])
                    assigned.add(j)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        return clusters
