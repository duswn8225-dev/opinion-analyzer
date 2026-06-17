import pandas as pd
from collections import Counter

class SentimentAnalyzer:
    """감정분석 클래스"""
    
    def __init__(self):
        # 긍정 단어
        self.positive_words = {
            '좋다', '좋습니다', '좋았다', '좋았습니다',
            '훌륭하다', '훌륭합니다', '훌륭했다',
            '만족', '만족합니다', '만족했습니다',
            '유용하다', '유용합니다', '유용했다',
            '실용적', '실용적이다', '실용적입니다',
            '도움', '도움이', '도움이 되었다', '도움이 되었습니다',
            '탁월하다', '탁월합니다',
            '훌륭합니다', '훌륭해요', '좋아요',
            '감사', '감사합니다', '고마워요', '고마웠어요',
            '편리하다', '편리합니다', '편리해요',
            '쉽다', '쉽습니다', '쉬워요',
            '명확하다', '명확합니다', '명확해요',
            '이해하기', '이해가', '이해하기 쉬웠다',
            '적절하다', '적절합니다', '적절해요',
            '빠르다', '빠릅니다', '빨라요',
            '친절하다', '친절합니다', '친절해요',
            '적극적', '열정적', '솔직한', '정직한',
            '매우', '정말', '아주', '매우 도움',
            '효과적', '효율적', '생산적', '긍정적',
            '완벽하다', '완벽합니다', '완벽해요',
            '훌륭해', '최고', '대박', '너무 좋아',
        }
        
        # 부정 단어
        self.negative_words = {
            '나쁘다', '나쁩니다', '나빴다', '나빴습니다',
            '별로', '별로다', '별로입니다',
            '어렵다', '어렵습니다', '어려워요',
            '부족하다', '부족합니다', '부족해요',
            '아쉽다', '아쉽습니다', '아쉬워요',
            '느리다', '느립니다', '느려요',
            '불편하다', '불편합니다', '불편해요',
            '복잡하다', '복잡합니다', '복잡해요',
            '이해가 안', '이해가 어렵다',
            '진행이 빠르다', '진행이 빨라요',
            '시간이 부족하다', '시간이 부족해요',
            '설명이 부족하다', '설명이 부족해요',
            '질문할 시간', '질문 시간',
            '예제가 부족하다', '예제가 부족해요',
            '실습이 부족하다', '실습이 부족해요',
            '지루하다', '지루합니다', '지루해요',
            '졸음이', '집중이 어렵다',
            '비추천', '추천 안', '별로입니다',
            '최악', '끔찍하다', '형편없다',
            '실망', '실망했다', '실망했습니다',
            '너무 어렵다', '너무 어려워요',
        }
        
        # 중립 단어 (해석 불필요)
        self.neutral_words = {
            '괜찮다', '괜찮습니다', '괜찮아요',
            '그럭저럭', '적당하다', '적당합니다',
            '보통', '평범하다', '평범합니다',
            '무난하다', '무난합니다',
        }
    
    def analyze_sentiment(self, text):
        """텍스트의 감정 분석"""
        if not isinstance(text, str):
            return 'neutral'
        
        text_lower = text.lower()
        
        # 각 감정 점수 계산
        positive_score = sum(1 for word in self.positive_words if word in text_lower)
        negative_score = sum(1 for word in self.negative_words if word in text_lower)
        
        # 감정 결정
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_batch(self, texts):
        """배치 감정분석"""
        sentiments = [self.analyze_sentiment(text) for text in texts]
        return sentiments
    
    def get_sentiment_distribution(self, texts):
        """감정 분포 계산"""
        sentiments = self.analyze_batch(texts)
        distribution = Counter(sentiments)
        
        total = len(sentiments)
        distribution_pct = {
            'positive': (distribution.get('positive', 0) / total * 100) if total > 0 else 0,
            'negative': (distribution.get('negative', 0) / total * 100) if total > 0 else 0,
            'neutral': (distribution.get('neutral', 0) / total * 100) if total > 0 else 0,
        }
        
        return distribution, distribution_pct
