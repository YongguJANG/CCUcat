# CCUcat
주식공부
# 📈 주식 선택을 위한 도구 (Stock Selection Tool)

import feedparser
from datetime import datetime

def fetch_news(ticker):
    """Yahoo Finance RSS로 뉴스 헤드라인과 요약, 발행일 수집"""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    records = []
    for entry in feed.entries:
        dt = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')
        date_str = dt.strftime("%Y-%m-%d")
        records.append({
            'ticker': ticker,
            'title': entry.title,
            'summary': entry.summary,
            'datetime': dt,
            'date': date_str
        })
    return pd.DataFrame(records)

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# FinBERT 모델 로드
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)

def analyze_sentiment(sentences):
    """FinBERT로 문장 리스트의 긍정 점수(확률) 반환"""
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    outputs = finbert(**inputs)
    logits = outputs.logits.detach()
    probs = F.softmax(logits, dim=1).numpy()
    # positive 클래스는 인덱스 1
    return [prob[1] for prob in probs]

df_news = fetch_news('PLTR')  # 예시: PLTR 뉴스
df_news['sentiment'] = analyze_sentiment(df_news['title'].tolist())
daily_sent = df_news.groupby('date')['sentiment'].mean().reset_index()

import yfinance as yf

def fetch_price(ticker):
    """yfinance를 사용해 1년치 일간 주가 데이터 불러오기"""
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period="1y", interval="1d", auto_adjust=False)
    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    return hist

df_price_pltr = fetch_price('PLTR')

import backtrader as bt

class SentimentStrategy(bt.Strategy):
    params = (("threshold", 0.6),)

    def __init__(self):
        # 각 데이터 피드에 대해 sentiment 지표 추출
        self.datas_sent = {data._name: data.sentiment for data in self.datas}

    def next(self):
        for data in self.datas:
            name = data._name
            sentiment = data.sentiment[0]
            position = self.getposition(data).size
            # 매수 조건: 아직 포지션 없고 감성 >= 임계값
            if not position and sentiment >= self.p.threshold:
                self.buy(data=data)
            # 매도 조건: 포지션 보유 중이면 모두 청산
            elif position:
                self.close(data=data)

cerebro = bt.Cerebro()
cerebro.broker.set_cash(100000)  # 시작 자본
tickers = ['PLTR','BBAI','SMCY','OPEN']
for ticker in tickers:
    df = prepare_merged_dataframe(ticker)  # 날짜, Open, Close, sentiment 등을 포함
    datafeed = bt.feeds.PandasData(dataname=df, name=ticker,
                                   datetime='Date', open='Open', high='High',
                                   low='Low', close='Close', volume='Volume',
                                   params=(('sentiment', 'sentiment'),))
    cerebro.adddata(datafeed)
cerebro.addstrategy(SentimentStrategy)
cerebro.run()

import streamlit as st
import pandas as pd

st.title("주식 뉴스 감성 기반 투자 전략 대시보드")

# 데이터 불러오기 (main.py에서 생성한 결과 파일들)
sentiment_df = pd.read_csv('data/daily_sentiment.csv', parse_dates=['date'], index_col='date')
portfolio_df = pd.read_csv('data/portfolio_returns.csv', parse_dates=['date'], index_col='date')

# 감성 점수 그래프
st.header("종목별 일별 감성 점수")
st.line_chart(sentiment_df)  # 각 열: PLTR, BBAI, SMCY, OPEN의 감성 점수

# 포트폴리오 누적 수익률 그래프
st.header("포트폴리오 누적 수익률")
st.line_chart(portfolio_df['cumulative_return'])

stock_sentiment_project/
├── main.py                 # 뉴스 수집, 감성분석, 백테스트 수행 스크립트
├── dashboard/
│   └── app.py             # Streamlit 대시보드 스크립트
├── src/
│   ├── data_collection.py  # 뉴스/주가 수집 함수 모듈
│   ├── sentiment.py        # FinBERT 감성분석 모듈
│   └── backtest.py         # Backtrader 전략 모듈
├── data/
│   ├── daily_sentiment.csv  # 일별 감성 점수
│   └── portfolio_returns.csv # 포트폴리오 수익률 데이터
├── requirements.txt        # 의존 라이브러리 목록
└── README.md               # 설치/실행 안내문
