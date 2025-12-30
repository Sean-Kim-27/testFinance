import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import timedelta, datetime

# ==========================================
# ★★★ 설정 (여기에 니 키를 박아둠) ★★★
# ==========================================
# 화면에는 절대 안 나옴. 너만 알고 있는 거임.
API_KEY = st.secrets["API_KEY"]

# ==========================================
# 1. 보조 함수: 기업 재무 정보 (OVERVIEW)
# ==========================================
def get_company_overview(ticker, api_key):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if "Symbol" not in data:
            return None
        return data
    except:
        return None


# ==========================================
# 2. 메인 분석 함수
# ==========================================
def get_ticker_analysis(ticker, api_key):
    # --- [A] 뉴스 데이터 ---
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}&limit=50"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        return None, None, None, f"통신 에러: {e}"

    if "feed" not in data:
        # API 키 한도 초과거나 틀렸을 때
        return None, None, None, "뉴스 데이터가 없거나 API 키 문제"

    news_list = []
    for item in data["feed"]:
        news_list.append({
            "time_published": item["time_published"],
            "title": item["title"],
            "summary": item["summary"],
            "url": item["url"],
            "sentiment_score": float(item["overall_sentiment_score"]),
            "sentiment_label": item["overall_sentiment_label"]
        })

    news_df = pd.DataFrame(news_list)

    # --- [B] 날짜 변환 ---
    news_df['datetime'] = pd.to_datetime(news_df['time_published'], format='%Y%m%dT%H%M%S')

    if news_df['datetime'].dt.tz is None:
        news_df['datetime'] = news_df['datetime'].dt.tz_localize('UTC')
    else:
        news_df['datetime'] = news_df['datetime'].dt.tz_convert('UTC')

    news_df['datetime'] = news_df['datetime'].dt.tz_convert('US/Eastern')

    def adjust_date(row):
        if row.hour >= 16:
            return (row + timedelta(days=1)).date()
        else:
            return row.date()

    news_df['date'] = news_df['datetime'].apply(adjust_date)
    news_df['date'] = pd.to_datetime(news_df['date'])

    # --- [C] 주가 데이터 ---
    start_date = news_df['date'].min() - timedelta(days=5)
    end_date = datetime.now().date() + timedelta(days=1)

    try:
        stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        return None, None, None, f"주가 다운로드 실패: {e}"

    if stock_df.empty:
        return None, None, None, "주가 데이터 없음 (티커 확인)"

    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)

    stock_df = stock_df.reset_index()
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
    news_df['date'] = pd.to_datetime(news_df['date']).dt.tz_localize(None)

    # --- [D] 현재가 및 등락률 ---
    latest_stock = stock_df.iloc[-1]
    prev_stock = stock_df.iloc[-2] if len(stock_df) > 1 else latest_stock

    current_price = float(latest_stock['Close'])
    prev_close = float(prev_stock['Close'])
    change_rate = ((current_price - prev_close) / prev_close) * 100

    stock_info = {
        "current_price": current_price,
        "change_rate": change_rate,
        "prev_close": prev_close
    }

    # --- [E] 상관관계 ---
    merged_df = pd.merge(news_df, stock_df, left_on='date', right_on='Date', how='inner')

    correlation = 0
    if not merged_df.empty:
        daily_analysis = merged_df.groupby('date').agg({
            'sentiment_score': 'mean',
            'Close': 'last',
            'Open': 'first'
        }).reset_index()

        daily_analysis['Daily_Return'] = (daily_analysis['Close'] - daily_analysis['Open']) / daily_analysis['Open']
        correlation = daily_analysis['sentiment_score'].corr(daily_analysis['Daily_Return'])

    # --- [F] 기업 개요 ---
    overview_data = get_company_overview(ticker, api_key)

    return merged_df, correlation, stock_info, overview_data


# ==========================================
# 3. 전망 생성기
# ==========================================
def generate_outlook(corr, recent_sentiment, stock_info, overview):
    outlook_msg = ""
    reasons = []

    # 1. 상관관계
    if abs(corr) > 0.3:
        if corr > 0:
            reasons.append(f"✅ **뉴스 민감도 높음**: 뉴스 좋으면 주가도 오름.")
        else:
            reasons.append(f"⚠️ **뉴스 민감도 낮음**: 뉴스와 주가가 반대로 감.")
    else:
        reasons.append(f"ℹ️ **뉴스 민감도 상관X**: 뉴스와 주가는 별개임.")

    # 2. 최근 뉴스 분위기
    if recent_sentiment > 0.15:
        reasons.append("🔥 **최근 분위기**: 뉴스 분위기 아주 좋음 (Bullish).")
        sentiment_score = 1
    elif recent_sentiment < -0.15:
        reasons.append("❄️ **최근 분위기**: 악재가 좀 있음 (Bearish).")
        sentiment_score = -1
    else:
        reasons.append("☁️ **최근 분위기**: 중립적임 (Neutral).")
        sentiment_score = 0

    # 3. 밸류에이션
    if overview:
        try:
            target_price = float(overview.get('AnalystTargetPrice', 0))
            current_price = stock_info['current_price']

            if target_price > 0:
                upside = ((target_price - current_price) / current_price) * 100
                reasons.append(f"💰 **목표 주가**: ${target_price} (상승 여력 {upside:.1f}%)")

                if upside > 20:
                    outlook_msg = "🚀 **강력 매수**"
                elif upside > 5:
                    outlook_msg = "↗️ **매수**"
                elif upside > -10:
                    outlook_msg = "⏸️ **보유**"
                else:
                    outlook_msg = "↘️ **매도 고민**"
            else:
                outlook_msg = "🤔 **판단 보류**"
        except:
            outlook_msg = "🤔 **판단 보류**"
    else:
        if sentiment_score == 1 and corr > 0.2:
            outlook_msg = "↗️ **단기 상승**"
        elif sentiment_score == -1 and corr > 0.2:
            outlook_msg = "↘️ **단기 하락**"
        else:
            outlook_msg = "⏸️ **관망**"

    return outlook_msg, reasons


# ==========================================
# 4. 스트림릿 UI (깔끔 버전)
# ==========================================
st.set_page_config(page_title="주식 분석기", layout="wide")

st.title("📈 해외주식 티커 분석기")
st.markdown("Bearish : 약세, Bullish : 강세, Neutral : 중립")

# 사이드바 없애버림

col1, col2 = st.columns([4, 1])
with col1:
    ticker_input = st.text_input("티커 입력", value="ORCL", label_visibility="collapsed", placeholder="티커 입력...")
with col2:
    st.markdown("<div style='margin-top: -5px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("분석", use_container_width=True)

if analyze_btn:
    if not ticker_input:
        st.error("티커를 넣어주세요 .")
    else:
        with st.spinner(f"'{ticker_input}' 분석 중..."):
            # 전역 변수 API_KEY 사용
            df, corr, stock_info, overview = get_ticker_analysis(ticker_input, API_KEY)

        if isinstance(overview, str):
            st.error(f"에러 발생: {overview}")
        elif df is None:
            st.error("API 키가 만료됐거나 티커가 잘못 입력되었습니다.")
        else:
            # 1. 현재가 & 등락률
            st.divider()
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    label="현재 주가",
                    value=f"${stock_info['current_price']:.2f}",
                    delta=f"{stock_info['change_rate']:.2f}%"
                )
            with m2:
                st.metric(label="뉴스 민감도", value=f"{corr:.2f}")

            with m3:
                if overview:
                    high52 = overview.get('52WeekHigh', '-')
                    st.metric(label="52주 최고가", value=f"${high52}")

            with m4:
                if overview:
                    pe = overview.get('PERatio', '-')
                    st.metric(label="PER", value=pe)

            # 2. 종합 전망
            st.divider()
            st.subheader("🤖 AI 종합 전망")

            recent_sentiment_avg = df['sentiment_score'].mean()
            outlook_title, reason_list = generate_outlook(corr, recent_sentiment_avg, stock_info, overview)

            st.success(f"### {outlook_title}")
            for reason in reason_list:
                st.markdown(f"- {reason}")

            # 3. 뉴스 리스트
            st.divider()
            st.subheader(f"📰 관련 뉴스 ({len(df)}건)")

            display_df = df[['date', 'title', 'sentiment_label', 'url', 'summary']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

            st.dataframe(
                display_df,
                column_config={
                    "date": "날짜",
                    "title": "제목",
                    "sentiment_label": "감성",
                    "url": st.column_config.LinkColumn("링크", display_text="기사보기"),
                    "summary": "요약"
                },
                hide_index=True,
                use_container_width=True
            )