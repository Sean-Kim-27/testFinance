import streamlit as st
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import feedparser
import google.generativeai as genai
from datetime import datetime

# ==========================================
# ★ 비밀키 로드
# ==========================================
try:
    API_KEYS = st.secrets["api_keys"]
    if isinstance(API_KEYS, str):
        API_KEYS = [API_KEYS]
except FileNotFoundError:
    st.error("야! .streamlit/secrets.toml 파일이 없잖아!")
    st.stop()
except KeyError:
    st.error("secrets.toml에 'api_keys'가 없다. 오타 냈냐?")
    st.stop()


# ==========================================
# 1. 보조 함수
# ==========================================
def analyze_sentiment(text):
    if not text: return 0
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def get_sentiment_label(score):
    if score > 0.1:
        return "Bullish (긍정)"
    elif score < -0.1:
        return "Bearish (부정)"
    else:
        return "Neutral (중립)"


def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        return not stock.history(period="1d").empty
    except:
        return False


def validate_crypto_ticker(ticker):
    try:
        if "-" not in ticker and not ticker.endswith("USD"):
            ticker = f"{ticker}-USD"
        coin = yf.Ticker(ticker)
        return not coin.history(period="1d").empty, ticker
    except:
        return False, ticker


# ==========================================
# 2. 데이터 수집
# ==========================================
def get_data_stock(ticker):
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    news_list = []
    for entry in feed.entries[:10]:
        title = entry.title
        score = analyze_sentiment(title)
        news_list.append({
            "date": datetime(*entry.published_parsed[:6]) if entry.published_parsed else datetime.now(),
            "title": title,
            "url": entry.link,
            "sentiment_score": score,
            "sentiment_label": get_sentiment_label(score)
        })
    news_df = pd.DataFrame(news_list)

    stock = yf.Ticker(ticker)
    info = stock.info
    curr = info.get('currentPrice') or info.get('regularMarketPrice') or 0
    prev = info.get('previousClose') or 0
    rate = ((curr - prev) / prev) * 100 if prev > 0 else 0

    stock_info = {
        "ticker": ticker, "type": "Stock", "current_price": curr, "change_rate": rate,
        "high52": info.get('fiftyTwoWeekHigh', 0), "pe_ratio": info.get('trailingPE', 'N/A'),
        "recommendation": info.get('recommendationKey', 'none').upper().replace('_', ' '),
        "target_price": info.get('targetMeanPrice', 0),
        "business_summary": info.get('longBusinessSummary', '정보 없음')[:300] + "..."
    }
    return news_df, stock_info


def get_data_crypto(ticker):
    rss_url = f"https://news.google.com/rss/search?q={ticker}+crypto&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    news_list = []
    for entry in feed.entries[:10]:
        title = entry.title
        score = analyze_sentiment(title)
        news_list.append({
            "date": datetime(*entry.published_parsed[:6]) if entry.published_parsed else datetime.now(),
            "title": title,
            "url": entry.link,
            "sentiment_score": score,
            "sentiment_label": get_sentiment_label(score)
        })
    news_df = pd.DataFrame(news_list)

    coin = yf.Ticker(ticker)
    info = coin.info
    curr = info.get('regularMarketPrice') or info.get('currentPrice') or 0
    prev = info.get('previousClose') or 0
    rate = ((curr - prev) / prev) * 100 if prev > 0 else 0

    stock_info = {
        "ticker": ticker, "type": "Crypto", "current_price": curr, "change_rate": rate,
        "high52": info.get('fiftyTwoWeekHigh', 0),
        "volume": info.get('volume24Hr') or info.get('regularMarketVolume') or 0,
        "market_cap": info.get('marketCap', 0),
        "circulating_supply": info.get('circulatingSupply', 0),
        "business_summary": info.get('description', '정보 없음')[:300] + "..."
    }
    return news_df, stock_info


# ==========================================
# 3. Gemini AI 분석 (★ 모델 3단 변신 로직)
# ==========================================
def get_ai_analysis(api_keys_list, market_info, news_df):
    # 1. 우선순위 모델 리스트 정의 (순서 중요)
    candidate_models = [
        "gemini-2.5-flash",  # 1순위
        "gemini-2.5-flash-lite",  # 2순위
        "gemini-3-flash"  # 3순위 (미래 or 고성능)
    ]

    # 2. 프롬프트 미리 구성 (모델 돌릴 때마다 만들면 낭비니까)
    news_txt = "\n".join([f"- {r['title']} ({r['sentiment_label']})" for _, r in
                          news_df.iterrows()]) if not news_df.empty else "뉴스 없음"

    asset_type = market_info.get('type', 'Stock')
    if asset_type == "Crypto":
        fund_txt = f"- 시총: ${market_info['market_cap']:,}\n- 거래량: {market_info['volume']:,}\n- 유통량: {market_info['circulating_supply']:,}"
        point = "온체인 데이터, 고래, 규제"
    else:
        fund_txt = f"- PER: {market_info['pe_ratio']}\n- 의견: {market_info['recommendation']}\n- 목표가: ${market_info['target_price']}"
        point = "실적, 금리, 펀더멘털"

    prompt = f"""
    너는 1타 애널리스트야. '{market_info['ticker']}' 투자 보고서를 써.

    [데이터]
    - 현재가: ${market_info['current_price']} ({market_info['change_rate']:.2f}%)
    {fund_txt}

    [뉴스]
    {news_txt}

    [요청]
    '{point}' 중점 심층 분석. 결론(매수/매도/관망) 도출. 언어는 한국어, 마크다운 형식으로 출력. 데이터에 근거하여 날카롭게 분석.
    """

    # 3. [키 순회] -> [모델 순회] 이중 루프
    for i, key in enumerate(api_keys_list):
        for model_name in candidate_models:
            try:
                # 키 설정 및 모델 생성
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)

                # 시도
                response = model.generate_content(prompt)

                # 성공 시 바로 리턴 (함수 종료)
                return response.text

            except Exception as e:
                # 실패 로그 찍고 continue (다음 모델 or 다음 키로 넘어감)
                print(f"Key #{i + 1} | Model '{model_name}' Fail: {e}")
                continue

    # 모든 키와 모델이 다 실패했을 때
    return f"🤯 모든 키가 전사했거나, 모델들({candidate_models})을 찾을 수 없다."


# ==========================================
# 4. UI 구성
# ==========================================
st.set_page_config(page_title="AI 투자 분석", layout="wide")

st.markdown("""
<style>
    .bubble {
        position: relative; background: #ffdddd; border: 2px solid #ff0000;
        color: #d8000c; font-weight: bold; padding: 10px; border-radius: 10px;
        margin-bottom: 15px; width: fit-content; animation: fadeIn 0.5s;
    }
    .bubble:after {
        content: ''; position: absolute; bottom: 0; left: 20px; width: 0; height: 0;
        border: 10px solid transparent; border-top-color: #ff0000; border-bottom: 0;
        margin-left: -10px; margin-bottom: -10px;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI 투자 분석 리포트")
st.caption(f"🔑 로드된 API 키: {len(API_KEYS)}개 | Models: 2.5-flash -> lite -> 3-flash")

tab_stock, tab_crypto = st.tabs(["📉 주식", "🪙 암호화폐"])

with tab_stock:
    e_stock = st.empty()
    c1, c2 = st.columns([4, 1])
    ticker = c1.text_input("티커 (예: TSLA)", "TSLA", key="s_in", label_visibility="collapsed")
    if c2.button("분석", key="s_btn", use_container_width=True):
        if not ticker:
            e_stock.markdown('<div class="bubble">입력해라.</div>', unsafe_allow_html=True)
        elif not validate_ticker(ticker):
            e_stock.markdown(f'<div class="bubble">\'{ticker}\' 없다.</div>', unsafe_allow_html=True)
        else:
            e_stock.empty()
            with st.spinner("분석 중..."):
                df, info = get_data_stock(ticker)
                rpt = get_ai_analysis(API_KEYS, info, df)

                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("현재가", f"${info['current_price']}", f"{info['change_rate']:.2f}%")
                m2.metric("목표가", f"${info['target_price']}")
                m3.metric("PER", info['pe_ratio'])
                m4.metric("의견", info['recommendation'])
                st.subheader(f"📝 {info['ticker']} 리포트")
                st.markdown(rpt)
                with st.expander("뉴스"): st.dataframe(df[['date', 'title', 'sentiment_label', 'url']], hide_index=True)

with tab_crypto:
    e_crypto = st.empty()
    c1, c2 = st.columns([4, 1])
    c_ticker = c1.text_input("코인 (예: BTC)", "BTC", key="c_in", label_visibility="collapsed")
    if c2.button("분석", key="c_btn", use_container_width=True):
        if not c_ticker:
            e_crypto.markdown('<div class="bubble">입력해라.</div>', unsafe_allow_html=True)
        else:
            valid, real_t = validate_crypto_ticker(c_ticker)
            if not valid:
                e_crypto.markdown(f'<div class="bubble">\'{c_ticker}\' 없다.</div>', unsafe_allow_html=True)
            else:
                e_crypto.empty()
                with st.spinner("분석 중..."):
                    df, info = get_data_crypto(real_t)
                    rpt = get_ai_analysis(API_KEYS, info, df)

                    st.divider()
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("현재가", f"${info['current_price']}", f"{info['change_rate']:.2f}%")
                    k2.metric("시가총액", f"${info['market_cap']:,}")
                    k3.metric("거래량", f"${info['volume']:,}")
                    k4.metric("유통량", f"{info['circulating_supply']:,}")
                    st.subheader(f"🪙 {info['ticker']} 리포트")
                    st.markdown(rpt)
                    with st.expander("뉴스"): st.dataframe(df[['date', 'title', 'sentiment_label', 'url']],
                                                         hide_index=True)