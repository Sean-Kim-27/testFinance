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
# 2. 데이터 수집 (★ 주가 히스토리 추가됨!)
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def get_data_stock(ticker):
    # [1] 뉴스
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

    # [2] 주가 정보 & ★ 히스토리 (1년치)
    stock = yf.Ticker(ticker)
    info = stock.info
    curr = info.get('currentPrice') or info.get('regularMarketPrice') or 0
    prev = info.get('previousClose') or 0
    rate = ((curr - prev) / prev) * 100 if prev > 0 else 0

    # 1년치 주가 데이터 가져오기
    history_df = stock.history(period="1y")

    stock_info = {
        "ticker": ticker, "type": "Stock", "current_price": curr, "change_rate": rate,
        "high52": info.get('fiftyTwoWeekHigh', 0), "pe_ratio": info.get('trailingPE', 'N/A'),
        "recommendation": info.get('recommendationKey', 'none').upper().replace('_', ' '),
        "target_price": info.get('targetMeanPrice', 0),
        "business_summary": info.get('longBusinessSummary', '정보 없음')[:300] + "..."
    }

    # [3] 재무제표 데이터
    financials = pd.DataFrame()
    financial_summary = "재무 데이터 없음"
    try:
        fin = stock.financials
        if not fin.empty:
            target_rows = ['Total Revenue', 'Operating Income', 'Net Income']
            available_rows = [r for r in target_rows if r in fin.index]
            if available_rows:
                financials = fin.loc[available_rows].T.sort_index()
                recent = financials.iloc[-1]
                financial_summary = f"최근 매출: {recent.get('Total Revenue', 0):,.0f}, 영업이익: {recent.get('Operating Income', 0):,.0f}, 순이익: {recent.get('Net Income', 0):,.0f}"
    except:
        pass

    # 리턴값에 history_df 추가됨
    return news_df, stock_info, financials, financial_summary, history_df


@st.cache_data(ttl=600, show_spinner=False)
def get_data_crypto(ticker):
    # 뉴스
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

    # 코인 정보 & ★ 히스토리
    coin = yf.Ticker(ticker)
    info = coin.info
    curr = info.get('regularMarketPrice') or info.get('currentPrice') or 0
    prev = info.get('previousClose') or 0
    rate = ((curr - prev) / prev) * 100 if prev > 0 else 0

    # 1년치 데이터
    history_df = coin.history(period="1y")

    stock_info = {
        "ticker": ticker, "type": "Crypto", "current_price": curr, "change_rate": rate,
        "high52": info.get('fiftyTwoWeekHigh', 0),
        "volume": info.get('volume24Hr') or info.get('regularMarketVolume') or 0,
        "market_cap": info.get('marketCap', 0),
        "circulating_supply": info.get('circulatingSupply', 0),
        "business_summary": info.get('description', '정보 없음')[:300] + "..."
    }
    return news_df, stock_info, history_df  # 코인은 재무제표 없으니까 3개만 리턴


# 거시경제 지표
@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_data():
    try:
        return yf.Ticker("^TNX").history(period="1y")['Close']
    except:
        return pd.Series()


# ==========================================
# 3. Gemini AI 분석
# ==========================================
def get_ai_analysis(api_keys_list, market_info, news_df, extra_context=""):
    candidate_models = ["gemma-3-27b-it", "gemini-2.5-flash"]

    if news_df.empty:
        news_txt = "특이 뉴스 없음."
    else:
        # 뉴스 제목 + 감성(긍정/부정)을 같이 줘서 판단을 도움
        news_txt = "\n".join([
            f"- {row['title']} (감성: {row['sentiment_label']})"
            for _, row in news_df.iterrows()
        ])

        # 자산 유형별 분석 포인트 설정
    asset_type = market_info.get('type', 'Stock')
    if asset_type == "Crypto":
        data_context = f"""
            - 시가총액: ${market_info['market_cap']:,}
            - 24시간 거래량: {market_info['volume']:,}
            - 유통 공급량: {market_info['circulating_supply']:,}
            """
        key_focus = "비트코인 도미넌스, 고래 지갑 이동, 규제 뉴스, 반감기 사이클"
    else:
        data_context = f"""
            - PER (주가수익비율): {market_info['pe_ratio']}
            - 월가 투자의견: {market_info['recommendation']}
            - 목표주가: ${market_info['target_price']}
            """
        key_focus = "매출/이익 성장세, 금리 민감도, 경쟁사 대비 우위, 밸류에이션 매력도"

    prompt = f"""
    너는 월가 헤지펀드의 수석 애널리스트다. 
    지금 당장 '{market_info['ticker']}' 종목에 대한 **매수/매도 보고서** 를 작성해야 한다.
    단순한 정보 나열은 해고 사유다. 제공된 데이터를 바탕으로 **날카로운 통찰(Insight)** 을 제시해라.
    그렇다고 정말 보고서 처럼 날짜, 제목을 정하지 마라.

    ### 1. [시장 데이터]
    - 현재가: ${market_info['current_price']}
    - 변동률: {market_info['change_rate']:.2f}% (오늘 흐름)
    {data_context}
    
    ### 2. [추가 컨텍스트 (재무/매크로)]
    {extra_context}

    ### 3. [최신 뉴스 헤드라인]
    {news_txt}

    ---
    ### [분석 지침]
    보고서는 다음 3단계 구조로 작성하고, 반드시 **한국어**로 출력하라.

    **1단계: 뉴스 및 재료 해석 (News Impact)**
    - 뉴스를 단순히 요약하지 마라.
    - 각 뉴스가 주가에 **상승 재료(Bullish)** 인지 **하락 재료(Bearish)** 인지, 아니면 **소음(Noise)** 인지 판별해라.
    - 해당 뉴스 별로 제목 + 해석을 일일이 작성하고 분석해라. 
    - 시장의 공포/탐욕 심리가 현재 가격에 반영되었는지 분석해라.

    **2단계: 펀더멘털 및 매크로 진단 (Valuation & Macro)**
    - 위 [시장 데이터]와 [추가 컨텍스트]를 연결해서 분석해라.
    - 예: "매출은 늘었는데 주가가 빠지는 이유는?", "금리 상승이 이 종목에 치명적인가?"
    - '{key_focus}' 관점에서 현재 위치를 평가해라.

    **3단계: 최종 투자 전략 (Final Verdict)**
    - 결론은 무조건 **[강력 매수 / 매수 / 관망 / 매도]** 중 하나로 명확히 시작해라. 또한 도출해낸 결론에 마크다운으로 볼드체와 글자 크기(## **내용** ##) 를 크게 작성해라.
    - 그 이유를 데이터 기반으로 정확하게 정리해라.
    - 목표가나 손절가에 대한 힌트가 있다면 포함해라.

    **[톤앤매너]**
    - 전문적이고 냉소적인 어조를 유지해라.
    - "좋을 수도 있고 나쁠 수도 있다"는 식의 애매한 말은 하지 마라.
    - 마크다운(Markdown)을 사용하여 가독성을 높여라.
    """

    for i, key in enumerate(api_keys_list):
        for model_name in candidate_models:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return f"🤖 **사용된 모델:** `{model_name}` (Key #{i + 1})\n\n" + response.text
            except Exception as e:
                continue

    return f"🤯 모든 키가 전사했거나, 모델들({candidate_models})을 찾을 수 없다."


# ==========================================
# 4. UI 구성
# ==========================================
st.set_page_config(page_title="AI 투자 분석 Pro", layout="wide")

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
st.caption(f"🔑 로드된 API 키: {len(API_KEYS)}개 | Model Priority: Gemma-3 -> Gemini-2.5")
st.caption("Made by sean-kim-27 | Powered by Gemini | 본 자료는 참고용이므로, 투자 시 발생하는 문제는 본인의 책임입니다.")

tab_stock, tab_crypto = st.tabs(["📉 주식", "🪙 암호화폐"])

# ----------------- 주식 탭 -----------------
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
            with st.spinner("주가 차트 그리는 중..."):
                # ★ 리턴값 5개로 늘어남 (history_df 추가)
                df, info, financials, fin_summary, history_df = get_data_stock(ticker)
                macro_data = get_macro_data()
                rpt = get_ai_analysis(API_KEYS, info, df, extra_context=f"재무요약: {fin_summary}")

                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("현재가", f"${info['current_price']}", f"{info['change_rate']:.2f}%")
                m2.metric("목표가", f"${info['target_price']}")
                m3.metric("PER", info['pe_ratio'])
                m4.metric("의견", info['recommendation'])

                # [UI] ★ 주가 차트 (Line Chart)
                st.subheader("📈 주가 추이 (1년)")
                if not history_df.empty:
                    # 종가(Close)만 뽑아서 그림
                    st.line_chart(history_df['Close'], color="#00FF00")
                else:
                    st.warning("주가 데이터가 없다.")

                # [UI] 실적 & 금리 차트
                st.subheader("📊 주요 실적 및 시장 지표")
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    if not financials.empty:
                        st.markdown("**💰 연간 실적 (매출/순이익)**")
                        financials.index = financials.index.strftime('%Y-%m')
                        st.bar_chart(financials[['Total Revenue', 'Net Income']])
                    else:
                        st.info("재무 데이터 없음")

                with chart_col2:
                    if not macro_data.empty:
                        st.markdown("**🇺🇸 미국 국채 10년물 금리**")
                        st.line_chart(macro_data, color="#ff4b4b")
                    else:
                        st.info("금리 데이터 없음")

                st.subheader(f"📝 {info['ticker']} 리포트")
                st.markdown(rpt)

                with st.expander("뉴스"):
                    st.dataframe(df[['date', 'title', 'sentiment_label', 'url']], hide_index=True)

# ----------------- 코인 탭 -----------------
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
                with st.spinner("차트 그리는 중..."):
                    # ★ 리턴값 3개 (history_df 추가)
                    df, info, history_df = get_data_crypto(real_t)
                    macro_data = get_macro_data()
                    rpt = get_ai_analysis(API_KEYS, info, df, extra_context="암호화폐 시장은 매크로(금리) 민감도가 높음.")

                    st.divider()
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("현재가", f"${info['current_price']}", f"{info['change_rate']:.2f}%")
                    k2.metric("시가총액", f"${info['market_cap']:,}")
                    k3.metric("거래량", f"${info['volume']:,}")
                    k4.metric("유통량", f"{info['circulating_supply']:,}")

                    # [UI] ★ 코인 차트 (Line Chart)
                    st.subheader("📈 시세 추이 (1년)")
                    if not history_df.empty:
                        st.line_chart(history_df['Close'], color="#00FF00")
                    else:
                        st.warning("차트 데이터 없음")

                    # 금리 차트
                    st.subheader("📊 시장 지표")
                    if not macro_data.empty:
                        st.markdown("**🇺🇸 미국 국채 10년물 금리**")
                        st.line_chart(macro_data, color="#ff4b4b")

                    st.subheader(f"🪙 {info['ticker']} 리포트")
                    st.markdown(rpt)

                    with st.expander("뉴스"):
                        st.dataframe(df[['date', 'title', 'sentiment_label', 'url']], hide_index=True)