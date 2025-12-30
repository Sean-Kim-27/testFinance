import streamlit as st
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import feedparser
import google.generativeai as genai
from datetime import datetime


try:
    GEMINI_KEY = st.secrets["API_KEY"]
except FileNotFoundError:
    st.error("API key not found.")
    st.stop()
except KeyError:
    st.error("API key not found.")
    st.stop()



# 1. 보조 함수

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

# 2. 유효성 검사 함수 (★ 추가됨)

def validate_ticker(ticker):
    """티커가 진짜 존재하는지 살짝 찔러보는 함수"""
    try:
        stock = yf.Ticker(ticker)
        # 1일치 데이터만 가져와서 데이터가 있는지 확인
        hist = stock.history(period="1d")
        if hist.empty:
            return False
        return True
    except:
        return False


# 3. 데이터 수집

def get_data(ticker):
    # --- [A] 구글 뉴스 ---
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    news_list = []
    for entry in feed.entries[:10]:
        title = entry.title
        link = entry.link
        pub_date = datetime(*entry.published_parsed[:6]) if entry.published_parsed else datetime.now()
        score = analyze_sentiment(title)

        news_list.append({
            "date": pub_date,
            "title": title,
            "url": link,
            "sentiment_score": score,
            "sentiment_label": get_sentiment_label(score)
        })

    news_df = pd.DataFrame(news_list)

    # --- [B] 주가 데이터 ---
    stock = yf.Ticker(ticker)
    info = stock.info

    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
    prev_close = info.get('previousClose') or 0

    change_rate = 0
    if prev_close > 0 and current_price > 0:
        change_rate = ((current_price - prev_close) / prev_close) * 100

    stock_info = {
        "ticker": ticker,
        "current_price": current_price,
        "change_rate": change_rate,
        "high52": info.get('fiftyTwoWeekHigh', 0),
        "pe_ratio": info.get('trailingPE', 'N/A'),
        "recommendation": info.get('recommendationKey', 'none').upper().replace('_', ' '),
        "target_price": info.get('targetMeanPrice', 0),
        "business_summary": info.get('longBusinessSummary', '정보 없음')[:300] + "..."
    }

    return news_df, stock_info



# 4. Gemini AI 분석

def get_ai_analysis(api_key, stock_info, news_df):
    genai.configure(api_key=api_key)

    model = None
    model_name_used = "Unknown"

    try:
        available_models = list(genai.list_models())
    except Exception as e:
        return f"모델 목록 조회 실패. 키 확인해라. 에러: {e}"

    # 모델 선택 로직 (Flash -> Pro -> Any)
    for m in available_models:
        if 'generateContent' in m.supported_generation_methods:
            if 'flash' in m.name:
                model = genai.GenerativeModel(m.name)
                model_name_used = m.name
                break

    if model is None:
        for m in available_models:
            if 'generateContent' in m.supported_generation_methods and 'pro' in m.name:
                model = genai.GenerativeModel(m.name)
                model_name_used = m.name
                break

    if model is None:
        for m in available_models:
            if 'generateContent' in m.supported_generation_methods:
                model = genai.GenerativeModel(m.name)
                model_name_used = m.name
                break

    if model is None:
        return "❌ 쓸 수 있는 모델이 하나도 없다."

    if news_df.empty:
        news_titles = "최근 관련 뉴스가 없습니다."
    else:
        news_titles = "\n".join([f"- {row['title']} ({row['sentiment_label']})" for _, row in news_df.iterrows()])

    prompt = f"""
    너는 월가에서 가장 냉철하고 분석적인 주식 애널리스트야. 
    아래 제공된 데이터를 바탕으로 '{stock_info['ticker']}' 종목에 대한 투자 보고서를 작성해줘.

    **작성 원칙:**
    1. 한국어로 작성할 것.
    2. 전문 용어를 적절히 섞되, 초보자도 이해할 수 있게 쉽게 설명할 것.
    3. 뻔한 소리 하지 말고, 데이터에 근거해서 날카롭게 분석할 것.
    4. 출력 형식은 가독성 좋은 마크다운(Markdown)으로.

    ---
    **[기업 개요]**
    - 현재가: ${stock_info['current_price']}
    - 52주 최고가: ${stock_info['high52']}
    - PER(주가수익비율): {stock_info['pe_ratio']}
    - 투자의견(컨센서스): {stock_info['recommendation']}
    - 목표주가: ${stock_info['target_price']}
    - 사업 요약: {stock_info['business_summary']}

    **[최신 뉴스 헤드라인 및 감성]**
    {news_titles}

    ---
    **[요청사항 - 보고서 목차]**
    1. **🧐 시장 분위기 및 뉴스 분석**: 뉴스들의 전반적인 톤앤매너와 주요 이슈 요약.
    2. **📊 펀더멘털 진단**: 현재 주가가 고평가인지 저평가인지, 목표주가 괴리율 등을 분석.
    3. **⚡ 최종 투자 의견**: 
       - 결론을 **[매수 / 매도 / 관망]** 중 하나로 명확히 내리고, 그 이유를 3줄 요약해줘.
    """

    try:
        response = model.generate_content(prompt)
        return f"🤖 **사용된 모델:** `{model_name_used}`\n\n" + response.text
    except Exception as e:
        return f"🤯 분석하다 터짐 ({model_name_used}): {e}"


# ==========================================
# 5. UI 구성
# ==========================================
st.set_page_config(page_title="AI 주식 분석", layout="wide")

# 말풍선 스타일 CSS 정의
st.markdown("""
<style>
    /* 말풍선 본체 */
    .bubble {
        position: relative;
        background: #ffdddd;
        border: 2px solid #ff0000;
        color: #d8000c;
        font-family: Arial, sans-serif;
        font-size: 14px;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 15px; /* 입력창이랑 간격 */
        width: fit-content;
        animation: fadeIn 0.5s;
    }

    /* 말풍선 꼬리 (아래쪽 화살표) */
    .bubble:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 20px; /* 꼬리 위치 */
        width: 0;
        height: 0;
        border: 10px solid transparent;
        border-top-color: #ff0000;
        border-bottom: 0;
        margin-left: -10px;
        margin-bottom: -10px;
    }

    /* 등장 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI 주식 분석 리포트")

# ★ 여기가 중요함: 에러 메시지가 뜰 공간(Placeholder)을 미리 잡아둠
error_placeholder = st.empty()

col1, col2 = st.columns([4, 1])
with col1:
    ticker = st.text_input("티커 입력", "TSLA", label_visibility="collapsed")
with col2:
    st.markdown("<div style='margin-top: -5px;'></div>", unsafe_allow_html=True)
    btn = st.button("분석", use_container_width=True)

if btn:
    if not ticker:
        # 티커 입력 안 했을 때 말풍선
        error_placeholder.markdown("""
            <div class="bubble"> 티커를 입력해주세요. </div>
        """, unsafe_allow_html=True)
    else:
        # 티커 검증
        with st.spinner("티커 확인 중..."):
            is_valid = validate_ticker(ticker)

        if not is_valid:
            # 존재하지 않는 티커일 때 말풍선
            error_placeholder.markdown(f"""
                <div class="bubble">'{ticker}'는 존재하지 않는 티커입니다.</div>
            """, unsafe_allow_html=True)
        else:
            # 정상일 때 에러 메시지 삭제하고 분석 시작
            error_placeholder.empty()

            with st.spinner(f"'{ticker}' 분석 중..."):
                df, s_info = get_data(ticker)
                ai_report = get_ai_analysis(GEMINI_KEY, s_info, df)

                st.divider()
                m1, m2, m3, m4 = st.columns(4)

                m1.metric("현재가", f"${s_info['current_price']}", f"{s_info['change_rate']:.2f}%")
                m2.metric("목표주가", f"${s_info['target_price']}")
                m3.metric("PER", s_info['pe_ratio'])
                m4.metric("월가 의견", s_info['recommendation'])

                st.subheader(f"📝 Gemini의 '{ticker}' 심층 분석")
                st.markdown(ai_report)

                with st.expander("📚 분석에 참고한 뉴스 원문 보기"):
                    if not df.empty:
                        st.dataframe(
                            df[['date', 'title', 'sentiment_label', 'url']],
                            column_config={
                                "date": st.column_config.DatetimeColumn("날짜", format="YYYY-MM-DD HH:mm"),
                                "title": "기사 제목",
                                "sentiment_label": "감성(AI분석)",
                                "url": st.column_config.LinkColumn("링크", display_text="기사 보기")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("뉴스 데이터가 없음.")

# ==========================================
# ★ 푸터 (배경색 자동 맞춤 + 선 제거)
# ==========================================
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--primary-background-color);
        color: var(--text-color);
        text-align: center;
        padding: 10px;
        font-size: 12px;
        z-index: 999;
        border-top: none; 
    }
    </style>
    <div class="footer">
        <p>Copyright © Made by sean-kim-27 (github) | Powered by Gemini | ⚠️ 투자는 본인의 선택이며 책임은 지지 않습니다.</p>
    </div>
    """,
    unsafe_allow_html=True
)