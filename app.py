import streamlit as st
import requests
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from statistics import mean
import yfinance as yf
import plotly.graph_objects as go

MAX_FREE_TIER_MAXIMIZE_FETCH = 100
MILLION = 1_000_000
BILLION = 1_000_000_000
TRILLION = 1_000_000_000_000
THOUSAND = 1_000

# Load FinBERT model and tokenizer (cached to avoid reloading)
@st.cache_resource
def load_finbert_model():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_finbert_model()

def get_stock_news(stock_symbol):
    API_KEY = st.secrets["news_api_key"] 
    url = 'https://newsapi.org/v2/everything'
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15)
    
    financial_domains = 'bloomberg.com,cnbc.com,reuters.com,wsj.com,finance.yahoo.com,fool.com,nasdaq.com,benzinga.com'
    
    params = {
        'q': stock_symbol + ' stock',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'apiKey': API_KEY,
        'language': 'en',
        'domains': financial_domains,
        'pageSize': MAX_FREE_TIER_MAXIMIZE_FETCH  
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()['articles']
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return None


def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_idx = probs.index(max(probs))
    if sentiment_idx == 0:  # negative
        compound = -probs[0]
    elif sentiment_idx == 2:  # positive
        compound = probs[2]
    else:  # neutral
        compound = 0.0
    return compound


def analyze_stock_sentiment(stock_symbol):
    st.write(f"Analyzing sentiment for {stock_symbol}...")
    
    articles = get_stock_news(stock_symbol)
    if not articles:
        return "No articles found or error occurred"
    
    sentiments = []
    article_details = []
    
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        url = article.get('url', '')
        
        text = f"{title} {description}"
        if text.strip():
            sentiment_score = analyze_sentiment(text)
            sentiments.append(sentiment_score)
            article_details.append({
                'title': title,
                'sentiment': sentiment_score,
                'url': url
            })
            st.write(f"Article: {title[:50]}... Sentiment: {sentiment_score:.3f}")
    
    if not sentiments:
        return "No valid content found for sentiment analysis"
    
    avg_sentiment = mean(sentiments)
    interpretation = ""
    if avg_sentiment > 0.05:
        interpretation = "Positive"
    elif avg_sentiment < -0.05:
        interpretation = "Negative"
    else:
        interpretation = "Neutral"
    
    output = f"""
**Stock:** {stock_symbol}  
**Average Sentiment Score:** {avg_sentiment:.3f}  
**Overall Sentiment:** {interpretation}  
**Based on:** {len(sentiments)} articles  

**Article Details:**  
"""
    for i, detail in enumerate(article_details, 1):
        output += f"{i}. {detail['title'][:70]}...\n"
        output += f"   **Sentiment:** {detail['sentiment']:.3f}\n"
        output += f"   **URL:** [{detail['url']}]({detail['url']})\n\n"
    
    return output


def plot_stock_price(symbol, timeframe):
    # Define timeframe parameters
    end_date = datetime.now()
    if timeframe == '1d':
        start_date = end_date - timedelta(days=1)
        interval = '5m'
    elif timeframe == '1w':
        start_date = end_date - timedelta(weeks=1)
        interval = '15m'
    elif timeframe == '1m':
        start_date = end_date - timedelta(days=30)
        interval = '1h'
    elif timeframe == '3m':
        start_date = end_date - timedelta(days=90)
        interval = '1d'
    elif timeframe == '6m':
        start_date = end_date - timedelta(days=180)
        interval = '1d'
    elif timeframe == 'ytd':
        start_date = datetime(end_date.year, 1, 1)
        interval = '1d'
    elif timeframe == '1y':
        start_date = end_date - timedelta(days=365)
        interval = '1d'
    elif timeframe == '3y':
        start_date = end_date - timedelta(days=1095)
        interval = '1d'
    elif timeframe == '5y':
        start_date = end_date - timedelta(days=1825)
        interval = '1d'
    else:  # max
        start_date = None
        interval = '1d'

    # Fetch stock data
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        st.error(f"No data available for {symbol}")
        return

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price ({timeframe})',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    return fig


def get_company_financials(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    
    # Extract relevant financial data
    revenue = info.get('totalRevenue', 'N/A')
    net_profit = info.get('netIncomeToCommon', 'N/A')
    cash = info.get('totalCash', 'N/A')
    market_cap = info.get('marketCap', 'N/A')
    trailing_pe_ratio = info.get('trailingPE', 'N/A')
    forward_pe_ratio = info.get('forwardPE', 'N/A')
    
    revenue = format_value(revenue)
    net_profit = format_value(net_profit)
    cash = format_value(cash)
    market_cap = format_value(market_cap)
    
    # Format the output
    financials = f"""
    **Company Financials for {symbol}:**
    - **Total Revenue (Trailing 12 months ending December 31, 2024):** ${revenue}
    - **Net Profit/Bottom Line (Trailing 12 months ending December 31, 2024):** ${net_profit}
    - **Total Cash (quarter ending December 31, 2024):** ${cash}
    - **Market Cap:** ${market_cap}
    - **Trailing PE Ratio:** {trailing_pe_ratio:.2f}
    - **Forward PE Ratio:** {forward_pe_ratio:.2f}
    """
    
    return financials


def is_in_millions(value):
    return MILLION <= value < BILLION


def in_millions(value):
    return f'{value / MILLION:.2f} million'


def is_in_billions(value):
    return BILLION <= value < TRILLION


def in_billions(value):
    return f'{value / BILLION:.2f} billion'


def is_in_trillions(value):
    return value >= TRILLION


def in_trillions(value):
    return f'{value / TRILLION:.2f} trillion'


def format_value(value):
    if is_in_millions(value):
        return in_millions(value)
    elif is_in_billions(value):
        return in_billions(value)
    elif is_in_trillions(value):
        return in_trillions(value)
    else:
        return f'{value:.2f}'


def main():
    st.title("Stock Analyzer")
    st.write("Enter a stock symbol to analyze sentiment based on recent financial news articles.")
    
    stock_symbol = st.text_input("Stock Symbol (e.g., AAPL for Apple)", "").upper()
    
    if stock_symbol:
        # Add timeframe selector
        timeframes = ['1d', '1w', '1m', '3m', '6m', 'ytd', '1y', '3y', '5y', 'max']
        selected_timeframe = st.selectbox('Select Timeframe', timeframes)
        
        # Plot stock price
        fig = plot_stock_price(stock_symbol, selected_timeframe)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display company financials
        st.header("Financials")
        financials = get_company_financials(stock_symbol)
        st.markdown(financials)
        
        st.header("Sentiment Analysis") 
        with st.spinner("Fetching and analyzing news..."):
            result = analyze_stock_sentiment(stock_symbol)
            st.markdown(result)
    else:
        st.warning("Please enter a stock symbol.")


if __name__ == "__main__":
    main()