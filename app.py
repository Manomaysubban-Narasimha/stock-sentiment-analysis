import streamlit as st
import requests
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from statistics import mean

MAX_FREE_TIER_MAXIMIZE_FETCH = 100

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


def main():
    st.title("Stock Sentiment Analyzer")
    st.write("Enter a stock symbol to analyze sentiment based on recent financial news articles.")
    
    stock_symbol = st.text_input("Stock Symbol (e.g., AAPL for Apple)", "").upper()
    
    if st.button("Analyze Sentiment"):
        if stock_symbol:
            with st.spinner("Fetching and analyzing news..."):
                result = analyze_stock_sentiment(stock_symbol)
                st.markdown(result)  
        else:
            st.warning("Please enter a stock symbol.")


if __name__ == "__main__":
    main()