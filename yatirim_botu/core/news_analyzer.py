import requests
import os
import sqlite3
from datetime import datetime

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

def init_news_db(db_path="yatirim_botu/data/news.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        news_text TEXT,
        analysis TEXT,
        sentiment REAL,
        created_at TEXT
    )''')
    conn.commit()
    conn.close()

def save_news_to_db(query, news_text, analysis, sentiment, db_path="yatirim_botu/data/news.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT INTO news (query, news_text, analysis, sentiment, created_at) VALUES (?, ?, ?, ?, ?)''',
              (query, news_text, analysis, sentiment, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def nlp_sentiment_score(text):
    if TextBlob is None:
        print("TextBlob yüklü değil. 'pip install textblob' ile yükleyin.")
        return 0
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 (negatif) ile +1 (pozitif) arası

def get_latest_news(query, api_key=None, db_path="yatirim_botu/data/news.db"):
    """
    Son 24 saatte ilgili query hakkında çıkan haberleri özetler ve piyasa etkisini değerlendirir.
    Ayrıca haber ve analiz sonuçlarını SQLite veritabanına kaydeder.
    NLP ile duygu skoru da ekler.
    """
    init_news_db(db_path)
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    url = "https://generativelanguage.googleapis.com/v1beta/models/text-bison-001:generateText"
    headers = {"Content-Type": "application/json"}

    prompt = f"""Son 24 saatte {query} hakkında çıkan haberleri özetle ve bu haberlerin piyasa üzerindeki etkisini değerlendir.\n- Pozitif mi negatif mi?\n- Kısa vadeli mi uzun vadeli mi etkisi olur?\n- Bu haber yatırım kararlarını nasıl etkileyebilir?"""

    data = {
        "prompt": {"text": prompt},
        "temperature": 0.7,
        "candidateCount": 1
    }

    response = requests.post(f"{url}?key={api_key}", headers=headers, json=data)

    if response.status_code == 200:
        analysis = response.json()["candidates"][0]["output"]
        # Haber metni yoksa query'yi kaydet
        news_text = query
        sentiment = nlp_sentiment_score(analysis)
        save_news_to_db(query, news_text, analysis, sentiment, db_path)
        return analysis, sentiment
    else:
        print("API Hatası:", response.text)
        return None, 0 