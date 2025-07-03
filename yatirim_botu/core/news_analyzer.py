import requests
import os

def get_latest_news(query, api_key=None):
    """
    Son 24 saatte ilgili query hakkında çıkan haberleri özetler ve piyasa etkisini değerlendirir.
    API key parametre olarak verilmezse, ortam değişkeninden alınır.
    
    Örnek:
        from core.news_analyzer import get_latest_news
        output = get_latest_news("NASDAQ")
        print(output)
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    url = "https://generativelanguage.googleapis.com/v1beta/models/text-bison-001:generateText"
    headers = {"Content-Type": "application/json"}

    prompt = f"""Son 24 saatte {query} hakkında çıkan haberleri özetle ve bu haberlerin piyasa üzerindeki etkisini değerlendir.
- Pozitif mi negatif mi?
- Kısa vadeli mi uzun vadeli mi etkisi olur?
- Bu haber yatırım kararlarını nasıl etkileyebilir?"""

    data = {
        "prompt": {"text": prompt},
        "temperature": 0.7,
        "candidateCount": 1
    }

    response = requests.post(f"{url}?key={api_key}", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["candidates"][0]["output"]
    else:
        print("API Hatası:", response.text)
        return None 