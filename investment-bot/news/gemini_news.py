# Gemini API news analysis implementation will go here 

def ask_gemini_sector_allocation(date, sectors):
    """
    Gemini API'ye (veya mock) şu soruyu sorar:
    'Bugün (date) yatırım yapacak olsan, hangi alana (sektöre) yatırım yapmamı önerirsin? Hangi sektöre yüzde kaç hisse ayırmamı önerirsin? Her sektör için 100 üzerinden puan ver.'
    sectors: sektör isimleri listesi
    return: {sektor: {'score': int, 'allocation': float}} (allocation: 0-1 arası oran)
    """
    # MOCK: Sektörleri rastgele puanla ve ağırlık ata (örnek)
    import random
    results = {}
    total = 0
    for s in sectors:
        score = random.randint(50, 100)
        alloc = random.uniform(0.05, 0.25)
        results[s] = {'score': score, 'allocation': alloc}
        total += alloc
    # Normalize allocation
    for s in results:
        results[s]['allocation'] /= total
    return results 