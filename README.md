# 💰 BankAI — Duygusal Finans Danışmanı

**BankAI**, yatırım kararlarını sadece fiyat hareketlerine değil duygulara da dayandırmak isteyen kullanıcılar için geliştirilmiş bir Streamlit uygulamasıdır. Tek ekrandan metin tabanlı duygu analizi, fiyat grafikleri ve davranışsal öneriler sunar.

https://bankaita.streamlit.app/

---

## ✨ Özellikler

| Başlık | Açıklama |
| --- | --- |
| 💬 Duygu Analizi | Kullanıcının metnini XLM-RoBERTa tabanlı modelle (Transformers) yorumlar, pozitif/negatif skorlar üretir. |
| 🤖 Gemini Yorumları | Google Gemini API açıksa duyguya göre kişiselleştirilmiş koçluk mesajı üretir; kapalıysa kural bazlı öneri verir. |
| 📈 Fiyat Grafiği | Seçilen varlık için fiyat, hareketli ortalama ve volatilite bandını çizer. |
| 🔁 Veri Kaynakları | Önce Binance (kripto), ardından yfinance ve yahooquery denenir; hepsi başarısızsa demo verisine düşer. |
| 🎛️ Kolay Arayüz | Dark tema, ayarlanabilir trend/volatilite pencereleri ve hazır örnek metin. |

---

## 🛠️ Kullanılan Teknolojiler

- Python 3.11+
- Streamlit
- yfinance & yahooquery (fiyat verisi)
- Binance REST API (kriptopara desteği)
- pandas, numpy, matplotlib
- transformers & torch
- google-generativeai
- python-dotenv

---

## ⚙️ Kurulum

```bash
# 1. Depoyu klonla
git clone https://github.com/grdonn/bankai.git
cd bankai

# 2. Sanal ortam (önerilir)
python3 -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate

# 3. Bağımlılıkları kur
pip install --upgrade pip
pip install -r requirements.txt
```

### 🔐 Ortam Değişkenleri
Uygulama `.env` dosyası üzerinden Google Gemini anahtarını okur:

```env
GOOGLE_API_KEY=senin_gemini_api_anahtarin
```

Anahtar tanımlı değilse uygulama otomatik olarak kural bazlı yorumlama yapar.

---

## ▶️ Çalıştırma

```bash
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

---

## 🔄 Veri Kaynağı Akışı

1. **Kripto varlıklar:** Binance → yfinance → yahooquery → demo veri  
2. **Hisse senetleri:** yfinance → yahooquery → demo veri  

Herhangi bir kaynak veri döndürdüğünde uyarı mesajı gösterilmez ve kullanılan kaynak alt başlıkta belirtilir. Erişim hataları durumunda loglar terminalde, kullanıcı mesajları ise Streamlit üzerinde görünür.

---

## 💡 Kullanım İpuçları

- Sol panelden varlık kodunu (örn. `BTC-USD`, `AKBNK.IS`), periyodu ve zaman aralığını seç.
- Trend ve volatilite pencere boyutları grafikteki hareketli ortalama ve bant genişliğini belirler.
- “Hızlı Test” butonu örnek bir duygu metni yükler.
- Gemini aktifken metin bazlı öneriler üretir; API anahtarı yoksa kural bazlı mesaj döner.

---

## 🗂️ Depo Yapısı

```
bankai/
├── app.py             # Streamlit uygulaması
├── requirements.txt   # Bağımlılıklar
├── .env               # (opsiyonel) Ortam değişkenleri
└── README.md          # Bu dosya
```

---

## ⚠️ Uyarı

Bu proje demo ve eğitim amaçlıdır; üretilen çıktılar yatırım tavsiyesi değildir. `.env` dosyasındaki API anahtarlarını paylaşmayın. Torch/Transformers modelleri ilk çalıştırmada indirildiği için başlatma süresi birkaç saniye sürebilir.
