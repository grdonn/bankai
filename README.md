# 💰 BankAI — Duygusal Finans Danışmanı

**BankAI**, yatırım kararlarını sadece sayılara değil **duygulara** da dayandıran bir yapay zekâ tabanlı finans koçudur.  
Bu proje, duygusal analiz ve finansal verileri bir araya getirerek yatırımcılara davranışsal farkındalık kazandırmayı amaçlar.

---

## 🚀 Amaç

Modern finansal karar verme süreçlerinde insanlar yalnızca veriyle değil; **duygu, stres, korku ve umut** gibi faktörlerle de hareket eder.  
**BankAI** bu duygusal faktörleri analiz ederek yatırımcıya:

- **Duygusal farkındalık** kazandırır,  
- **Davranışsal finans** ilkelerine göre sakin kararlar almasına yardımcı olur,  
- **Panik, acelecilik veya aşırı iyimserlik** gibi hatalı karar kalıplarını fark ettirir.

---

## 🧠 Özellikler

| Alan | Açıklama |
|------|-----------|
| 💬 **Duygu Analizi** | Kullanıcının yazdığı finansal duygu/karar metnini analiz eder. (Türkçe destekli XLM-RoBERTa modeli ile) |
| 🤖 **Gemini Entegrasyonu** | Google Gemini API kullanılarak duygu durumuna göre kişiselleştirilmiş finansal farkındalık önerileri üretir. |
| 📈 **Veri Analizi** | Seçilen hisse veya kripto varlık için `yfinance` üzerinden fiyat geçmişini çeker ve trend + volatilite grafiği üretir. |
| 📊 **Volatilite ve Trend Göstergesi** | Hareketli ortalama ve volatilite bantlarıyla piyasa davranışını görselleştirir. |
| 🧩 **Kural Bazlı Yorumlama** | API kapalı olsa dahi sistem, davranışsal finans ilkelerine göre otomatik yorumlama yapar. |
| 🌙 **Modern UI / Dark Mode** | Streamlit + özel CSS tasarımıyla sade, profesyonel bir kullanıcı arayüzü. |

---

## 🏗️ Kullanılan Teknolojiler

- **Python 3.13**
- **Streamlit** — Web arayüzü
- **yFinance** — Piyasa verileri
- **Matplotlib** — Grafik çizimi
- **Transformers (Hugging Face)** — Türkçe çok dilli duygu analizi (XLM-RoBERTa)
- **Google Gemini API** — Akıllı metin yorumlama
- **dotenv** — Ortam değişkenleri yönetimi

---

## ⚙️ Kurulum

```bash
1️⃣ Depoyu klonla
git clone https://github.com/<kullanici-adin>/bankai.git
cd bankai

2️⃣ Sanal ortam oluştur ve etkinleştir
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Bağımlılıkları yükle
pip install -r requirements.txt

4️⃣ .env dosyasını oluştur

.env dosyan şu şekilde olmalı:

GOOGLE_API_KEY=senin_gemini_api_anahtarin
GEMINI_MODEL=gemini-2.5-flash

5️⃣ Uygulamayı başlat
streamlit run app.py


💡 Uygulama genellikle http://localhost:8501 adresinde çalışır.
```
💡 Örnek Senaryo

Kullanıcı girişi:
“Piyasa düşüyor diye satış yapmak istiyorum ama emin değilim.”

BankAI:

Duygusal tonun nötr/kararsız olduğunu tespit eder,

Fiyat grafiğini analiz ederek volatilitenin yüksek olduğunu gösterir,

“Panik satıştan kaçın, planına sadık kal, kademeli ilerle.” gibi öneriler üretir.
```bash
🧭 Proje Yapısı
bankai/
│
├── app.py                # Ana Streamlit uygulaması
├── requirements.txt      # Gerekli bağımlılıklar
├── .env                  # API anahtarları
├── .gitignore            # Git için hariç tutulan dosyalar
└── README.md             # Bu dosya
```
🔐 Güvenlik Notu

.env dosyanı asla paylaşma — içinde kişisel API anahtarları bulunur.

Gemini API çıktıları yalnızca davranışsal analiz amaçlıdır; yatırım tavsiyesi değildir.

✨ Gelecek Planları

📊 Portföy analizi entegrasyonu (gerçek kazanç/zarar farkındalığı)

🧩 Kullanıcı duygularını zaman içinde izleyen grafikler

💬 Telegram veya Discord bot entegrasyonu

🧠 Türkçe finans verisiyle fine-tuning model geliştirmesi

👨‍💻 Geliştirici

Taha Akgül
Yapay Zekâ ve Finans Teknolojileri Meraklısı
📍 Türkiye

Proje, Akbank’ın yenilikçi yapay zekâ projeleri kapsamında geliştirilmiştir.
Bu yazılım yalnızca demo ve eğitim amaçlıdır; yatırım tavsiyesi olarak değerlendirilmemelidir.

