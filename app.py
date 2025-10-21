# app.py — BankAI (Türkçe Duygusal Finans Danışmanı)
# Çalıştırmak için:
#   streamlit run app.py

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# Çok dilli duygu analizi için
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)

# ==================== .env YÜKLE ====================
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def _load_env():
    if load_dotenv is None:
        return
    for p in [Path.cwd()/".env", Path(__file__).parent/".env", Path(__file__).parent.parent/".env"]:
        if p.exists():
            load_dotenv(p, override=False)
_load_env()

# ==================== SAYFA AYARI ====================
st.set_page_config(page_title="BankAI — Duygusal Finans Danışmanı", layout="wide", page_icon="💰")

st.markdown("""
<style>
:root {
  --bg: #0f172a;
  --panel: #0b1220;
  --card: #111827;
  --muted: #93a3b8;
  --brand: #22d3ee;
  --good: #34d399;
  --warn: #f59e0b;
  --bad: #fb7185;
}
.block-container {padding-top: 1rem;}
.header-bar {
  background: linear-gradient(90deg, rgba(34,211,238,.15), rgba(99,102,241,.15));
  border: 1px solid rgba(255,255,255,.05);
  border-radius: 14px;
  padding: 12px 16px;
  margin-bottom: 12px;
}
.card {
  background: var(--card);
  border: 1px solid rgba(255,255,255,.05);
  border-radius: 14px;
  padding: 16px 18px;
  margin-bottom: 10px;
}
.kpi {background:#0f172a; border-radius:14px; text-align:center; padding:14px;}
.kpi .label{color:var(--muted); font-size:0.85rem;}
.kpi .value{font-weight:700; font-size:1.4rem; color:#e5e7eb;}
.badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;}
.badge.good{background:rgba(16,185,129,.15);color:#34d399;}
.badge.bad{background:rgba(239,68,68,.15);color:#f87171;}
.badge.neutral{background:rgba(148,163,184,.15);color:#94a3b8;}
.bar-wrap{height:10px;background:rgba(148,163,184,.2);border-radius:999px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='header-bar'>
  <h1 style="margin:0 0 6px 0">💰 BankAI — Duygusal Finans Danışmanı</h1>
  <div style="color:#9fb0c8">Kararlarını sadece veriye değil, duyguna göre de anlamlandır.</div>
</div>
""", unsafe_allow_html=True)

# ==================== VERİ ÇEKME ====================
def fetch_prices(ticker, period, interval):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return None, f"Veri bulunamadı ({ticker})"
        df = df.rename(columns={"Close": "close"})[["close"]]
        df.index = pd.to_datetime(df.index)
        return df, None
    except Exception as e:
        return None, str(e)

# ==================== GRAFİK ====================
def plot_prices(df, trend_win, vol_win, title):
    s = df["close"]
    ma = s.rolling(trend_win).mean()
    std = s.rolling(vol_win).std()
    upper, lower = ma + 2*std, ma - 2*std

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s, label="Fiyat", color="#fcd34d", linewidth=2)
    ax.plot(ma.index, ma, "--", label="Ortalama", color="#93c5fd")
    ax.fill_between(ma.index, lower, upper, color="#60a5fa", alpha=0.15, label="Vol Bant")
    ax.set_title(title, color="#cbd5e1")
    ax.legend(loc="upper left", frameon=False)
    st.pyplot(fig)

# ==================== XLM-RoBERTa DUYGU MODELİ ====================
@st.cache_resource(show_spinner=False)
def _load_sentiment_pipeline():
    model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline("sentiment-analysis", model=mdl, tokenizer=tok, framework="pt")

def analyze_sentiment(text):
    text = text.strip()
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "label": "nötr / kararsız"}
    try:
        nlp = _load_sentiment_pipeline()
        out = nlp(text[:512])[0]
        label = out["label"].lower()
        score = float(out["score"])
        pos = score if "pos" in label else (1-score)/2
        neg = score if "neg" in label else (1-score)/2
        comp = np.clip(pos - neg, -1, 1)
        lbl = "pozitif / iyimser" if comp > 0.2 else ("negatif / kötümser" if comp < -0.2 else "nötr / kararsız")
        return {"compound": round(comp,2), "pos": round(pos,2), "neg": round(neg,2), "label": lbl}
    except Exception as e:
        st.info(f"Duygu modeli yüklenemedi: {e}")
        return {"compound":0,"pos":0,"neg":0,"label":"nötr / kararsız"}

# ==================== GEMINI ====================
def gemini_comment(prompt):
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        return "Gemini kapalı: GOOGLE_API_KEY bulunamadı, kural bazlı yorum kullanılıyor."
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        sys = "Sen bir duygusal finans koçusun. Yatırım tavsiyesi verme, davranışsal öneri sun."
        resp = model.generate_content([sys, prompt])
        txt = getattr(resp, "text", "") or ""
        if not txt.strip():
            return "Gemini uyarısı: model boş yanıt döndürdü (finish_reason=2 olabilir)."
        return txt.strip()
    except Exception as e:
        return f"Gemini uyarısı: {e}"

# ==================== UI ====================
with st.sidebar:
    st.header("⚙️ Ayarlar")
    ticker = st.text_input("Varlık / Hisse", "BTC-USD")
    period = st.selectbox("Periyot", ["1mo", "3mo", "6mo"], index=0)
    interval = st.selectbox("Aralık", ["1d", "1wk"], index=0)
    trend_win = st.slider("Trend penceresi", 5, 200, 50)
    vol_win = st.slider("Volatilite penceresi", 5, 60, 20)
    gemini_on = st.checkbox("🤖 Gemini Aktif", True)
    st.caption(f"GOOGLE_API_KEY set: {'✅' if bool(os.getenv('GOOGLE_API_KEY')) else '❌'}")

# -------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("💬 Karar / Düşünceni Yaz")
user_text = st.text_area("Duygunu yaz:", "Piyasa düşüyor. Satış yapmayı düşünüyorum.", height=100)
col1, col2 = st.columns(2)
with col1: run = st.button("🔍 Analiz Et")
with col2:
    if st.button("⚡ Hızlı Test"):
        user_text = "Endişeliyim ama sabırlı kalmaya çalışıyorum."
        st.session_state["_txt"] = user_text
        st.success("Örnek metin yüklendi. Analiz Et’e bas.")
st.markdown("</div>", unsafe_allow_html=True)

if "_txt" in st.session_state:
    user_text = st.session_state.pop("_txt")

# ==================== FİYAT GRAFİĞİ ====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Fiyat Grafiği")
df, err = fetch_prices(ticker, period, interval)
if df is None:
    st.error(err)
else:
    plot_prices(df, trend_win, vol_win, f"{ticker} — {period} {interval}")
st.markdown("</div>", unsafe_allow_html=True)

# ==================== DUYGU ====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 Duygu Analizi")
senti = analyze_sentiment(user_text)
comp, pos, neg, label = senti["compound"], senti["pos"], senti["neg"], senti["label"]
k1, k2, k3, k4 = st.columns([1,1,1,2])
with k1: st.markdown(f"<div class='kpi'><div class='label'>VADER · Compound</div><div class='value'>{comp:.2f}</div></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='kpi'><div class='label'>Pozitif</div><div class='value'>{pos:.2f}</div></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='kpi'><div class='label'>Negatif</div><div class='value'>{neg:.2f}</div></div>", unsafe_allow_html=True)
color = "good" if comp>0.2 else ("bad" if comp<-0.2 else "neutral")
with k4:
    st.markdown(f"<span class='badge {color}'>{label}</span>", unsafe_allow_html=True)
    pct = int((comp+1)*50)
    st.markdown(f"<div class='bar-wrap'><div style='width:{pct}%;background:#10b981;height:10px;'></div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==================== BANKAI YORUMU ====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧩 BankAI Yorumu")
if gemini_on:
    prompt = f"Kullanıcı metni: {user_text}\nDuygu: {label}\nCompound: {comp}"
    gtxt = gemini_comment(prompt)
    st.markdown(f"<div style='background:#0b2540;padding:14px;border-radius:12px;color:#cfe5ff'>{gtxt}</div>", unsafe_allow_html=True)
else:
    st.markdown("Gemini kapalı. Kural bazlı değerlendirme etkin.")
    st.info("Piyasa gürültüsünü duygudan ayır, plana sadık kal. Hedef/zarar-kes belirle, kademeli ilerle.")
st.markdown("</div>", unsafe_allow_html=True)

# ==================== DAVRANIŞSAL ÖZET ====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Davranışsal Özet & Öneriler")
st.markdown("""
- Piyasa gürültüsünü duygudan ayır, plana sadık kal.  
- Hedef/zarar-kes seviyelerini yaz.  
- Pozisyonu kademelendir, riski dağıt.  
- Not: Bu bir yatırım tavsiyesi değildir.
""")
st.markdown("</div>", unsafe_allow_html=True)