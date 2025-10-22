# app.py — BankAI (Tek Dosya)
# Çalıştır: streamlit run app.py

import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests

# Uyarıları sadeleştir
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- .env ----------------
try:
    from dotenv import load_dotenv
    for p in [Path.cwd()/".env", Path(__file__).parent/".env", Path(__file__).parent.parent/".env"]:
        if p.exists():
            load_dotenv(p, override=False)
except Exception:
    pass

# ---------------- UI ----------------
st.set_page_config(page_title="BankAI — Duygusal Finans Danışmanı", layout="wide", page_icon="💰")
st.markdown("""
<style>
:root { --card:#111827; --muted:#93a3b8; --good:#34d399; --warn:#f59e0b; --bad:#fb7185; --ink:#e5e7eb; }
.block-container {padding-top: 0.8rem;}
.card { background: var(--card); border: 1px solid rgba(255,255,255,.06); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;}
.kpi{background:#0f172a;border-radius:14px;text-align:center;padding:14px;}
.kpi .label{color:var(--muted);font-size:.85rem}
.kpi .value{font-weight:700;font-size:1.3rem;color:var(--ink)}
.badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px}
.badge.good{background:rgba(16,185,129,.15);color:#34d399}
.badge.bad{background:rgba(239,68,68,.15);color:#f87171}
.badge.neutral{background:rgba(148,163,184,.15);color:#94a3b8}
.bar-wrap{height:10px;background:rgba(148,163,184,.2);border-radius:999px;overflow:hidden}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='card' style="background:linear-gradient(90deg, rgba(34,211,238,.12), rgba(99,102,241,.12));">
  <h1 style="margin:0 0 6px 0">💰 BankAI — Duygusal Finans Danışmanı</h1>
  <div style="color:#9fb0c8">Kararlarını sadece veriye değil, duyguna göre de anlamlandır.</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def _to_tz_naive_datetime_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    i = pd.DatetimeIndex(pd.to_datetime(idx))
    if i.tz is not None:
        i = i.tz_convert("UTC").tz_localize(None)
    return i

def _make_demo_series(days=30, start_price=100.0, seed=42):
    rng = np.random.default_rng(seed)
    # utcnow() uyarısını gidermek için:
    utc_now = datetime.now(timezone.utc)
    dates = pd.date_range(end=utc_now, periods=days, freq="D")
    rets = rng.normal(loc=0.0005, scale=0.01, size=days)
    prices = start_price * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": prices}, index=_to_tz_naive_datetime_index(dates))
    return df

def _binance_symbol(ticker: str) -> str | None:
    ticker = (ticker or "").upper().replace("/", "-").strip()
    if "-" not in ticker:
        return None
    base, quote = ticker.split("-", 1)
    if not base.isalpha():
        return None
    quote_map = {"USD": "USDT", "USDT": "USDT", "USDC": "USDC", "EUR": "EUR"}
    quote = quote_map.get(quote, None)
    if not quote:
        return None
    return f"{base}{quote}"

def _try_fetch_binance(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    symbol = _binance_symbol(ticker)
    if not symbol:
        return None
    interval_map = {"1d": "1d", "1wk": "1w"}
    binance_interval = interval_map.get(interval)
    if not binance_interval:
        return None
    period_days = {"1mo": 30, "3mo": 90, "6mo": 180}
    interval_days = {"1d": 1, "1wk": 7}
    total_days = period_days.get(period)
    step = interval_days.get(interval)
    if not total_days or not step:
        return None
    limit = min(1000, max(2, int(total_days / step)))
    url = "https://api.binance.com/api/v3/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": binance_interval, "limit": limit}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or not data:
        return None
    records = []
    for item in data:
        try:
            open_time = datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc)
            close_price = float(item[4])
            records.append((open_time, close_price))
        except Exception:
            continue
    if not records:
        return None
    idx = pd.DatetimeIndex([r[0] for r in records])
    series = pd.Series([r[1] for r in records], index=idx, name="close")
    df = pd.DataFrame(series).sort_index()
    df.index = _to_tz_naive_datetime_index(df.index)
    return df

# ---------------- Data (yfinance -> yahooquery -> demo) ----------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(ticker: str, period: str, interval: str):
    meta = {"source": None, "message": None}
    errors = []
    prefer_binance = _binance_symbol(ticker) is not None

    # 0) Binance öncelik (kripto sembolleri için)
    if prefer_binance:
        try:
            df = _try_fetch_binance(ticker, period, interval)
            if df is not None and not df.empty:
                meta["source"] = "binance"
                return df, meta
        except Exception as e:
            errors.append(f"binance hata: {e}")

    # 1) yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True, actions=False)
        if df is None or df.empty:
            alt = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=True, actions=False, progress=False)
            df = alt
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.swaplevel(axis=1)[ticker]
                except KeyError:
                    df = df.xs(ticker, level=-1, axis=1, drop_level=True)
            rename_map = {c: c.lower() for c in df.columns}
            df = df.rename(columns=rename_map)
            for candidate in ("close", "adj close", "adjclose"):
                if candidate in df.columns:
                    close_col = candidate
                    break
            else:
                close_col = None
            if close_col:
                df = df[[close_col]].rename(columns={close_col: "close"})
                df.index = _to_tz_naive_datetime_index(df.index)
                meta["source"] = "yfinance"
                return df, meta
        errors.append("yfinance boş döndü.")
    except Exception as e:
        errors.append(f"yfinance hata: {e}")

    # 2) yahooquery (bazı sürümlerde backoff_exponent yok—kullanma)
    try:
        from yahooquery import Ticker as YqTicker
        yq = YqTicker(ticker)  # sade
        hist = yq.history(period=period, interval=interval)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.reset_index()
                hist = hist[hist["symbol"].astype(str).str.upper() == ticker.upper()]
                hist["date"] = pd.to_datetime(hist["date"], utc=True)
                hist = hist.set_index("date")
            elif hist.index.tz is None:
                hist.index = pd.DatetimeIndex(pd.to_datetime(hist.index, utc=True))
            close_col = "adjclose" if "adjclose" in hist.columns else ("close" if "close" in hist.columns else None)
            if close_col:
                hist.index = _to_tz_naive_datetime_index(hist.index)
                df = pd.DataFrame({"close": hist[close_col].astype(float)})
                if not df.empty and df["close"].notna().sum() > 1:
                    meta["source"] = "yahooquery"
                    return df, meta
        errors.append("yahooquery: veri alınamadı.")
    except Exception as e:
        errors.append(f"yahooquery hata: {e}")

    # 3) Binance (kripto için ek kaynak)
    try:
        df = _try_fetch_binance(ticker, period, interval)
        if df is not None and not df.empty:
            meta["source"] = "binance"
            meta["message"] = None
            return df, meta
    except Exception as e:
        errors.append(f"binance hata: {e}")

    # 4) Demo
    df = _make_demo_series(days=30, start_price=100.0)
    meta["source"] = "demo"
    msg = " | ".join(errors)
    if msg:
        msg += " | "
    meta["message"] = msg + "Gerçek fiyat verisine ulaşılamadı; demo seri gösteriliyor."
    return df, meta

# ---------------- Plot ----------------
def plot_prices(df: pd.DataFrame, trend_win: int, vol_win: int, title: str):
    s = df["close"].astype(float)
    ma = s.rolling(trend_win, min_periods=max(2, trend_win//3)).mean()
    std = s.rolling(vol_win, min_periods=max(2, vol_win//3)).std()
    upper, lower = ma + 2*std, ma - 2*std

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, label="Fiyat", linewidth=2)
    ax.plot(ma.index, ma.values, "--", label="Ortalama")
    ax.fill_between(ma.index, lower.values, upper.values, alpha=0.15, label="Vol Bant")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False)
    st.pyplot(fig)

# ---------------- Sentiment ----------------
@st.cache_resource(show_spinner=False)
def _load_sentiment_pipeline():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    # Uyarıyı sessize almak için CPU'da çalıştır:
    nlp = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, framework="pt",
                   return_all_scores=True, device=-1)
    id2label = getattr(mdl.config, "id2label", {0: "negative", 1: "neutral", 2: "positive"})
    return nlp, id2label

def analyze_sentiment(text: str):
    text = (text or "").strip()
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "label": "nötr / kararsız"}
    try:
        nlp, id2label = _load_sentiment_pipeline()
        scores = nlp(text[:512])[0]
        probs = {}
        for item in scores:
            lbl = item["label"].lower()
            if lbl.startswith("label_"):
                try:
                    idx = int(lbl.split("_")[-1])
                    lbl = id2label.get(idx, lbl).lower()
                except Exception:
                    pass
            probs[lbl] = float(item["score"])
        pos = probs.get("positive", 0.0)
        neg = probs.get("negative", 0.0)
        neu = probs.get("neutral", 0.0)
        comp = float(pos - neg)
        if comp >= 0.15:
            lbl = "pozitif / iyimser"
        elif comp <= -0.15:
            lbl = "negatif / kötümser"
        else:
            lbl = "nötr / kararsız" if neu >= max(pos, neg) else ("pozitif / iyimser" if pos >= neg else "negatif / kötümser")
        return {"compound": round(comp, 2), "pos": round(pos, 2), "neg": round(neg, 2), "label": lbl}
    except Exception as e:
        st.info(f"Duygu modeli yüklenemedi: {e}")
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "label": "nötr / kararsız"}

# ---------------- Gemini ----------------
def gemini_comment(prompt: str) -> str:
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        return "Gemini kapalı: GOOGLE_API_KEY bulunamadı, kural bazlı yorum kullanılıyor."
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        sys = "Sen bir duygusal finans koçusun. Yatırım tavsiyesi verme; davranışsal öneri sun."
        resp = model.generate_content([sys, prompt])
        return (getattr(resp, "text", "") or "").strip() or "Gemini uyarısı: model boş yanıt döndürdü."
    except Exception as e:
        return f"Gemini uyarısı: {e}"

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    ticker = st.text_input("Varlık / Hisse", "BTC-USD")
    period = st.selectbox("Periyot", ["1mo", "3mo", "6mo"], index=0)
    interval = st.selectbox("Aralık", ["1d", "1wk"], index=0)
    trend_win = st.slider("Trend penceresi", 5, 200, 50)
    vol_win = st.slider("Volatilite penceresi", 5, 60, 20)
    gemini_on = st.checkbox("🤖 Gemini Aktif", True)
    st.caption(f"GOOGLE_API_KEY set: {'✅' if bool(os.getenv('GOOGLE_API_KEY')) else '❌'}")

# ---------------- Karar Metni ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("💬 Karar / Düşünceni Yaz")
user_text = st.text_area("Duygunu yaz:", "Piyasa yükseliyor. alım yapacağım.", height=110)
col1, col2 = st.columns(2)
with col1:
    run = st.button("🔍 Analiz Et")
with col2:
    if st.button("⚡ Hızlı Test"):
        st.session_state["_txt"] = "Endişeliyim ama sabırlı kalmaya çalışıyorum."
        st.success("Örnek metin yüklendi. Analiz Et’e bas.")
st.markdown("</div>", unsafe_allow_html=True)
if "_txt" in st.session_state:
    user_text = st.session_state.pop("_txt")

# ---------------- Fiyat Grafiği ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Fiyat Grafiği")

df, meta = fetch_prices(ticker, period, interval)
if meta.get("message"):
    st.warning(meta["message"])

title = f"{ticker.upper()} — {period} {interval}"
plot_prices(df, trend_win, vol_win, title)
st.caption(f"Veri kaynağı: {meta['source']}")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Duygu ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 Duygu Analizi")
senti = analyze_sentiment(user_text)
comp, pos, neg, label = senti["compound"], senti["pos"], senti["neg"], senti["label"]

k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
with k1:
    st.markdown(f"<div class='kpi'><div class='label'>XLM-R · Compound</div><div class='value'>{comp:.2f}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='label'>Pozitif</div><div class='value'>{pos:.2f}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='label'>Negatif</div><div class='value'>{neg:.2f}</div></div>", unsafe_allow_html=True)

color = "good" if comp >= 0.15 else ("bad" if comp <= -0.15 else "neutral")
with k4:
    st.markdown(f"<span class='badge {color}'>{label}</span>", unsafe_allow_html=True)
    pct = int((comp + 1) * 50)
    st.markdown(f"<div class='bar-wrap'><div style='width:{pct}%;background:#10b981;height:10px;'></div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BankAI Yorumu ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧩 BankAI Yorumu")
if gemini_on:
    prompt = f"Kullanıcı metni: {user_text}\nDuygu: {label}\nCompound: {comp}"
    st.markdown(f"<div style='background:#0b2540;padding:14px;border-radius:12px;color:#cfe5ff'>{gemini_comment(prompt)}</div>", unsafe_allow_html=True)
else:
    st.markdown("Gemini kapalı. Kural bazlı değerlendirme:", unsafe_allow_html=True)
    st.info("Piyasa gürültüsünü duygudan ayır; plana sadık kal. Hedef/zarar-kes yaz, kademeli ilerle. Bu bir yatırım tavsiyesi değildir.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Özet ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Davranışsal Özet & Öneriler")
st.markdown("""
- Planını yaz ve yalnızca plana göre hareket et.  
- Hedef / zarar-kes seviyelerini önceden belirle.  
- Pozisyonu kademelendir, tek işlemle risk alma.  
- Not: Bu bir yatırım tavsiyesi değildir.
""")
st.markdown("</div>", unsafe_allow_html=True)
