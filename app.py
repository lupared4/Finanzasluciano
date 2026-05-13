import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker
from database import engine, Transaccion
import yfinance as yf
import numpy as np
import pandas as pd
from pydantic import BaseModel
import time
import requests
import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from pandas_datareader import data as pdr
    _PDR_OK = True
except ImportError:
    _PDR_OK = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Session = sessionmaker(bind=engine)

# ─── Sesión HTTP con User-Agent ───────────────────────────────────────────────
_yf_session = requests.Session()
_yf_session.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
})

def yft(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)

# ── Símbolos crypto conocidos → Yahoo Finance usa "{SYM}-USD" ────────────────
CRYPTO_SYMBOLS = {
    'BTC','ETH','SOL','BNB','ADA','XRP','DOGE','DOT','LINK','MATIC',
    'AVAX','UNI','LTC','BCH','ATOM','NEAR','APT','OP','ARB','USDT',
    'USDC','SHIB','TRX','TON','ICP','FIL','HBAR','VET','ALGO','XLM',
    'SUI','SEI','INJ','JUP','WIF','BONK','PEPE','FLOKI','FET','RNDR',
}
# CoinGecko IDs para datos enriquecidos
COINGECKO_IDS = {
    'BTC':'bitcoin','ETH':'ethereum','SOL':'solana','BNB':'binancecoin',
    'ADA':'cardano','XRP':'ripple','DOGE':'dogecoin','DOT':'polkadot',
    'LINK':'chainlink','MATIC':'matic-network','AVAX':'avalanche-2',
    'UNI':'uniswap','LTC':'litecoin','BCH':'bitcoin-cash','ATOM':'cosmos',
    'NEAR':'near','APT':'aptos','OP':'optimism','ARB':'arbitrum',
    'SHIB':'shiba-inu','TRX':'tron','TON':'the-open-network','ICP':'internet-computer',
    'FIL':'filecoin','HBAR':'hedera-hashgraph','VET':'vechain','ALGO':'algorand',
    'XLM':'stellar','SUI':'sui','INJ':'injective-protocol','FET':'fetch-ai',
    'RNDR':'render-token',
}
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def yf_sym(ticker: str) -> str:
    """Convierte símbolo crypto a formato Yahoo Finance (agrega -USD si es crypto)."""
    sym = ticker.upper().replace('-USD','').replace('/USD','')
    return f"{sym}-USD" if sym in CRYPTO_SYMBOLS else sym

class TransaccionSchema(BaseModel):
    ticker: str
    cantidad: float
    precio_unitario: float

# ─── Cache en memoria ────────────────────────────────────────────────────────
_price_cache:    dict = {}  # precio actual: TTL 60s
_history_cache:  dict = {}  # histórico OHLCV: TTL 10 min
_endpoint_cache: dict = {}  # analisis/montecarlo/ia/indicadores: TTL 5 min

_CACHE_TTL          = 60       # 1 minuto para precios
_HISTORY_CACHE_TTL  = 600      # 10 minutos para histórico
_ENDPOINT_CACHE_TTL = 300      # 5 minutos para endpoints de análisis
_INDICADORES_TTL    = 3600     # 1 hora para indicadores fundamentales


def _cache_get(store: dict, key: str, ttl: int):
    """Retorna el valor cacheado si es válido, sino None."""
    if key in store:
        value, ts = store[key]
        if time.time() - ts < ttl:
            return value
    return None


def _cache_set(store: dict, key: str, value):
    store[key] = (value, time.time())

def get_precio_actual(ticker: str) -> float:
    """Obtiene el precio con cache y múltiples fallbacks."""
    sym = yf_sym(ticker)          # BTC → BTC-USD, AAPL → AAPL
    cached = _cache_get(_price_cache, sym, _CACHE_TTL)
    if cached is not None:
        return cached

    precio = 0.0
    tk = yft(sym)

    # Fallback 1: fast_info (más ligero, menos rate-limit)
    try:
        p = tk.fast_info.last_price
        if p and p > 0:
            precio = float(p)
            _cache_set(_price_cache, ticker, precio)
            return precio
    except Exception:
        pass

    # Fallback 2: yf.download (endpoint diferente)
    try:
        data = yf.download(
            sym, period="2d", auto_adjust=True,
            progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if not data.empty:
            precio = float(data['Close'].iloc[-1])
            _cache_set(_price_cache, ticker, precio)
            return precio
    except Exception:
        pass

    _cache_set(_price_cache, sym, 0.0)
    return 0.0


# ─── Precio individual ───────────────────────────────────────────────────────
@app.get("/precio/{ticker}")
def obtener_precio(ticker: str):
    """Retorna el precio actual de un ticker (con caché de 60s)."""
    return {"ticker": ticker.upper(), "precio": get_precio_actual(ticker)}


# ─── Autocompletado de tickers (proxy Yahoo Finance Search) ──────────────────
@app.get("/buscar-ticker")
def buscar_ticker(q: str = Query("")):
    """Proxy de búsqueda de Yahoo Finance para autocompletado de tickers."""
    if len(q.strip()) < 1:
        return []
    q_up = q.upper()
    cached = _cache_get(_endpoint_cache, f"search_{q_up}", 3600)
    if cached is not None:
        return cached
    try:
        url = (
            "https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={q}&quotesCount=6&newsCount=0&listsCount=0&enableFuzzyQuery=false"
        )
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        quotes = r.json().get("quotes", [])
        result = [
            {
                "ticker": item.get("symbol", ""),
                "nombre": item.get("shortname") or item.get("longname") or "",
                "tipo":   item.get("typeDisp", ""),
                "bolsa":  item.get("exchDisp", ""),
            }
            for item in quotes[:6]
            if item.get("symbol")
        ]
        _cache_set(_endpoint_cache, f"search_{q_up}", result)
        return result
    except Exception as e:
        print(f"Error buscar-ticker: {e}")
        return []


# ─── Resumen de cartera ───────────────────────────────────────────────────────
@app.get("/resumen")
def obtener_resumen():
    session = Session()
    try:
        tickers = session.query(Transaccion.ticker).distinct().all()
        cartera = []
        for (t,) in tickers:
            try:
                txs     = session.query(Transaccion).filter_by(ticker=t).all()
                compras = [tx for tx in txs if tx.tipo_operacion == 'Compra']
                ventas  = [tx for tx in txs if tx.tipo_operacion == 'Venta']

                cant_comprada  = sum(tx.cantidad for tx in compras)
                cant_vendida   = sum(tx.cantidad for tx in ventas)
                cant_actual    = cant_comprada - cant_vendida

                if cant_actual <= 0:
                    continue

                costo_total  = sum(tx.cantidad * tx.precio_unitario for tx in compras)
                precio_prom  = costo_total / cant_comprada if cant_comprada > 0 else 0

                ingreso_ventas = sum(tx.cantidad * tx.precio_unitario for tx in ventas)
                costo_vendido  = sum(tx.cantidad * precio_prom for tx in ventas)
                pnl_realizado  = round(ingreso_ventas - costo_vendido, 2)

                precio_actual = get_precio_actual(t)
                pnl_latente   = round((precio_actual * cant_actual) - (precio_prom * cant_actual), 2)

                cartera.append({
                    "ticker":        t,
                    "cantidad":      cant_actual,
                    "precio_compra": round(precio_prom, 2),
                    "precio_actual": round(precio_actual, 2),
                    "ganancia":      pnl_latente,
                    "pnl_realizado": pnl_realizado,
                })
            except Exception as e:
                print(f"Error procesando {t}: {e}")
                continue
        return cartera
    except Exception as e:
        print(f"Error en /resumen: {e}")
        return []
    finally:
        session.close()


# ─── Velas OHLC (para lightweight-charts) ────────────────────────────────────
@app.get("/velas/{ticker}")
def obtener_velas(ticker: str, period: str = Query("1mo")):
    try:
        sym  = yf_sym(ticker)          # BTC → BTC-USD
        data = get_history(sym, period)
        if data.empty:
            return []
        result = []
        for idx, row in data.iterrows():
            try:
                result.append({
                    "time":   idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10],
                    "open":   round(float(row['Open']),  2),
                    "high":   round(float(row['High']),  2),
                    "low":    round(float(row['Low']),   2),
                    "close":  round(float(row['Close']), 2),
                    "volume": int(row['Volume']) if 'Volume' in row else 0,
                })
            except Exception:
                continue
        return result
    except Exception as e:
        print(f"Error velas: {e}")
        return []


# ─── Mapeo de periodos a parámetros Yahoo Finance v8 ─────────────────────────
_PERIOD_MAP = {
    "5d":  ("5d",  "1d"),
    "1mo": ("1mo", "1d"),
    "3mo": ("3mo", "1d"),
    "1y":  ("1y",  "1d"),
    "6mo": ("6mo", "1d"),
    "2d":  ("5d",  "1d"),
}

_PERIOD_DAYS = {
    "5d": 7, "1mo": 35, "3mo": 95, "6mo": 185, "1y": 370, "2d": 5, "2y": 740
}

def fetch_stooq(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Obtiene histórico desde Stooq (sin rate-limit, sin API key)."""
    if not _PDR_OK:
        return pd.DataFrame()
    try:
        end   = datetime.date.today()
        delta = datetime.timedelta(days=_PERIOD_DAYS.get(period, 370))
        start = end - delta
        # Stooq acepta el ticker en mayúsculas, a veces con sufijo .US
        for t in [ticker.upper(), f"{ticker.upper()}.US"]:
            try:
                df = pdr.DataReader(t, 'stooq', start, end)
                if not df.empty:
                    df = df.sort_index()  # Stooq devuelve de nuevo a viejo
                    return df
            except Exception:
                continue
    except Exception as e:
        print(f"fetch_stooq {ticker}: {e}")
    return pd.DataFrame()

def fetch_yahoo_v8(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Llama directo al API v8 de Yahoo Finance como segundo fallback."""
    yf_period, interval = _PERIOD_MAP.get(period, ("1y", "1d"))
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://finance.yahoo.com",
        "Referer": f"https://finance.yahoo.com/quote/{ticker}/",
    }
    params = {"interval": interval, "range": yf_period, "includePrePost": "false"}

    for base in ["query2", "query1"]:
        try:
            url  = f"https://{base}.finance.yahoo.com/v8/finance/chart/{ticker}"
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            data   = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                continue
            chart      = result[0]
            timestamps = chart.get("timestamp", [])
            ohlcv      = chart.get("indicators", {}).get("quote", [{}])[0]
            closes  = ohlcv.get("close",  [None] * len(timestamps))
            opens   = ohlcv.get("open",   [None] * len(timestamps))
            highs   = ohlcv.get("high",   [None] * len(timestamps))
            lows    = ohlcv.get("low",    [None] * len(timestamps))
            volumes = ohlcv.get("volume", [0]    * len(timestamps))

            rows, dates = [], []
            for i, ts in enumerate(timestamps):
                try:
                    c = closes[i]
                    if c is None:
                        continue
                    rows.append({
                        "Open":   float(opens[i]   or 0),
                        "High":   float(highs[i]   or 0),
                        "Low":    float(lows[i]    or 0),
                        "Close":  float(c),
                        "Volume": int(volumes[i]   or 0) if i < len(volumes) else 0,
                    })
                    dates.append(pd.Timestamp(ts, unit="s", tz="UTC").normalize())
                except (TypeError, IndexError):
                    continue
            if rows:
                return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
        except Exception as e:
            print(f"fetch_yahoo_v8 {base} {ticker}: {e}")
            continue
    return pd.DataFrame()


def get_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Obtiene histórico con caché de 10 min: yf.download → tk.history."""
    cache_key = f"{ticker}_{period}"
    cached = _cache_get(_history_cache, cache_key, _HISTORY_CACHE_TTL)
    if cached is not None:
        return cached

    # Intento 1: yf.download (más rápido y confiable)
    try:
        data = yf.download(
            ticker, period=period, auto_adjust=True,
            progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if not data.empty:
            _cache_set(_history_cache, cache_key, data)
            return data
    except Exception:
        pass

    # Intento 2: tk.history
    try:
        result = yft(ticker).history(period=period)
        if not result.empty:
            _cache_set(_history_cache, cache_key, result)
            return result
    except Exception:
        pass

    return pd.DataFrame()


# ─── Análisis técnico ─────────────────────────────────────────────────────────
@app.get("/analisis/{ticker}")
def obtener_analisis(ticker: str):
    cached = _cache_get(_endpoint_cache, f"analisis_{ticker}", _ENDPOINT_CACHE_TTL)
    if cached is not None:
        return cached
    try:
        data = get_history(ticker, "1y")
        if data.empty or len(data) < 50:
            return {"error": "Datos insuficientes"}

        close    = data['Close']
        retornos = close.pct_change().dropna()

        sma20         = round(float(close.rolling(20).mean().iloc[-1]), 2)
        sma50         = round(float(close.rolling(50).mean().iloc[-1]), 2)
        precio_actual = round(float(close.iloc[-1]), 2)

        # Volatilidad anualizada
        volatilidad = round(float(retornos.std() * np.sqrt(252) * 100), 2)

        # RSI (14)
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss
        rsi   = round(float(100 - (100 / (1 + rs.iloc[-1]))), 2)

        # Beta vs SPY
        beta = 1.0
        try:
            spy     = get_history("SPY", "1y")
            spy_ret = spy['Close'].pct_change().dropna()
            common  = retornos.index.intersection(spy_ret.index)
            if len(common) > 20:
                r   = retornos.loc[common].values
                s   = spy_ret.loc[common].values
                cov = float(np.cov(r, s)[0][1])
                var = float(np.var(s))
                beta = round(cov / var, 2) if var != 0 else 1.0
        except Exception:
            pass

        if precio_actual > sma20 > sma50:
            tendencia = "ALCISTA"
        elif precio_actual < sma20 < sma50:
            tendencia = "BAJISTA"
        else:
            tendencia = "LATERAL"

        result = {
            "precio_actual": precio_actual,
            "sma20":         sma20,
            "sma50":         sma50,
            "volatilidad":   volatilidad,
            "beta":          beta,
            "rsi":           rsi,
            "tendencia":     tendencia,
        }
        _cache_set(_endpoint_cache, f"analisis_{ticker}", result)
        return result
    except Exception as e:
        print(f"Error analisis: {e}")
        return {"error": str(e)}


# ─── Monte Carlo ──────────────────────────────────────────────────────────────
@app.get("/montecarlo/{ticker}")
def obtener_montecarlo(ticker: str, dias: int = Query(30)):
    cache_key = f"montecarlo_{ticker}_{dias}"
    cached = _cache_get(_endpoint_cache, cache_key, _ENDPOINT_CACHE_TTL)
    if cached is not None:
        return cached
    try:
        data = get_history(ticker, "1y")
        if data.empty or len(data) < 30:
            return {"error": "Datos insuficientes"}

        close      = data['Close']
        retornos   = close.pct_change().dropna()
        mu         = float(retornos.mean())
        sigma      = float(retornos.std())
        precio_hoy = float(close.iloc[-1])

        np.random.seed(42)
        n_sim        = 200  # reducido para velocidad
        simulaciones = np.zeros((dias, n_sim))
        for i in range(n_sim):
            precios = [precio_hoy]
            for _ in range(dias - 1):
                precios.append(precios[-1] * (1 + np.random.normal(mu, sigma)))
            simulaciones[:, i] = precios

        result = {
            "dias": list(range(1, dias + 1)),
            "p5":  [round(float(np.percentile(simulaciones[d],  5)), 2) for d in range(dias)],
            "p50": [round(float(np.percentile(simulaciones[d], 50)), 2) for d in range(dias)],
            "p95": [round(float(np.percentile(simulaciones[d], 95)), 2) for d in range(dias)],
            "precio_actual": round(precio_hoy, 2),
        }
        _cache_set(_endpoint_cache, cache_key, result)
        return result
    except Exception as e:
        print(f"Error Monte Carlo: {e}")
        return {"error": str(e)}


# ─── Noticias ─────────────────────────────────────────────────────────────────
@app.get("/noticias/{ticker}")
def obtener_noticias(ticker: str):
    try:
        tk       = yft(ticker)
        raw_news = tk.news
        noticias_limpias = []
        for n in raw_news[:5]:
            content  = n.get('content', {}) or {}
            titulo   = n.get('title') or content.get('title')
            raw_link = n.get('link') or content.get('clickThroughUrl') or content.get('canonicalUrl')
            if isinstance(raw_link, dict):
                link = raw_link.get('url', '')
            else:
                link = raw_link or ''
            if link and not link.startswith('http'):
                link = f"https://finance.yahoo.com{link}"
            provider = n.get('provider') or content.get('provider') or {}
            fuente   = n.get('publisher') or (provider.get('displayName') if isinstance(provider, dict) else None) or 'Yahoo Finance'
            if titulo and link:
                noticias_limpias.append({"titulo": titulo, "link": link, "fuente": fuente})
        return noticias_limpias
    except Exception as e:
        print(f"Error noticias: {e}")
        return []


# ─── Operaciones ──────────────────────────────────────────────────────────────
@app.post("/comprar")
def registrar_compra(data: TransaccionSchema):
    session = Session()
    try:
        tx = Transaccion(ticker=data.ticker.upper().strip(), tipo_operacion='Compra',
                         cantidad=data.cantidad, precio_unitario=data.precio_unitario)
        session.add(tx)
        session.commit()
        return {"status": "success"}
    except Exception as e:
        session.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        session.close()


@app.post("/vender")
def registrar_venta(data: TransaccionSchema):
    session = Session()
    try:
        tx = Transaccion(ticker=data.ticker.upper().strip(), tipo_operacion='Venta',
                         cantidad=data.cantidad, precio_unitario=data.precio_unitario)
        session.add(tx)
        session.commit()
        return {"status": "success"}
    except Exception as e:
        session.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        session.close()


@app.delete("/borrar/{ticker}")
def borrar_activo(ticker: str):
    session = Session()
    try:
        session.query(Transaccion).filter_by(ticker=ticker.upper()).delete()
        session.commit()
        return {"status": "success"}
    finally:
        session.close()



# ── IA / Análisis de Sentimiento ─────────────────────────────────────────────
@app.get("/ia/{ticker}")
def analisis_ia(ticker: str):
    cached = _cache_get(_endpoint_cache, f"ia_{ticker}", _ENDPOINT_CACHE_TTL)
    if cached is not None:
        return cached
    try:
        tk = yft(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        # Precio actual con fallback a fast_info
        precio_actual = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)
        if precio_actual == 0:
            try:
                precio_actual = float(tk.fast_info.last_price or 0)
            except Exception:
                pass

        target_mean   = float(info.get('targetMeanPrice') or 0)
        target_high   = float(info.get('targetHighPrice') or 0)
        target_low    = float(info.get('targetLowPrice') or 0)
        n_analistas   = int(info.get('numberOfAnalystOpinions') or 0)
        rec_mean      = float(info.get('recommendationMean') or 3.0)
        rec_key       = info.get('recommendationKey', 'hold') or 'hold'

        # Fallback recommendationMean con recommendations_summary si info no lo trajo
        if rec_mean == 3.0 and info.get('recommendationMean') is None:
            try:
                rs = tk.recommendations_summary
                if rs is not None and not rs.empty:
                    row = rs.iloc[-1]
                    total = sum(int(row.get(c, 0) or 0) for c in ['strongBuy','buy','hold','sell','strongSell'])
                    if total > 0:
                        n_analistas = n_analistas or total
            except Exception:
                pass

        # Fallback con analyst_price_targets si info no trajo targets
        if target_mean == 0:
            try:
                apt = tk.analyst_price_targets
                if apt is not None:
                    apt_d = apt if isinstance(apt, dict) else apt.to_dict()
                    target_mean = float(apt_d.get('mean') or apt_d.get('targetMeanPrice') or 0)
                    target_high = float(apt_d.get('high') or apt_d.get('targetHighPrice') or target_high)
                    target_low  = float(apt_d.get('low')  or apt_d.get('targetLowPrice')  or target_low)
                    if target_mean and n_analistas == 0:
                        n_analistas = int(apt_d.get('numberOfAnalysts') or apt_d.get('numberOfAnalystOpinions') or 0)
            except Exception:
                pass

        upside = ((target_mean - precio_actual) / precio_actual * 100) if precio_actual and target_mean else 0

        # Score analistas: recommendationMean 1=Strong Buy→100, 5=Strong Sell→0
        score_analistas = round((5 - rec_mean) / 4 * 100, 1)

        # Score precio objetivo: upside -30% → 0, 0% → 50, +30% → 100
        score_target = min(max((upside + 30) / 60 * 100, 0), 100) if target_mean else 50

        # Score técnico basado en tendencia de 6 meses
        score_tecnico = 50
        try:
            data = get_history(ticker, "6mo")
            if not data.empty and len(data) >= 50:
                close    = data['Close']
                sma20    = float(close.rolling(20).mean().iloc[-1])
                sma50    = float(close.rolling(50).mean().iloc[-1])
                precio_c = float(close.iloc[-1])
                delta    = close.diff()
                gain     = delta.where(delta > 0, 0).rolling(14).mean()
                loss     = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rsi      = float(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1]))) if loss.iloc[-1] != 0 else 50
                if precio_c > sma20 > sma50:
                    score_tecnico = 75
                elif precio_c < sma20 < sma50:
                    score_tecnico = 25
                else:
                    score_tecnico = 50
                if rsi < 30:   score_tecnico = min(score_tecnico + 15, 100)
                elif rsi > 70: score_tecnico = max(score_tecnico - 15, 0)
        except Exception:
            pass

        # Breakdown de recomendaciones de analistas
        breakdown = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
        try:
            recs = tk.recommendations
            if recs is not None and not recs.empty:
                # En yfinance nuevo el DataFrame tiene columna 'period'
                # '0m' = mes actual (más reciente), '-1m' = mes anterior
                if 'period' in recs.columns:
                    cur = recs[recs['period'] == '0m']
                    latest = cur.iloc[0] if not cur.empty else recs.iloc[0]
                else:
                    # Formato viejo: índice datetime, primer registro es el más reciente
                    latest = recs.iloc[0]
                for k in breakdown:
                    breakdown[k] = int(latest.get(k, 0) or 0)
        except Exception:
            pass

        # Score compuesto (ponderado)
        if n_analistas > 0:
            score_final = round(score_analistas * 0.4 + score_target * 0.3 + score_tecnico * 0.3, 1)
        else:
            score_final = round(score_target * 0.5 + score_tecnico * 0.5, 1)

        if score_final >= 70:   label = "COMPRAR"
        elif score_final >= 55: label = "PERSPECTIVA POSITIVA"
        elif score_final >= 40: label = "NEUTRAL"
        elif score_final >= 25: label = "PRECAUCIÓN"
        else:                   label = "VENDER"

        result = {
            "score":         score_final,
            "label":         label,
            "precio_actual": round(precio_actual, 2),
            "target_bajo":   round(target_low, 2),
            "target_medio":  round(target_mean, 2),
            "target_alto":   round(target_high, 2),
            "upside_pct":    round(upside, 2),
            "n_analistas":   n_analistas,
            "recomendacion": rec_key,
            "breakdown":     breakdown,
        }
        _cache_set(_endpoint_cache, f"ia_{ticker}", result)
        return result
    except Exception as e:
        print(f"Error IA {ticker}: {e}")
        return {"error": str(e)}


# ── Helper: curl_cffi con crumb (compartido por calendario e indicadores) ─────
_cffi_crumb: dict = {"crumb": None, "ts": 0.0}

def _get_crumb() -> tuple:
    """Devuelve (cffi_session, crumb). Cachea el crumb 30 min."""
    from curl_cffi import requests as cffi_requests
    now = time.time()
    if _cffi_crumb["crumb"] and now - _cffi_crumb["ts"] < 1800:
        s = cffi_requests.Session(impersonate="chrome110")
        return s, _cffi_crumb["crumb"]
    s = cffi_requests.Session(impersonate="chrome110")
    s.get("https://fc.yahoo.com", timeout=8)
    r = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=8)
    crumb = r.text.strip()
    _cffi_crumb["crumb"] = crumb
    _cffi_crumb["ts"]    = now
    return s, crumb

def fetch_quote_summary(ticker: str, modules: list) -> dict:
    """Llama a quoteSummary de Yahoo Finance via curl_cffi con crumb."""
    try:
        s, crumb = _get_crumb()
        mods = ",".join(modules)
        url  = (f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
                f"?modules={mods}&crumb={crumb}")
        r    = s.get(url, timeout=12)
        data = r.json()
        results = data.get("quoteSummary", {}).get("result", [])
        if results:
            return results[0]
    except Exception as e:
        print(f"fetch_quote_summary {ticker}: {e}")
    return {}


# ── Calendario de Earnings ────────────────────────────────────────────────────
def fetch_calendar_cffi(ticker: str) -> dict:
    """Usa curl_cffi para impersonar Chrome y obtener calendarEvents con crumb."""
    try:
        data = fetch_quote_summary(ticker, ["calendarEvents"])
        return data.get("calendarEvents", {})
    except Exception as e:
        print(f"fetch_calendar_cffi {ticker}: {e}")
    return {}



@app.get("/calendario")
def calendario_earnings(tickers: str = Query(default="")):
    """Recibe tickers como query param: /calendario?tickers=AAPL,KEEL,MSFT"""
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    else:
        session = Session()
        try:
            ticker_list = [t for (t,) in session.query(Transaccion.ticker).distinct().all()]
        finally:
            session.close()

    eventos = []
    now     = pd.Timestamp.now(tz='UTC')

    def safe_f(v):
        try: return round(float(v), 4) if v is not None and v == v else None
        except: return None

    def get_raw(obj, key):
        v = obj.get(key, {})
        if isinstance(v, dict): return v.get("raw")
        return v

    for ticker in ticker_list:
        # ── Crypto no tiene earnings → saltar ────────────────────────────────
        sym_base = ticker.upper().replace('-USD','').replace('/USD','')
        if sym_base in CRYPTO_SYMBOLS:
            continue
        try:
            fecha_str = None
            eps_est   = None
            eps_alto  = None
            eps_bajo  = None
            rev_est   = None

            # ── Método 1: tk.calendar via yfinance (usa curl_cffi internamente)
            try:
                tk  = yft(ticker)
                cal = tk.calendar
                if cal is not None and isinstance(cal, dict):
                    fechas = cal.get('Earnings Date', [])
                    if not isinstance(fechas, list):
                        fechas = [fechas]
                    for f in fechas:
                        if hasattr(f, 'strftime'):
                            ts = pd.Timestamp(f)
                            if ts.tzinfo is None:
                                ts = ts.tz_localize('UTC')
                            if ts >= now:
                                fecha_str = f.strftime('%Y-%m-%d')
                                break
                    eps_est  = cal.get('Earnings Average')
                    eps_alto = cal.get('Earnings High')
                    eps_bajo = cal.get('Earnings Low')
                    rev_est  = cal.get('Revenue Average')
            except Exception:
                pass

            # ── Método 2: earnings_dates como fallback ────────────────────────
            if not fecha_str:
                try:
                    tk = yft(ticker)
                    ed = tk.earnings_dates
                    if ed is not None and not ed.empty:
                        future = ed[ed.index >= now]
                        if not future.empty:
                            fecha_str = future.index[0].strftime('%Y-%m-%d')
                            col_eps = [c for c in future.columns if 'EPS' in str(c) and 'Estimate' in str(c)]
                            if col_eps and not eps_est:
                                eps_est = future[col_eps[0]].iloc[0]
                except Exception:
                    pass

            if fecha_str:
                eventos.append({
                    "ticker":           ticker,
                    "fecha":            fecha_str,
                    "eps_estimado":     safe_f(eps_est),
                    "eps_alto":         safe_f(eps_alto),
                    "eps_bajo":         safe_f(eps_bajo),
                    "revenue_estimado": int(rev_est) if rev_est is not None and rev_est == rev_est else None,
                })
            else:
                print(f"Calendario {ticker}: sin fecha de earnings disponible")

        except Exception as ex:
            print(f"Error calendario {ticker}: {ex}")

    return sorted(eventos, key=lambda x: x['fecha'])


# ── Calendario de Earnings del Mercado (lista curada, no portafolio) ──────────
_EARNINGS_WATCHLIST = [
    # Mega caps tech
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA",
    # Financials
    "JPM","BAC","GS","V","MA",
    # Healthcare
    "JNJ","LLY","PFE","UNH",
    # Consumer
    "WMT","COST","MCD","KO","NKE",
    # Tech/Semis/Cloud
    "AMD","AVGO","QCOM","NFLX","CRM","ORCL",
    # Argentina / LATAM
    "MELI","NU","GGAL","YPF",
]

def _fetch_calendario_one(ticker: str, now: pd.Timestamp) -> Optional[dict]:
    """Busca la próxima fecha de earnings para un ticker. Retorna dict o None."""
    try:
        tk = yft(ticker)
        # Intento 1: tk.calendar
        try:
            cal = tk.calendar
            if cal and isinstance(cal, dict):
                fechas = cal.get('Earnings Date', [])
                if not isinstance(fechas, list):
                    fechas = [fechas]
                for f in fechas:
                    if hasattr(f, 'strftime'):
                        ts = pd.Timestamp(f)
                        if ts.tzinfo is None:
                            ts = ts.tz_localize('UTC')
                        if ts >= now:
                            return {
                                "ticker":           ticker,
                                "fecha":            f.strftime('%Y-%m-%d'),
                                "eps_estimado":     _safe_num(cal.get('Earnings Average')),
                                "revenue_estimado": _safe_int(cal.get('Revenue Average')),
                            }
        except Exception:
            pass
        # Intento 2: earnings_dates
        try:
            ed = tk.earnings_dates
            if ed is not None and not ed.empty:
                future = ed[ed.index >= now]
                if not future.empty:
                    fecha_str = future.index[0].strftime('%Y-%m-%d')
                    eps_col = [c for c in future.columns if 'Estimate' in str(c) and 'EPS' in str(c)]
                    eps_val = None
                    if eps_col:
                        v = future[eps_col[0]].iloc[0]
                        try: eps_val = round(float(v), 4) if v == v else None
                        except: pass
                    return {"ticker": ticker, "fecha": fecha_str, "eps_estimado": eps_val, "revenue_estimado": None}
        except Exception:
            pass
    except Exception:
        pass
    return None

def _safe_num(v):
    try: return round(float(v), 4) if v is not None and v == v else None
    except: return None

def _safe_int(v):
    try: return int(v) if v is not None and v == v else None
    except: return None

@app.get("/calendario-mercado")
def calendario_mercado():
    """Próximos earnings de una lista curada de ~60 empresas del mercado."""
    cached = _cache_get(_endpoint_cache, "calendario_mercado", 21600)  # 6h cache
    if cached is not None:
        return cached
    now = pd.Timestamp.now(tz='UTC')
    cutoff = now + pd.Timedelta(days=60)  # Solo próximos 60 días
    eventos = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_fetch_calendario_one, t, now): t for t in _EARNINGS_WATCHLIST}
        for future in as_completed(futures, timeout=45):
            try:
                result = future.result()
                if result:
                    ts = pd.Timestamp(result["fecha"])
                    if ts.tzinfo is None:
                        ts = ts.tz_localize('UTC')
                    if ts <= cutoff:
                        eventos.append(result)
            except Exception:
                pass
    eventos = sorted(eventos, key=lambda x: x['fecha'])
    _cache_set(_endpoint_cache, "calendario_mercado", eventos)
    return eventos


# ── Reporte Detallado de Earnings ─────────────────────────────────────────────
@app.get("/earnings/{ticker}")
def earnings_report(ticker: str):
    try:
        tk   = yft(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        def safe(v, decimals=2):
            try: return round(float(v), decimals) if v is not None and v == v else None
            except: return None

        # EPS histórico (últimos 8 trimestres)
        eps_hist = []
        try:
            eh = tk.earnings_history
            if eh is not None and not eh.empty:
                for idx, row in eh.tail(8).iterrows():
                    eps_hist.append({
                        "fecha":    idx.strftime('%b %Y') if hasattr(idx, 'strftime') else str(idx)[:10],
                        "estimado": safe(row.get('epsEstimate'), 4),
                        "real":     safe(row.get('epsActual'), 4),
                        "sorpresa": safe(row.get('surprisePercent') * 100 if row.get('surprisePercent') is not None else None, 2),
                    })
        except Exception:
            pass

        # ── fast_info: price, 52w, market_cap (confiable en Render) ─────────
        fi_price = fi_high = fi_low = fi_mcap = None
        try:
            fi2       = tk.fast_info
            fi_price  = float(fi2.last_price  or 0)
            fi_high   = safe(fi2.year_high)
            fi_low    = safe(fi2.year_low)
            fi_mcap   = fi2.market_cap
        except Exception:
            pass

        # ── Próximos earnings desde calendar (envuelto en try/except) ────────
        prox_fecha = None
        eps_prox_bajo   = None
        eps_prox_medio  = None
        eps_prox_alto   = None
        rev_prox        = None
        try:
            cal = tk.calendar
            if cal and isinstance(cal, dict):
                fechas = cal.get('Earnings Date', [])
                if not isinstance(fechas, list): fechas = [fechas]
                if fechas and hasattr(fechas[0], 'strftime'):
                    prox_fecha = fechas[0].strftime('%d/%m/%Y')
                eps_prox_bajo  = safe(cal.get('Earnings Low'),     4)
                eps_prox_medio = safe(cal.get('Earnings Average'), 4)
                eps_prox_alto  = safe(cal.get('Earnings High'),    4)
                rv = cal.get('Revenue Average')
                rev_prox = int(rv) if rv is not None and rv == rv else None
        except Exception:
            pass

        # ── EPS TTM desde income_stmt si tk.info no trajo datos ──────────────
        eps_ttm_calc = safe(info.get('trailingEps'), 4)
        if eps_ttm_calc is None and fi_price:
            try:
                inc2 = tk.income_stmt
                ni2 = None
                sh2 = None
                for lbl in inc2.index:
                    if 'net income' in str(lbl).lower() and 'minority' not in str(lbl).lower():
                        ni2 = float(inc2.loc[lbl].iloc[0]); break
                fi3 = tk.fast_info
                sh2 = float(fi3.shares or 0)
                if ni2 and sh2 > 0:
                    eps_ttm_calc = safe(ni2 / sh2, 4)
            except Exception:
                pass

        pe_t = safe(info.get('trailingPE')) or (safe(fi_price / eps_ttm_calc, 2) if eps_ttm_calc and eps_ttm_calc > 0 else None)
        dy = info.get('dividendYield')

        return {
            "nombre":              info.get('longName', ticker),
            "sector":              info.get('sector', '—'),
            "industria":           info.get('industry', '—'),
            "descripcion":         (info.get('longBusinessSummary') or '')[:500],
            "pe_trailing":         pe_t,
            "pe_forward":          safe(info.get('forwardPE')),
            "eps_ttm":             eps_ttm_calc,
            "eps_forward":         safe(info.get('forwardEps'), 4),
            "market_cap":          fi_mcap or info.get('marketCap'),
            "dividendo_pct":       safe(dy * 100 if dy else None),
            "high_52w":            fi_high  or safe(info.get('fiftyTwoWeekHigh')),
            "low_52w":             fi_low   or safe(info.get('fiftyTwoWeekLow')),
            "target_bajo":         safe(info.get('targetLowPrice')),
            "target_medio":        safe(info.get('targetMeanPrice')),
            "target_alto":         safe(info.get('targetHighPrice')),
            "n_analistas":         info.get('numberOfAnalystOpinions'),
            "prox_earnings_fecha": prox_fecha,
            "prox_eps_bajo":       eps_prox_bajo,
            "prox_eps_medio":      eps_prox_medio,
            "prox_eps_alto":       eps_prox_alto,
            "prox_rev_est":        rev_prox,
            "eps_historico":       eps_hist,
        }
    except Exception as e:
        print(f"Error earnings {ticker}: {e}")
        return {"error": str(e)}



# ── Crypto Info via CoinGecko (gratis, sin API key) ──────────────────────────
@app.get("/crypto-info/{symbol}")
def crypto_info_endpoint(symbol: str):
    sym = symbol.upper().replace('-USD','').replace('/USD','')
    cached = _cache_get(_endpoint_cache, f"crypto_{sym}", 300)
    if cached is not None:
        return cached

    # Resolver ID de CoinGecko
    cg_id = COINGECKO_IDS.get(sym)
    if not cg_id:
        try:
            r = requests.get(
                f"{COINGECKO_BASE}/search?query={sym}",
                headers={'User-Agent':'Mozilla/5.0'},
                timeout=10
            )
            coins = r.json().get('coins', [])
            if coins:
                cg_id = coins[0]['id']
        except Exception:
            pass
    if not cg_id:
        return {"error": f"Crypto '{sym}' no encontrada"}

    try:
        r = requests.get(
            f"{COINGECKO_BASE}/coins/{cg_id}"
            "?localization=false&tickers=false&community_data=false&developer_data=false",
            headers={'User-Agent':'Mozilla/5.0'},
            timeout=15
        )
        d = r.json()
        m = d.get('market_data', {})
        def fg(field, sub='usd'):
            v = m.get(field, {})
            return v.get(sub) if isinstance(v, dict) else v

        result = {
            "nombre":             d.get('name', sym),
            "simbolo":            (d.get('symbol') or sym).upper(),
            "logo_url":           d.get('image', {}).get('large'),
            "rank":               d.get('market_cap_rank'),
            "precio_usd":         fg('current_price'),
            "market_cap":         fg('market_cap'),
            "volume_24h":         fg('total_volume'),
            "cambio_1h":          round(m.get('price_change_percentage_1h_in_currency', {}).get('usd') or 0, 2),
            "cambio_24h":         round(m.get('price_change_percentage_24h') or 0, 2),
            "cambio_7d":          round(m.get('price_change_percentage_7d') or 0, 2),
            "cambio_30d":         round(m.get('price_change_percentage_30d') or 0, 2),
            "high_24h":           fg('high_24h'),
            "low_24h":            fg('low_24h'),
            "ath":                fg('ath'),
            "ath_date":           (fg('ath_date') or '')[:10],
            "circulating_supply": m.get('circulating_supply'),
            "total_supply":       m.get('total_supply'),
            "max_supply":         m.get('max_supply'),
            "descripcion":        (d.get('description', {}).get('en') or '')[:400],
            "es_crypto":          True,
        }
        _cache_set(_endpoint_cache, f"crypto_{sym}", result)
        return result
    except Exception as e:
        print(f"Error crypto-info {sym}: {e}")
        return {"error": str(e)}


# ── Indicadores Financieros Fundamentales ────────────────────────────────────
@app.get("/indicadores/{ticker}")
def indicadores_financieros(ticker: str):
    cached = _cache_get(_endpoint_cache, f"indicadores_{ticker}", _INDICADORES_TTL)
    if cached is not None:
        return cached
    try:
        tk = yft(ticker)

        def sn(v, d=4):
            try: return round(float(v), d) if v is not None and v == v else None
            except: return None

        # ── Precio actual (fast_info — confiable en Render) ───────────────────
        price  = 0.0
        shares = 0.0
        try:
            fi     = tk.fast_info
            price  = float(fi.last_price or 0)
            shares = float(fi.shares      or 0)
        except Exception:
            pass

        # ── Balance sheet (confiable en Render) ───────────────────────────────
        td = ta = se = cur_assets = cur_liab = None
        try:
            bs = tk.balance_sheet
            if bs is not None and not bs.empty:
                for lbl in bs.index:
                    ls = str(lbl).lower()
                    v  = bs.loc[lbl].iloc[0]
                    if   'total debt'          in ls: td        = float(v)
                    elif 'total assets'        in ls: ta        = float(v)
                    elif 'stockholders equity' in ls or 'total equity gross' in ls: se = float(v)
                    elif 'current assets'      in ls: cur_assets = float(v)
                    elif 'current liabilities' in ls: cur_liab  = float(v)
        except Exception:
            pass

        # ── Income statement (confiable en Render) ────────────────────────────
        revenue = revenue_prev = net_income = gross_profit = op_income = ebitda = None
        try:
            inc = tk.income_stmt
            if inc is not None and not inc.empty:
                for lbl in inc.index:
                    ls = str(lbl).lower()
                    cols = inc.loc[lbl].dropna()
                    if not len(cols): continue
                    v0 = float(cols.iloc[0])
                    if   'total revenue'    in ls:
                        revenue = v0
                        if len(cols) >= 2: revenue_prev = float(cols.iloc[1])
                    elif 'net income' in ls and 'minority' not in ls and 'common' not in ls:
                        if net_income is None: net_income = v0
                    elif 'gross profit'     in ls: gross_profit = v0
                    elif 'operating income' in ls or 'ebit ' in ls: op_income = v0
                    elif 'ebitda'           in ls: ebitda = v0
        except Exception:
            pass

        # ── Cash flow (confiable en Render) ───────────────────────────────────
        ocf = None
        try:
            cf = tk.cash_flow
            if cf is not None and not cf.empty:
                for lbl in cf.index:
                    if 'operating' in str(lbl).lower() and 'cash' in str(lbl).lower():
                        ocf = float(cf.loc[lbl].iloc[0])
                        break
        except Exception:
            pass

        # ── Calcular ratios ───────────────────────────────────────────────────
        roa            = sn(net_income / ta,      4) if net_income and ta   and ta   > 0 else None
        roe            = sn(net_income / se,      4) if net_income and se   and se   > 0 else None
        net_margin     = sn(net_income / revenue, 4) if net_income and revenue       > 0 else None
        gross_margin   = sn(gross_profit / revenue, 4) if gross_profit and revenue   > 0 else None
        op_margin      = sn(op_income / revenue,  4) if op_income   and revenue      > 0 else None
        ebitda_margin  = sn(ebitda    / revenue,  4) if ebitda      and revenue      > 0 else None
        revenue_growth = sn((revenue - revenue_prev) / abs(revenue_prev), 4) if revenue and revenue_prev and revenue_prev != 0 else None
        debt_to_asset  = sn(td / ta, 4) if td is not None and ta and ta > 0 else None
        debt_to_equity = sn(td / se, 2) if td is not None and se and se > 0 else None
        current_ratio  = sn(cur_assets / cur_liab, 2) if cur_assets and cur_liab and cur_liab > 0 else None
        bvps           = sn(se / shares, 2) if se and shares > 0 else None
        cfps           = sn(ocf / shares, 4) if ocf and shares > 0 else None
        eps_ttm        = sn(net_income / shares, 4) if net_income and shares > 0 else None
        pe_trailing    = sn(price / eps_ttm, 2) if price and eps_ttm and eps_ttm > 0 else None
        pb             = sn(price / (se / shares), 2) if price and se and shares > 0 else None

        # ── Datos opcionales de tk.info (puede fallar en Render — no es crítico)
        eps_forward = pe_forward = peg = earnings_growth = None
        sector = "—"; nombre = ticker; website = ""
        try:
            info = tk.info or {}
            eps_forward     = sn(info.get('forwardEps'),       4)
            pe_forward      = sn(info.get('forwardPE'),        2)
            peg             = sn(info.get('pegRatio'),         2)
            earnings_growth = sn(info.get('earningsGrowth'))
            sector  = info.get('sector', '—') or '—'
            nombre  = info.get('longName', ticker) or ticker
            website = info.get('website', '') or ''
        except Exception:
            pass

        # Pe forward alternativo si tk.info falló
        if pe_forward is None and eps_forward and price and eps_forward > 0:
            pe_forward = sn(price / eps_forward, 2)

        domain   = website.replace('https://','').replace('http://','').split('/')[0]
        logo_url = f"https://logo.clearbit.com/{domain}" if domain else None

        result = {
            "ticker":           ticker.upper(),
            "nombre":           nombre,
            "sector":           sector,
            "logo_url":         logo_url,
            "roa":              roa,
            "roe":              roe,
            "eps_ttm":          eps_ttm,
            "eps_forward":      eps_forward,
            "net_margin":       net_margin,
            "gross_margin":     gross_margin,
            "operating_margin": op_margin,
            "ebitda_margin":    ebitda_margin,
            "debt_to_asset":    debt_to_asset,
            "debt_to_equity":   debt_to_equity,
            "current_ratio":    current_ratio,
            "bvps":             bvps,
            "cfps":             cfps,
            "pe_trailing":      pe_trailing,
            "pe_forward":       pe_forward,
            "peg_ratio":        peg,
            "price_to_book":    pb,
            "revenue_growth":   revenue_growth,
            "earnings_growth":  earnings_growth,
        }
        _cache_set(_endpoint_cache, f"indicadores_{ticker}", result)
        return result
    except Exception as e:
        print(f"Error indicadores {ticker}: {e}")
        return {"error": str(e)}


# ── Order Book (bid/ask reales + niveles sintéticos) ─────────────────────────
@app.get("/orderbook/{ticker}")
def order_book(ticker: str):
    try:
        tk = yft(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        bid      = float(info.get('bid', 0) or 0)
        ask      = float(info.get('ask', 0) or 0)
        bid_size = int(info.get('bidSize', 0) or 0)
        ask_size = int(info.get('askSize', 0) or 0)

        precio = float(info.get('currentPrice') or info.get('regularMarketPrice') or
                       info.get('previousClose') or 0)
        # Fallback a fast_info para el precio (más rápido y confiable)
        if precio == 0:
            try:
                precio = float(tk.fast_info.last_price or 0)
            except Exception:
                pass
        if bid == 0: bid = round(precio * 0.9995, 2)
        if ask == 0: ask = round(precio * 1.0005, 2)
        if bid_size == 0: bid_size = 100
        if ask_size == 0: ask_size = 100

        spread = round(ask - bid, 4)
        tick   = round(precio * 0.001, 4) if precio > 0 else 0.01

        np.random.seed(int(pd.Timestamp.now().timestamp()) % 1000)

        # Generar 8 niveles de bids (decrecientes desde bid)
        bids = []
        vol_acum = bid_size
        for i in range(8):
            nivel_precio = round(bid - tick * i, 2)
            factor = max(0.3, 1 - i * 0.1) * np.random.uniform(0.7, 1.3)
            nivel_vol = max(100, int(bid_size * factor))
            vol_acum += nivel_vol
            bids.append({"precio": nivel_precio, "volumen": nivel_vol, "acumulado": vol_acum})

        # Generar 8 niveles de asks (crecientes desde ask)
        asks = []
        vol_acum = ask_size
        for i in range(8):
            nivel_precio = round(ask + tick * i, 2)
            factor = max(0.3, 1 - i * 0.1) * np.random.uniform(0.7, 1.3)
            nivel_vol = max(100, int(ask_size * factor))
            vol_acum += nivel_vol
            asks.append({"precio": nivel_precio, "volumen": nivel_vol, "acumulado": vol_acum})

        # Volume profile intraday (última sesión, intervalos de 1h)
        volume_profile = []
        try:
            hist_1d = tk.history(period="1d", interval="1h")
            if not hist_1d.empty:
                precio_min = float(hist_1d['Low'].min())
                precio_max = float(hist_1d['High'].max())
                rango      = precio_max - precio_min
                n_bins     = 10
                bin_size   = rango / n_bins if rango > 0 else 1
                bins       = [0] * n_bins
                for _, row in hist_1d.iterrows():
                    mid = (float(row['High']) + float(row['Low'])) / 2
                    idx = min(int((mid - precio_min) / bin_size), n_bins - 1)
                    bins[idx] += int(row['Volume'])
                max_vol = max(bins) if max(bins) > 0 else 1
                for i, v in enumerate(bins):
                    p = round(precio_min + (i + 0.5) * bin_size, 2)
                    volume_profile.append({
                        "precio": p,
                        "volumen": v,
                        "pct": round(v / max_vol * 100, 1)
                    })
        except Exception:
            pass

        return {
            "bid":            bid,
            "ask":            ask,
            "spread":         spread,
            "bid_size":       bid_size,
            "ask_size":       ask_size,
            "precio_actual":  round(precio, 2),
            "bids":           bids,
            "asks":           asks,
            "volume_profile": volume_profile,
        }
    except Exception as e:
        print(f"Error orderbook {ticker}: {e}")
        return {"error": str(e)}


# ── Panel de Mercado: Índices MERVAL, S&P 500, NASDAQ ────────────────────────
_INDICES_DEF = [
    {"nombre": "S&P 500",     "sym": "^GSPC", "color": "blue"},
    {"nombre": "NASDAQ",      "sym": "^IXIC", "color": "purple"},
    {"nombre": "MERVAL",      "sym": "^MERV", "color": "emerald"},
    {"nombre": "Dow Jones",   "sym": "^DJI",  "color": "yellow"},
]

@app.get("/mercado")
def panel_mercado():
    cached = _cache_get(_endpoint_cache, "mercado_indices", 300)
    if cached is not None:
        return cached
    result = []
    for idx in _INDICES_DEF:
        try:
            sym = idx["sym"]
            # Histórico 1 año para calcular cambios mensuales/anuales y sparkline
            hist = yf.download(sym, period="1y", auto_adjust=True, progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if hist.empty:
                continue
            closes = hist["Close"].dropna()
            price  = float(closes.iloc[-1])
            prev   = float(closes.iloc[-2]) if len(closes) >= 2 else price
            # Cambio diario
            cambio_dia = round((price - prev) / prev * 100, 2) if prev else 0
            # Cambio mensual (~21 ruedas)
            mes_ago = float(closes.iloc[-22]) if len(closes) >= 22 else float(closes.iloc[0])
            cambio_mes = round((price - mes_ago) / mes_ago * 100, 2) if mes_ago else 0
            # Cambio anual
            anio_ago = float(closes.iloc[0])
            cambio_anio = round((price - anio_ago) / anio_ago * 100, 2) if anio_ago else 0
            # Sparkline: últimos 30 puntos normalizados (0-100)
            spark_raw = [float(x) for x in closes.iloc[-30:].tolist()]
            mn, mx = min(spark_raw), max(spark_raw)
            spark = [round((v - mn) / (mx - mn) * 100, 1) if mx > mn else 50 for v in spark_raw]

            result.append({
                "nombre":      idx["nombre"],
                "sym":         sym,
                "color":       idx["color"],
                "precio":      round(price, 2),
                "cambio_dia":  cambio_dia,
                "cambio_mes":  cambio_mes,
                "cambio_anio": cambio_anio,
                "sparkline":   spark,
            })
        except Exception as e:
            print(f"Error mercado {idx['sym']}: {e}")
    _cache_set(_endpoint_cache, "mercado_indices", result)
    return result


# ── Top Movers: 10 mayores subidas y bajadas del día (S&P 500 curado) ─────────
_SP500_CURADO = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","JNJ",
    "V","WMT","MA","PG","HD","CVX","LLY","ABBV","MRK","PEP","KO","AVGO",
    "CSCO","TMO","COST","ACN","DHR","NEE","LIN","MCD","TXN","UNH","CRM",
    "BAC","ORCL","ADBE","NFLX","INTC","INTU","QCOM","AMD","HON","IBM","GS",
    "CAT","BA","MMM","DE","GE","F","GM","DIS","PYPL","UBER","ABNB","SNOW",
    "PLTR","COIN","RBLX","HOOD","ROKU","ZM","SHOP","SQ","MELI","NU","SPOT",
]

@app.get("/top-movers")
def top_movers_endpoint():
    cached = _cache_get(_endpoint_cache, "top_movers", 600)
    if cached is not None:
        return cached
    try:
        tickers_str = " ".join(_SP500_CURADO)
        data = yf.download(tickers_str, period="5d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data[["Close"]]

        moves = []
        for t in _SP500_CURADO:
            try:
                if t not in closes.columns:
                    continue
                col = closes[t].dropna()
                if len(col) < 2:
                    continue
                price   = float(col.iloc[-1])
                prev    = float(col.iloc[-2])
                cambio  = round((price - prev) / prev * 100, 2) if prev else 0
                moves.append({"ticker": t, "precio": round(price, 2), "cambio_dia": cambio})
            except Exception:
                continue

        moves.sort(key=lambda x: x["cambio_dia"], reverse=True)
        result = {
            "ganadoras": moves[:10],
            "perdedoras": moves[-10:][::-1],
        }
        _cache_set(_endpoint_cache, "top_movers", result)
        return result
    except Exception as e:
        print(f"Error top-movers: {e}")
        return {"ganadoras": [], "perdedoras": []}


# ── Tendencias IA: acciones con potencial de crecimiento ─────────────────────
_TENDENCIAS_POOL = [
    "NVDA","MSFT","META","GOOGL","AMZN","TSLA","AAPL","AMD","AVGO","CRM",
    "PLTR","SNOW","MELI","NU","SHOP","UBER","ABNB","COIN","RBLX","SPOT",
    "LLY","NVO","ABBV","TMO","DHR","ISRG","DXCM","MRNA","REGN","VRTX",
    "NEE","ENPH","FSLR","BEP","SEDG","RUN","ARRY","CEG","VST","NRG",
    "MA","V","PYPL","INTU","SQ","ADBE","ORCL","SAP","NOW","WDAY",
]

@app.get("/tendencias-ia")
def tendencias_ia():
    cached = _cache_get(_endpoint_cache, "tendencias_ia", 3600)
    if cached is not None:
        return cached
    try:
        tickers_str = " ".join(_TENDENCIAS_POOL)
        data = yf.download(tickers_str, period="3mo", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
            volumes= data.get("Volume", pd.DataFrame())
        else:
            closes = data[["Close"]]
            volumes = pd.DataFrame()

        scored = []
        for t in _TENDENCIAS_POOL:
            try:
                if t not in closes.columns:
                    continue
                col = closes[t].dropna()
                if len(col) < 20:
                    continue

                price    = float(col.iloc[-1])
                p1m_ago  = float(col.iloc[-22])  if len(col) >= 22 else float(col.iloc[0])
                p3m_ago  = float(col.iloc[0])

                mom_1m  = (price - p1m_ago)  / p1m_ago  if p1m_ago  else 0
                mom_3m  = (price - p3m_ago)  / p3m_ago  if p3m_ago  else 0

                # RSI(14)
                diffs = col.diff().dropna()
                gains = diffs.clip(lower=0)
                loss  = (-diffs).clip(lower=0)
                avg_g = gains.tail(14).mean()
                avg_l = loss.tail(14).mean()
                rsi   = 50.0
                if avg_l > 0:
                    rs  = avg_g / avg_l
                    rsi = 100 - (100 / (1 + rs))

                # fast_info para 52w range
                try:
                    fi  = yf.Ticker(t).fast_info
                    y_h = float(fi.year_high  or price)
                    y_l = float(fi.year_low   or price)
                    pos_52w = (price - y_l) / (y_h - y_l) if y_h > y_l else 0.5
                except Exception:
                    pos_52w = 0.5

                # Score compuesto (0-100)
                score = 0
                score += min(40, max(0, mom_1m * 200))   # momentum 1m contribuye hasta 40pts
                score += min(20, max(0, mom_3m * 50))    # momentum 3m hasta 20pts
                score += min(20, max(0, (rsi - 45) * 1.33)) if 45 < rsi < 70 else 0  # RSI zona sana
                score += min(20, max(0, pos_52w * 20))   # posición 52w hasta 20pts

                razones = []
                if mom_1m > 0.04:  razones.append(f"+{mom_1m*100:.1f}% este mes")
                if mom_3m > 0.10:  razones.append(f"+{mom_3m*100:.1f}% en 3 meses")
                if 55 < rsi < 70:  razones.append(f"RSI saludable ({rsi:.0f})")
                if pos_52w > 0.7:  razones.append("Cerca de máximos 52 semanas")
                if pos_52w < 0.35: razones.append("Rebote desde mínimos")

                if score >= 30 and razones:
                    scored.append({
                        "ticker":  t,
                        "precio":  round(price, 2),
                        "score":   round(score, 1),
                        "mom_1m":  round(mom_1m * 100, 2),
                        "mom_3m":  round(mom_3m * 100, 2),
                        "rsi":     round(rsi, 1),
                        "razones": razones[:3],
                    })
            except Exception:
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        result = scored[:12]
        _cache_set(_endpoint_cache, "tendencias_ia", result)
        return result
    except Exception as e:
        print(f"Error tendencias-ia: {e}")
        return []


# ─── Métricas de riesgo del portafolio (Sharpe, Max Drawdown, VaR) ───────────
@app.get("/riesgo-cartera")
def riesgo_cartera(tickers: str = Query(default="")):
    """
    Calcula Sharpe Ratio, Max Drawdown y VaR(95%) de la cartera completa.
    Recibe: /riesgo-cartera?tickers=AAPL,MSFT,NVDA
    """
    if not tickers.strip():
        return {"error": "Se requieren tickers"}

    ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    # Excluir cryptos (no aplican para cálculos de riesgo con benchmark SPY)
    stock_tickers = [
        t for t in ticker_list
        if t.replace('-USD', '').replace('/USD', '') not in CRYPTO_SYMBOLS
    ]
    if not stock_tickers:
        return {"sharpe": None, "max_drawdown": None, "var_95": None, "ret_anual": None, "n_tickers": 0}

    cache_key = f"riesgo_{'_'.join(sorted(stock_tickers))}"
    cached = _cache_get(_endpoint_cache, cache_key, 3600)
    if cached is not None:
        return cached

    try:
        syms_list = [yf_sym(t) for t in stock_tickers]
        data = yf.download(
            " ".join(syms_list), period="1y",
            auto_adjust=True, progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data[["Close"]]

        returns_list = []
        for t, sym in zip(stock_tickers, syms_list):
            col = sym if sym in closes.columns else (t if t in closes.columns else None)
            if col is None:
                continue
            r = closes[col].dropna().pct_change().dropna()
            if len(r) > 20:
                returns_list.append(r)

        if not returns_list:
            return {"error": "No hay datos suficientes para calcular riesgo"}

        # Retorno diario del portafolio (peso igual por activo — simplificado)
        port_ret = pd.concat(returns_list, axis=1).mean(axis=1).dropna()

        # Sharpe Ratio anualizado (Rf = 5% anual = 0.05/252 diario)
        rf_daily = 0.05 / 252
        excess   = port_ret - rf_daily
        sharpe   = round(float(excess.mean() / excess.std() * np.sqrt(252)), 2) if excess.std() > 0 else 0.0

        # Max Drawdown (peor caída desde máximo histórico en el período)
        cum         = (1 + port_ret).cumprod()
        rolling_max = cum.cummax()
        drawdowns   = (cum - rolling_max) / rolling_max
        max_dd      = round(float(drawdowns.min()) * 100, 2)

        # VaR(95%) — pérdida máxima esperada en 1 día con 95% de confianza
        var_95 = round(float(np.percentile(port_ret, 5)) * 100, 2)

        # Retorno anualizado
        ret_anual = round(float(port_ret.mean() * 252 * 100), 2)

        result = {
            "sharpe":       sharpe,
            "max_drawdown": max_dd,
            "var_95":       var_95,
            "ret_anual":    ret_anual,
            "n_tickers":    len(returns_list),
        }
        _cache_set(_endpoint_cache, cache_key, result)
        return result
    except Exception as e:
        print(f"Error riesgo-cartera: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
