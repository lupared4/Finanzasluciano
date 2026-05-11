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
    return yf.Ticker(ticker, session=_yf_session)

class TransaccionSchema(BaseModel):
    ticker: str
    cantidad: float
    precio_unitario: float

# ─── Cache en memoria para precios (TTL 60 segundos) ─────────────────────────
_price_cache: dict = {}
_CACHE_TTL = 60

def get_precio_actual(ticker: str) -> float:
    """Obtiene el precio con cache y múltiples fallbacks."""
    now = time.time()
    if ticker in _price_cache:
        precio, ts = _price_cache[ticker]
        if now - ts < _CACHE_TTL:
            return precio

    precio = 0.0
    tk = yft(ticker)

    # Fallback 1: fast_info (más ligero, menos rate-limit)
    try:
        p = tk.fast_info.last_price
        if p and p > 0:
            precio = float(p)
            _price_cache[ticker] = (precio, now)
            return precio
    except Exception:
        pass

    # Fallback 2: yf.download (endpoint diferente)
    try:
        data = yf.download(
            ticker, period="2d", auto_adjust=True,
            progress=False, session=_yf_session
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if not data.empty:
            precio = float(data['Close'].iloc[-1])
            _price_cache[ticker] = (precio, now)
            return precio
    except Exception:
        pass

    # Fallback 3: devolver 0 sin crashear
    _price_cache[ticker] = (0.0, now)
    return 0.0


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
        data = get_history(ticker, period)
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
    """Obtiene histórico: Stooq → Yahoo v8 API → yf.download → tk.history."""
    # Intento 1: Stooq (sin rate-limit desde cualquier servidor)
    df = fetch_stooq(ticker, period)
    if not df.empty:
        return df
    # Intento 2: API v8 directa de Yahoo
    df = fetch_yahoo_v8(ticker, period)
    if not df.empty:
        return df
    # Intento 3: yf.download
    try:
        data = yf.download(
            ticker, period=period, auto_adjust=True,
            progress=False, session=_yf_session
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if not data.empty:
            return data
    except Exception:
        pass
    # Intento 4: tk.history
    try:
        return yft(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()


# ─── Análisis técnico ─────────────────────────────────────────────────────────
@app.get("/analisis/{ticker}")
def obtener_analisis(ticker: str):
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

        return {
            "precio_actual": precio_actual,
            "sma20":         sma20,
            "sma50":         sma50,
            "volatilidad":   volatilidad,
            "beta":          beta,
            "rsi":           rsi,
            "tendencia":     tendencia,
        }
    except Exception as e:
        print(f"Error analisis: {e}")
        return {"error": str(e)}


# ─── Monte Carlo ──────────────────────────────────────────────────────────────
@app.get("/montecarlo/{ticker}")
def obtener_montecarlo(ticker: str, dias: int = Query(30)):
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
        n_sim        = 500
        simulaciones = np.zeros((dias, n_sim))
        for i in range(n_sim):
            precios = [precio_hoy]
            for _ in range(dias - 1):
                precios.append(precios[-1] * (1 + np.random.normal(mu, sigma)))
            simulaciones[:, i] = precios

        return {
            "dias": list(range(1, dias + 1)),
            "p5":  [round(float(np.percentile(simulaciones[d],  5)), 2) for d in range(dias)],
            "p50": [round(float(np.percentile(simulaciones[d], 50)), 2) for d in range(dias)],
            "p95": [round(float(np.percentile(simulaciones[d], 95)), 2) for d in range(dias)],
            "precio_actual": round(precio_hoy, 2),
        }
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

        return {
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
    except Exception as e:
        print(f"Error IA {ticker}: {e}")
        return {"error": str(e)}


# ── Calendario de Earnings ────────────────────────────────────────────────────
@app.get("/calendario")
def calendario_earnings():
    session = Session()
    try:
        tickers = [t for (t,) in session.query(Transaccion.ticker).distinct().all()]
        eventos = []
        now     = pd.Timestamp.now(tz='UTC')

        for ticker in tickers:
            try:
                tk        = yft(ticker)
                cal       = tk.calendar
                fecha_str = None
                eps_est   = None
                eps_alto  = None
                eps_bajo  = None
                rev_est   = None

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

                if not fecha_str:
                    try:
                        ed = tk.earnings_dates
                        if ed is not None and not ed.empty:
                            future = ed[ed.index >= now]
                            if not future.empty:
                                fecha_str = future.index[0].strftime('%Y-%m-%d')
                                col_eps = [c for c in future.columns if 'EPS' in str(c) and 'Estimate' in str(c)]
                                if col_eps:
                                    eps_est = future[col_eps[0]].iloc[0]
                    except Exception:
                        pass

                def safe_f(v):
                    try: return round(float(v), 4) if v is not None and v == v else None
                    except: return None

                if fecha_str:
                    eventos.append({
                        "ticker":           ticker,
                        "fecha":            fecha_str,
                        "eps_estimado":     safe_f(eps_est),
                        "eps_alto":         safe_f(eps_alto),
                        "eps_bajo":         safe_f(eps_bajo),
                        "revenue_estimado": int(rev_est) if rev_est is not None and rev_est == rev_est else None,
                    })
            except Exception as ex:
                print(f"Error calendario {ticker}: {ex}")

        return sorted(eventos, key=lambda x: x['fecha'])
    finally:
        session.close()


# ── Reporte Detallado de Earnings ─────────────────────────────────────────────
@app.get("/earnings/{ticker}")
def earnings_report(ticker: str):
    try:
        tk   = yft(ticker)
        info = tk.info or {}

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
                        "sorpresa": safe(row.get('surprisePercent'), 2),
                    })
        except Exception:
            pass

        # Próximos earnings desde calendar
        cal        = tk.calendar
        prox_fecha = None
        eps_prox_bajo   = None
        eps_prox_medio  = None
        eps_prox_alto   = None
        rev_prox        = None

        if cal and isinstance(cal, dict):
            fechas = cal.get('Earnings Date', [])
            if not isinstance(fechas, list): fechas = [fechas]
            if fechas and hasattr(fechas[0], 'strftime'):
                prox_fecha = fechas[0].strftime('%d/%m/%Y')
            eps_prox_bajo  = safe(cal.get('Earnings Low'),     4)
            eps_prox_medio = safe(cal.get('Earnings Average'), 4)
            eps_prox_alto  = safe(cal.get('Earnings High'),    4)
            rev_prox = int(cal.get('Revenue Average')) \
                if cal.get('Revenue Average') is not None and cal.get('Revenue Average') == cal.get('Revenue Average') \
                else None

        dy = info.get('dividendYield')
        return {
            "nombre":              info.get('longName', ticker),
            "sector":              info.get('sector', '—'),
            "industria":           info.get('industry', '—'),
            "descripcion":         (info.get('longBusinessSummary') or '')[:500],
            "pe_trailing":         safe(info.get('trailingPE')),
            "pe_forward":          safe(info.get('forwardPE')),
            "eps_ttm":             safe(info.get('trailingEps'), 4),
            "eps_forward":         safe(info.get('forwardEps'), 4),
            "market_cap":          info.get('marketCap'),
            "dividendo_pct":       safe(dy * 100 if dy else None),
            "high_52w":            safe(info.get('fiftyTwoWeekHigh')),
            "low_52w":             safe(info.get('fiftyTwoWeekLow')),
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



# ── Indicadores Financieros Fundamentales ────────────────────────────────────
@app.get("/indicadores/{ticker}")
def indicadores_financieros(ticker: str):
    try:
        tk = yft(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        def safe(v, decimals=4):
            try: return round(float(v), decimals) if v is not None and v == v else None
            except: return None

        roa = safe(info.get('returnOnAssets'))
        roe = safe(info.get('returnOnEquity'))
        net_margin = safe(info.get('profitMargins'))
        current_ratio = safe(info.get('currentRatio'), 2)
        book_value = safe(info.get('bookValue'), 2)
        eps_ttm = safe(info.get('trailingEps'), 4)
        eps_forward = safe(info.get('forwardEps'), 4)
        pe_trailing = safe(info.get('trailingPE'), 2)
        pe_forward = safe(info.get('forwardPE'), 2)
        revenue_growth = safe(info.get('revenueGrowth'))
        earnings_growth = safe(info.get('earningsGrowth'))
        gross_margins = safe(info.get('grossMargins'))
        operating_margins = safe(info.get('operatingMargins'))
        ebitda_margins = safe(info.get('ebitdaMargins'))
        debt_to_equity = safe(info.get('debtToEquity'), 2)
        peg_ratio = safe(info.get('pegRatio'), 2)
        price_to_book = safe(info.get('priceToBook'), 2)

        # CFPS = operatingCashflow / sharesOutstanding
        cfps = None
        try:
            ocf    = info.get('operatingCashflow')
            shares = info.get('sharesOutstanding')
            if ocf and shares and shares > 0:
                cfps = round(float(ocf) / float(shares), 4)
        except Exception:
            pass

        # Debt to Asset = totalDebt / totalAssets
        debt_to_asset = None
        try:
            bs = tk.balance_sheet
            if bs is not None and not bs.empty:
                total_debt = None
                total_assets = None
                for label in bs.index:
                    ls = str(label).lower()
                    if 'total debt' in ls or 'totaldebt' in ls:
                        total_debt = float(bs.loc[label].iloc[0])
                    if 'total assets' in ls or 'totalassets' in ls:
                        total_assets = float(bs.loc[label].iloc[0])
                if total_debt is not None and total_assets and total_assets > 0:
                    debt_to_asset = round(total_debt / total_assets, 4)
        except Exception:
            pass

        # BVPS directo de info o book_value ya cargado
        bvps = book_value

        # Logo: clearbit fallback a yahoo
        website = info.get('website', '') or ''
        domain  = website.replace('https://','').replace('http://','').split('/')[0]
        logo_url = f"https://logo.clearbit.com/{domain}" if domain else None
        if not logo_url:
            logo_url = info.get('logo_url') or None

        return {
            "ticker":            ticker.upper(),
            "nombre":            info.get('longName', ticker),
            "sector":            info.get('sector', '—'),
            "logo_url":          logo_url,
            "roa":               roa,
            "roe":               roe,
            "eps_ttm":           eps_ttm,
            "eps_forward":       eps_forward,
            "net_margin":        net_margin,
            "gross_margin":      gross_margins,
            "operating_margin":  operating_margins,
            "ebitda_margin":     ebitda_margins,
            "debt_to_asset":     debt_to_asset,
            "debt_to_equity":    debt_to_equity,
            "current_ratio":     current_ratio,
            "bvps":              bvps,
            "cfps":              cfps,
            "pe_trailing":       pe_trailing,
            "pe_forward":        pe_forward,
            "peg_ratio":         peg_ratio,
            "price_to_book":     price_to_book,
            "revenue_growth":    revenue_growth,
            "earnings_growth":   earnings_growth,
        }
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


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
