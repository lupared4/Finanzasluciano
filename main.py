import yfinance as yf
import warnings
from sqlalchemy.orm import sessionmaker
from database import engine, Transaccion, Activo 

# 1. Ignoramos avisos visuales molestos
warnings.simplefilter(action='ignore', category=FutureWarning)

# 2. Configuramos la conexión a tu base de datos
Session = sessionmaker(bind=engine)
session = Session()

def consultar_activo(ticker):
    """
    Esta función va a internet, busca el precio y te lo devuelve limpio.
    """
    try:
        # Descargamos datos de los últimos 5 días
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        
        if not data.empty and len(data) >= 2:
            # Extraemos los precios de forma segura
            precio_actual = float(data['Close'].iloc[-1])
            precio_anterior = float(data['Close'].iloc[-2])
            variacion = ((precio_actual / precio_anterior) - 1) * 100
            
            return {
                "ticker": ticker,
                "precio": round(precio_actual, 2),
                "variacion": round(variacion, 2)
            }
        else:
            return None
    except Exception as e:
        return None

# --- AQUÍ PROBAMOS SI FUNCIONA ---
if __name__ == "__main__":
    test_ticker = "AAPL"
    resultado = consultar_activo(test_ticker)
    if resultado:
        print(f"Resultado para {test_ticker}: {resultado}")
    else:
        print(f"No se pudo encontrar datos para {test_ticker}")
        

def registrar_compra(ticker, cantidad, precio_compra):
    """
    Guarda una operación de compra en la base de datos.
    """
    try:
        # 1. Verificamos si el activo ya existe en nuestro 'diccionario'
        # Si no existe (ej: es la primera vez que comprás ese ticker), lo creamos
        activo = session.query(Activo).filter_by(ticker=ticker).first()
        if not activo:
            nuevo_activo = Activo(ticker=ticker, nombre=ticker, tipo="Acción/Bono")
            session.add(nuevo_activo)
            session.commit()

        # 2. Registramos la transacción
        nueva_tx = Transaccion(
            ticker=ticker, 
            tipo_operacion='Compra', 
            cantidad=cantidad, 
            precio_unitario=precio_compra
        )
        session.add(nueva_tx)
        session.commit()
        print(f"✅ Compra registrada: {cantidad} unidades de {ticker} a ${precio_compra}")
    
    except Exception as e:
        print(f"❌ Error al registrar: {e}")
        session.rollback()

# --- PROBEMOS REGISTRAR ALGO ---
if __name__ == "__main__":
    # Vamos a simular que compramos 10 acciones de Apple a 280 USD
    registrar_compra("AAPL", 10, 280.0)
    
def mostrar_resumen_cartera():
    """
    Calcula la tenencia actual y la ganancia/pérdida total.
    """
    print("\n" + "="*40)
    print("📊 RESUMEN DE TU CARTERA")
    print("="*40)
    
    # 1. Obtenemos todos los activos que el usuario compró (sin repetir)
    tickers_en_cartera = session.query(Transaccion.ticker).distinct().all()
    
    total_cartera_actual = 0
    total_invertido = 0

    for (t,) in tickers_en_cartera:
        # Sumamos todas las compras de este ticker
        transacciones = session.query(Transaccion).filter_by(ticker=t).all()
        cantidad_total = sum(tx.cantidad for tx in transacciones)
        # Calculamos el costo promedio ponderado
        costo_total = sum(tx.cantidad * tx.precio_unitario for tx in transacciones)
        precio_promedio = costo_total / cantidad_total
        
        # Consultamos el precio real de mercado
        data_mercado = consultar_activo(t)
        
        if data_mercado:
            precio_actual = data_mercado['precio']
            valor_actual = cantidad_total * precio_actual
            ganancia_neta = valor_actual - costo_total
            rendimiento = ((precio_actual / precio_promedio) - 1) * 100
            
            total_cartera_actual += valor_actual
            total_invertido += costo_total
            
            print(f"🔹 {t}: {cantidad_total} unidades")
            print(f"   Costo Promedio: ${precio_promedio:.2f} | Precio Actual: ${precio_actual:.2f}")
            print(f"   Valuación: ${valor_actual:.2f} | Ganancia: ${ganancia_neta:.2f} ({rendimiento:.2f}%)")
            print("-" * 20)

    lucro_total = total_cartera_actual - total_invertido
    rendimiento_total = ((total_cartera_actual / total_invertido) - 1) * 100 if total_invertido > 0 else 0
    
    print("="*40)
    print(f"💰 VALOR TOTAL DE CARTERA: ${total_cartera_actual:.2f}")
    print(f"📈 GANANCIA TOTAL: ${lucro_total:.2f} ({rendimiento_total:.2f}%)")
    print("="*40 + "\n")

# --- PRUEBA FINAL ---
if __name__ == "__main__":
    # Comentamos la compra para no duplicar la que ya hicimos
    # registrar_compra("AAPL", 10, 280.0) 
    
    # Mostramos cómo está nuestra plata hoy
    mostrar_resumen_cartera()
        
