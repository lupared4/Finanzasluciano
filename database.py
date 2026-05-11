from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# 1. Creamos el archivo de la base de datos
engine = create_engine('sqlite:///cartera_inversion.db', echo=True)
Base = declarative_base()

# 2. Definimos la tabla de Activos (Diccionario)
class Activo(Base):
    __tablename__ = 'activos'
    ticker = Column(String, primary_key=True) # Ej: AAPL, AL30
    nombre = Column(String)
    tipo = Column(String) # Acción, Bono, etc.

# 3. Definimos la tabla de Transacciones (El historial del usuario)
class Transaccion(Base):
    __tablename__ = 'transacciones'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, ForeignKey('activos.ticker'))
    tipo_operacion = Column(String) # 'Compra' o 'Venta'
    cantidad = Column(Float)
    precio_unitario = Column(Float)
    fecha = Column(DateTime, default=datetime.datetime.utcnow)

# 4. Crear las tablas en el archivo .db
Base.metadata.create_all(engine)

print("¡Base de datos y tablas creadas con éxito!")
