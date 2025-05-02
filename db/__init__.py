import os
import psycopg2
from sqlalchemy import Engine, Connection, create_engine, text, CursorResult
from dotenv import load_dotenv

PG_USER = os.getenv('PG_USER')
PG_PASS = os.getenv('PG_PASS')
PG_HOST = os.getenv('PG_HOST')
PG_DB = os.getenv('PG_DB')
PG_PORT = os.getenv('PG_PORT')
load_dotenv()

engine: Engine = create_engine(f'postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}')
