from db import engine, Engine, text, Connection, CursorResult

class Postgres:
    def __init__(self, engine: Engine = engine):
        self.engine = engine

    def connect(self):
        return self.engine.connect()

    def disconnect(self, conn: Connection):
        conn.close()

    def execute_query(self, query: str):
        query = text(query)
        with self.connect() as conn:
            result: CursorResult = conn.execute(query)
            return result.fetchall()

    def execute_update(self, query: str):
        query = text(query)
        with self.connect() as conn:
            conn.execute(query)
            conn.commit()