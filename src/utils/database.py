import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional
from src.utils.config import DATA_DIR

DB_PATH = DATA_DIR / "penid.db"


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._ensure_dir()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_tables()

    def _ensure_dir(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                sample_count INTEGER DEFAULT 0,
                has_embedding INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                person_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        """)
        self.conn.commit()

    def add_person(self, name: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
        self.conn.commit()
        return cursor.lastrowid

    def get_persons(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, sample_count, has_embedding FROM persons ORDER BY id")
        rows = cursor.fetchall()
        return [
            {"id": r[0], "name": r[1], "sample_count": r[2], "has_embedding": r[3]}
            for r in rows
        ]

    def get_person(self, person_id: int) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, sample_count, has_embedding FROM persons WHERE id=?", (person_id,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "sample_count": row[2], "has_embedding": row[3]}
        return None

    def get_person_by_name(self, name: str) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, sample_count, has_embedding FROM persons WHERE name=?", (name,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "sample_count": row[2], "has_embedding": row[3]}
        return None

    def update_sample_count(self, person_id: int, count: int):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE persons SET sample_count=? WHERE id=?", (count, person_id))
        self.conn.commit()

    def delete_person(self, person_id: int):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM embeddings WHERE person_id=?", (person_id,))
        cursor.execute("DELETE FROM persons WHERE id=?", (person_id,))
        self.conn.commit()

    def save_embedding(self, person_id: int, embedding: np.ndarray):
        blob = embedding.astype(np.float32).tobytes()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO embeddings (person_id, embedding) VALUES (?, ?)",
            (person_id, blob),
        )
        cursor.execute("UPDATE persons SET has_embedding=1 WHERE id=?", (person_id,))
        self.conn.commit()

    def load_embedding(self, person_id: int) -> Optional[np.ndarray]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE person_id=?", (person_id,))
        row = cursor.fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def load_all_embeddings(self) -> dict[int, np.ndarray]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT person_id, embedding FROM embeddings")
        result = {}
        for row in cursor.fetchall():
            result[row[0]] = np.frombuffer(row[1], dtype=np.float32)
        return result

    def close(self):
        self.conn.close()
