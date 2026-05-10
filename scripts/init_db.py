#!/usr/bin/env python3
"""Скрипт инициализации базы данных для AI-Terminator.

Создаёт SQLite-базу с таблицами concepts, parameters, metadata,
индексами и записью версии схемы.
"""

import sqlite3
from pathlib import Path
import sys

# Добавляем корень проекта в path для импортов
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config


def init_db(db_path: str) -> None:
    """Инициализировать базу данных.

    Args:
        db_path: Путь к файлу базы данных.
    """
    db_file = Path(db_path)

    # Проверка и создание родительской папки
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Подключение к базе
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    cursor = conn.cursor()

    # Создание таблицы concepts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            term TEXT NOT NULL,
            domain TEXT,
            embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Создание таблицы parameters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_id TEXT REFERENCES concepts(id) ON DELETE CASCADE,
            name TEXT,
            label_ru TEXT,
            type TEXT CHECK(type IN ('string','integer','float','boolean','enum')),
            description TEXT,
            unit TEXT,
            enum_values TEXT
        )
    """)

    # Создание таблицы metadata для версионирования
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Запись версии схемы
    cursor.execute(
        "INSERT OR IGNORE INTO metadata (key, value) VALUES ('schema_version', '1')"
    )

    # Создание индексов
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_parameters_concept_id ON parameters(concept_id)"
    )

    # Фиксация и закрытие
    conn.commit()
    conn.close()

    print(f"База данных инициализирована: {db_path}")


if __name__ == "__main__":
    # Загрузка конфигурации
    config = Config.from_json("configs/config.json")
    init_db(config.db_path)
