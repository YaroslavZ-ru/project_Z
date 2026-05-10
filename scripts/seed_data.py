#!/usr/bin/env python3
"""Скрипт наполнения базы данных тестовыми данными для AI-Terminator.

Добавляет несколько понятий-примеров с их параметрами.
Эмбеддинги генерируются случайными (нормализованными).
"""

import sqlite3
import numpy as np
from pathlib import Path
import sys
import logging

# Добавляем корень проекта в path для импортов
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config

logger = logging.getLogger(__name__)


def random_embedding(dim: int = 300) -> bytes:
    """Сгенерировать случайный нормализованный вектор.

    Args:
        dim: Размерность вектора (по умолчанию 300).

    Returns:
        Байтовое представление вектора (little-endian float32).
    """
    vec = np.random.randn(dim).astype("<f4")
    vec /= np.linalg.norm(vec)
    return vec.tobytes()


def seed(config: Config, force: bool = False) -> None:
    """Наполнить базу данных тестовыми данными.

    Args:
        config: Экземпляр конфигурации.
        force: Если True, перезаписать существующие данные.
    """
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Проверка на существующие данные
    cursor.execute("SELECT COUNT(*) FROM concepts")
    count = cursor.fetchone()[0]

    if count > 0 and not force:
        logger.info("База уже содержит данные. Пропускаем seed. Используйте force=True для перезаписи.")
        conn.close()
        return

    if force:
        # Очистка таблиц
        cursor.execute("DELETE FROM parameters")
        cursor.execute("DELETE FROM concepts")
        logger.info("Таблицы очищены")

    # Вставка понятий
    concepts = [
        ("concept_001", "ключ гаечный", "слесарный инструмент", random_embedding()),
        ("concept_002", "ключ разводной", "слесарный инструмент", random_embedding()),
        ("concept_003", "ключ скрипичный", "музыка", random_embedding()),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO concepts (id, term, domain, embedding) VALUES (?, ?, ?, ?)",
        concepts
    )
    logger.info(f"Добавлено {len(concepts)} понятий")

    # Вставка параметров
    parameters = [
        ("concept_001", "size_mm", "Размер в мм", "float", "Диаметр зева", "мм", None),
        ("concept_001", "material", "Материал", "string", "Сталь, титан", None, None),
        ("concept_002", "size_range_mm", "Диапазон размеров", "string", "От 6 до 24 мм", None, None),
        ("concept_002", "material", "Материал", "string", "Хромованадиевая сталь", None, None),
        ("concept_003", "clef_type", "Тип ключа", "enum", "Нотный ключ", None, '["скрипичный","басовый","альтовый"]'),
    ]
    cursor.executemany(
        """
        INSERT OR IGNORE INTO parameters 
        (concept_id, name, label_ru, type, description, unit, enum_values)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        parameters
    )
    logger.info(f"Добавлено {len(parameters)} параметров")

    # Фиксация и закрытие
    conn.commit()
    conn.close()
    logger.info("База данных наполнена тестовыми данными")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Загрузка конфигурации
    config = Config.from_json("configs/config.json")
    seed(config, force=True)
