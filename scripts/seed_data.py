#!/usr/bin/env python3
"""Скрипт наполнения базы данных тестовыми данными для AI-Terminator.

Добавляет несколько понятий-примеров с их параметрами.
Эмбеддинги вычисляются через compute_concept_embedding().
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
from src.embeddings import FastTextWrapper
from src.synonyms import SynonymDict
from src.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


def compute_real_embedding(kb: KnowledgeBase, term: str) -> bytes:
    """Вычислить реальный эмбеддинг для термина.

    Args:
        kb: Экземпляр KnowledgeBase.
        term: Термин для вычисления эмбеддинга.

    Returns:
        Байтовое представление вектора (little-endian float32).
    """
    try:
        embedding = kb.compute_concept_embedding(term)
        return embedding.tobytes()
    except Exception as e:
        logger.warning(f"Ошибка вычисления эмбеддинга для '{term}': {e}. Используем случайный вектор.")
        vec = np.random.randn(300).astype("<f4")
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
        cursor.execute("DELETE FROM relations")
        cursor.execute("DELETE FROM sessions")
        cursor.execute("DELETE FROM concept_constraints")
        logger.info("Таблицы очищены")

    # Инициализация компонентов для вычисления эмбеддингов
    try:
        emb_model = FastTextWrapper(
            str(config.fasttext_model_path),
            str(config.db_path.parent / "models" / "static_embeddings.npy"),
            cache_size=config.word_vector_cache_size
        )
        synonym_dict = SynonymDict(str(config.synonyms_path))
        kb = KnowledgeBase(str(config.db_path), emb_model, synonym_dict)
    except Exception as e:
        logger.warning(f"Ошибка инициализации компонентов: {e}. Используем случайные векторы.")
        kb = None

    # Вставка понятий
    concepts = [
        ("concept_001", "ключ гаечный", "слесарный инструмент"),
        ("concept_002", "ключ разводной", "слесарный инструмент"),
        ("concept_003", "ключ скрипичный", "музыка"),
    ]

    for cid, term, domain in concepts:
        if kb:
            embedding = compute_real_embedding(kb, term)
        else:
            vec = np.random.randn(300).astype("<f4")
            vec /= np.linalg.norm(vec)
            embedding = vec.tobytes()
        
        cursor.execute(
            "INSERT OR IGNORE INTO concepts (id, term, domain, embedding) VALUES (?, ?, ?, ?)",
            (cid, term, domain, embedding)
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

    # Вставка связей (relations)
    relations = [
        ("concept_001", "concept_002", "related_to", 0.8),
        ("concept_001", "concept_003", "related_to", 0.3),
    ]
    cursor.executemany(
        """
        INSERT OR IGNORE INTO relations 
        (source_concept_id, target_concept_id, relation_type, confidence)
        VALUES (?, ?, ?, ?)
        """,
        relations
    )
    logger.info(f"Добавлено {len(relations)} связей")

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
