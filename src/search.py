"""Модуль поиска ближайших понятий в базе знаний для AI-Terminator.

Предоставляет функции для поиска по векторному представлению запроса.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


class KnowledgeBase:
    """База знаний понятий с векторными представлениями."""

    def __init__(self, db_path: str):
        """Инициализация базы знаний.

        Args:
            db_path: Путь к SQLite базе данных.
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Подключиться к базе данных."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._create_tables()

    def close(self) -> None:
        """Закрыть соединение с базой данных."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        """Создать таблицы если они не существуют."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                term TEXT NOT NULL,
                domain TEXT,
                embedding BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                id INTEGER PRIMARY KEY,
                concept_id TEXT REFERENCES concepts(id),
                name TEXT,
                label_ru TEXT,
                type TEXT CHECK(type IN ('string','integer','float','boolean','enum')),
                description TEXT,
                unit TEXT,
                enum_values TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain)
        """)

        self._conn.commit()

    def add_concept(
        self,
        concept_id: str,
        term: str,
        domain: str,
        embedding: np.ndarray,
        parameters: Optional[list[dict]] = None,
    ) -> None:
        """Добавить понятие в базу знаний.

        Args:
            concept_id: Уникальный идентификатор.
            term: Термин понятия.
            domain: Предметная область.
            embedding: Вектор эмбеддинга (300 чисел).
            parameters: Список параметров (опционально).
        """
        cursor = self._conn.cursor()

        # Сохраняем вектор как бинарные данные
        embedding_bytes = embedding.tobytes()

        cursor.execute(
            """
            INSERT OR REPLACE INTO concepts (id, term, domain, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (concept_id, term, domain, embedding_bytes),
        )

        if parameters:
            for param in parameters:
                cursor.execute(
                    """
                    INSERT INTO parameters (
                        concept_id, name, label_ru, type, description, unit, enum_values
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        concept_id,
                        param.get("name", ""),
                        param.get("label_ru", ""),
                        param.get("type", "string"),
                        param.get("description", ""),
                        param.get("unit", ""),
                        json.dumps(param.get("enum_values", [])),
                    ),
                )

        self._conn.commit()

    def get_concept(self, concept_id: str) -> Optional[dict]:
        """Получить понятие по ID.

        Args:
            concept_id: Идентификатор понятия.

        Returns:
            Словарь с данными понятия или None.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
        row = cursor.fetchone()

        if row:
            return {
                "id": row["id"],
                "term": row["term"],
                "domain": row["domain"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32),
            }
        return None

    def get_all_concepts(self) -> list[dict]:
        """Получить все понятия из базы.

        Returns:
            Список словарей с данными понятий.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM concepts")
        rows = cursor.fetchall()

        concepts = []
        for row in rows:
            concepts.append({
                "id": row["id"],
                "term": row["term"],
                "domain": row["domain"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32),
            })

        return concepts

    def get_concepts_by_domain(self, domain: str) -> list[dict]:
        """Получить понятия по домену.

        Args:
            domain: Название домена.

        Returns:
            Список понятий в домене.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM concepts WHERE domain = ?", (domain,))
        rows = cursor.fetchall()

        concepts = []
        for row in rows:
            concepts.append({
                "id": row["id"],
                "term": row["term"],
                "domain": row["domain"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32),
            })

        return concepts


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Вычислить косинусное сходство между двумя векторами.

    Векторы должны быть L2-нормализованы.

    Args:
        vec1: Первый вектор.
        vec2: Второй вектор.

    Returns:
        Косинусное сходство (от -1 до 1).
    """
    return float(np.dot(vec1, vec2))


def search_similar_concepts(
    query_vector: np.ndarray,
    knowledge_base: KnowledgeBase,
    min_confidence: float = 0.3,
    max_candidates: int = 20,
) -> list[dict]:
    """Найти ближайшие понятия в базе знаний.

    Args:
        query_vector: Вектор запроса (нормализованный).
        knowledge_base: Экземпляр KnowledgeBase.
        min_confidence: Минимальное косинусное сходство (по умолчанию 0.3).
        max_candidates: Максимальное количество кандидатов (по умолчанию 20).

    Returns:
        Список кандидатов с полями:
            - concept_id: str
            - term: str
            - domain: str
            - similarity: float
            - parameters: list
    """
    # Подключаемся к БД
    knowledge_base.connect()

    try:
        concepts = knowledge_base.get_all_concepts()
    finally:
        knowledge_base.close()

    if not concepts:
        return []

    # Вычисляем сходство для каждого понятия
    candidates = []
    for concept in concepts:
        similarity = cosine_similarity(query_vector, concept["embedding"])

        if similarity >= min_confidence:
            candidates.append({
                "concept_id": concept["id"],
                "term": concept["term"],
                "domain": concept["domain"],
                "similarity": similarity,
                "parameters": [],  # Пока пусто, будет заполнено позже
            })

    # Сортируем по убыванию сходства
    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    # Ограничиваем количество кандидатов
    candidates = candidates[:max_candidates]

    # Если кандидатов меньше 3, снижаем порог
    if len(candidates) < 3 and min_confidence > 0.2:
        return search_similar_concepts(
            query_vector, knowledge_base, min_confidence=0.2, max_candidates=max_candidates
        )

    return candidates


def search_similar_concepts_fast(
    query_vector: np.ndarray,
    concepts: list[dict],
    min_confidence: float = 0.3,
    max_candidates: int = 20,
) -> list[dict]:
    """Быстрый поиск кандидатов (без подключения к БД).

    Args:
        query_vector: Вектор запроса (нормализованный).
        concepts: Список понятий с векторами.
        min_confidence: Минимальное косинусное сходство.
        max_candidates: Максимальное количество кандидатов.

    Returns:
        Список кандидатов.
    """
    candidates = []
    for concept in concepts:
        similarity = cosine_similarity(query_vector, concept["embedding"])

        if similarity >= min_confidence:
            candidates.append({
                "concept_id": concept.get("id", ""),
                "term": concept.get("term", ""),
                "domain": concept.get("domain", ""),
                "similarity": similarity,
                "parameters": concept.get("parameters", []),
            })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    candidates = candidates[:max_candidates]

    # Если кандидатов мало, снижаем порог
    if len(candidates) < 3 and min_confidence > 0.2:
        return search_similar_concepts_fast(
            query_vector, concepts, min_confidence=0.2, max_candidates=max_candidates
        )

    return candidates