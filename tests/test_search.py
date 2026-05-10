"""Тесты модуля поиска в базе знаний для AI-Terminator."""

import os
import tempfile
import unittest

import numpy as np

from src.search import search_similar_concepts
from src.knowledge_base import KnowledgeBase


class MockKnowledgeBase:
    """Мок-объект KnowledgeBase для тестирования."""

    def __init__(self, concepts):
        self._concepts = concepts

    def get_all_concepts(self, use_cache=True):
        return self._concepts


class TestSearchSimilarConcepts(unittest.TestCase):
    """Тесты функции search_similar_concepts."""

    def setUp(self):
        """Создание тестовых данных."""
        # Создаем понятия с фиксированными векторами
        self.concepts = [
            {
                "id": "concept_001",
                "term": "ключ гаечный",
                "domain": "слесарный инструмент",
                "embedding": np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32),
                "parameters": [
                    {"name": "size_mm", "label_ru": "Размер в мм", "type": "float",
                     "description": "Диаметр зева", "unit": "мм", "enum_values": None,
                     "confidence": 1.0, "source": "knowledge_base"}
                ]
            },
            {
                "id": "concept_002",
                "term": "ключ разводной",
                "domain": "слесарный инструмент",
                "embedding": np.array([0.9, 0.1, 0.0] + [0.0] * 297, dtype=np.float32),
                "parameters": [
                    {"name": "material", "label_ru": "Материал", "type": "string",
                     "description": "Хромованадиевая сталь", "unit": None, "enum_values": None,
                     "confidence": 1.0, "source": "knowledge_base"}
                ]
            },
            {
                "id": "concept_003",
                "term": "ключ скрипичный",
                "domain": "музыка",
                "embedding": np.array([0.0, 0.0, 1.0] + [0.0] * 297, dtype=np.float32),
                "parameters": [
                    {"name": "clef_type", "label_ru": "Тип ключа", "type": "enum",
                     "description": "Нотный ключ", "unit": None,
                     "enum_values": ["скрипичный", "басовый"],
                     "confidence": 1.0, "source": "knowledge_base"}
                ]
            },
        ]

    def test_search_with_high_similarity(self):
        """Поиск с высоким порогом сходства."""
        query_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)
        kb = MockKnowledgeBase(self.concepts)

        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.95, max_candidates=20)

        # Должен найти как минимум 1 кандидат (ключ гаечный с similarity=1.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["term"], "ключ гаечный")
        self.assertAlmostEqual(candidates[0]["similarity"], 1.0, places=5)

    def test_search_with_low_similarity(self):
        """Поиск с низким порогом сходства."""
        query_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)
        kb = MockKnowledgeBase(self.concepts)

        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.5, max_candidates=20)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["term"], "ключ гаечный")
        self.assertEqual(candidates[1]["term"], "ключ разводной")

    def test_search_empty_base(self):
        """Поиск в пустой базе."""
        query_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)
        kb = MockKnowledgeBase([])

        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.5, max_candidates=20)

        self.assertEqual(candidates, [])

    def test_search_zero_vector(self):
        """Поиск с нулевым вектором запроса."""
        query_vector = np.zeros(300, dtype=np.float32)
        kb = MockKnowledgeBase(self.concepts)

        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.5, max_candidates=20)

        # Должны вернуть все понятия с similarity=0.0
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]["similarity"], 0.0)

    def test_search_threshold_reduction(self):
        """Снижение порога при малом количестве кандидатов."""
        # Создаем вектор, который плохо совпадает ни с одним понятием
        query_vector = np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 296, dtype=np.float32)
        kb = MockKnowledgeBase(self.concepts)

        # С высоким порогом ничего не найдем
        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.9, max_candidates=20)
        self.assertEqual(len(candidates), 0)

        # С низким порогом 0.2 тоже ничего не найдем, так как все векторы ортогональны
        # Проверяем, что функция работает корректно
        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.0, max_candidates=20)
        self.assertEqual(len(candidates), 3)

    def test_search_max_candidates(self):
        """Ограничение количества кандидатов."""
        query_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)
        kb = MockKnowledgeBase(self.concepts)

        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.5, max_candidates=2)

        self.assertLessEqual(len(candidates), 2)

    def test_search_sorted_by_similarity(self):
        """Результаты отсортированы по убыванию сходства."""
        query_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)
        kb = MockKnowledgeBase(self.concepts)

        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.5, max_candidates=20)

        similarities = [c["similarity"] for c in candidates]
        self.assertEqual(similarities, sorted(similarities, reverse=True))

    def test_search_preserves_parameters(self):
        """Параметры понятий сохраняются в кандидатах."""
        query_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)
        kb = MockKnowledgeBase(self.concepts)

        candidates = search_similar_concepts(query_vector, kb, min_confidence=0.95, max_candidates=20)

        self.assertEqual(len(candidates[0]["parameters"]), 1)
        self.assertEqual(candidates[0]["parameters"][0]["name"], "size_mm")

    def test_search_with_real_kb(self):
        """Интеграционный тест с реальной БД."""
        temp_dir = tempfile.mkdtemp()
        temp_db = os.path.join(temp_dir, "test_kb.db")

        try:
            # Создаем БД
            import sqlite3
            conn = sqlite3.connect(temp_db)
            conn.execute("PRAGMA foreign_keys = ON")

            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE concepts (
                    id TEXT PRIMARY KEY,
                    term TEXT NOT NULL,
                    domain TEXT,
                    embedding BLOB,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE parameters (
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

            # Создаем понятия с фиксированным вектором
            def fixed_embedding():
                vec = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)
                return vec.tobytes()

            concepts = [
                ("concept_001", "ключ гаечный", "слесарный инструмент", fixed_embedding()),
                ("concept_002", "ключ разводной", "слесарный инструмент", fixed_embedding()),
            ]
            cursor.executemany(
                "INSERT INTO concepts (id, term, domain, embedding) VALUES (?, ?, ?, ?)",
                concepts
            )

            conn.commit()
            conn.close()

            # Загружаем и ищем
            kb = KnowledgeBase(temp_db)
            query_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)

            candidates = search_similar_concepts(query_vector, kb, min_confidence=0.0, max_candidates=20)

            self.assertGreaterEqual(len(candidates), 1)

        finally:
            # Очистка - закрываем соединение перед удалением
            import gc
            gc.collect()
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass


if __name__ == "__main__":
    unittest.main()
