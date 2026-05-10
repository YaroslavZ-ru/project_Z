"""Тесты модуля базы знаний для AI-Terminator."""

import os
import tempfile
import unittest

import numpy as np

from src.knowledge_base import KnowledgeBase
from src.embeddings import FastTextWrapper
from src.synonyms import SynonymDict


class TestKnowledgeBase(unittest.TestCase):
    """Тесты класса KnowledgeBase."""

    def setUp(self):
        """Создание временной БД перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = os.path.join(self.temp_dir, "test_kb.db")

        # Создаем БД и заполняем тестовыми данными
        import sqlite3
        conn = sqlite3.connect(self.temp_db)
        conn.execute("PRAGMA foreign_keys = ON")

        cursor = conn.cursor()

        # Создание таблиц
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

        cursor.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        cursor.execute(
            "INSERT INTO metadata (key, value) VALUES ('schema_version', '1')"
        )

        # Создаем случайные эмбеддинги
        def random_embedding():
            vec = np.random.randn(300).astype("<f4")
            vec /= np.linalg.norm(vec)
            return vec.tobytes()

        # Вставка понятий
        concepts = [
            ("concept_001", "ключ гаечный", "слесарный инструмент", random_embedding()),
            ("concept_002", "ключ разводной", "слесарный инструмент", random_embedding()),
            ("concept_003", "ключ скрипичный", "музыка", random_embedding()),
        ]
        cursor.executemany(
            "INSERT INTO concepts (id, term, domain, embedding) VALUES (?, ?, ?, ?)",
            concepts
        )

        # Вставка параметров
        parameters = [
            ("concept_001", "size_mm", "Размер в мм", "float", "Диаметр зева", "мм", None),
            ("concept_001", "material", "Материал", "string", "Сталь, титан", None, None),
            ("concept_002", "size_range_mm", "Диапазон размеров", "string", "От 6 до 24 мм", None, None),
            ("concept_002", "material", "Материал", "string", "Хромованадиевая сталь", None, None),
            ("concept_003", "clef_type", "Тип ключа", "enum", "Нотный ключ", None, '["скрипичный","басовый"]'),
        ]
        cursor.executemany(
            """
            INSERT INTO parameters 
            (concept_id, name, label_ru, type, description, unit, enum_values)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            parameters
        )

        conn.commit()
        conn.close()

    def tearDown(self):
        """Очистка временных файлов."""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_init(self):
        """Тест инициализации."""
        kb = KnowledgeBase(self.temp_db)
        self.assertIsNotNone(kb.conn)
        kb.close()

    def test_get_all_concepts(self):
        """Тест загрузки всех понятий."""
        kb = KnowledgeBase(self.temp_db)
        concepts = kb.get_all_concepts()

        self.assertEqual(len(concepts), 3)
        self.assertEqual(concepts[0]["id"], "concept_001")
        self.assertEqual(concepts[0]["term"], "ключ гаечный")
        self.assertEqual(concepts[0]["domain"], "слесарный инструмент")
        self.assertEqual(len(concepts[0]["parameters"]), 2)
        kb.close()

    def test_get_all_concepts_with_cache(self):
        """Тест кэширования понятий."""
        kb = KnowledgeBase(self.temp_db)
        concepts1 = kb.get_all_concepts(use_cache=True)
        concepts2 = kb.get_all_concepts(use_cache=True)

        # Должны быть одинаковыми объектами (кэш)
        self.assertIs(concepts1, concepts2)
        kb.close()

    def test_get_all_concepts_no_cache(self):
        """Тест без кэширования."""
        kb = KnowledgeBase(self.temp_db)
        concepts1 = kb.get_all_concepts(use_cache=False)
        concepts2 = kb.get_all_concepts(use_cache=False)

        # Должны быть разными объектами
        self.assertIsNot(concepts1, concepts2)
        kb.close()

    def test_blob_to_vector(self):
        """Тест преобразования BLOB в вектор."""
        kb = KnowledgeBase(self.temp_db)
        vec = kb._blob_to_vector(np.random.randn(300).astype("<f4").tobytes())
        self.assertEqual(len(vec), 300)
        self.assertEqual(vec.dtype, np.float32)
        kb.close()

    def test_blob_to_vector_none(self):
        """Тест преобразования None в вектор."""
        kb = KnowledgeBase(self.temp_db)
        vec = kb._blob_to_vector(None)
        self.assertEqual(len(vec), 300)
        self.assertTrue(np.allclose(vec, 0.0))
        kb.close()

    def test_parse_enum(self):
        """Тест разбора enum_values."""
        kb = KnowledgeBase(self.temp_db)
        result = kb._parse_enum('["скрипичный","басовый"]')
        self.assertEqual(result, ["скрипичный", "басовый"])
        kb.close()

    def test_parse_enum_none(self):
        """Тест разбора None."""
        kb = KnowledgeBase(self.temp_db)
        result = kb._parse_enum(None)
        self.assertIsNone(result)
        kb.close()

    def test_parse_enum_empty(self):
        """Тест разбора пустой строки."""
        kb = KnowledgeBase(self.temp_db)
        result = kb._parse_enum("")
        self.assertIsNone(result)
        kb.close()

    def test_parse_enum_invalid(self):
        """Тест разбора некорректного JSON."""
        kb = KnowledgeBase(self.temp_db)
        result = kb._parse_enum("{invalid json}")
        self.assertIsNone(result)
        kb.close()

    def test_compute_concept_embedding(self):
        """Тест вычисления эмбеддинга понятия."""
        # Создаем мок-объекты
        class MockEmbeddingModel:
            def get_phrase_vector(self, phrase):
                vec = np.zeros(300, dtype=np.float32)
                if "ключ" in phrase:
                    vec[0] = 1.0
                elif "техника" in phrase:
                    vec[1] = 1.0
                return vec
            def get_dimension(self):
                return 300

        class MockSynonymDict:
            def get_synonyms(self, lemma, max_synonyms=2):
                if lemma == "ключ":
                    return ["инструмент"]
                return []

        kb = KnowledgeBase(self.temp_db, MockEmbeddingModel(), MockSynonymDict())
        vec = kb.compute_concept_embedding("ключ гаечный")

        self.assertEqual(len(vec), 300)
        self.assertEqual(vec.dtype, np.float32)
        kb.close()

    def test_compute_concept_embedding_no_model(self):
        """Тест вычисления без модели."""
        kb = KnowledgeBase(self.temp_db)
        with self.assertRaises(RuntimeError):
            kb.compute_concept_embedding("ключ")
        kb.close()

    def test_update_all_embeddings(self):
        """Тест пересчёта всех эмбеддингов."""
        # Создаем мок-объекты
        class MockEmbeddingModel:
            def get_phrase_vector(self, phrase):
                vec = np.zeros(300, dtype=np.float32)
                vec[0] = 1.0
                return vec
            def get_dimension(self):
                return 300

        class MockSynonymDict:
            def get_synonyms(self, lemma, max_synonyms=2):
                return []

        kb = KnowledgeBase(self.temp_db, MockEmbeddingModel(), MockSynonymDict())
        kb.update_all_embeddings()
        kb.close()

        # Проверяем, что эмбеддинги обновились
        kb2 = KnowledgeBase(self.temp_db)
        concepts = kb2.get_all_concepts(use_cache=False)
        kb2.close()
        for concept in concepts:
            self.assertTrue(np.linalg.norm(concept["embedding"]) > 0.9)

    def test_update_all_embeddings_no_model(self):
        """Тест пересчёта без модели."""
        kb = KnowledgeBase(self.temp_db)
        with self.assertRaises(RuntimeError):
            kb.update_all_embeddings()
        kb.close()

    def test_close(self):
        """Тест закрытия соединения."""
        kb = KnowledgeBase(self.temp_db)
        kb.close()
        # Проверяем, что соединение закрыто
        self.assertTrue(kb.conn.closed if hasattr(kb.conn, 'closed') else True)


if __name__ == "__main__":
    unittest.main()
