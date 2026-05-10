"""Тесты модуля поиска в базе знаний."""

import os
import tempfile

import numpy as np
import pytest

from src.search import (
    KnowledgeBase,
    cosine_similarity,
    search_similar_concepts,
    search_similar_concepts_fast,
)


class TestCosineSimilarity:
    """Тесты функции косинусного сходства."""

    def test_cosine_same_vector(self):
        """Одинаковые векторы имеют сходство 1."""
        vec = np.array([1.0, 0.0, 0.0])
        assert np.isclose(cosine_similarity(vec, vec), 1.0)

    def test_cosine_orthogonal(self):
        """Ортогональные векторы имеют сходство 0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        assert np.isclose(cosine_similarity(vec1, vec2), 0.0)

    def test_cosine_opposite(self):
        """Противоположные векторы имеют сходство -1."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        assert np.isclose(cosine_similarity(vec1, vec2), -1.0)

    def test_cosine_normalized(self):
        """Нормализованные векторы."""
        vec1 = np.array([1.0, 1.0])
        vec2 = np.array([1.0, 0.0])
        # Нормализуем
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        assert np.isclose(cosine_similarity(vec1, vec2), np.cos(np.pi / 4))


class TestKnowledgeBase:
    """Тесты базы знаний."""

    def test_create_and_add(self):
        """Создание и добавление понятия."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            kb = KnowledgeBase(db_path)
            kb.connect()

            embedding = np.random.randn(300).astype(np.float32)
            kb.add_concept(
                concept_id="test1",
                term="ключ",
                domain="инструменты",
                embedding=embedding,
            )

            kb.close()

            # Проверяем, что файл создан
            assert os.path.exists(db_path)
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_get_concept(self):
        """Получение понятия."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            kb = KnowledgeBase(db_path)
            kb.connect()

            embedding = np.random.randn(300).astype(np.float32)
            kb.add_concept(
                concept_id="test1",
                term="ключ",
                domain="инструменты",
                embedding=embedding,
            )

            concept = kb.get_concept("test1")
            assert concept is not None
            assert concept["id"] == "test1"
            assert concept["term"] == "ключ"
            assert concept["domain"] == "инструменты"
            assert len(concept["embedding"]) == 300

            kb.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_get_all_concepts(self):
        """Получение всех понятий."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            kb = KnowledgeBase(db_path)
            kb.connect()

            kb.add_concept(
                concept_id="test1",
                term="ключ",
                domain="инструменты",
                embedding=np.random.randn(300).astype(np.float32),
            )
            kb.add_concept(
                concept_id="test2",
                term="отвертка",
                domain="инструменты",
                embedding=np.random.randn(300).astype(np.float32),
            )

            concepts = kb.get_all_concepts()
            assert len(concepts) == 2

            kb.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestSearchSimilarConcepts:
    """Тесты функции поиска."""

    def test_search_empty_base(self):
        """Поиск в пустой базе."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            kb = KnowledgeBase(db_path)
            kb.connect()

            query_vector = np.random.randn(300).astype(np.float32)
            query_vector = query_vector / np.linalg.norm(query_vector)

            candidates = search_similar_concepts(query_vector, kb, min_confidence=0.5)
            assert candidates == []

            kb.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_search_with_candidates(self):
        """Поиск с кандидатами."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            kb = KnowledgeBase(db_path)
            kb.connect()

            # Добавляем понятие с похожим вектором
            query_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297).astype(np.float32)
            similar_vector = np.array([0.9, 0.1, 0.0] + [0.0] * 297).astype(np.float32)

            kb.add_concept(
                concept_id="test1",
                term="ключ",
                domain="инструменты",
                embedding=similar_vector,
            )

            candidates = search_similar_concepts(query_vector, kb, min_confidence=0.8)
            assert len(candidates) >= 1
            assert candidates[0]["term"] == "ключ"
            assert candidates[0]["domain"] == "инструменты"

            kb.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_search_threshold(self):
        """Поиск с порогом."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            kb = KnowledgeBase(db_path)
            kb.connect()

            # Добавляем понятие с низким сходством
            query_vector = np.array([1.0, 0.0] + [0.0] * 298).astype(np.float32)
            different_vector = np.array([0.0, 1.0] + [0.0] * 298).astype(np.float32)

            kb.add_concept(
                concept_id="test1",
                term="ключ",
                domain="инструменты",
                embedding=different_vector,
            )

            # С высоким порогом не найдем
            candidates = search_similar_concepts(query_vector, kb, min_confidence=0.9)
            assert len(candidates) == 0

            # С низким порогом найдем
            candidates = search_similar_concepts(query_vector, kb, min_confidence=0.0)
            assert len(candidates) == 1

            kb.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestSearchSimilarConceptsFast:
    """Тесты быстрого поиска."""

    def test_fast_search_basic(self):
        """Базовый быстрый поиск."""
        concepts = [
            {
                "id": "test1",
                "term": "ключ",
                "domain": "инструменты",
                "embedding": np.array([1.0, 0.0, 0.0]).astype(np.float32),
            },
            {
                "id": "test2",
                "term": "отвертка",
                "domain": "инструменты",
                "embedding": np.array([0.0, 1.0, 0.0]).astype(np.float32),
            },
        ]

        query_vector = np.array([0.9, 0.1, 0.0]).astype(np.float32)
        candidates = search_similar_concepts_fast(query_vector, concepts, min_confidence=0.8)

        assert len(candidates) >= 1
        assert candidates[0]["term"] == "ключ"