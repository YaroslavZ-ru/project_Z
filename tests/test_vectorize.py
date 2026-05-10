"""Тесты модуля векторизации."""

import numpy as np
import pytest

from src.vectorize import vectorize, get_embedding_model, EmbeddingModel


class TestEmbeddingModel:
    """Тесты базовой модели эмбеддингов."""

    def test_get_vector_default_dimension(self):
        """Получить вектор с размерностью по умолчанию."""
        model = EmbeddingModel()
        vector = model.get_vector("тест")
        assert len(vector) == 300
        assert vector.dtype == np.float32

    def test_get_vector_custom_dimension(self):
        """Получить вектор с кастомной размерностью."""
        model = EmbeddingModel(dimension=100)
        vector = model.get_vector("тест")
        assert len(vector) == 100

    def test_get_vector_consistency(self):
        """Одинаковые слова дают одинаковые векторы."""
        model = EmbeddingModel()
        v1 = model.get_vector("тест")
        v2 = model.get_vector("тест")
        assert np.allclose(v1, v2)

    def test_get_vector_different_words(self):
        """Разные слова дают разные векторы."""
        model = EmbeddingModel()
        v1 = model.get_vector("тест1")
        v2 = model.get_vector("тест2")
        # Скорее всего векторы будут разными (случайная инициализация)
        assert not np.allclose(v1, v2)


class TestVectorize:
    """Тесты функции vectorize."""

    def test_vectorize_basic(self):
        """Базовая векторизация."""
        tokens = [("ключ", 1.0), ("техника", 1.0)]
        vector = vectorize(tokens)
        assert vector.shape == (300,)
        assert np.isclose(np.linalg.norm(vector), 1.0)  # L2 норма = 1

    def test_vectorize_with_weights(self):
        """Векторизация с весами."""
        tokens = [("ключ", 0.7), ("техника", 0.3)]
        vector = vectorize(tokens)
        assert vector.shape == (300,)
        assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_vectorize_empty(self):
        """Векторизация пустого списка."""
        tokens = []
        vector = vectorize(tokens)
        assert vector.shape == (300,)

    def test_vectorize_single_token(self):
        """Векторизация одного токена."""
        tokens = [("ключ", 1.0)]
        vector = vectorize(tokens)
        assert vector.shape == (300,)
        assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_vectorize_multiple_tokens(self):
        """Векторизация нескольких токенов."""
        tokens = [("ключ", 1.0), ("техника", 1.0), ("вращение", 1.0)]
        vector = vectorize(tokens)
        assert vector.shape == (300,)
        assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_vectorize_different_weights(self):
        """Векторизация с разными весами."""
        tokens = [("ключ", 0.7), ("техника", 0.15), ("вращение", 0.15)]
        vector = vectorize(tokens)
        assert vector.shape == (300,)
        assert np.isclose(np.linalg.norm(vector), 1.0)