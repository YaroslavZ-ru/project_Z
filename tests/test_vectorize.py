"""Тесты модуля векторизации."""

import numpy as np
import pytest

from src.vectorize import vectorize
from src.embeddings import FastTextWrapper


class MockEmbedding:
    """Мок-объект для тестирования векторизации."""

    def __init__(self, dimension: int = 300):
        self._dimension = dimension

    def get_phrase_vector(self, phrase: str) -> np.ndarray:
        """Вернуть фиксированный вектор для теста."""
        vec = np.zeros(self._dimension, dtype=np.float32)
        if phrase == "a":
            vec[0] = 1.0
        elif phrase == "b":
            vec[1] = 1.0
        elif phrase == "ключ":
            vec[0] = 1.0
        elif phrase == "техника":
            vec[1] = 1.0
        elif phrase == "вращение":
            vec[2] = 1.0
        return vec

    def get_dimension(self) -> int:
        """Получить размерность вектора."""
        return self._dimension


class TestVectorize:
    """Тесты функции vectorize."""

    def test_vectorize_no_normalization(self):
        """Векторизация без нормализации."""
        emb = MockEmbedding()
        # Один токен вес 1.0 - без нормализации должен быть [1,0,0,...]
        q = {"tokens_with_weights": [("a", 1.0)]}
        v = vectorize(q, emb, normalize=False)
        assert v[0] == 1.0
        assert np.allclose(v[1:], 0.0)

    def test_vectorize_with_normalization(self):
        """Векторизация с нормализацией (без деления на сумму весов)."""
        emb = MockEmbedding()
        # Веса: 0.7 для "a", 0.3 для "b"
        # weighted_sum = 0.7*[1,0,0] + 0.3*[0,1,0] = [0.7, 0.3, 0...]
        # norm = sqrt(0.7^2 + 0.3^2) = sqrt(0.58) = 0.761577
        # result = [0.919, 0.394, 0...]
        q = {"tokens_with_weights": [("a", 0.7), ("b", 0.3)]}
        v = vectorize(q, emb, normalize=True)
        assert abs(v[0] - 0.919) < 1e-2
        assert abs(v[1] - 0.394) < 1e-2
        assert np.isclose(np.linalg.norm(v), 1.0)

    def test_vectorize_empty(self):
        """Векторизация пустого списка токенов."""
        emb = MockEmbedding()
        q = {"tokens_with_weights": []}
        v = vectorize(q, emb)
        assert np.allclose(v, 0.0)

    def test_vectorize_single_token(self):
        """Векторизация одного токена."""
        emb = MockEmbedding()
        q = {"tokens_with_weights": [("ключ", 1.0)]}
        v = vectorize(q, emb)
        assert v.shape == (300,)
        assert np.isclose(np.linalg.norm(v), 1.0)

    def test_vectorize_multiple_tokens(self):
        """Векторизация нескольких токенов с весами."""
        emb = MockEmbedding()
        # Термин "ключ" (0.7), подсказки "техника" и "вращение" (0.15 каждая)
        q = {"tokens_with_weights": [
            ("ключ", 0.7),
            ("техника", 0.15),
            ("вращение", 0.15)
        ]}
        v = vectorize(q, emb)
        assert v.shape == (300,)
        assert np.isclose(np.linalg.norm(v), 1.0)

    def test_vectorize_invalid_weight(self):
        """Векторизация с некорректным весом (NaN)."""
        emb = MockEmbedding()
        q = {"tokens_with_weights": [("a", float("nan")), ("b", 1.0)]}
        v = vectorize(q, emb)
        # Должен пропустить NaN и использовать только "b"
        assert v.shape == (300,)

    def test_vectorize_infinite_weight(self):
        """Векторизация с некорректным весом (Inf)."""
        emb = MockEmbedding()
        q = {"tokens_with_weights": [("a", float("inf")), ("b", 1.0)]}
        v = vectorize(q, emb)
        # Должен пропустить Inf и использовать только "b"
        assert v.shape == (300,)

    def test_vectorize_phrase_support(self):
        """Векторизация с поддержкой фраз (усреднение)."""
        emb = MockEmbedding()
        # Фраза "ключ гаечный" будет усреднена
        q = {"tokens_with_weights": [("ключ гаечный", 1.0)]}
        v = vectorize(q, emb)
        assert v.shape == (300,)
        assert np.isclose(np.linalg.norm(v), 1.0)

    def test_vectorize_weighted_sum_no_division(self):
        """Проверка: веса не делятся на сумму (прямое взвешивание)."""
        emb = MockEmbedding()
        # Два одинаковых токена с весом 1.0
        q1 = {"tokens_with_weights": [("a", 1.0), ("a", 1.0)]}
        v1 = vectorize(q1, emb, normalize=False)
        # Два токена с весом 2.0 (в 2 раза больше)
        q2 = {"tokens_with_weights": [("a", 2.0), ("a", 2.0)]}
        v2 = vectorize(q2, emb, normalize=False)
        # Без нормализации v2 должен быть в 2 раза больше v1
        assert np.allclose(v2, 2.0 * v1)

    def test_vectorize_with_normalize_false(self):
        """Векторизация без нормализации (normalize=False)."""
        emb = MockEmbedding()
        q = {"tokens_with_weights": [("a", 0.5), ("b", 0.5)]}
        v = vectorize(q, emb, normalize=False)
        # Без нормализации: [0.5, 0.5, 0...]
        assert np.isclose(v[0], 0.5)
        assert np.isclose(v[1], 0.5)
        # L2 норма не должна быть 1
        assert not np.isclose(np.linalg.norm(v), 1.0)