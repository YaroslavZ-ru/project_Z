"""Тесты модуля кэширования (QueryVectorCache)."""

import logging
import numpy as np
import pytest

from src.cache import QueryVectorCache


class MockConfig:
    """Мок-конфигурация для тестов."""

    def __init__(self, use_synonyms: bool = True, max_synonyms_per_token: int = 2):
        self.use_synonyms = use_synonyms
        self.max_synonyms_per_token = max_synonyms_per_token


class TestQueryVectorCache:
    """Тесты класса QueryVectorCache."""

    def test_cache_initially_empty(self):
        """Кэш изначально пуст."""
        cache = QueryVectorCache(max_size=10)
        result = cache.get("тест", ["подсказка"], MockConfig())
        assert result is None

    def test_cache_put_and_get(self):
        """Добавление и получение из кэша."""
        cache = QueryVectorCache(max_size=10)
        term = "ключ"
        hints = ["техника"]
        config = MockConfig()
        vector = np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32)
        result_data = {"preprocessed": {"status": "ok"}}

        cache.set(term, hints, config, vector, result_data)
        cached = cache.get(term, hints, config)

        assert cached is not None
        cached_vec, cached_data = cached
        assert np.allclose(cached_vec, vector)
        assert cached_data == result_data

    def test_cache_different_hints_order(self):
        """Разный порядок подсказок дает попадание (сортировка ключа)."""
        cache = QueryVectorCache(max_size=10)
        term = "ключ"
        hints1 = ["техника", "вращение"]
        hints2 = ["вращение", "техника"]
        config = MockConfig()
        vector = np.array([1.0, 0.0] + [0.0] * 298, dtype=np.float32)

        cache.set(term, hints1, config, vector, {"data": 1})
        cached = cache.get(term, hints2, config)

        assert cached is not None  # Должно быть попадание

    def test_cache_different_config(self):
        """Изменение конфигурации дает промах."""
        cache = QueryVectorCache(max_size=10)
        term = "ключ"
        hints = ["техника"]
        config1 = MockConfig(use_synonyms=True)
        config2 = MockConfig(use_synonyms=False)
        vector = np.array([1.0, 0.0] + [0.0] * 298, dtype=np.float32)

        cache.set(term, hints, config1, vector, {"data": 1})
        cached = cache.get(term, hints, config2)

        assert cached is None  # Должен быть промах

    def test_cache_lru_eviction(self):
        """LRU-вытеснение при превышении размера."""
        cache = QueryVectorCache(max_size=3)
        config = MockConfig()

        # Добавляем 3 элемента
        for i in range(3):
            cache.set(f"term{i}", ["hint"], config, np.zeros(300, dtype=np.float32), {"i": i})

        # Добавляем 4-й элемент - должен вытеснить первый
        cache.set("term3", ["hint"], config, np.zeros(300, dtype=np.float32), {"i": 3})

        # term0 должен быть вытеснен
        assert cache.get("term0", ["hint"], config) is None
        # term3 должен быть в кэше
        assert cache.get("term3", ["hint"], config) is not None

    def test_cache_lru_access_order(self):
        """LRU-вытеснение учитывает порядок доступа."""
        cache = QueryVectorCache(max_size=3)
        config = MockConfig()

        # Добавляем 3 элемента
        cache.set("term0", ["hint"], config, np.zeros(300, dtype=np.float32), {})
        cache.set("term1", ["hint"], config, np.zeros(300, dtype=np.float32), {})
        cache.set("term2", ["hint"], config, np.zeros(300, dtype=np.float32), {})

        # Доступ к term0 - он становится "свежим"
        cache.get("term0", ["hint"], config)

        # Добавляем 4-й элемент - должен вытеснить term1 (самый старый)
        cache.set("term3", ["hint"], config, np.zeros(300, dtype=np.float32), {})

        # term0 и term2 должны быть в кэше, term1 - вытеснен
        assert cache.get("term0", ["hint"], config) is not None
        assert cache.get("term1", ["hint"], config) is None
        assert cache.get("term2", ["hint"], config) is not None
        assert cache.get("term3", ["hint"], config) is not None

    def test_cache_clear(self):
        """Очистка кэша."""
        cache = QueryVectorCache(max_size=10)
        config = MockConfig()

        cache.set("term", ["hint"], config, np.zeros(300, dtype=np.float32), {})
        assert cache.get("term", ["hint"], config) is not None

        cache.clear()
        assert cache.get("term", ["hint"], config) is None

    def test_cache_empty_term(self):
        """Кэш с пустым термином."""
        cache = QueryVectorCache(max_size=10)
        config = MockConfig()
        vector = np.array([1.0, 0.0] + [0.0] * 298, dtype=np.float32)

        cache.set("", ["hint"], config, vector, {})
        cached = cache.get("", ["hint"], config)

        assert cached is not None

    def test_cache_multiple_hints(self):
        """Кэш с несколькими подсказками."""
        cache = QueryVectorCache(max_size=10)
        config = MockConfig()
        vector = np.array([1.0, 0.0] + [0.0] * 298, dtype=np.float32)

        hints = ["техника", "вращение", "гаечный"]
        cache.set("ключ", hints, config, vector, {})
        cached = cache.get("ключ", hints, config)

        assert cached is not None

    def test_cache_vector_dtype(self):
        """Вектор возвращается как float32."""
        cache = QueryVectorCache(max_size=10)
        config = MockConfig()
        vector = np.array([1.0, 0.0] + [0.0] * 298, dtype=np.float32)

        cache.set("term", ["hint"], config, vector, {})
        cached = cache.get("term", ["hint"], config)

        assert cached is not None
        cached_vec, _ = cached
        assert cached_vec.dtype == np.float32
