"""Тесты модуля эмбеддингов (FastTextWrapper)."""

import logging
import numpy as np
import pytest
from pathlib import Path

from src.embeddings import FastTextWrapper


class TestFastTextWrapper:
    """Тесты класса FastTextWrapper."""

    def test_get_dimension_default(self):
        """Получить размерность по умолчанию (300)."""
        # Создаем с фиктивным путем (не будет загружена)
        emb = FastTextWrapper("models/nonexistent.bin")
        assert emb.get_dimension() == 300

    def test_get_word_vector_caching_logs(self, caplog):
        """Проверка логов кэширования (DEBUG)."""
        emb = FastTextWrapper("models/nonexistent.bin")
        
        with caplog.at_level(logging.DEBUG):
            # Первый вызов - промах
            emb.get_word_vector("тест1")
            assert any("Кэш промах" in record.message for record in caplog.records)
            
            # Очистка логов
            caplog.clear()
            
            # Второй вызов - попадание
            emb.get_word_vector("тест1")
            assert any("Кэш попадание" in record.message for record in caplog.records)

    def test_get_word_vector_returns_array(self):
        """get_word_vector возвращает numpy array."""
        emb = FastTextWrapper("models/nonexistent.bin")
        vec = emb.get_word_vector("тест")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    def test_get_word_vector_dimension(self):
        """Вектор имеет правильную размерность."""
        emb = FastTextWrapper("models/nonexistent.bin")
        vec = emb.get_word_vector("тест")
        assert len(vec) == 300

    def test_get_word_vector_empty_string(self):
        """Пустая строка возвращает нулевой вектор."""
        emb = FastTextWrapper("models/nonexistent.bin")
        vec = emb.get_word_vector("")
        assert np.allclose(vec, 0.0)

    def test_get_word_vector_whitespace(self):
        """Строка из пробелов возвращает нулевой вектор."""
        emb = FastTextWrapper("models/nonexistent.bin")
        vec = emb.get_word_vector("   ")
        assert np.allclose(vec, 0.0)

    def test_get_word_vector_caching(self):
        """Векторы кэшируются (одинаковые слова дают одинаковые векторы)."""
        emb = FastTextWrapper("models/nonexistent.bin")
        v1 = emb.get_word_vector("тест")
        v2 = emb.get_word_vector("тест")
        assert np.allclose(v1, v2)

    def test_get_word_vector_different_words(self):
        """Разные слова дают нулевые векторы (в режиме без модели и fallback)."""
        emb = FastTextWrapper("models/nonexistent.bin")
        v1 = emb.get_word_vector("тест1")
        v2 = emb.get_word_vector("тест2")
        # В режиме без модели и fallback возвращаются нулевые векторы
        assert np.allclose(v1, 0.0)
        assert np.allclose(v2, 0.0)

    def test_get_phrase_vector(self):
        """get_phrase_vector возвращает усредненный вектор."""
        emb = FastTextWrapper("models/nonexistent.bin")
        vec = emb.get_phrase_vector("ключ техника")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 300

    def test_get_phrase_vector_empty(self):
        """Пустая фраза возвращает нулевой вектор."""
        emb = FastTextWrapper("models/nonexistent.bin")
        vec = emb.get_phrase_vector("")
        assert np.allclose(vec, 0.0)

    def test_get_phrase_vector_single_word(self):
        """Фраза из одного слова."""
        emb = FastTextWrapper("models/nonexistent.bin")
        vec1 = emb.get_word_vector("ключ")
        vec2 = emb.get_phrase_vector("ключ")
        assert np.allclose(vec1, vec2)

    def test_get_phrase_vector_multiple_words(self):
        """Фраза из нескольких слов усредняется."""
        emb = FastTextWrapper("models/nonexistent.bin")
        # Векторы для отдельных слов
        v1 = emb.get_word_vector("ключ")
        v2 = emb.get_word_vector("техника")
        # Вектор фразы
        phrase_vec = emb.get_phrase_vector("ключ техника")
        # Должен быть близок к усреднению
        expected = (v1 + v2) / 2
        assert np.allclose(phrase_vec, expected)

    def test_get_word_vector_lowercase(self):
        """Векторы не зависят от регистра."""
        emb = FastTextWrapper("models/nonexistent.bin")
        v1 = emb.get_word_vector("ТЕСТ")
        v2 = emb.get_word_vector("тест")
        assert np.allclose(v1, v2)

    def test_get_word_vector_fallback_mode(self):
        """Fallback-режим работает при отсутствии модели."""
        # Создаем временный словарь
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "static_embeddings.npy"
            # Создаем фиктивный словарь
            dummy_dict = {
                "тест": np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32),
                "ключ": np.array([0.0, 1.0, 0.0] + [0.0] * 297, dtype=np.float32),
            }
            np.save(fallback_path, dummy_dict)

            emb = FastTextWrapper("models/nonexistent.bin", str(fallback_path))
            vec = emb.get_word_vector("тест")
            # Должен использовать fallback
            assert vec[0] == 1.0
            assert np.allclose(vec[1:], 0.0)

    def test_get_word_vector_fallback_not_found(self):
        """Если слово не найдено в fallback, возвращается нулевой вектор."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "static_embeddings.npy"
            dummy_dict = {"тест": np.array([1.0, 0.0] + [0.0] * 298, dtype=np.float32)}
            np.save(fallback_path, dummy_dict)

            emb = FastTextWrapper("models/nonexistent.bin", str(fallback_path))
            vec = emb.get_word_vector("не_в_словаре")
            # Должен вернуть нулевой вектор
            assert np.allclose(vec, 0.0)

    def test_get_word_vector_no_fallback(self):
        """Если fallback не указан, возвращаются нулевые векторы."""
        emb = FastTextWrapper("models/nonexistent.bin")
        vec = emb.get_word_vector("тест")
        # Должен вернуть нулевой вектор
        assert np.allclose(vec, 0.0)
