"""Тесты модуля предобработки текста."""

import pytest

from src.preprocess import clean_text, lemmatize, preprocess


class TestCleanText:
    """Тесты функции clean_text."""

    def test_clean_text_basic(self):
        """Базовая очистка текста."""
        assert clean_text("Привет, мир!") == "Приветмир"
        assert clean_text("Термин-123") == "Термин-123"
        assert clean_text("Слово (в скобках)") == "Слововскобках"

    def test_clean_text_empty(self):
        """Очистка пустой строки."""
        assert clean_text("") == ""

    def test_clean_text_only_special(self):
        """Очистка только специальных символов."""
        assert clean_text("!@#$%^&*()") == ""

    def test_clean_text_mixed(self):
        """Очистка смешанного текста."""
        assert clean_text("Ключ123-Вращение!") == "Ключ123-Вращение"


class TestLemmatize:
    """Тесты функции lemmatize."""

    def test_lemmatize_plural(self):
        """Лемматизация множественного числа."""
        morph = MorphAnalyzer()
        assert lemmatize("ключи", morph) == "ключ"
        assert lemmatize("столы", morph) == "стол"

    def test_lemmatize_verb(self):
        """Лемматизация глагола."""
        morph = MorphAnalyzer()
        assert lemmatize("бежал", morph) == "бежать"
        assert lemmatize("пишет", morph) == "писать"

    def test_lemmatize_empty(self):
        """Лемматизация пустой строки."""
        morph = MorphAnalyzer()
        assert lemmatize("", morph) == ""


class TestPreprocess:
    """Тесты функции preprocess."""

    def test_preprocess_basic(self):
        """Базовая предобработка."""
        result = preprocess("ключи", ["техника", "вращение"])
        assert result["term_lemma"] == "ключ"
        assert "техника" in result["hint_lemmas"]
        assert "вращение" in result["hint_lemmas"]
        assert len(result["tokens"]) == 3

    def test_preprocess_empty_term(self):
        """Пустой термин."""
        with pytest.raises(ValueError, match="Пустой термин"):
            preprocess("   ")

    def test_preprocess_no_hints(self):
        """Без подсказок."""
        result = preprocess("ключ")
        assert result["term_lemma"] == "ключ"
        assert result["hint_lemmas"] == []

    def test_preprocess_too_many_hints(self):
        """Слишком много подсказок."""
        result = preprocess("ключ", ["техника", "вращение", "ручной", "электрический"])
        assert len(result["hint_lemmas"]) == 3
        assert "Подсказок больше 3" in str(result["warnings"])


# Импорт MorphAnalyzer после определения классов, чтобы избежать проблем с импортами
from pymorphy3 import MorphAnalyzer