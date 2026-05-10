"""Тесты модуля лемматизатора для AI-Terminator."""

import unittest

from src.lemmatizer import Lemmatizer


class TestLemmatizer(unittest.TestCase):
    """Тесты класса Lemmatizer."""

    def setUp(self):
        """Инициализация лемматизатора перед каждым тестом."""
        self.lemmatizer = Lemmatizer()

    def test_lemmatize_word_single(self):
        """Тест лемматизации одного слова."""
        # Обычные случаи
        self.assertEqual(self.lemmatizer.lemmatize_word("ключи"), "ключ")
        self.assertEqual(self.lemmatizer.lemmatize_word("гаечные"), "гаечный")
        self.assertEqual(self.lemmatizer.lemmatize_word("вращение"), "вращение")
        self.assertEqual(self.lemmatizer.lemmatize_word("техника"), "техника")

    def test_lemmatize_word_case_insensitive(self):
        """Тест регистронезависимости."""
        self.assertEqual(self.lemmatizer.lemmatize_word("КЛЮЧИ"), "ключ")
        self.assertEqual(self.lemmatizer.lemmatize_word("Ключ"), "ключ")
        self.assertEqual(self.lemmatizer.lemmatize_word("КЛЮЧ"), "ключ")

    def test_lemmatize_word_empty(self):
        """Тест обработки пустой строки."""
        self.assertEqual(self.lemmatizer.lemmatize_word(""), "")

    def test_lemmatize_word_numbers(self):
        """Тест обработки чисел."""
        # Числа не лемматизируются, возвращаются в нижнем регистре
        self.assertEqual(self.lemmatizer.lemmatize_word("123"), "123")
        self.assertEqual(self.lemmatizer.lemmatize_word("100"), "100")

    def test_lemmatize_word_special_chars(self):
        """Тест обработки слов со специальными символами."""
        # Специальные символы удаляются, оставшаяся часть лемматизируется
        self.assertEqual(self.lemmatizer.lemmatize_word("ключ-гаечный"), "ключ-гаечный")
        self.assertEqual(self.lemmatizer.lemmatize_word("ключ_гаечный"), "ключ_гаечный")

    def test_lemmatize_phrase_simple(self):
        """Тест лемматизации простой фразы."""
        result = self.lemmatizer.lemmatize_phrase("ключи гаечные")
        self.assertEqual(result, ["ключ", "гаечный"])

    def test_lemmatize_phrase_empty(self):
        """Тест обработки пустой фразы."""
        self.assertEqual(self.lemmatizer.lemmatize_phrase(""), [])
        self.assertEqual(self.lemmatizer.lemmatize_phrase("   "), [])

    def test_lemmatize_phrase_multiple_words(self):
        """Тест лемматизации фразы с несколькими словами."""
        result = self.lemmatizer.lemmatize_phrase("ключ гаечный размер 12")
        self.assertEqual(result, ["ключ", "гаечный", "размер", "12"])

    def test_lemmatize_phrase_case_insensitive(self):
        """Тест регистронезависимости для фраз."""
        result = self.lemmatizer.lemmatize_phrase("Ключи Гаечные")
        self.assertEqual(result, ["ключ", "гаечный"])

    def test_lemmatize_phrase_with_extra_spaces(self):
        """Тест обработки фразы с лишними пробелами."""
        result = self.lemmatizer.lemmatize_phrase("  ключи   гаечные  ")
        self.assertEqual(result, ["ключ", "гаечный"])

    def test_singleton_behavior(self):
        """Тест, что Lemmatizer является синглтоном."""
        lemmatizer1 = Lemmatizer()
        lemmatizer2 = Lemmatizer()
        self.assertIs(lemmatizer1, lemmatizer2)

    def test_caching(self):
        """Тест кэширования результатов."""
        # Первый вызов
        result1 = self.lemmatizer.lemmatize_word("ключи")
        # Второй вызов должен вернуть из кэша
        result2 = self.lemmatizer.lemmatize_word("ключи")
        self.assertEqual(result1, result2)
        self.assertEqual(result1, "ключ")

    def test_full_preprocess_example(self):
        """Тест полного примера из ТЗ."""
        # Пример из ТЗ: "ключи гаечные" → ["ключ", "гаечный"]
        result = self.lemmatizer.lemmatize_phrase("ключи гаечные")
        self.assertEqual(result, ["ключ", "гаечный"])


if __name__ == "__main__":
    unittest.main()
