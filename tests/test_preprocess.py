"""Тесты модуля предобработки текста для AI-Terminator."""

import unittest

from src.preprocess import preprocess


class TestPreprocess(unittest.TestCase):
    """Тесты функции preprocess."""

    def test_preprocess_basic(self):
        """Базовая предобработка."""
        result = preprocess("ключи", ["техника", "вращение"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["clean_term"], "ключи")
        self.assertEqual(result["term_lemmas"], ["ключ"])
        self.assertIn("техника", result["clean_hints"])
        self.assertIn("вращение", result["clean_hints"])

    def test_preprocess_original_values(self):
        """Проверка исходных значений."""
        result = preprocess("Ключи!", ["Техника", "Вращение"])
        self.assertEqual(result["original_term"], "Ключи!")
        self.assertEqual(result["original_hints"], ["Техника", "Вращение"])

    def test_preprocess_empty_term(self):
        """Пустой термин."""
        result = preprocess("   ")
        self.assertEqual(result["status"], "error")
        self.assertIn("Пустой термин", result["message"])

    def test_preprocess_no_hints(self):
        """Без подсказок."""
        result = preprocess("ключ")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["term_lemmas"], ["ключ"])
        self.assertEqual(result["clean_hints"], [])
        self.assertEqual(result["hints_lemmas"], [])

    def test_preprocess_too_many_hints(self):
        """Слишком много подсказок."""
        result = preprocess("ключ", ["техника", "вращение", "ручной", "электрический"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["clean_hints"]), 3)
        self.assertIn("Подсказок больше 3", result["warnings"][0])

    def test_preprocess_duplicate_hints(self):
        """Дубликаты подсказок."""
        result = preprocess("ключ", ["техника", "техника", "вращение"])
        self.assertEqual(result["status"], "ok")
        # Дубликаты должны быть удалены
        self.assertEqual(result["clean_hints"], ["техника", "вращение"])

    def test_preprocess_empty_hints(self):
        """Пустые подсказки."""
        result = preprocess("ключ", ["", "  ", None])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["clean_hints"], [])

    def test_preprocess_long_term(self):
        """Термин слишком длинный."""
        long_term = "a" * 101
        result = preprocess(long_term)
        self.assertEqual(result["status"], "error")
        self.assertIn("слишком длинный", result["message"])

    def test_preprocess_long_hint(self):
        """Подсказка слишком длинная."""
        long_hint = "a" * 51
        result = preprocess("ключ", [long_hint])
        self.assertEqual(result["status"], "ok")
        self.assertIn("слишком длинная", result["warnings"][0])
        self.assertLessEqual(len(result["clean_hints"][0]), 50)

    def test_preprocess_special_chars(self):
        """Очистка специальных символов."""
        result = preprocess("ключ-гаечный!", ["техника", "вращение"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["clean_term"], "ключ-гаечный")

    def test_preprocess_case_insensitive(self):
        """Регистронезависимость."""
        result = preprocess("КЛЮЧИ", ["ТЕХНИКА"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["clean_term"], "ключи")
        self.assertEqual(result["clean_hints"], ["техника"])

    def test_preprocess_all_lemmas(self):
        """Проверка всех лемм."""
        result = preprocess("ключ гаечный", ["техника"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["term_lemmas"], ["ключ", "гаечный"])
        self.assertEqual(result["hints_lemmas"], [["техника"]])
        self.assertEqual(result["all_lemmas"], ["ключ", "гаечный", "техника"])

    def test_preprocess_multilingual(self):
        """Мультиязычный текст."""
        result = preprocess("test ключ", ["tech"])
        self.assertEqual(result["status"], "ok")
        self.assertIn("test", result["clean_term"])
        self.assertIn("ключ", result["clean_term"])
        self.assertIn("tech", result["clean_hints"])

    def test_preprocess_whitespace_handling(self):
        """Обработка лишних пробелов."""
        result = preprocess("  ключ  ", ["  техника  "])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["clean_term"], "ключ")
        self.assertEqual(result["clean_hints"], ["техника"])

    def test_preprocess_hyphen_handling(self):
        """Обработка дефисов."""
        result = preprocess("ключ-гаечный", ["разводной-инструмент"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["clean_term"], "ключ-гаечный")
        self.assertEqual(result["clean_hints"], ["разводной-инструмент"])

    def test_preprocess_numbers(self):
        """Числа в термине."""
        result = preprocess("ключ 123", ["размер 10"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["clean_term"], "ключ 123")
        self.assertEqual(result["clean_hints"], ["размер 10"])


if __name__ == "__main__":
    unittest.main()
