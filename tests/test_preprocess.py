"""Тесты модуля предобработки текста для AI-Terminator."""

import unittest

from src.preprocess import preprocess
from src.synonyms import SynonymDict


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

    def test_preprocess_with_synonyms(self):
        """Предобработка с синонимами."""
        # Создаем временный словарь синонимов
        import tempfile
        import os
        import json

        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "synonyms.json")
        test_data = {
            "ключ": ["инструмент", "отмычка"],
            "техника": ["механизм"],
        }
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        synonym_dict = SynonymDict(temp_file)

        result = preprocess("ключ", ["техника"], synonym_dict=synonym_dict)

        self.assertEqual(result["status"], "ok")
        self.assertIn("tokens_with_weights", result)

        # Проверяем веса
        tokens_weights = result["tokens_with_weights"]
        token_dict = dict(tokens_weights)

        # Термин "ключ" должен иметь вес 0.7
        self.assertAlmostEqual(token_dict.get("ключ", 0), 0.7, places=5)

        # Подсказка "техника" должна иметь вес 0.3
        self.assertAlmostEqual(token_dict.get("техника", 0), 0.3, places=5)

        # Синонимы должны иметь суммарный вес 0.1
        # "инструмент", "отмычка", "механизм" = 3 синонима, каждый 0.1/3
        self.assertAlmostEqual(token_dict.get("инструмент", 0), 0.1 / 3, places=5)
        self.assertAlmostEqual(token_dict.get("отмычка", 0), 0.1 / 3, places=5)
        self.assertAlmostEqual(token_dict.get("механизм", 0), 0.1 / 3, places=5)

        # Очистка
        os.remove(temp_file)
        os.rmdir(temp_dir)

    def test_preprocess_without_synonyms(self):
        """Предобработка без синонимов."""
        result = preprocess("ключ", ["техника"])

        self.assertEqual(result["status"], "ok")
        self.assertIn("tokens_with_weights", result)

        tokens_weights = result["tokens_with_weights"]
        token_dict = dict(tokens_weights)

        # Термин "ключ" должен иметь вес 0.7
        self.assertAlmostEqual(token_dict.get("ключ", 0), 0.7, places=5)

        # Подсказка "техника" должна иметь вес 0.3
        self.assertAlmostEqual(token_dict.get("техника", 0), 0.3, places=5)

        # Синонимов нет, суммарный вес синонимов = 0
        self.assertEqual(len([t for t in tokens_weights if t[0] not in ["ключ", "техника"]]), 0)

    def test_preprocess_multiple_term_words(self):
        """Термин с несколькими словами."""
        result = preprocess("ключ гаечный", ["техника"], synonym_dict=None)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["term_lemmas"], ["ключ", "гаечный"])

        tokens_weights = result["tokens_with_weights"]
        token_dict = dict(tokens_weights)

        # Каждое слово термина должно иметь вес 0.7 / 2 = 0.35
        self.assertAlmostEqual(token_dict.get("ключ", 0), 0.35, places=5)
        self.assertAlmostEqual(token_dict.get("гаечный", 0), 0.35, places=5)

        # Подсказка должна иметь вес 0.3
        self.assertAlmostEqual(token_dict.get("техника", 0), 0.3, places=5)


if __name__ == "__main__":
    unittest.main()
