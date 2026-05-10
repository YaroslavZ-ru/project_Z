"""Тесты модуля словаря синонимов для AI-Terminator."""

import json
import os
import tempfile
import unittest

from src.synonyms import SynonymDict


class TestSynonymDict(unittest.TestCase):
    """Тесты класса SynonymDict."""

    def setUp(self):
        """Создание временного файла со синонимами."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "synonyms.json")

        # Создаем тестовый словарь
        test_data = {
            "ключ": ["инструмент", "отмычка"],
            "гаечный": ["разводной"],
            "техника": ["механизм", "устройство", "аппарат"],
        }
        with open(self.temp_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        """Очистка временных файлов."""
        # Удаляем все файлы в temp_dir, затем саму папку
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_load_synonyms(self):
        """Тест загрузки словаря синонимов."""
        synonym_dict = SynonymDict(self.temp_file)
        self.assertIn("ключ", synonym_dict._data)
        self.assertIn("гаечный", synonym_dict._data)

    def test_get_synonyms_basic(self):
        """Тест получения синонимов для леммы."""
        synonym_dict = SynonymDict(self.temp_file)
        synonyms = synonym_dict.get_synonyms("ключ")
        self.assertEqual(synonyms, ["инструмент", "отмычка"])

    def test_get_synonyms_max_limit(self):
        """Тест ограничения количества синонимов."""
        synonym_dict = SynonymDict(self.temp_file)
        # Для "техника" в файле 3 синонима, но max_synonyms=2
        synonyms = synonym_dict.get_synonyms("техника", max_synonyms=2)
        self.assertEqual(len(synonyms), 2)
        self.assertEqual(synonyms, ["механизм", "устройство"])

    def test_get_synonyms_not_found(self):
        """Тест для леммы без синонимов."""
        synonym_dict = SynonymDict(self.temp_file)
        synonyms = synonym_dict.get_synonyms("неизвестное")
        self.assertEqual(synonyms, [])

    def test_get_synonyms_empty_file(self):
        """Тест для пустого словаря."""
        # Создаем пустой файл
        empty_file = os.path.join(self.temp_dir, "empty.json")
        with open(empty_file, "w", encoding="utf-8") as f:
            json.dump({}, f)

        synonym_dict = SynonymDict(empty_file)
        self.assertEqual(synonym_dict.get_synonyms("ключ"), [])

    def test_get_all_synonyms(self):
        """Тест получения всех синонимов для списка лемм."""
        synonym_dict = SynonymDict(self.temp_file)
        lemmas = ["ключ", "гаечный"]
        all_synonyms = synonym_dict.get_all_synonyms(lemmas)
        # "ключ" -> ["инструмент", "отмычка"], "гаечный" -> ["разводной"]
        self.assertEqual(set(all_synonyms), {"инструмент", "отмычка", "разводной"})

    def test_get_all_synonyms_unique(self):
        """Тест, что синонимы уникальны (без дубликатов)."""
        # Создаем файл с дублирующимися синонимами
        dup_file = os.path.join(self.temp_dir, "duplicates.json")
        test_data = {
            "ключ": ["инструмент", "отмычка"],
            "инструмент": ["ключ", "отмычка"],  # "отмычка" дублируется
        }
        with open(dup_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        synonym_dict = SynonymDict(dup_file)
        lemmas = ["ключ", "инструмент"]
        all_synonyms = synonym_dict.get_all_synonyms(lemmas, max_synonyms=2)
        # Должно быть 3 уникальных синонима: "инструмент", "отмычка", "ключ"
        self.assertEqual(len(all_synonyms), 3)

    def test_has_synonyms(self):
        """Тест проверки наличия синонимов."""
        synonym_dict = SynonymDict(self.temp_file)
        self.assertTrue(synonym_dict.has_synonyms("ключ"))
        self.assertTrue(synonym_dict.has_synonyms("гаечный"))
        self.assertFalse(synonym_dict.has_synonyms("неизвестное"))

    def test_has_synonyms_empty(self):
        """Тест проверки для леммы без синонимов в файле."""
        # Создаем файл с пустым списком
        empty_list_file = os.path.join(self.temp_dir, "empty_list.json")
        test_data = {"ключ": []}
        with open(empty_list_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        synonym_dict = SynonymDict(empty_list_file)
        self.assertFalse(synonym_dict.has_synonyms("ключ"))

    def test_statistics(self):
        """Тест получения статистики."""
        synonym_dict = SynonymDict(self.temp_file)
        stats = synonym_dict.get_statistics()
        self.assertEqual(stats["total_lemmas"], 3)
        self.assertEqual(stats["total_synonyms"], 6)  # 2 + 1 + 3
        self.assertEqual(stats["max_synonyms_per_lemma"], 3)

    def test_statistics_empty(self):
        """Тест статистики для пустого словаря."""
        empty_file = os.path.join(self.temp_dir, "empty.json")
        with open(empty_file, "w", encoding="utf-8") as f:
            json.dump({}, f)

        synonym_dict = SynonymDict(empty_file)
        stats = synonym_dict.get_statistics()
        self.assertEqual(stats["total_lemmas"], 0)
        self.assertEqual(stats["total_synonyms"], 0)
        self.assertEqual(stats["max_synonyms_per_lemma"], 0)

    def test_file_not_found(self):
        """Тест обработки отсутствующего файла."""
        # Используем несуществующий путь
        synonym_dict = SynonymDict("/nonexistent/path/synonyms.json")
        self.assertEqual(synonym_dict._data, {})
        self.assertEqual(synonym_dict.get_synonyms("ключ"), [])

    def test_invalid_json(self):
        """Тест обработки некорректного JSON."""
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("{invalid json content}")

        synonym_dict = SynonymDict(invalid_file)
        self.assertEqual(synonym_dict._data, {})

    def test_integration_with_lemmatizer(self):
        """Интеграционный тест с лемматизатором."""
        from src.lemmatizer import Lemmatizer

        synonym_dict = SynonymDict(self.temp_file)
        lemmatizer = Lemmatizer()

        # Лемматизируем и получаем синонимы
        phrase = "Ключи гаечные"
        lemmas = lemmatizer.lemmatize_phrase(phrase)
        self.assertEqual(lemmas, ["ключ", "гаечный"])

        # Получаем синонимы для лемм
        all_synonyms = synonym_dict.get_all_synonyms(lemmas, max_synonyms=2)
        self.assertIn("инструмент", all_synonyms)
        self.assertIn("отмычка", all_synonyms)
        self.assertIn("разводной", all_synonyms)


if __name__ == "__main__":
    unittest.main()
