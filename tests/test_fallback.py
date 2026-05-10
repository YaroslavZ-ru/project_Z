"""Тесты модуля fallback-режима для AI-Terminator."""

import json
import os
import tempfile
import unittest

from src.fallback import generate_template_response, detect_domain, load_templates


class TestDetectDomain(unittest.TestCase):
    """Тесты функции detect_domain."""

    def test_detect_domain_technique(self):
        """Определение домена 'техника'."""
        term_lemmas = ["ключ"]
        hints_lemmas = [["техника"], ["вращение"]]
        domain = detect_domain(term_lemmas, hints_lemmas)
        self.assertEqual(domain, "техника")

    def test_detect_domain_music(self):
        """Определение домена 'музыка'."""
        term_lemmas = ["скрипичный"]
        hints_lemmas = [["нота"], ["аккорд"]]
        domain = detect_domain(term_lemmas, hints_lemmas)
        self.assertEqual(domain, "музыка")

    def test_detect_domain_general(self):
        """Определение домена 'общее' (по умолчанию)."""
        term_lemmas = ["неизвестное"]
        hints_lemmas = [["другое"]]
        domain = detect_domain(term_lemmas, hints_lemmas)
        self.assertEqual(domain, "общее")

    def test_detect_domain_empty(self):
        """Определение домена для пустых лемм."""
        term_lemmas = []
        hints_lemmas = []
        domain = detect_domain(term_lemmas, hints_lemmas)
        self.assertEqual(domain, "общее")

    def test_detect_domain_multiple_hints(self):
        """Определение домена с несколькими подсказками."""
        term_lemmas = ["ключ"]
        hints_lemmas = [["техника"], ["механизм"], ["инструмент"]]
        domain = detect_domain(term_lemmas, hints_lemmas)
        self.assertEqual(domain, "техника")

    def test_detect_domain_keyword_in_lemma(self):
        """Определение домена по вхождению ключевого слова в лемму."""
        # "техник" не входит в "технический" как подстрока
        # Используем "технический" с ключевым словом "техник" - не сработает
        # Используем "техника" с ключевым словом "техник" - сработает
        term_lemmas = ["техника"]
        hints_lemmas = []
        domain = detect_domain(term_lemmas, hints_lemmas)
        self.assertEqual(domain, "техника")

    def test_detect_domain_priority(self):
        """Приоритет доменов (техника > музыка)."""
        term_lemmas = ["музыкант"]
        hints_lemmas = [["техника"]]
        domain = detect_domain(term_lemmas, hints_lemmas)
        # "музык" входит в "музыкант", поэтому музыка
        self.assertEqual(domain, "музыка")


class TestLoadTemplates(unittest.TestCase):
    """Тесты функции load_templates."""

    def setUp(self):
        """Создание временного файла шаблонов."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "templates.json")

        test_data = {
            "музыка": {
                "domain": "музыка",
                "parameters": [
                    {"name": "genre", "label_ru": "Жанр", "type": "string", "description": "Музыкальное направление"}
                ]
            },
            "техника": {
                "domain": "техника",
                "parameters": [
                    {"name": "power_source", "label_ru": "Источник энергии", "type": "string", "description": "Ручной, электрический"}
                ]
            }
        }
        with open(self.temp_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        """Очистка временных файлов."""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_load_templates(self):
        """Загрузка шаблонов."""
        templates = load_templates(self.temp_file)
        self.assertIn("музыка", templates)
        self.assertIn("техника", templates)

    def test_load_templates_not_found(self):
        """Загрузка несуществующего файла."""
        templates = load_templates("/nonexistent/path/templates.json")
        self.assertEqual(templates, {})

    def test_load_templates_invalid_json(self):
        """Загрузка некорректного JSON."""
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("{invalid json}")

        templates = load_templates(invalid_file)
        self.assertEqual(templates, {})


class TestGenerateTemplateResponse(unittest.TestCase):
    """Тесты функции generate_template_response."""

    def setUp(self):
        """Создание шаблонов."""
        self.templates = {
            "музыка": {
                "domain": "музыка",
                "parameters": [
                    {"name": "genre", "label_ru": "Жанр", "type": "string", "description": "Музыкальное направление"},
                    {"name": "composer", "label_ru": "Композитор", "type": "string", "description": "Автор произведения"}
                ]
            },
            "техника": {
                "domain": "техника",
                "parameters": [
                    {"name": "power_source", "label_ru": "Источник энергии", "type": "string", "description": "Ручной, электрический"},
                    {"name": "material", "label_ru": "Материал", "type": "string", "description": "Материал изготовления"}
                ]
            },
            "общее": {
                "domain": "общее",
                "parameters": [
                    {"name": "name", "label_ru": "Название", "type": "string", "description": "Имя объекта"}
                ]
            }
        }

    def test_generate_response_technique(self):
        """Генерация ответа для техники."""
        processed_query = {
            "term_lemmas": ["ключ"],
            "hints_lemmas": [["техника"]]
        }

        response = generate_template_response("ключ", ["техника"], processed_query, self.templates)

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["term"], "ключ")
        self.assertEqual(response["selected_context"]["domain"], "техника")
        self.assertEqual(response["selected_context"]["confidence"], 0.3)
        self.assertEqual(len(response["parameters"]), 2)
        self.assertEqual(response["parameters"][0]["name"], "power_source")
        self.assertEqual(response["parameters"][0]["confidence"], 0.3)
        self.assertEqual(response["parameters"][0]["source"], "template")
        self.assertIn("Термин не найден в базе знаний", response["warnings"][0])

    def test_generate_response_music(self):
        """Генерация ответа для музыки."""
        processed_query = {
            "term_lemmas": ["скрипичный"],
            "hints_lemmas": [["нота"]]
        }

        response = generate_template_response("скрипичный ключ", ["нота"], processed_query, self.templates)

        self.assertEqual(response["selected_context"]["domain"], "музыка")
        self.assertEqual(len(response["parameters"]), 2)

    def test_generate_response_general(self):
        """Генерация ответа для общего случая."""
        processed_query = {
            "term_lemmas": ["неизвестное"],
            "hints_lemmas": [["другое"]]
        }

        response = generate_template_response("неизвестное", ["другое"], processed_query, self.templates)

        self.assertEqual(response["selected_context"]["domain"], "общее")
        self.assertEqual(len(response["parameters"]), 1)

    def test_generate_response_empty_templates(self):
        """Генерация ответа без шаблонов."""
        processed_query = {
            "term_lemmas": ["ключ"],
            "hints_lemmas": [["техника"]]
        }

        response = generate_template_response("ключ", ["техника"], processed_query, {})

        # Без шаблонов detect_domain вернет "техника" (из hints)
        # Но в generate_template_response используется templates.get(domain, {})
        # Если domain="техника" и templates={}, то parameters=[]
        self.assertEqual(response["selected_context"]["domain"], "техника")
        self.assertEqual(len(response["parameters"]), 0)

    def test_generate_response_no_hints(self):
        """Генерация ответа без подсказок."""
        processed_query = {
            "term_lemmas": ["ключ"],
            "hints_lemmas": []
        }

        response = generate_template_response("ключ", [], processed_query, self.templates)

        self.assertIn(response["selected_context"]["domain"], ["техника", "общее"])


if __name__ == "__main__":
    unittest.main()
