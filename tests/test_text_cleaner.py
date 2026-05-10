"""Тесты модуля очистки текста для AI-Terminator."""

import unittest

from src.text_cleaner import clean_text


class TestCleanText(unittest.TestCase):
    """Тесты функции clean_text."""

    def test_basic_cleaning(self):
        """Тест базовой очистки."""
        self.assertEqual(clean_text("Ключ"), "ключ")
        self.assertEqual(clean_text("Гаечный"), "гаечный")

    def test_remove_special_chars(self):
        """Тест удаления специальных символов."""
        self.assertEqual(clean_text("ключ!"), "ключ")
        self.assertEqual(clean_text("ключ@"), "ключ")
        self.assertEqual(clean_text("ключ#"), "ключ")
        self.assertEqual(clean_text("ключ$"), "ключ")

    def test_remove_punctuation(self):
        """Тест удаления знаков препинания."""
        self.assertEqual(clean_text("ключ, гаечный"), "ключ гаечный")
        self.assertEqual(clean_text("ключ. гаечный"), "ключ гаечный")
        self.assertEqual(clean_text("ключ: гаечный"), "ключ гаечный")
        self.assertEqual(clean_text("ключ; гаечный"), "ключ гаечный")
        self.assertEqual(clean_text("ключ? гаечный!"), "ключ гаечный")

    def test_remove_brackets(self):
        """Тест удаления скобок."""
        self.assertEqual(clean_text("ключ (гаечный)"), "ключ гаечный")
        self.assertEqual(clean_text("ключ [гаечный]"), "ключ гаечный")
        self.assertEqual(clean_text("ключ {гаечный}"), "ключ гаечный")

    def test_preserve_spaces(self):
        """Тест сохранения пробелов."""
        self.assertEqual(clean_text("ключ  гаечный"), "ключ гаечный")
        self.assertEqual(clean_text("  ключ  гаечный  "), "ключ гаечный")

    def test_preserve_hyphens(self):
        """Тест сохранения дефисов внутри слова."""
        self.assertEqual(clean_text("ключ-гаечный"), "ключ-гаечный")
        self.assertEqual(clean_text("ключ - гаечный"), "ключ - гаечный")

    def test_remove_edge_hyphens(self):
        """Тест удаления дефисов по краям."""
        self.assertEqual(clean_text("-ключ"), "ключ")
        self.assertEqual(clean_text("ключ-"), "ключ")
        self.assertEqual(clean_text("-ключ-"), "ключ")
        self.assertEqual(clean_text("--ключ--"), "ключ")

    def test_preserve_numbers(self):
        """Тест сохранения цифр."""
        self.assertEqual(clean_text("ключ 123"), "ключ 123")
        self.assertEqual(clean_text("размер 10мм"), "размер 10мм")

    def test_preserve_latin_letters(self):
        """Тест сохранения латинских букв."""
        self.assertEqual(clean_text("ключ ABC"), "ключ abc")
        self.assertEqual(clean_text("test ключ"), "test ключ")

    def test_empty_string(self):
        """Тест пустой строки."""
        self.assertEqual(clean_text(""), "")

    def test_only_special_chars(self):
        """Тест только специальных символов."""
        self.assertEqual(clean_text("!@#$%^&*()"), "")
        self.assertEqual(clean_text("!!!"), "")

    def test_only_spaces(self):
        """Тест только пробелов."""
        self.assertEqual(clean_text("   "), "")
        self.assertEqual(clean_text("  \t  \n  "), "")

    def test_mixed_content(self):
        """Тест смешанного содержимого."""
        self.assertEqual(
            clean_text("  -Ключ-гаечный! (размер 12)  "),
            "ключ-гаечный размер 12",
        )

    def test_cyrillic_and_latin_mixed(self):
        """Тест смешанных кириллицы и латиницы."""
        self.assertEqual(clean_text("ключABCгаечный"), "ключabcгаечный")
        self.assertEqual(clean_text("test123ключ"), "test123ключ")

    def test_multiple_hyphens(self):
        """Тест нескольких дефисов подряд."""
        self.assertEqual(clean_text("ключ--гаечный"), "ключ--гаечный")
        self.assertEqual(clean_text("ключ---"), "ключ")

    def test_unicode_whitespace(self):
        """Тест удаления unicode пробелов."""
        # Табуляция
        self.assertEqual(clean_text("ключ\tгаечный"), "ключ гаечный")
        # Новая строка
        self.assertEqual(clean_text("ключ\nгаечный"), "ключ гаечный")

    def test_complex_example(self):
        """Тест сложного примера из ТЗ."""
        # Пример из ТЗ: "  -Ключ-гаечный! (размер 12)  " → "ключ-гаечный размер 12"
        result = clean_text("  -Ключ-гаечный! (размер 12)  ")
        self.assertEqual(result, "ключ-гаечный размер 12")

    def test_emoji_removal(self):
        """Тест удаления эмодзи."""
        # Эмодзи заменяются на пробел, затем сжимаются
        self.assertEqual(clean_text("ключ 🔧 гаечный"), "ключ гаечный")
        self.assertEqual(clean_text("ключ😀гаечный"), "ключ гаечный")

    def test_math_symbols(self):
        """Тест удаления математических символов."""
        # Символы заменяются на пробел, затем сжимаются
        self.assertEqual(clean_text("ключ + гаечный"), "ключ гаечный")
        self.assertEqual(clean_text("ключ = гаечный"), "ключ гаечный")
        self.assertEqual(clean_text("ключ > гаечный"), "ключ гаечный")


if __name__ == "__main__":
    unittest.main()
