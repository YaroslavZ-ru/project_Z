"""Модуль предобработки текста для AI-Terminator.

Предоставляет функции для лемматизации, очистки и расширения токенов.
"""

import re
from typing import Optional

from pymorphy3 import MorphAnalyzer

# Статический словарь синонимов (загружается из файла при необходимости)
# Формат: {"лемма": [{"syn": "синоним", "weight": 0.4}, ...]}
SYNONYMS: dict[str, list[dict[str, float | str]]] = {}


def load_synonyms(synonyms_path: str) -> dict[str, list[dict[str, float | str]]]:
    """Загрузить словарь синонимов из JSON-файла.

    Args:
        synonyms_path: Путь к JSON-файлу со синонимами.

    Returns:
        dict: Словарь синонимов в формате {"лемма": [{"syn": "...", "weight": ...}, ...]}.
    """
    import json
    from pathlib import Path

    global SYNONYMS
    if SYNONYMS:
        return SYNONYMS

    synonyms_file = Path(synonyms_path)
    if not synonyms_file.exists():
        # Возвращаем пустой словарь, если файл не найден
        return {}

    with open(synonyms_file, "r", encoding="utf-8") as f:
        SYNONYMS = json.load(f)

    return SYNONYMS


def clean_text(text: str) -> str:
    """Очистить текст от ненужных символов.

    Оставляет только буквы, цифры и дефис.

    Args:
        text: Исходный текст.

    Returns:
        Очищенный текст.
    """
    # Удаляем все, кроме букв, цифр и дефиса
    return re.sub(r"[^a-zA-Zа-яА-Я0-9\-]", "", text)


def lemmatize(token: str, morph: MorphAnalyzer) -> str:
    """Лемматизировать одно слово.

    Args:
        token: Исходное слово.
        morph: Экземпляр MorphAnalyzer.

    Returns:
        Лемма (нормальная форма слова).
    """
    if not token:
        return token

    # Лемматизация
    parsed = morph.parse(token)
    if parsed:
        return parsed[0].normal_form

    return token


def expand_with_synonyms(
    tokens: list[str], synonyms: dict[str, list[dict[str, float | str]]], morph: MorphAnalyzer
) -> list[tuple[str, float]]:
    """Расширить токены синонимами.

    Args:
        tokens: Список исходных токенов.
        synonyms: Словарь синонимов.
        morph: Экземпляр MorphAnalyzer для лемматизации синонимов.

    Returns:
        Список кортежей (токен, вес).
    """
    result: list[tuple[str, float]] = []

    for token in tokens:
        # Оригинальный токен с весом 1.0
        result.append((token, 1.0))

        # Синонимы с понижающим весом 0.4
        if token in synonyms:
            for syn_info in synonyms[token]:
                syn_word = syn_info.get("syn", "")
                if syn_word:
                    # Лемматизируем синоним
                    syn_lemma = lemmatize(syn_word, morph)
                    result.append((syn_lemma, 0.4))

    return result


def preprocess(term: str, hints: Optional[list[str]] = None) -> dict:
    """Предобработать входные данные.

    Выполняет:
    1. Валидацию термина
    2. Очистку текста
    3. Лемматизацию
    4. Расширение синонимами

    Args:
        term: Анализируемый термин.
        hints: Список уточняющих слов (0-3 слова).

    Returns:
        dict: Словарь с подготовленными данными:
            - tokens: list[tuple[str, float]] - токены с весами
            - term_lemma: str - лемма термина
            - hint_lemmas: list[str] - леммы подсказок
            - warnings: list[str] - предупреждения

    Raises:
        ValueError: Если термин пустой после очистки.
    """
    if hints is None:
        hints = []

    warnings: list[str] = []

    # 1. Валидация термина
    if not term or not term.strip():
        raise ValueError("Пустой термин. Введите значимое слово.")

    # 2. Очистка и приведение к нижнему регистру
    term_clean = clean_text(term.lower())
    if not term_clean:
        raise ValueError("Пустой термин после очистки.")

    # Очистка подсказок
    hints_clean = []
    for hint in hints:
        if hint and hint.strip():
            hint_clean = clean_text(hint.lower())
            if hint_clean:
                hints_clean.append(hint_clean)

    # Ограничение на 3 подсказки
    if len(hints_clean) > 3:
        hints_clean = hints_clean[:3]
        warnings.append("Подсказок больше 3, использованы первые 3")

    # 3. Лемматизация
    morph = MorphAnalyzer()
    term_lemma = lemmatize(term_clean, morph)

    hint_lemmas = [lemmatize(h, morph) for h in hints_clean]

    # 4. Загрузка синонимов (если файл существует)
    # Синонимы будут загружены при необходимости в следующих шагах
    # Здесь просто возвращаем пустой список, синонимы добавятся при векторизации

    # 5. Формирование итоговых токенов с весами
    # Оригинальные токены: термин + подсказки
    tokens: list[tuple[str, float]] = [(term_lemma, 1.0)]
    for h in hint_lemmas:
        tokens.append((h, 1.0))

    return {
        "tokens": tokens,
        "term_lemma": term_lemma,
        "hint_lemmas": hint_lemmas,
        "warnings": warnings,
    }