"""Модуль предобработки текста для AI-Terminator.

Предоставляет функции для лемматизации, очистки и расширения токенов.
"""

import logging
from typing import Optional

from pymorphy3 import MorphAnalyzer

from src.lemmatizer import Lemmatizer
from src.text_cleaner import clean_text

logger = logging.getLogger(__name__)

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


def preprocess(term: str, hints: Optional[list[str]] = None) -> dict:
    """Предобработать входные данные.

    Выполняет:
    1. Валидацию термина
    2. Очистку текста (с проверкой длины)
    3. Лемматизацию через Lemmatizer
    4. Удаление дубликатов подсказок

    Args:
        term: Анализируемый термин.
        hints: Список уточняющих слов (0-3 слова).

    Returns:
        dict: Словарь с подготовленными данными:
            - status: "ok" или "error"
            - original_term: исходный термин
            - original_hints: исходные подсказки
            - clean_term: очищенный термин
            - clean_hints: очищенные подсказки
            - term_lemmas: список лемм термина
            - hints_lemmas: список списков лемм для каждой подсказки
            - all_lemmas: список всех лемм
            - warnings: список предупреждений

    Note:
        Если термин пустой после очистки, возвращается {"status": "error", "message": "..."}.
    """
    if hints is None:
        hints = []

    warnings: list[str] = []

    # 1. Валидация термина
    if not term or not term.strip():
        return {
            "status": "error",
            "message": "Пустой термин. Введите значимое слово.",
            "original_term": term,
            "original_hints": hints,
        }

    # 2. Очистка текста
    clean_term = clean_text(term)
    if not clean_term:
        return {
            "status": "error",
            "message": "Пустой термин после очистки.",
            "original_term": term,
            "original_hints": hints,
        }

    # Очистка подсказок
    clean_hints = []
    for hint in hints:
        if hint and hint.strip():
            hint_clean = clean_text(hint)
            if hint_clean:
                clean_hints.append(hint_clean)

    # Ограничение на 3 подсказки
    if len(clean_hints) > 3:
        warnings.append("Подсказок больше 3, использованы первые 3")
        clean_hints = clean_hints[:3]

    # 3. Проверка максимальной длины
    if len(clean_term) > 100:
        return {
            "status": "error",
            "message": f"Термин слишком длинный (максимум 100 символов, получено {len(clean_term)})",
            "original_term": term,
            "original_hints": hints,
        }

    for i, hint in enumerate(clean_hints):
        if len(hint) > 50:
            warnings.append(f"Подсказка #{i + 1} слишком длинная (максимум 50 символов), обрезана")
            clean_hints[i] = hint[:50]

    # 4. Удаление дубликатов подсказок (сохраняя порядок)
    original_hint_count = len(clean_hints)
    clean_hints = list(dict.fromkeys(clean_hints))
    if len(clean_hints) < original_hint_count:
        logger.info(f"Удалены дубликаты подсказок: {original_hint_count} -> {len(clean_hints)}")

    # 5. Лемматизация через Lemmatizer
    lemmatizer = Lemmatizer()
    term_lemmas = lemmatizer.lemmatize_phrase(clean_term)

    if not term_lemmas:
        return {
            "status": "error",
            "message": "Термин не содержит значимых слов после лемматизации",
            "original_term": term,
            "original_hints": hints,
        }

    # Лемматизация подсказок (каждая подсказка может содержать несколько слов)
    hints_lemmas = []
    for hint in clean_hints:
        hint_lemmas = lemmatizer.lemmatize_phrase(hint)
        if hint_lemmas:  # Пропускаем пустые результаты
            hints_lemmas.append(hint_lemmas)

    # 6. Сбор всех лемм
    all_lemmas = term_lemmas.copy()
    for hint_lemmas_list in hints_lemmas:
        all_lemmas.extend(hint_lemmas_list)

    logger.info(f"Предобработка завершена: term='{clean_term}', hints={clean_hints}")

    return {
        "status": "ok",
        "original_term": term,
        "original_hints": hints,
        "clean_term": clean_term,
        "clean_hints": clean_hints,
        "term_lemmas": term_lemmas,
        "hints_lemmas": hints_lemmas,
        "all_lemmas": all_lemmas,
        "warnings": warnings,
    }
