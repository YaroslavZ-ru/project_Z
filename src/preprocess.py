"""Модуль предобработки текста для AI-Terminator.

Предоставляет функции для лемматизации, очистки и расширения токенов.
"""

import logging
from typing import Optional

from pymorphy3 import MorphAnalyzer

from src.lemmatizer import Lemmatizer
from src.synonyms import SynonymDict
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


def preprocess(
    term: str, hints: Optional[list[str]] = None, synonym_dict: Optional[SynonymDict] = None
) -> dict:
    """Предобработать входные данные.

    Выполняет:
    1. Валидацию термина
    2. Очистку текста (с проверкой длины)
    3. Лемматизацию через Lemmatizer
    4. Удаление дубликатов подсказок
    5. Расширение синонимами с правильными весами

    Args:
        term: Анализируемый термин.
        hints: Список уточняющих слов (0-3 слова).
        synonym_dict: Экземпляр SynonymDict для получения синонимов (опционально).

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
            - tokens_with_weights: список кортежей (токен, вес) для векторизации
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

    # 7. Расширение синонимами с весами
    tokens_with_weights = _expand_with_synonyms(
        term_lemmas, hints_lemmas, synonym_dict, lemmatizer
    )

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
        "tokens_with_weights": tokens_with_weights,
        "warnings": warnings,
    }


def _expand_with_synonyms(
    term_lemmas: list[str],
    hints_lemmas: list[list[str]],
    synonym_dict: Optional[SynonymDict],
    lemmatizer: Lemmatizer,
) -> list[tuple[str, float]]:
    """Расширить токены синонимами с правильными весами.

    Формула весов (согласно ТЗ):
    - Вес исходных токенов терма: 0.7 / len(term_lemmas)
    - Вес исходных токенов подсказок: 0.3 / total_hint_words
    - Суммарный вес всех синонимов: 0.1 (равномерно между всеми синонимами)

    Args:
        term_lemmas: Список лемм термина.
        hints_lemmas: Список списков лемм для каждой подсказки.
        synonym_dict: Экземпляр SynonymDict для получения синонимов.
        lemmatizer: Экземпляр Lemmatizer для лемматизации синонимов.

    Returns:
        Список кортежей (токен, вес).
    """
    tokens_with_weights: list[tuple[str, float]] = []

    # Вес для токенов термина
    term_weight_per_word = 0.7 / len(term_lemmas)
    for lemma in term_lemmas:
        tokens_with_weights.append((lemma, term_weight_per_word))

    # Сбор синонимов для термина
    all_synonyms: set[str] = set()
    for lemma in term_lemmas:
        if synonym_dict:
            for syn in synonym_dict.get_synonyms(lemma, max_synonyms=2):
                # Лемматизируем синоним
                syn_lemma = lemmatizer.lemmatize_word(syn)
                if syn_lemma:
                    all_synonyms.add(syn_lemma)

    # Вес для токенов подсказок
    total_hint_words = sum(len(lst) for lst in hints_lemmas)
    if total_hint_words > 0:
        hint_weight_per_word = 0.3 / total_hint_words
        for hint_list in hints_lemmas:
            for lemma in hint_list:
                tokens_with_weights.append((lemma, hint_weight_per_word))
                # Сбор синонимов для подсказок
                if synonym_dict:
                    for syn in synonym_dict.get_synonyms(lemma, max_synonyms=2):
                        syn_lemma = lemmatizer.lemmatize_word(syn)
                        if syn_lemma:
                            all_synonyms.add(syn_lemma)
    else:
        # Если нет подсказок, собираем синонимы только с терма
        pass

    # Добавляем синонимы
    if all_synonyms:
        synonym_weight = 0.1 / len(all_synonyms)
        for syn in all_synonyms:
            tokens_with_weights.append((syn, synonym_weight))

    return tokens_with_weights
