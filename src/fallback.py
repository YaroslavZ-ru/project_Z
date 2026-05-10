"""Модуль fallback-режима для AI-Terminator.

Предоставляет функции для определения предметной области
и генерации шаблонных параметров при отсутствии в базе знаний.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_templates(templates_path: str) -> Dict[str, Any]:
    """Загрузить шаблоны предметных областей из JSON-файла.

    Args:
        templates_path: Путь к JSON-файлу с шаблонами.

    Returns:
        Словарь шаблонов. При ошибке возвращает пустой словарь.
    """
    path = Path(templates_path)

    if not path.exists():
        logger.warning(f"Файл шаблонов не найден: {templates_path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON в {templates_path}: {e}")
        return {}


def detect_domain(
    term_lemmas: List[str],
    hints_lemmas: List[List[str]],
) -> str:
    """Определить предметную область по ключевым словам.

    Args:
        term_lemmas: Леммы термина.
        hints_lemmas: Леммы подсказок (список списков).

    Returns:
        Название предметной области (например, "музыка", "техника", "общее").
    """
    # Объединить все леммы в одно множество
    all_lemmas = set(term_lemmas)
    for hint_list in hints_lemmas:
        all_lemmas.update(hint_list)

    # Словарь ключевых слов для каждой области
    KEYWORDS: Dict[str, List[str]] = {
        "музыка": ["музык", "нот", "скрипич", "бас", "аккорд"],
        "техника": ["техник", "инструмент", "механизм", "прибор", "устройств"],
    }

    # Проверить каждую область
    for domain, keywords in KEYWORDS.items():
        for lemma in all_lemmas:
            for keyword in keywords:
                # Проверка вхождения ключевого слова в лемму (как подстрока)
                if keyword in lemma:
                    return domain

    # По умолчанию возвращаем "общее"
    return "общее"


def generate_template_response(
    term: str,
    hints: List[str],
    processed_query: Dict[str, Any],
    templates: Dict[str, Any],
) -> Dict[str, Any]:
    """Сгенерировать ответ в fallback-режиме.

    Args:
        term: Анализируемый термин.
        hints: Подсказки пользователя.
        processed_query: Результат предобработки.
        templates: Шаблоны предметных областей.

    Returns:
        Словарь ответа с шаблонными параметрами.
    """
    # Извлечь леммы
    term_lemmas = processed_query.get("term_lemmas", [])
    hints_lemmas = processed_query.get("hints_lemmas", [])

    # Определить домен
    domain = detect_domain(term_lemmas, hints_lemmas)

    # Получить параметры из шаблона
    domain_template = templates.get(domain, {})
    template_params = domain_template.get("parameters", [])

    # Добавить поля confidence и source к параметрам
    parameters = []
    for p in template_params:
        param = p.copy()
        param["confidence"] = 0.3
        param["source"] = "template"
        parameters.append(param)

    # Сформировать ответ
    response = {
        "status": "ok",
        "term": term,
        "selected_context": {
            "domain": domain,
            "confidence": 0.3,
        },
        "parameters": parameters,
        "suggested_refinements": [],
        "warnings": [
            "Термин не найден в базе знаний, параметры предложены на основе шаблона предметной области"
        ],
    }

    return response
