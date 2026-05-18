"""Модуль fallback-режима для AI-Terminator.

Предоставляет функции для определения предметной области
и генерации шаблонных параметров при отсутствии в базе знаний.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

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


def load_domain_keywords(keywords_path: str) -> Dict[str, List[str]]:
    """Загрузить ключевые слова для определения домена из JSON-файла.

    Args:
        keywords_path: Путь к JSON-файлу с ключевыми словами.

    Returns:
        Словарь {domain: [keywords]}. При ошибке возвращает пустой словарь.
    """
    path = Path(keywords_path)

    if not path.exists():
        logger.warning(f"Файл ключевых слов не найден: {keywords_path}. Используем встроенные ключевые слова.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON в {keywords_path}: {e}")
        return {}


def detect_domain(
    term_lemmas: List[str],
    hints_lemmas: List[List[str]],
    domain_keywords: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Определить предметную область по ключевым словам.

    Args:
        term_lemmas: Леммы термина.
        hints_lemmas: Леммы подсказок (список списков).
        domain_keywords: Словарь ключевых слов для доменов (опционально).

    Returns:
        Название предметной области (например, "музыка", "техника", "общее").
    """
    # Объединить все леммы в одно множество
    all_lemmas = set(term_lemmas)
    for hint_list in hints_lemmas:
        all_lemmas.update(hint_list)

    # Словарь ключевых слов для каждой области
    DEFAULT_KEYWORDS: Dict[str, List[str]] = {
        "музыка": ["музык", "нот", "скрипич", "бас", "аккорд"],
        "техника": ["техник", "инструмент", "механизм", "прибор", "устройств"],
    }

    # Использовать переданный словарь или встроенный
    keywords = domain_keywords if domain_keywords else DEFAULT_KEYWORDS

    # Проверить каждую область
    best_domain = "общее"
    best_score = 0

    for domain, domain_keywords_list in keywords.items():
        score = 0
        for lemma in all_lemmas:
            for keyword in domain_keywords_list:
                # Проверка вхождения ключевого слова в лемму (как подстрока)
                if keyword in lemma:
                    score += 1
        if score > best_score:
            best_score = score
            best_domain = domain

    logger.info(f"Определен домен: {best_domain} (score={best_score})")
    return best_domain


def detect_domain_with_centroids(
    term_lemmas: List[str],
    hints_lemmas: List[List[str]],
    kb: "KnowledgeBase",
    query_vector: np.ndarray,
    domain_keywords: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Определить предметную область с использованием центроидов доменов.

    Сначала проверяет ключевые слова, затем использует центроиды для уточнения.

    Args:
        term_lemmas: Леммы термина.
        hints_lemmas: Леммы подсказок (список списков).
        kb: Экземпляр KnowledgeBase для получения центроидов.
        query_vector: Вектор запроса.
        domain_keywords: Словарь ключевых слов для доменов (опционально).

    Returns:
        Название предметной области.
    """
    import numpy as np

    # Сначала определяем домен по ключевым словам
    domain_by_keywords = detect_domain(term_lemmas, hints_lemmas, domain_keywords)

    # Если домен "общее" или есть центроиды, используем их для уточнения
    if domain_by_keywords == "общее" or kb is not None:
        try:
            # Пытаемся получить центроиды
            centroids = kb.get_domain_centroids() if kb else {}
            
            if centroids and len(centroids) > 0:
                # Используем ближайший центроид
                domain_by_centroid = kb.get_closest_domain(query_vector)
                
                if domain_by_centroid:
                    logger.info(
                        f"Домен уточнен по центроиду: {domain_by_centroid} "
                        f"(было: {domain_by_keywords})"
                    )
                    return domain_by_centroid
        except Exception as e:
            logger.warning(f"Ошибка при определении домена по центроидам: {e}")

    return domain_by_keywords


def generate_template_response(
    term: str,
    hints: List[str],
    processed_query: Dict[str, Any],
    templates: Dict[str, Any],
    domain_keywords: Optional[Dict[str, List[str]]] = None,
    max_parameters: int = 15,
) -> Dict[str, Any]:
    """Сгенерировать ответ в fallback-режиме.

    Args:
        term: Анализируемый термин.
        hints: Подсказки пользователя.
        processed_query: Результат предобработки.
        templates: Шаблоны предметных областей.
        domain_keywords: Словарь ключевых слов для доменов (опционально).
        max_parameters: Максимальное количество параметров.

    Returns:
        Словарь ответа с шаблонными параметрами.
    """
    # Извлечь леммы
    term_lemmas = processed_query.get("term_lemmas", [])
    hints_lemmas = processed_query.get("hints_lemmas", [])

    # Определить домен
    domain = detect_domain(term_lemmas, hints_lemmas, domain_keywords)

    # Получить параметры из шаблона
    domain_template = templates.get(domain, {})
    template_params = domain_template.get("parameters", [])

    # Ограничить количество параметров
    template_params = template_params[:max_parameters]

    # Добавить поля confidence и source к параметрам
    parameters = []
    for p in template_params:
        param = p.copy()
        param["confidence"] = 0.3
        param["source"] = "template"
        parameters.append(param)

    # Генерация suggested_refinements на основе подсказок
    suggested_refinements = []
    if hints:
        # Проверить, не противоречат ли подсказки выбранному домену
        if domain == "общее":
            suggested_refinements.append(
                "Уточните контекст: добавьте тематические подсказки для определения предметной области"
            )
        elif domain == "техника":
            suggested_refinements.append(
                "Можно добавить параметр 'мощность' или 'тип привода'"
            )
        elif domain == "музыка":
            suggested_refinements.append(
                "Можно добавить параметр 'жанр' или 'исполнитель'"
            )

    # Сформировать ответ
    response = {
        "status": "ok",
        "term": term,
        "selected_context": {
            "domain": domain,
            "confidence": 0.3,
        },
        "parameters": parameters,
        "suggested_refinements": suggested_refinements,
        "warnings": [
            "Термин не найден в базе знаний, параметры предложены на основе шаблона предметной области"
        ],
    }

    return response
