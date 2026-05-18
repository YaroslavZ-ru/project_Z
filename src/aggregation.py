"""Модуль агрегации параметров для AI-Terminator.

Предоставляет функции для агрегации параметров из списка кандидатов,
определения доминирующего контекста и формирования итогового ответа.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def aggregate_parameters(
    candidates: List[Dict[str, Any]],
    hints_lemmas: List[List[str]],
    max_parameters: int = 15,
) -> List[Dict[str, Any]]:
    """Агрегировать параметры из списка кандидатов.

    Группирует параметры по имени, вычисляет интегральный скор и ранжирует.

    Args:
        candidates: Список кандидатов от search_similar_concepts.
            Каждый кандидат содержит parameters и similarity.
        hints_lemmas: Список списков лемм подсказок (из preprocess_full).
        max_parameters: Максимальное количество возвращаемых параметров.

    Returns:
        Список параметров с полями:
        - name, label_ru, type, description, unit, enum_values
        - confidence (нормированный скор)
        - source = "knowledge_base"
    """
    if not candidates:
        return []

    # Сбор всех параметров с информацией о родительском кандидате
    all_params = []
    for candidate in candidates:
        similarity = candidate.get("similarity", 0.0)
        for param in candidate.get("parameters", []):
            all_params.append({
                "param": param,
                "similarity": similarity,
            })

    if not all_params:
        return []

    # Группировка по name
    groups: Dict[str, List[Dict]] = {}
    for item in all_params:
        name = item["param"]["name"]
        if name not in groups:
            groups[name] = {
                "param": item["param"],
                "similarities": [],
            }
        groups[name]["similarities"].append(item["similarity"])

    # Вычисление скора для каждой группы
    scored_params = []
    max_freq = max(len(g["similarities"]) for g in groups.values())

    for name, group in groups.items():
        freq = len(group["similarities"])
        avg_similarity = sum(group["similarities"]) / len(group["similarities"])

        # Вычисление hint_match
        hint_match = _compute_hint_match(
            group["param"], hints_lemmas
        )

        # Нормализация частоты
        freq_norm = freq / max_freq if max_freq > 0 else 0

        # Вычисление скора по формуле
        score = 0.6 * freq_norm + 0.3 * avg_similarity + 0.1 * hint_match

        scored_params.append({
            "param": group["param"],
            "score": score,
            "freq": freq,
            "avg_similarity": avg_similarity,
        })

    # Сортировка по убыванию скора
    scored_params.sort(key=lambda x: x["score"], reverse=True)

    # Отбор топ-max_parameters
    top_params = scored_params[:max_parameters]

    # Нормировка confidence
    max_score = max(p["score"] for p in top_params) if top_params else 1.0

    result = []
    for item in top_params:
        param = item["param"].copy()
        param["confidence"] = item["score"] / max_score if max_score > 0 else 0.0
        param["source"] = "knowledge_base"
        result.append(param)

    logger.info(
        f"Агрегация параметров: {len(candidates)} кандидатов, "
        f"{len(result)} параметров отобрано"
    )

    return result


def _compute_hint_match(
    param: Dict[str, Any],
    hints_lemmas: List[List[str]],
) -> float:
    """Вычислить степень совпадения параметра с подсказками.

    Args:
        param: Параметр с полями label_ru и description.
        hints_lemmas: Список списков лемм подсказок.

    Returns:
        Доля токенов подсказок, встречающихся в label_ru или description.
    """
    if not hints_lemmas:
        return 0.0

    # Объединить все леммы подсказок
    all_hint_lemmas = set()
    for hint_list in hints_lemmas:
        all_hint_lemmas.update(hint_list)

    if not all_hint_lemmas:
        return 0.0

    # Текст параметра для поиска
    param_text = (
        (param.get("label_ru", "") or "").lower() + " " +
        (param.get("description", "") or "").lower()
    )

    # Подсчет совпадений
    matches = 0
    for lemma in all_hint_lemmas:
        if lemma.lower() in param_text:
            matches += 1

    return matches / len(all_hint_lemmas)


def determine_context(candidates: List[Dict[str, Any]], threshold_omonymy: float = 0.1) -> Dict[str, Any]:
    """Определить доминирующий контекст (предметную область).

    Args:
        candidates: Список кандидатов.
        threshold_omonymy: Порог для определения омонимии (разница в confidence).

    Returns:
        Словарь {"domain": str, "confidence": float} или
        {"context_candidates": [...]} при омонимии.
    """
    if not candidates:
        return {"domain": "не определено", "confidence": 0.0}

    # Подсчет суммы similarity по доменам
    domain_scores: Dict[str, float] = {}
    domain_counts: Dict[str, int] = {}

    for candidate in candidates:
        domain = candidate.get("domain", "не определено")
        similarity = candidate.get("similarity", 0.0)

        if domain not in domain_scores:
            domain_scores[domain] = 0.0
            domain_counts[domain] = 0

        domain_scores[domain] += similarity
        domain_counts[domain] += 1

    # Вычисление средней confidence для каждого домена
    domain_confidences = {
        d: domain_scores[d] / domain_counts[d] 
        for d in domain_scores
    }

    # Проверка на омонимию (несколько доменов с высокой confidence)
    sorted_domains = sorted(domain_confidences.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_domains) >= 2:
        best_conf = sorted_domains[0][1]
        second_conf = sorted_domains[1][1]
        
        if best_conf - second_conf < threshold_omonymy and best_conf > 0.7:
            # Омонимия: возвращаем кандидатов
            context_candidates = [
                {"domain": d, "confidence": c}
                for d, c in sorted_domains[:3]
            ]
            logger.info(
                f"Обнаружена омонимия: {context_candidates}"
            )
            return {"context_candidates": context_candidates}

    # Выбор домена с максимальной суммой
    best_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
    best_score = domain_scores[best_domain]
    best_count = domain_counts[best_domain]

    # Confidence = средняя similarity для лучшего домена
    confidence = best_score / best_count if best_count > 0 else 0.0

    logger.info(
        f"Определен контекст: domain='{best_domain}', confidence={confidence:.3f}"
    )

    return {"domain": best_domain, "confidence": confidence}
