"""Модуль поиска в базе знаний для AI-Terminator.

Предоставляет функцию search_similar_concepts для поиска похожих понятий.
"""

import numpy as np
import logging
from typing import List, Dict, Optional

from src.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# Попытка импорта FAISS (опционально)
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS доступен, будет использоваться для быстрого поиска")
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS не найден, будет использоваться линейный поиск")


def _search_with_threshold(
    query_vec: np.ndarray,
    concepts: List[Dict],
    threshold: float,
    max_candidates: int,
) -> List[Dict]:
    """Поиск понятий с заданным порогом сходства.

    Args:
        query_vec: Вектор запроса (нормализованный).
        concepts: Список понятий из базы знаний.
        threshold: Минимальное косинусное сходство.
        max_candidates: Максимальное количество кандидатов.

    Returns:
        Список кандидатов, отсортированных по убыванию сходства.
    """
    results = []
    for c in concepts:
        # Косинусное сходство (векторы нормализованы)
        sim = float(np.dot(query_vec, c["embedding"]))
        if sim >= threshold:
            results.append({
                "concept_id": c["id"],
                "term": c["term"],
                "domain": c["domain"],
                "similarity": sim,
                "parameters": c["parameters"],
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_candidates]


def search_similar_concepts(
    query_vector: np.ndarray,
    kb: KnowledgeBase,
    min_confidence: float,
    max_candidates: int = 20,
    use_faiss: bool = False,
    domain_filter: Optional[str] = None,
) -> List[Dict]:
    """Найти похожие понятия в базе знаний.

    Args:
        query_vector: Вектор запроса.
        kb: Экземпляр KnowledgeBase.
        min_confidence: Минимальное пороговое значение сходства.
        max_candidates: Максимальное количество кандидатов.
        use_faiss: Использовать FAISS для поиска (опционально).
        domain_filter: Фильтр по домену (опционально).

    Returns:
        Список кандидатов с их параметрами.
    """
    # Защита от нулевого вектора
    norm_q = np.linalg.norm(query_vector)
    if norm_q < 1e-6:
        logger.warning("Вектор запроса нулевой. Возвращаем все понятия с similarity=0.0")
        concepts = kb.get_all_concepts()
        results = [
            {
                "concept_id": c["id"],
                "term": c["term"],
                "domain": c["domain"],
                "similarity": 0.0,
                "parameters": c["parameters"],
            }
            for c in concepts
        ][:max_candidates]
        
        # Применить фильтр по домену
        if domain_filter:
            results = [c for c in results if c["domain"] == domain_filter]
        
        return results

    # Нормализуем на всякий случай
    query_vector = query_vector / norm_q

    concepts = kb.get_all_concepts()
    if not concepts:
        logger.warning("База знаний пуста")
        return []

    # Если FAISS доступен и включен, используем его для больших БД
    if FAISS_AVAILABLE and use_faiss and len(concepts) > 1000:
        results = _search_with_faiss(query_vector, concepts, min_confidence, max_candidates)
    else:
        results = _search_with_threshold(
            query_vector, concepts, min_confidence, max_candidates
        )

    # Если найдено мало кандидатов, снижаем порог
    if len(results) < 3 and min_confidence > 0.2:
        logger.info(
            f"Найдено только {len(results)} кандидатов, снижаем порог до 0.2"
        )
        results = _search_with_threshold(
            query_vector, concepts, 0.2, max_candidates
        )

    # Применить фильтр по домену
    if domain_filter:
        original_count = len(results)
        results = [c for c in results if c["domain"] == domain_filter]
        filtered_count = len(results)
        
        if filtered_count < original_count:
            logger.info(
                f"Фильтрация по домену '{domain_filter}': {original_count} -> {filtered_count} кандидатов"
            )
        
        # Если после фильтрации ничего не осталось, предупредить
        if not results and original_count > 0:
            logger.warning(
                f"После фильтрации по домену '{domain_filter}' не осталось кандидатов"
            )

    logger.info(f"Поиск завершён: {len(results)} кандидатов")
    return results


def _search_with_faiss(
    query_vec: np.ndarray,
    concepts: List[Dict],
    threshold: float,
    max_candidates: int,
) -> List[Dict]:
    """Поиск с использованием FAISS индекса.

    Args:
        query_vec: Вектор запроса (нормализованный).
        concepts: Список понятий из базы знаний.
        threshold: Минимальное косинусное сходство.
        max_candidates: Максимальное количество кандидатов.

    Returns:
        Список кандидатов, отсортированных по убыванию сходства.
    """
    if not FAISS_AVAILABLE:
        logger.warning("FAISS не доступен, используем линейный поиск")
        return _search_with_threshold(query_vec, concepts, threshold, max_candidates)

    # Извлекаем векторы в матрицу
    embeddings = np.array([c["embedding"] for c in concepts], dtype=np.float32)
    
    # Создаем FAISS индекс (косинусное сходство)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product для косинусного сходства
    index.add(embeddings)

    # Поиск
    query_vec_normalized = query_vec / np.linalg.norm(query_vec)
    D, I = index.search(query_vec_normalized.reshape(1, -1), max_candidates)

    results = []
    for i, idx in enumerate(I[0]):
        if i < len(concepts) and D[0][i] >= threshold:
            c = concepts[idx]
            results.append({
                "concept_id": c["id"],
                "term": c["term"],
                "domain": c["domain"],
                "similarity": float(D[0][i]),
                "parameters": c["parameters"],
            })

    return results


def expand_candidates_with_relations(
    candidates: List[Dict],
    kb: KnowledgeBase,
    max_depth: int = 1,
    decay_factor: float = 0.5,
    relation_types: Optional[List[str]] = None,
) -> List[Dict]:
    """Расширить список кандидатов через связи между понятиями.

    Args:
        candidates: Исходный список кандидатов (каждый должен содержать concept_id и similarity).
        kb: Экземпляр KnowledgeBase для получения связей.
        max_depth: Максимальная глубина обхода связей.
        decay_factor: Коэффициент затухания similarity при переходе по связям.
        relation_types: Список типов связей для использования (по умолчанию: related_to, synonym).

    Returns:
        Расширенный список кандидатов, отсортированный по similarity.
    """
    if not candidates:
        return []

    if relation_types is None:
        relation_types = ["related_to", "synonym"]

    # Множество уже просмотренных понятий
    seen_concept_ids: set = {c["concept_id"] for c in candidates}
    
    # Расширенный список кандидатов
    expanded: List[Dict] = list(candidates)

    # Текущий уровень обхода
    current_level = list(candidates)

    for depth in range(max_depth):
        next_level: List[Dict] = []

        for candidate in current_level:
            source_id = candidate["concept_id"]
            original_similarity = candidate.get("similarity", 0.0)

            # Получить связи для текущего понятия
            relations = kb.get_all_relations(source_id)

            for rel in relations:
                if rel["relation_type"] not in relation_types:
                    continue

                target_id = rel["target_concept_id"]

                # Пропустить, если уже добавлено
                if target_id in seen_concept_ids:
                    continue

                # Вычислить новую similarity с учетом затухания
                # Затухание применяется на каждом шаге глубины
                new_similarity = original_similarity * (decay_factor ** (depth + 1))

                # Получить информацию о целевом понятии
                target_concept = kb.get_all_concepts(use_cache=False)
                target_info = None
                for c in target_concept:
                    if c["id"] == target_id:
                        target_info = c
                        break

                if target_info:
                    new_candidate = {
                        "concept_id": target_id,
                        "term": target_info.get("term", ""),
                        "domain": target_info.get("domain", ""),
                        "similarity": new_similarity,
                        "parameters": target_info.get("parameters", []),
                        "relation_type": rel["relation_type"],
                        "relation_confidence": rel["confidence"],
                    }
                    expanded.append(new_candidate)
                    seen_concept_ids.add(target_id)
                    next_level.append(new_candidate)

        current_level = next_level

        # Если на следующем уровне ничего нет, завершаем
        if not current_level:
            break

    # Сортировка по similarity (первичный) и confidence связи (вторичный)
    expanded.sort(
        key=lambda x: (x["similarity"], x.get("relation_confidence", 0.0)),
        reverse=True
    )

    return expanded
