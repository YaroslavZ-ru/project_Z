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
) -> List[Dict]:
    """Найти похожие понятия в базе знаний.

    Args:
        query_vector: Вектор запроса.
        kb: Экземпляр KnowledgeBase.
        min_confidence: Минимальное пороговое значение сходства.
        max_candidates: Максимальное количество кандидатов.
        use_faiss: Использовать FAISS для поиска (опционально).

    Returns:
        Список кандидатов с их параметрами.
    """
    # Защита от нулевого вектора
    norm_q = np.linalg.norm(query_vector)
    if norm_q < 1e-6:
        logger.warning("Вектор запроса нулевой. Возвращаем все понятия с similarity=0.0")
        concepts = kb.get_all_concepts()
        return [
            {
                "concept_id": c["id"],
                "term": c["term"],
                "domain": c["domain"],
                "similarity": 0.0,
                "parameters": c["parameters"],
            }
            for c in concepts
        ][:max_candidates]

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
