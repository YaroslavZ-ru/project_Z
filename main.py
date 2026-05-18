#!/usr/bin/env python3
"""Точка входа в приложение AI-Terminator.

Принимает входные данные (термин и подсказки), возвращает структурированный
ответ с параметрами понятия.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Добавляем корень проекта в path для импортов
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.preprocess import preprocess
from src.embeddings import FastTextWrapper
from src.vectorize import vectorize
from src.knowledge_base import KnowledgeBase
from src.search import search_similar_concepts
from src.fallback import generate_template_response, load_templates, load_domain_keywords
from src.synonyms import SynonymDict
from src.aggregation import aggregate_parameters, determine_context
from src.cache import QueryVectorCache
from src.generative import GenerativeHelper
from src.sessions import SessionManager


def run_pipeline(
    term: str,
    hints: list[str],
    config: Config,
    debug: bool = False,
    cache: Optional[QueryVectorCache] = None,
    generative_helper: Optional[GenerativeHelper] = None,
    session_manager: Optional[SessionManager] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Запустить полный конвейер обработки термина.

    Args:
        term: Анализируемый термин.
        hints: Список уточняющих слов (0-3 слова).
        config: Экземпляр конфигурации.
        debug: Режим отладки (добавить промежуточные данные в ответ).
        cache: Экземпляр QueryVectorCache для кэширования (опционально).

    Returns:
        dict: Структурированный ответ.
    """
    logger.info(f"Pipeline запущен: term='{term}', hints_count={len(hints)}")

    # Обработка session_id для интерактивных сессий
    accumulated_hints = hints.copy()
    domain_filter = None

    if session_manager and session_id:
        # Получаем существующую сессию
        existing_session = session_manager.get_session(session_id)
        if existing_session is None:
            return {
                "status": "error",
                "message": "Session not found or expired",
                "term": term,
                "selected_context": {"domain": "не определено", "confidence": 0.0},
                "parameters": [],
                "suggested_refinements": [],
                "warnings": ["Сессия не найдена или истекла"],
            }

        # Объединяем подсказки из сессии и текущие
        accumulated_hints = existing_session.get("accumulated_hints", []).copy()
        accumulated_hints.extend(hints)  # Добавляем новые подсказки

        # Домен для фильтрации (приоритет: параметр > сессия)
        domain_filter = existing_session.get("selected_domain")

        logger.debug(f"Используем сессию {session_id}: hints={accumulated_hints}, domain={domain_filter}")

    # Инициализация компонентов
    synonym_dict = SynonymDict(config.synonyms_path)
    emb_model = FastTextWrapper(
        str(config.fasttext_model_path),
        str(config.db_path.parent / "models" / "static_embeddings.npy"),
        cache_size=config.word_vector_cache_size
    )
    generative_helper = GenerativeHelper(
        use_generative=config.use_generative,
        model_name=config.generative_model,
        max_new_tokens=config.generative_max_new_tokens,
        temperature=config.generative_temperature,
        max_new_params=config.generative_max_new_params,
        timeout_seconds=config.generative_timeout_seconds,
        keywords=config.generative_keywords,
    )
    
    # Инициализация SessionManager
    session_manager = SessionManager(
        session_ttl_seconds=config.session_ttl_seconds,
        session_cache_size=config.session_cache_size,
        session_cleanup_interval_seconds=config.session_cleanup_interval_seconds,
        auto_save_domain_on_ok=config.auto_save_domain_on_ok,
    )
    
    # Предобработка
    processed = preprocess(term, hints, synonym_dict, config)

    if processed.get("status") == "error":
        return {
            "status": "error",
            "message": processed.get("message", "Ошибка предобработки"),
            "term": term,
            "selected_context": {"domain": "не определено", "confidence": 0.0},
            "parameters": [],
            "suggested_refinements": [],
            "warnings": [processed.get("message", "")],
        }

    # Проверка кэша
    if cache is not None:
        cached_result = cache.get(term, hints, config)
        if cached_result is not None:
            query_vector, _ = cached_result
            logger.debug(f"Кэш использован для term='{term}'")
        else:
            # Векторизация
            query_vector = vectorize(processed, emb_model, normalize=True)
            # Сохраняем в кэш
            cache.set(term, hints, config, query_vector, {"preprocessed": processed})
    else:
        # Векторизация без кэша
        query_vector = vectorize(processed, emb_model, normalize=True)

    # Инициализация базы знаний
    kb = KnowledgeBase(config.db_path, emb_model, synonym_dict)
    
    # Пересчёт эмбеддингов при первом запуске (если они случайные)
    # Проверяем, нулевые ли эмбеддинги
    concepts = kb.get_all_concepts(use_cache=False)
    if concepts and np.linalg.norm(concepts[0]['embedding']) < 0.1:
        logger.info("Эмбеддинги случайные, пересчитываем...")
        kb.update_all_embeddings()
        logger.info("Эмбеддинги пересчитаны")

    # Поиск похожих понятий
    candidates = search_similar_concepts(
        query_vector, kb, config.min_confidence, config.max_candidates,
        use_faiss=getattr(config, "use_faiss", False)
    )
    
    # Если все векторы нулевые (нет fastText модели), используем fallback
    if np.linalg.norm(query_vector) < 1e-6:
        logger.info("Вектор запроса нулевой, используем fallback")
        candidates = []  # Явно пустой список для fallback

    # Формирование ответа
    if candidates:
        # Агрегация параметров
        parameters = aggregate_parameters(
            candidates,
            processed.get("hints_lemmas", []),
            config.max_parameters
        )
        
        # Генеративное достраивание (если параметров мало и включен режим)
        if generative_helper and generative_helper.is_available():
            min_params_for_gen = getattr(config, "min_parameters_for_generative", 5)
            if len(parameters) < min_params_for_gen:
                new_params, refinements = generative_helper.generate_suggestions(
                    term, hints, parameters
                )
                if new_params:
                    parameters.extend(new_params)
                    # Ограничиваем общее количество параметров
                    if len(parameters) > config.max_parameters:
                        parameters = parameters[:config.max_parameters]
                    logger.info(f"Генеративное достраивание: добавлено {len(new_params)} параметров")
                if refinements:
                    suggested_refinements.extend(refinements)
        
        # Определение контекста
        selected_context = determine_context(candidates)
        
        # Получение связанных терминов (если есть таблица relations)
        related_terms = []
        if candidates:
            first_candidate_id = candidates[0].get("concept_id")
            if first_candidate_id:
                related_terms = kb.get_related_terms(first_candidate_id, max_terms=3)
        
        # Получение ограничений (если есть таблица concept_constraints)
        constraints = []
        if candidates:
            first_candidate_id = candidates[0].get("concept_id")
            if first_candidate_id:
                constraints = kb.get_constraints(first_candidate_id)
        
        # Генерация suggested_refinements
        suggested_refinements = []
        
        # Если омонимия, добавляем подсказку для уточнения
        if "context_candidates" in selected_context:
            suggested_refinements.append(
                "Уточните контекст: выберите домен или добавьте тематическую подсказку"
            )
        
        # Добавляем подсказки на основе подсказок пользователя
        if processed.get("hints_lemmas"):
            # Проверяем, не противоречат ли подсказки выбранному домену
            if "context_candidates" not in selected_context:
                domain = selected_context.get("domain", "")
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
                elif domain == "слесарный инструмент":
                    suggested_refinements.append(
                        "Можно добавить параметр 'покрытие' (хромирование, фосфатирование)"
                    )
        
        # Добавление предупреждений
        warnings = []
        
        # Если омонимия
        if "context_candidates" in selected_context:
            warnings.append(
                "Подсказки имеют низкую семантическую связность, возможен выбор неверного контекста"
            )
        
        # Если параметров мало
        if len(parameters) < 3:
            warnings.append(
                "Слишком мало параметров. Уточните контекст или добавьте подсказки"
            )
        
        # Обновление сессии (если используется)
        if session_manager and session_id:
            session_manager.update_session(
                session_id,
                new_hints=hints,
                selected_domain=domain_filter
            )
        
        response = {
            "status": "ok",
            "term": term,
            "selected_context": selected_context,
            "parameters": parameters,
            "suggested_refinements": suggested_refinements,
            "warnings": warnings,
            "related_terms": related_terms,
            "constraints": constraints,
        }
    else:
        # Fallback-режим
        templates = load_templates(config.domain_templates_path)
        domain_keywords = load_domain_keywords(config.domain_keywords_path)
        response = generate_template_response(
            term, hints, processed, templates, domain_keywords, config.max_parameters
        )
        
        # Добавление предупреждений для fallback
        if "warnings" not in response:
            response["warnings"] = []
        response["warnings"].append(
            "Понятие не найдено в базе знаний, параметры предположительные"
        )
        
        # Обновление сессии (если используется)
        if session_manager and session_id:
            session_manager.update_session(
                session_id,
                new_hints=hints,
                selected_domain=domain_filter
            )

    if debug:
        response["debug_info"] = {
            "query_vector": query_vector[:10].tolist(),
            "candidates_raw": candidates[:5] if candidates else [],
            "scores_distribution": [c["similarity"] for c in candidates[:10]] if candidates else [],
        }

    return response


def main():
    """Основная точка входа."""
    # Пример входных данных
    test_input = {
        "term": "ключи",
        "hints": ["техника", "вращение"],
        "debug": False,
    }

    print("AI-Terminator: запуск...")

    # Загрузка конфигурации
    try:
        config = Config.from_json("configs/config.json")
        print(f"Конфигурация загружена: {config.log_level}")
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return

    # Обработка входных данных
    # Создаем кэш для векторов запросов
    cache = QueryVectorCache(max_size=config.query_cache_size)
    
    # Создаем SessionManager для поддержки интерактивных сессий
    session_manager = SessionManager(
        session_ttl_seconds=config.session_ttl_seconds,
        session_cache_size=config.session_cache_size,
        session_cleanup_interval_seconds=config.session_cleanup_interval_seconds,
        auto_save_domain_on_ok=config.auto_save_domain_on_ok,
    )
    
    # Получаем session_id из входных данных (если есть)
    session_id = test_input.get("session_id")
    
    result = run_pipeline(
        test_input["term"],
        test_input.get("hints", []),
        config,
        debug=test_input.get("debug", False),
        cache=cache,
        session_manager=session_manager,
        session_id=session_id
    )
    
    # Если сессия была создана, добавляем session_id в ответ
    if session_id is None and result.get("status") == "ok":
        # Создаем новую сессию
        new_session_id = session_manager.create_session(
            test_input["term"],
            test_input.get("hints", [])
        )
        result["session_id"] = new_session_id
        logger.info(f"Создана новая сессия: {new_session_id}")

    # Вывод результата
    print("\nРезультат обработки:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()