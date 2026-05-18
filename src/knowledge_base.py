"""Модуль базы знаний для AI-Terminator.

Предоставляет класс KnowledgeBase для доступа к SQLite-базе данных.
"""

import sqlite3
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Any

from src.lemmatizer import Lemmatizer
from src.synonyms import SynonymDict
from src.embeddings import FastTextWrapper

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Обёртка над SQLite-базой данных для работы с понятиями и параметрами.

    Предоставляет методы:
    - get_all_concepts(): загрузка всех понятий с параметрами
    - compute_concept_embedding(): вычисление вектора понятия
    - update_all_embeddings(): пересчёт всех эмбеддингов
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: Optional[FastTextWrapper] = None,
        synonym_dict: Optional[SynonymDict] = None,
    ):
        """Инициализация базы знаний.

        Args:
            db_path: Путь к файлу базы данных.
            embedding_model: Экземпляр FastTextWrapper для вычисления эмбеддингов.
            synonym_dict: Экземпляр SynonymDict для получения синонимов.
        """
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.embedding_model = embedding_model
        self.synonym_dict = synonym_dict
        self._lemmatizer = Lemmatizer()
        self._cache: Optional[List[Dict[str, Any]]] = None
        self.logger = logger

    def _blob_to_vector(self, blob: Optional[bytes]) -> np.ndarray:
        """Преобразовать BLOB в вектор numpy.

        Args:
            blob: Байтовое представление вектора.

        Returns:
            Вектор numpy (float32, размерность 300).
        """
        if blob is None:
            return np.zeros(300, dtype=np.float32)

        try:
            vec = np.frombuffer(blob, dtype="<f4")
            if len(vec) != 300:
                self.logger.warning(
                    f"Некорректная размерность вектора: {len(vec)}, ожидается 300"
                )
                return np.zeros(300, dtype=np.float32)
            return vec.copy()
        except Exception as e:
            self.logger.warning(f"Ошибка преобразования BLOB в вектор: {e}")
            return np.zeros(300, dtype=np.float32)

    def _parse_enum(self, value: Optional[str]) -> Optional[List[str]]:
        """Разобрать JSON-строку с enum значениями.

        Args:
            value: JSON-строка или None.

        Returns:
            Список значений или None.
        """
        if not value:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Ошибка парсинга enum_values: {e}")
            return None

    def get_all_concepts(
        self, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Загрузить все понятия с их параметрами.

        Args:
            use_cache: Использовать кэш (по умолчанию True).

        Returns:
            Список понятий с параметрами.
        """
        if use_cache and self._cache is not None:
            return self._cache

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.id, c.term, c.domain, c.embedding,
                   p.name, p.label_ru, p.type, p.description, p.unit, p.enum_values, p.confidence
            FROM concepts c
            LEFT JOIN parameters p ON c.id = p.concept_id
            ORDER BY c.id, p.id
        """)

        concepts_dict: Dict[str, Dict[str, Any]] = {}

        for row in cursor:
            cid = row["id"]
            if cid not in concepts_dict:
                concepts_dict[cid] = {
                    "id": cid,
                    "term": row["term"],
                    "domain": row["domain"],
                    "embedding": self._blob_to_vector(row["embedding"]),
                    "parameters": [],
                }

            if row["name"] is not None:
                param = {
                    "name": row["name"],
                    "label_ru": row["label_ru"],
                    "type": row["type"],
                    "description": row["description"] or "",
                    "unit": row["unit"],
                    "enum_values": self._parse_enum(row["enum_values"]),
                    "confidence": row["confidence"] if row["confidence"] else 1.0,
                    "source": "knowledge_base",
                }
                concepts_dict[cid]["parameters"].append(param)

        result = list(concepts_dict.values())

        if use_cache:
            self._cache = result

        return result

    def compute_concept_embedding(self, term: str) -> np.ndarray:
        """Вычислить эмбеддинг понятия по его термину.

        Использует ту же логику, что и для запроса:
        - Вес слов термина: 0.7 / len(lemmas)
        - Вес синонимов: 0.3 / len(synonyms)

        Args:
            term: Термин понятия.

        Returns:
            Нормализованный вектор (float32, размерность 300).
        """
        if self.embedding_model is None or self.synonym_dict is None:
            raise RuntimeError(
                "embedding_model и synonym_dict должны быть инициализированы"
            )

        # Получить леммы терма
        lemmas = self._lemmatizer.lemmatize_phrase(term)
        if not lemmas:
            return np.zeros(300, dtype=np.float32)

        # Веса для слов терма
        term_weight_per_word = 0.7 / len(lemmas)
        tokens_weights: List[tuple[str, float]] = []
        for lemma in lemmas:
            tokens_weights.append((lemma, term_weight_per_word))

        # Сбор синонимов (извлекаем только слова из кортежей)
        all_synonyms: set[str] = set()
        for lemma in lemmas:
            for syn in self.synonym_dict.get_synonyms(lemma, max_synonyms=2):
                # syn - это кортеж (word, weight), берем только слово
                if isinstance(syn, tuple) and len(syn) >= 1:
                    syn_word = syn[0]
                    # Лемматизируем синоним
                    syn_lemma = self._lemmatizer.lemmatize_word(syn_word)
                    if syn_lemma:
                        all_synonyms.add(syn_lemma)
                elif isinstance(syn, str):
                    syn_lemma = self._lemmatizer.lemmatize_word(syn)
                    if syn_lemma:
                        all_synonyms.add(syn_lemma)

        # Веса для синонимов
        if all_synonyms:
            syn_weight = 0.3 / len(all_synonyms)
            for syn in all_synonyms:
                tokens_weights.append((syn, syn_weight))

        # Взвешенная сумма векторов
        weighted_sum = np.zeros(300, dtype=np.float64)
        for token, w in tokens_weights:
            vec = self.embedding_model.get_phrase_vector(token)
            weighted_sum += w * vec

        # L2-нормализация
        norm = np.linalg.norm(weighted_sum)
        if norm > 1e-9:
            weighted_sum = weighted_sum / norm
        else:
            self.logger.warning(f"Нулевой вектор для термина '{term}'")

        return weighted_sum.astype("<f4")

    def update_all_embeddings(self) -> None:
        """Пересчитать эмбеддинги всех понятий."""
        if self.embedding_model is None or self.synonym_dict is None:
            raise RuntimeError(
                "embedding_model и synonym_dict должны быть инициализированы"
            )

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, term FROM concepts")
        concepts = cursor.fetchall()

        updated = 0
        for row in concepts:
            concept_id = row["id"]
            term = row["term"]

            try:
                embedding = self.compute_concept_embedding(term)
                cursor.execute(
                    "UPDATE concepts SET embedding = ? WHERE id = ?",
                    (embedding.tobytes(), concept_id),
                )
                updated += 1
            except Exception as e:
                self.logger.error(f"Ошибка вычисления эмбеддинга для '{term}': {e}")

        self.conn.commit()
        self._cache = None
        self.logger.info(f"Обновлено {updated} эмбеддингов")

    def get_all_relations(self, concept_id: str) -> List[Dict[str, Any]]:
        """Получить все связи для понятия.

        Args:
            concept_id: Идентификатор понятия.

        Returns:
            Список связей.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_concept_id, target_concept_id, relation_type, confidence
            FROM relations
            WHERE source_concept_id = ?
            ORDER BY confidence DESC
        """, (concept_id,))
        
        relations = []
        for row in cursor:
            relations.append({
                "source_concept_id": row["source_concept_id"],
                "target_concept_id": row["target_concept_id"],
                "relation_type": row["relation_type"],
                "confidence": row["confidence"],
            })
        return relations

    def get_related_terms(self, concept_id: str, max_terms: int = 3) -> List[Dict[str, Any]]:
        """Получить связанные термины для понятия.

        Args:
            concept_id: Идентификатор понятия.
            max_terms: Максимальное количество связанных терминов.

        Returns:
            Список связанных терминов с их типами связи.
        """
        relations = self.get_all_relations(concept_id)
        if not relations:
            return []
        
        cursor = self.conn.cursor()
        related = []
        for rel in relations[:max_terms]:
            cursor.execute(
                "SELECT id, term, domain FROM concepts WHERE id = ?",
                (rel["target_concept_id"],)
            )
            row = cursor.fetchone()
            if row:
                related.append({
                    "concept_id": row["id"],
                    "term": row["term"],
                    "domain": row["domain"],
                    "relation_type": rel["relation_type"],
                    "confidence": rel["confidence"],
                })
        return related

    def get_constraints(self, concept_id: str) -> List[Dict[str, Any]]:
        """Получить ограничения для понятия.

        Args:
            concept_id: Идентификатор понятия.

        Returns:
            Список ограничений.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT type, parameter, value, description
            FROM concept_constraints
            WHERE concept_id = ?
        """, (concept_id,))
        
        constraints = []
        for row in cursor:
            constraints.append({
                "type": row["type"],
                "parameter": row["parameter"],
                "value": row["value"],
                "description": row["description"],
            })
        return constraints

    def close(self) -> None:
        """Закрыть соединение с базой данных."""
        if self.conn:
            self.conn.close()
