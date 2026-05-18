"""Модуль конфигурации проекта AI-Terminator."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Set


@dataclass
class Config:
    """Класс для загрузки и управления конфигурацией."""

    db_path: Path
    fasttext_model_path: Path
    synonyms_path: Path
    domain_templates_path: Path
    domain_keywords_path: Path
    min_confidence: float
    max_candidates: int
    max_parameters: int
    use_generative: bool
    generative_model: str
    generative_max_new_tokens: int
    generative_temperature: float
    generative_max_new_params: int
    generative_timeout_seconds: float
    min_parameters_for_generative: int
    generative_keywords: list[str]
    timeout_seconds: float
    cache_embeddings: bool
    log_level: str
    cache_lemma_size: int
    max_synonyms_per_token: int
    use_synonyms: bool
    max_term_length: int
    max_hint_length: int
    word_vector_cache_size: int
    query_cache_size: int
    use_faiss: bool = False
    faiss_index_path: str = ""
    session_ttl_seconds: int = 1800
    session_cache_size: int = 1000
    session_cleanup_interval_seconds: int = 60
    auto_save_domain_on_ok: bool = True
    ambiguity_threshold: float = 0.7
    ambiguity_delta: float = 0.1
    domain_centroid_threshold: float = 0.3
    auto_save_domain_on_fallback: bool = False
    use_relations: bool = False
    relation_max_depth: int = 1
    relation_decay_factor: float = 0.5
    domain_centroids_min_concepts: int = 2

    # Допустимые уровни логирования
    VALID_LOG_LEVELS: Set[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})

    def __post_init__(self):
        """Валидация загруженных значений."""
        self._validate()

    def _validate(self) -> None:
        """Проверка корректности значений конфигурации."""
        # min_confidence должен быть в диапазоне [0, 1]
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence должен быть в диапазоне [0, 1], получено: {self.min_confidence}"
            )

        # max_candidates должен быть положительным
        if self.max_candidates <= 0:
            raise ValueError(
                f"max_candidates должен быть положительным, получено: {self.max_candidates}"
            )

        # max_parameters должен быть положительным
        if self.max_parameters <= 0:
            raise ValueError(
                f"max_parameters должен быть положительным, получено: {self.max_parameters}"
            )

        # timeout_seconds должен быть положительным
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds должен быть положительным, получено: {self.timeout_seconds}"
            )

        # log_level должен быть допустимым
        if self.log_level not in self.VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level должен быть одним из {self.VALID_LOG_LEVELS}, получено: {self.log_level}"
            )

        # cache_lemma_size должен быть положительным
        if self.cache_lemma_size <= 0:
            raise ValueError(
                f"cache_lemma_size должен быть положительным, получено: {self.cache_lemma_size}"
            )

        # max_synonyms_per_token должен быть >= 1
        if self.max_synonyms_per_token < 1:
            raise ValueError(
                f"max_synonyms_per_token должен быть >= 1, получено: {self.max_synonyms_per_token}"
            )

        # max_term_length должен быть положительным
        if self.max_term_length <= 0:
            raise ValueError(
                f"max_term_length должен быть положительным, получено: {self.max_term_length}"
            )

        # max_hint_length должен быть положительным
        if self.max_hint_length <= 0:
            raise ValueError(
                f"max_hint_length должен быть положительным, получено: {self.max_hint_length}"
            )

    @classmethod
    def from_json(
        cls, json_path: str, project_root: Optional[Path] = None
    ) -> "Config":
        """Загрузить конфигурацию из JSON-файла.

        Args:
            json_path: Путь к JSON-файлу (относительный или абсолютный).
            project_root: Корневая директория проекта. Если None, определяется автоматически.

        Returns:
            Config: Объект конфигурации с валидированными значениями.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            json.JSONDecodeError: Если файл содержит некорректный JSON.
            ValueError: Если значения не прошли валидацию.
        """
        json_file = Path(json_path)

        # Если путь относительный, ищем относительно project_root
        if not json_file.is_absolute():
            if project_root is None:
                # Автоматическое определение корня: ищем папку configs
                current = json_file.resolve()
                while current.parent != current:
                    if (current.parent / "configs").exists():
                        project_root = current.parent
                        break
                    current = current.parent
                if project_root is None:
                    project_root = Path.cwd()
            else:
                project_root = Path(project_root).resolve()
            json_file = project_root / json_path

        if not json_file.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Преобразование путей в абсолютные
        if project_root is None:
            project_root = json_file.parent

        def make_absolute(path_str: str) -> Path:
            p = Path(path_str)
            if not p.is_absolute():
                return project_root / p
            return p.resolve()

        return cls(
            db_path=make_absolute(data["db_path"]),
            fasttext_model_path=make_absolute(data["fasttext_model_path"]),
            synonyms_path=make_absolute(data["synonyms_path"]),
            domain_templates_path=make_absolute(data["domain_templates_path"]),
            domain_keywords_path=make_absolute(data.get("domain_keywords_path", "configs/domain_keywords.json")),
            min_confidence=float(data["min_confidence"]),
            max_candidates=int(data["max_candidates"]),
            max_parameters=int(data["max_parameters"]),
            use_generative=bool(data["use_generative"]),
            generative_model=str(data["generative_model"]),
            generative_max_new_tokens=int(data.get("generative_max_new_tokens", 100)),
            generative_temperature=float(data.get("generative_temperature", 0.7)),
            generative_max_new_params=int(data.get("generative_max_new_params", 3)),
            generative_timeout_seconds=float(data.get("generative_timeout_seconds", 2.0)),
            min_parameters_for_generative=int(data.get("min_parameters_for_generative", 5)),
            generative_keywords=list(data.get("generative_keywords", [
                "материал", "размер", "тип", "скорость", "мощность", "вес",
                "длина", "цвет", "напряжение", "ёмкость", "частота", "температура",
                "давление", "форма", "покрытие", "привод", "источник энергии",
            ])),
            timeout_seconds=float(data["timeout_seconds"]),
            cache_embeddings=bool(data["cache_embeddings"]),
            log_level=str(data["log_level"]),
            cache_lemma_size=int(data.get("cache_lemma_size", 1000)),
            max_synonyms_per_token=int(data.get("max_synonyms_per_token", 2)),
            use_synonyms=bool(data.get("use_synonyms", True)),
            max_term_length=int(data.get("max_term_length", 100)),
            max_hint_length=int(data.get("max_hint_length", 50)),
            word_vector_cache_size=int(data.get("word_vector_cache_size", 20000)),
            query_cache_size=int(data.get("query_cache_size", 100)),
            use_faiss=bool(data.get("use_faiss", False)),
            faiss_index_path=str(data.get("faiss_index_path", "")),
            session_ttl_seconds=int(data.get("session_ttl_seconds", 1800)),
            session_cache_size=int(data.get("session_cache_size", 1000)),
            session_cleanup_interval_seconds=int(data.get("session_cleanup_interval_seconds", 60)),
            auto_save_domain_on_ok=bool(data.get("auto_save_domain_on_ok", True)),
            ambiguity_threshold=float(data.get("ambiguity_threshold", 0.7)),
            ambiguity_delta=float(data.get("ambiguity_delta", 0.1)),
            domain_centroid_threshold=float(data.get("domain_centroid_threshold", 0.3)),
            auto_save_domain_on_fallback=bool(data.get("auto_save_domain_on_fallback", False)),
            use_relations=bool(data.get("use_relations", False)),
            relation_max_depth=int(data.get("relation_max_depth", 1)),
            relation_decay_factor=float(data.get("relation_decay_factor", 0.5)),
            domain_centroids_min_concepts=int(data.get("domain_centroids_min_concepts", 2)),
        )

    def to_dict(self) -> dict:
        """Преобразовать конфигурацию в словарь."""
        return {
            "db_path": str(self.db_path),
            "fasttext_model_path": str(self.fasttext_model_path),
            "synonyms_path": str(self.synonyms_path),
            "domain_templates_path": str(self.domain_templates_path),
            "domain_keywords_path": str(self.domain_keywords_path),
            "min_confidence": self.min_confidence,
            "max_candidates": self.max_candidates,
            "max_parameters": self.max_parameters,
            "use_generative": self.use_generative,
            "generative_model": self.generative_model,
            "generative_max_new_tokens": self.generative_max_new_tokens,
            "generative_temperature": self.generative_temperature,
            "generative_max_new_params": self.generative_max_new_params,
            "generative_timeout_seconds": self.generative_timeout_seconds,
            "min_parameters_for_generative": self.min_parameters_for_generative,
            "generative_keywords": self.generative_keywords,
            "timeout_seconds": self.timeout_seconds,
            "cache_embeddings": self.cache_embeddings,
            "log_level": self.log_level,
            "cache_lemma_size": self.cache_lemma_size,
            "max_synonyms_per_token": self.max_synonyms_per_token,
            "use_synonyms": self.use_synonyms,
            "max_term_length": self.max_term_length,
            "max_hint_length": self.max_hint_length,
            "word_vector_cache_size": self.word_vector_cache_size,
            "query_cache_size": self.query_cache_size,
            "use_faiss": self.use_faiss,
            "faiss_index_path": self.faiss_index_path,
            "session_ttl_seconds": self.session_ttl_seconds,
            "session_cache_size": self.session_cache_size,
            "session_cleanup_interval_seconds": self.session_cleanup_interval_seconds,
            "auto_save_domain_on_ok": self.auto_save_domain_on_ok,
            "ambiguity_threshold": self.ambiguity_threshold,
            "ambiguity_delta": self.ambiguity_delta,
            "domain_centroid_threshold": self.domain_centroid_threshold,
            "auto_save_domain_on_fallback": self.auto_save_domain_on_fallback,
            "use_relations": self.use_relations,
            "relation_max_depth": self.relation_max_depth,
            "relation_decay_factor": self.relation_decay_factor,
            "domain_centroids_min_concepts": self.domain_centroids_min_concepts,
        }