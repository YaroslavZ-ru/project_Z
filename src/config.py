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
    min_confidence: float
    max_candidates: int
    max_parameters: int
    use_generative: bool
    generative_model: str
    timeout_seconds: float
    cache_embeddings: bool
    log_level: str

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
            min_confidence=float(data["min_confidence"]),
            max_candidates=int(data["max_candidates"]),
            max_parameters=int(data["max_parameters"]),
            use_generative=bool(data["use_generative"]),
            generative_model=str(data["generative_model"]),
            timeout_seconds=float(data["timeout_seconds"]),
            cache_embeddings=bool(data["cache_embeddings"]),
            log_level=str(data["log_level"]),
        )

    def to_dict(self) -> dict:
        """Преобразовать конфигурацию в словарь."""
        return {
            "db_path": str(self.db_path),
            "fasttext_model_path": str(self.fasttext_model_path),
            "synonyms_path": str(self.synonyms_path),
            "domain_templates_path": str(self.domain_templates_path),
            "min_confidence": self.min_confidence,
            "max_candidates": self.max_candidates,
            "max_parameters": self.max_parameters,
            "use_generative": self.use_generative,
            "generative_model": self.generative_model,
            "timeout_seconds": self.timeout_seconds,
            "cache_embeddings": self.cache_embeddings,
            "log_level": self.log_level,
        }