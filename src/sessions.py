"""Модуль управления сессиями для AI-Terminator.

Предоставляет класс SessionManager для работы с интерактивными сессиями
пользователей, накопления подсказок и запоминания выбранных доменов.
"""

import logging
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SessionManager:
    """Менеджер сессий для интерактивного уточнения понятий.

    Хранит сессии в памяти с LRU-вытеснением при переполнении.
    Поддерживает TTL для автоматического удаления неактивных сессий.
    """

    def __init__(
        self,
        session_ttl_seconds: int = 1800,
        session_cache_size: int = 1000,
        session_cleanup_interval_seconds: int = 60,
        auto_save_domain_on_ok: bool = True,
    ):
        """Инициализация менеджера сессий.

        Args:
            session_ttl_seconds: Время жизни сессии без активности (по умолчанию 30 мин).
            session_cache_size: Максимальное количество сессий в памяти.
            session_cleanup_interval_seconds: Интервал фоновой очистки.
            auto_save_domain_on_ok: Автоматически сохранять домен из успешного ответа.
        """
        self.session_ttl_seconds = session_ttl_seconds
        self.session_cache_size = session_cache_size
        self.session_cleanup_interval_seconds = session_cleanup_interval_seconds
        self.auto_save_domain_on_ok = auto_save_domain_on_ok

        # OrderedDict для LRU-вытеснения
        self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()  # Для потокобезопасности

        # Фоновый поток для очистки (опционально)
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

    def _generate_session_id(self) -> str:
        """Сгенерировать уникальный идентификатор сессии.

        Returns:
            UUID4 без дефисов (32 символа).
        """
        return uuid.uuid4().hex

    def _cleanup_expired(self) -> None:
        """Удалить просроченные сессии."""
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if current_time - session["last_accessed"] > self.session_ttl_seconds:
                    expired_keys.append(session_id)

            for key in expired_keys:
                del self._sessions[key]
                logger.info(f"Сессия удалена по TTL: {key}")

    def create_session(self, term: str, initial_hints: List[str]) -> str:
        """Создать новую сессию.

        Args:
            term: Исходный термин.
            initial_hints: Начальные подсказки.

        Returns:
            session_id новой сессии.
        """
        session_id = self._generate_session_id()

        # Уникализация подсказок (case-insensitive)
        seen_hints = set()
        unique_hints = []
        for hint in initial_hints:
            hint_lower = hint.lower().strip()
            if hint_lower and hint_lower not in seen_hints:
                seen_hints.add(hint_lower)
                unique_hints.append(hint)

        session = {
            "session_id": session_id,
            "term": term,
            "accumulated_hints": unique_hints,
            "selected_domain": None,
            "created_at": time.time(),
            "last_accessed": time.time(),
        }

        with self._lock:
            # Если кэш полон, вытесняем самую старую
            while len(self._sessions) >= self.session_cache_size:
                oldest_key = next(iter(self._sessions))
                del self._sessions[oldest_key]
                logger.warning(f"Сессия вытеснена (кэш полон): {oldest_key}")

            self._sessions[session_id] = session

        logger.info(
            f"Создана новая сессия: {session_id}, term='{term}', hints={len(unique_hints)}"
        )
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Получить сессию по ID.

        Args:
            session_id: Идентификатор сессии.

        Returns:
            Копия сессии или None, если не найдена/просрочена.
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Сессия не найдена: {session_id}")
                return None

            session = self._sessions[session_id]
            current_time = time.time()

            # Проверка TTL
            if current_time - session["last_accessed"] > self.session_ttl_seconds:
                del self._sessions[session_id]
                logger.info(f"Сессия истекла: {session_id}")
                return None

            # Обновляем last_accessed
            session["last_accessed"] = current_time
            # Перемещаем в конец OrderedDict (LRU)
            self._sessions.move_to_end(session_id)

        # Возвращаем копию (чтобы нельзя было напрямую изменить)
        return session.copy()

    def update_session(
        self,
        session_id: str,
        new_hints: Optional[List[str]] = None,
        selected_domain: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Обновить сессию.

        Args:
            session_id: Идентификатор сессии.
            new_hints: Новые подсказки для добавления.
            selected_domain: Выбранный домен.

        Returns:
            Обновленная сессия или None, если не найдена.
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Сессия не найдена для обновления: {session_id}")
                return None

            session = self._sessions[session_id]
            current_time = time.time()
            session["last_accessed"] = current_time
            self._sessions.move_to_end(session_id)

            # Добавление новых подсказок
            if new_hints:
                seen_hints = {h.lower().strip() for h in session["accumulated_hints"]}
                for hint in new_hints:
                    hint_lower = hint.lower().strip()
                    if hint_lower and hint_lower not in seen_hints:
                        seen_hints.add(hint_lower)
                        session["accumulated_hints"].append(hint)
                        logger.debug(f"Добавлена подска��ка в сессию {session_id}: {hint}")

            # Обновление домена
            if selected_domain:
                session["selected_domain"] = selected_domain
                logger.debug(f"Обновлен домен в сессии {session_id}: {selected_domain}")

        return session.copy()

    def delete_session(self, session_id: str) -> bool:
        """Удалить сессию.

        Args:
            session_id: Идентификатор сессии.

        Returns:
            True при успешном удалении, False если не найдена.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Сессия удалена: {session_id}")
                return True
            return False

    def cleanup_expired(self) -> int:
        """Удалить все просроченные сессии.

        Returns:
            Количество удаленных сессий.
        """
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if current_time - session["last_accessed"] > self.session_ttl_seconds:
                    expired_keys.append(session_id)

            for key in expired_keys:
                del self._sessions[key]

        if expired_keys:
            logger.info(f"Удалено {len(expired_keys)} просроченных сессий")

        return len(expired_keys)

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Получить все активные сессии.

        Returns:
            Список копий сессий.
        """
        with self._lock:
            return [s.copy() for s in self._sessions.values()]

    def start_cleanup_thread(self) -> None:
        """Запустить фоновый поток для очистки просроченных сессий."""
        if self._cleanup_thread is not None:
            return

        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="SessionCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Фоновый поток очистки сессий запущен")

    def stop_cleanup_thread(self) -> None:
        """Остановить фоновый поток для очистки сессий."""
        if self._cleanup_thread is not None:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=2)
            self._cleanup_thread = None
            logger.info("Фоновый поток очистки сессий остановлен")

    def _cleanup_loop(self) -> None:
        """Цикл фоновой очистки."""
        while not self._stop_cleanup.wait(timeout=self.session_cleanup_interval_seconds):
            self.cleanup_expired()
