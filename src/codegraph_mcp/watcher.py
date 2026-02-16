"""File watcher — monitors KB directories for markdown changes, auto-reindexes.

Inspired by memsearch (zilliztech). Uses watchdog + threading.Timer debounce.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_MS = 1500


class _MarkdownHandler(FileSystemEventHandler):
    """Dispatch markdown file events to a callback with debounce."""

    def __init__(
        self,
        callback: Callable[[str, Path], None],
        extensions: tuple[str, ...] = (".md", ".markdown"),
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
    ) -> None:
        self._callback = callback
        self._extensions = extensions
        self._debounce_s = debounce_ms / 1000.0
        self._timers: dict[str, threading.Timer] = {}
        self._pending: dict[str, str] = {}  # path -> latest event_type
        self._lock = threading.Lock()

    def _is_markdown(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._extensions

    def _is_ignored(self, path: str) -> bool:
        ignore = (".git/", ".venv/", "node_modules/", ".solo/", "archive/")
        return any(p in path for p in ignore)

    def _schedule(self, event_type: str, path: str) -> None:
        with self._lock:
            self._pending[path] = event_type
            if path in self._timers:
                self._timers[path].cancel()
            timer = threading.Timer(self._debounce_s, self._fire, args=(path,))
            self._timers[path] = timer
            timer.start()

    def _fire(self, path: str) -> None:
        with self._lock:
            event_type = self._pending.pop(path, None)
            self._timers.pop(path, None)
        if event_type:
            logger.debug("Debounced %s: %s", event_type, path)
            self._callback(event_type, Path(path))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path) and not self._is_ignored(event.src_path):
            self._schedule("created", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path) and not self._is_ignored(event.src_path):
            self._schedule("modified", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path) and not self._is_ignored(event.src_path):
            self._schedule("deleted", event.src_path)

    def cancel_all(self) -> None:
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()
            self._pending.clear()


class KBWatcher:
    """Watch KB directories for markdown changes and auto-reindex.

    Parameters
    ----------
    paths:
        Directories to watch (typically the KB root).
    callback:
        Called with ``(event_type, file_path)`` on change.
        ``event_type`` is one of ``"created"``, ``"modified"``, ``"deleted"``.
    debounce_ms:
        Debounce delay in milliseconds. Multiple events for the same
        file within this window are collapsed into one callback.
        Defaults to 1500 ms.
    """

    def __init__(
        self,
        paths: list[str | Path],
        callback: Callable[[str, Path], None],
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
    ) -> None:
        self._paths = [Path(p).expanduser().resolve() for p in paths]
        self._handler = _MarkdownHandler(callback, debounce_ms=debounce_ms)
        self._observer = Observer()

    def start(self) -> None:
        for p in self._paths:
            if p.is_dir():
                self._observer.schedule(self._handler, str(p), recursive=True)
                logger.info("Watching %s", p)
        self._observer.start()

    def stop(self) -> None:
        self._handler.cancel_all()
        self._observer.stop()
        self._observer.join()

    def __enter__(self) -> KBWatcher:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
