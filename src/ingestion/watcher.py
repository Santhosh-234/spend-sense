"""Directory watcher — triggers the pipeline when a receipt image lands in data/raw."""
import time
from pathlib import Path
from typing import Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from loguru import logger


_VALID_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"})


class _ReceiptEventHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[Path], None]) -> None:
        self._callback = callback
        super().__init__()

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in _VALID_EXTENSIONS:
            logger.debug(f"Ignored non-image file: {path.name}")
            return
        logger.info(f"Receipt detected → {path.name}")
        try:
            self._callback(path)
        except Exception:
            logger.exception(f"Pipeline failed for {path.name}")


class ReceiptWatcher:
    """Blocking watcher — call .start() to enter the watch loop."""

    def __init__(self, config: dict, pipeline_callback: Callable[[Path], None]) -> None:
        self._watch_dir = Path(config["paths"]["raw_data"])
        self._observer = Observer()
        self._handler = _ReceiptEventHandler(pipeline_callback)

    def start(self) -> None:
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        self._observer.schedule(self._handler, str(self._watch_dir), recursive=False)
        self._observer.start()
        logger.info(f"Watching {self._watch_dir} for receipts — Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
        logger.info("Watcher stopped cleanly.")
