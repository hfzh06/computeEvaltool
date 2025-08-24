# logging_utils.py
from __future__ import annotations

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple

init_loggers = {}
_init_lock = threading.Lock()

DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - "
    "%(levelname)s - %(process)d - rank=%(rank)s - %(message)s"
)
SIMPLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Choose default level via env, fallback INFO
DEFAULT_LEVEL = logging.DEBUG if os.getenv("COMPUTEEVALTOOL_LOG_LEVEL", "INFO").upper() == "DEBUG" else logging.INFO

# Global formatters; datefmt is left default (local time) unless utc=True on get_logger
detailed_formatter = logging.Formatter(DETAILED_FORMAT)
simple_formatter = logging.Formatter(SIMPLE_FORMAT)

# Optional: don’t stomp root config unless explicitly allowed
if os.getenv("COMPUTEEVALTOOL_FORCE_BASICCONFIG", "1") == "1":
    logging.basicConfig(format=SIMPLE_FORMAT, level=DEFAULT_LEVEL, force=True)

# Quiet some noisy libs
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("modelscope").setLevel(logging.ERROR)


def _resolve_level(level: int | str | None) -> int:
    if level is None:
        return DEFAULT_LEVEL
    if isinstance(level, int):
        return level
    try:
        return getattr(logging, str(level).upper())
    except Exception:
        return DEFAULT_LEVEL


def _dist_info() -> Tuple[bool, int, int, bool]:
    """Return (is_dist, rank, world_size, is_master). Uses torch.distributed or env fallback."""
    # torch.distributed path
    try:
        import torch.distributed as dist  # type: ignore

        if getattr(dist, "is_available", lambda: False)() and getattr(dist, "is_initialized", lambda: False)():
            rank = dist.get_rank()
            world = dist.get_world_size()
            return True, rank, world, (rank == 0)
    except Exception:
        pass
    # Env fallback (works for torchrun/mpirun setups that export RANK/WORLD_SIZE)
    try:
        rank = int(os.getenv("RANK", "0"))
        world = int(os.getenv("WORLD_SIZE", "1"))
        return (world > 1), rank, world, (rank == 0)
    except Exception:
        return False, 0, 1, True


class _RankFilter(logging.Filter):
    """Injects `rank` into records so format strings can include %(rank)s."""
    def __init__(self, rank: int):
        super().__init__()
        self._rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "rank"):
            record.rank = self._rank
        return True


def _make_file_handler(
    path: str,
    mode: str,
    level: int,
    use_detailed: bool,
    rotate: bool,
    max_bytes: int,
    backup_count: int,
    utc: bool,
    rank: int,
) -> logging.Handler:
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    if rotate:
        fh: logging.Handler = RotatingFileHandler(path, mode=mode, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    else:
        fh = logging.FileHandler(path, mode=mode, encoding="utf-8")

    # Set formatter
    formatter = logging.Formatter(DETAILED_FORMAT if use_detailed else SIMPLE_FORMAT)
    if utc:
        # Make formatter output UTC
        import time

        formatter.converter = time.gmtime
    fh.setFormatter(formatter)
    fh.setLevel(level)
    fh.addFilter(_RankFilter(rank))
    return fh


def _make_stream_handler(level: int, use_detailed: bool, utc: bool, rank: int) -> logging.Handler:
    sh = logging.StreamHandler()
    formatter = logging.Formatter(DETAILED_FORMAT if use_detailed else SIMPLE_FORMAT)
    if utc:
        import time

        formatter.converter = time.gmtime
    sh.setFormatter(formatter)
    sh.setLevel(level)
    sh.addFilter(_RankFilter(rank))
    return sh


def get_logger(
    log_file: Optional[str] = None,
    log_level: int | str = DEFAULT_LEVEL,
    file_mode: str = "w",
    force: bool = False,
    *,
    name: Optional[str] = None,
    rotate: bool = False,
    rotate_max_bytes: int = 10 * 1024 * 1024,  # 10MB
    rotate_backups: int = 5,
    utc: bool = False,
    replace_file_handler: bool = False,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Args:
        log_file: If set, add a file handler (on master rank only).
        log_level: int or string ("DEBUG"/"INFO"/...).
        file_mode: File open mode for the file handler.
        force: If True, update existing handlers to the new level/format and add file handler if missing.
        name: Logger name (default: top-level package of this module).
        rotate: Use RotatingFileHandler instead of FileHandler.
        rotate_max_bytes: Max file size before rotation (when rotate=True).
        rotate_backups: Number of rotated backups to keep.
        utc: Timestamps in UTC if True, else local time.
        replace_file_handler: If True and an existing FileHandler targets a different file, replace it.
    """
    level = _resolve_level(log_level)
    use_detailed = level <= logging.DEBUG
    is_dist, rank, _world, is_master = _dist_info()

    logger_name = name or __name__.split(".")[0]
    with _init_lock:
        logger = logging.getLogger(logger_name)
        logger.propagate = False

        if logger_name in init_loggers and not force:
            return logger

        # (Re)configure core level
        logger.setLevel(level if is_master else logging.ERROR)

        # Build/refresh handlers
        if force:
            # Update existing handlers
            for h in list(logger.handlers):
                # Optionally replace/retarget file handler
                if isinstance(h, (logging.FileHandler, RotatingFileHandler)):
                    if replace_file_handler and isinstance(h, logging.FileHandler) and log_file:
                        logger.removeHandler(h)
                        h.close()
                        continue  # we'll add the new one below
                # Refresh formatter & level
                formatter = logging.Formatter(DETAILED_FORMAT if use_detailed else SIMPLE_FORMAT)
                if utc:
                    import time

                    formatter.converter = time.gmtime
                h.setFormatter(formatter)
                h.setLevel(level if is_master else logging.ERROR)
        else:
            # Fresh init: remove any inherited handlers, then add our own
            logger.handlers[:] = []
            # Console on master only to avoid duplicate noisy output from all ranks
            if is_master:
                logger.addHandler(_make_stream_handler(level, use_detailed, utc, rank))

        # Add file handler if requested and we’re master
        if is_master and log_file:
            has_file = any(isinstance(h, (logging.FileHandler, RotatingFileHandler)) for h in logger.handlers)
            needs_add = (not has_file) or (force and replace_file_handler)
            if needs_add:
                fh = _make_file_handler(
                    path=log_file,
                    mode=file_mode,
                    level=level,
                    use_detailed=use_detailed,
                    rotate=rotate,
                    max_bytes=rotate_max_bytes,
                    backup_count=rotate_backups,
                    utc=utc,
                    rank=rank,
                )
                logger.addHandler(fh)

        init_loggers[logger_name] = True
        return logger


def configure_logging(debug: bool, log_file: Optional[str] = None) -> None:
    """
    Backwards-compatible convenience helper.
    - If log_file is set, ensure a file handler is attached.
    - If debug is True, flip to DEBUG + detailed format.
    """
    if log_file:
        get_logger(log_file=log_file, force=True, replace_file_handler=False)
    if debug:
        get_logger(log_level=logging.DEBUG, force=True)

