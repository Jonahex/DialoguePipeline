from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class ProcessingCancelled(BaseException):
    """Raised when the active UI processing operation requests cancellation."""


_state = threading.local()


@contextmanager
def cancellation_scope(event: threading.Event) -> Iterator[None]:
    previous = getattr(_state, "event", None)
    _state.event = event
    try:
        yield
    finally:
        _state.event = previous


def check_processing_cancelled() -> None:
    event = getattr(_state, "event", None)
    if event is not None and event.is_set():
        raise ProcessingCancelled("Processing cancelled by user.")
