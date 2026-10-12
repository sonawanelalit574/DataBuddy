"""Resource guards for analytical tool calls."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class CalculationTimeout(Exception):
    def __init__(self) -> None:
        super().__init__("calculation timed out")
        self.user_message = (
            "That calculation took too long and was stopped. Try a smaller file or a narrower question."
        )


def run_with_timeout(func: Callable[[], T], seconds: float) -> T:
    """Stop a tool that runs past the limit on platforms that support timers.

    Windows has no SIGALRM, so the call runs without a wall-clock interrupt
    there. File size and row limits still apply.
    """

    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return func()

    def _handle(signum, frame):  # noqa: ARG001
        raise CalculationTimeout()

    previous = signal.signal(signal.SIGALRM, _handle)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
