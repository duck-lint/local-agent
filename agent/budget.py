from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised when a budget cap is hit pre-generation. During generation,
    callers use BudgetTracker.check_during_generation() which returns a
    degrade flag instead of raising."""

    def __init__(self, *, code: str, label: str, kind: str, detail: str = "") -> None:
        super().__init__(f"{code}: {kind} budget exceeded at {label} ({detail})")
        self.code = code
        self.label = label
        self.kind = kind
        self.detail = detail


@dataclass
class BudgetTracker:
    """Per-query budget enforcement.

    wall_clock_s and max_prompt_tokens are hard caps. The tracker is
    deterministic, does no I/O, and uses time.perf_counter() for timing
    (monotonic, immune to wall-clock adjustments).
    """

    wall_clock_s: float
    max_prompt_tokens: int
    _start_perf: float = field(default_factory=time.perf_counter)
    _prompt_tokens_used: int = 0
    _degrade_during_generation: bool = False

    def __post_init__(self) -> None:
        if self.wall_clock_s <= 0:
            raise ValueError(f"wall_clock_s must be positive, got {self.wall_clock_s}")
        if self.max_prompt_tokens <= 0:
            raise ValueError(f"max_prompt_tokens must be positive, got {self.max_prompt_tokens}")

    def elapsed_ms(self) -> int:
        return int(round((time.perf_counter() - self._start_perf) * 1000))

    def remaining_ms(self) -> int:
        return max(0, int(self.wall_clock_s * 1000) - self.elapsed_ms())

    def prompt_tokens_used(self) -> int:
        return self._prompt_tokens_used

    def degrade_flag(self) -> bool:
        return self._degrade_during_generation

    def record_prompt_tokens(self, n: int) -> None:
        new_total = self._prompt_tokens_used + int(n)
        if new_total > self.max_prompt_tokens:
            raise BudgetExceededError(
                code="BUDGET_EXCEEDED",
                label="record_prompt_tokens",
                kind="tokens",
                detail=f"used={new_total} max={self.max_prompt_tokens}",
            )
        self._prompt_tokens_used = new_total

    def check_pre_generation(self, label: str) -> None:
        if self.elapsed_ms() >= int(self.wall_clock_s * 1000):
            raise BudgetExceededError(
                code="BUDGET_EXCEEDED",
                label=label,
                kind="wall_clock",
                detail=f"elapsed_ms={self.elapsed_ms()} wall_clock_s={self.wall_clock_s}",
            )

    def check_during_generation(self) -> bool:
        if self.elapsed_ms() >= int(self.wall_clock_s * 1000):
            self._degrade_during_generation = True
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "wall_clock_s": self.wall_clock_s,
            "max_prompt_tokens": self.max_prompt_tokens,
            "elapsed_ms": self.elapsed_ms(),
            "remaining_ms": self.remaining_ms(),
            "prompt_tokens_used": self.prompt_tokens_used(),
            "degrade_during_generation": self.degrade_flag(),
        }
