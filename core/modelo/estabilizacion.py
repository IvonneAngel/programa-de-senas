from __future__ import annotations

from collections import Counter, deque

from core.base.mensajes import SHOW_HAND_TEXT


class TranslationStabilizer:
    def __init__(
        self,
        window_size: int = 7,
        min_votes: int = 4,
        reset_labels: set[str] | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if min_votes < 1:
            raise ValueError("min_votes must be at least 1")
        if min_votes > window_size:
            raise ValueError("min_votes cannot be greater than window_size")

        self.window_size = window_size
        self.min_votes = min_votes
        self.reset_labels = reset_labels or {SHOW_HAND_TEXT, "MUESTRA TU MANO"}
        self._history: deque[str] = deque(maxlen=window_size)
        self._stable_label = ""

    @property
    def stable_label(self) -> str:
        return self._stable_label

    def reset(self, label: str = "") -> str:
        self._history.clear()
        self._stable_label = label
        return self._stable_label

    def update(self, candidate: str) -> str:
        if candidate in self.reset_labels:
            return self.reset(candidate)

        self._history.append(candidate)
        label, votes = Counter(self._history).most_common(1)[0]
        if votes >= self.min_votes:
            self._stable_label = label

        return self._stable_label or candidate