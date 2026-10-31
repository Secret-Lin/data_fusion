class FrameGroupAssigner:
    """Assign nearby captures from distinct cameras to an explicit sync group."""

    def __init__(self, tolerance_seconds: float = 0.02) -> None:
        if tolerance_seconds <= 0:
            raise ValueError("tolerance_seconds must be positive")
        self._tolerance = tolerance_seconds
        self._group = -1
        self._anchor: float | None = None
        self._sources: set[str] = set()

    def assign(self, source: str, capture_time: float) -> int:
        if self._anchor is None or source in self._sources or abs(capture_time - self._anchor) > self._tolerance:
            self._group += 1
            self._anchor = capture_time
            self._sources.clear()
        self._sources.add(source)
        return self._group
