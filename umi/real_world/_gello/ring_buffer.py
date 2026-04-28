import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ThreadSafeRingBuffer:
    """Thread-safe fixed-size ring buffer for timestamped records."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._lock = threading.Lock()
        self._arrays: Dict[str, np.ndarray] = {}
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._dtypes: Dict[str, np.dtype] = {}
        self._keys: Optional[Tuple[str, ...]] = None
        self._write_index = 0
        self._count = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def put(self, record: Dict[str, Any]) -> None:
        if "timestamp" not in record:
            raise ValueError("record must contain a 'timestamp' field")
        if not record:
            raise ValueError("record must not be empty")

        converted = {key: np.asarray(value) for key, value in record.items()}

        with self._lock:
            if self._keys is None:
                self._initialize(converted)
            elif set(converted.keys()) != set(self._keys):
                raise ValueError(
                    f"record keys changed: expected {self._keys}, "
                    f"got {tuple(converted.keys())}"
                )

            for key, value in converted.items():
                if value.shape != self._shapes[key]:
                    raise ValueError(
                        f"record field '{key}' shape changed: expected "
                        f"{self._shapes[key]}, got {value.shape}"
                    )
                if not np.can_cast(value.dtype, self._dtypes[key], casting="same_kind"):
                    raise ValueError(
                        f"record field '{key}' dtype {value.dtype} cannot be cast "
                        f"to buffer dtype {self._dtypes[key]}"
                    )
                self._arrays[key][self._write_index] = value

            self._write_index = (self._write_index + 1) % self._capacity
            self._count = min(self._count + 1, self._capacity)

    def get(self, k: int) -> Dict[str, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")

        with self._lock:
            if self._keys is None or self._count == 0:
                raise RuntimeError("buffer is empty")

            n = min(int(k), self._count)
            start = (self._write_index - n) % self._capacity
            if start + n <= self._capacity:
                indices = np.arange(start, start + n)
            else:
                indices = np.concatenate(
                    (
                        np.arange(start, self._capacity),
                        np.arange(0, (start + n) % self._capacity),
                    )
                )

            return {key: self._arrays[key][indices].copy() for key in self._keys}

    def _initialize(self, record: Dict[str, np.ndarray]) -> None:
        self._keys = tuple(record.keys())
        for key, value in record.items():
            self._shapes[key] = value.shape
            self._dtypes[key] = value.dtype
            self._arrays[key] = np.empty(
                (self._capacity,) + value.shape,
                dtype=value.dtype,
            )
