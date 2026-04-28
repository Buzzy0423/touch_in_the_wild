import atexit
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from umi.real_world._gello.elgato_reset import reset_all_elgato_devices
from umi.real_world._gello.ring_buffer import ThreadSafeRingBuffer


_ELGATO_RESET_LOCK = threading.Lock()
_ELGATO_RESET_DONE = False
_ELGATO_RESET_SETTLE_TIMEOUT_S = 8.0
_ELGATO_OPEN_RETRY_INTERVAL_S = 0.25


def _reset_elgato_once() -> None:
    global _ELGATO_RESET_DONE
    with _ELGATO_RESET_LOCK:
        if _ELGATO_RESET_DONE:
            return
        failures = reset_all_elgato_devices(raise_on_error=False)
        for device_path, error in failures:
            print(f"[UvcCamera] Warning: failed to reset {device_path}: {error}")
        _ELGATO_RESET_DONE = True


class UvcCamera:
    """Threaded UVC camera producer backed by a timestamped ring buffer."""

    def __init__(
        self,
        device_path: str,
        resolution: Tuple[int, int] = (2560, 1440),
        fps: int = 60,
        fourcc: str = "NV12",
        receive_latency: float = 0.125,
        buffer_size: int = 60,
        reset_elgato: bool = True,
        flip: bool = False,
    ):
        if reset_elgato:
            _reset_elgato_once()

        self._device_path = device_path
        self._resolution = tuple(resolution)
        self._fps = int(fps)
        self._fourcc = fourcc
        self._receive_latency = float(receive_latency)
        self._flip = bool(flip)
        self._ring_buffer = ThreadSafeRingBuffer(capacity=buffer_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if reset_elgato:
            self._wait_for_device_path()

        self._cap = self._open_capture()

        if len(fourcc) != 4:
            raise ValueError("fourcc must be a four-character string")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._warmup_capture()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        atexit.register(self.close)
        print(
            f"[UvcCamera] Opened {device_path} "
            f"({self._resolution[0]}x{self._resolution[1]}@{self._fps}fps, "
            f"fourcc={fourcc}, receive_latency={self._receive_latency})"
        )

    def get_buffer(self, k: int) -> Dict[str, np.ndarray]:
        return self._ring_buffer.get(k)

    def read_latest(self) -> Dict[str, Any]:
        record = self.get_buffer(k=1)
        return {
            "color": record["color"][-1],
            "timestamp": float(record["timestamp"][-1]),
            "camera_receive_timestamp": float(record["camera_receive_timestamp"][-1]),
        }

    def read_with_timestamp(self) -> Tuple[np.ndarray, float, float]:
        record = self.read_latest()
        return (
            record["color"],
            float(record["timestamp"]),
            float(record["camera_receive_timestamp"]),
        )

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        record = self.read_latest()
        color = record["color"]
        if img_size is not None:
            color = cv2.resize(color, (img_size[1], img_size[0]))
        depth = np.zeros((color.shape[0], color.shape[1], 1), dtype=np.uint16)
        return color, depth

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap.isOpened():
            self._cap.release()
        print(f"[UvcCamera] Released {self._device_path}")

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            self._capture_one()

    def _wait_for_device_path(self) -> None:
        deadline = time.time() + _ELGATO_RESET_SETTLE_TIMEOUT_S
        while time.time() < deadline:
            if os.path.exists(self._device_path):
                # Give the driver a short moment after the node reappears.
                time.sleep(0.5)
                return
            time.sleep(_ELGATO_OPEN_RETRY_INTERVAL_S)
        raise RuntimeError(f"Timed out waiting for camera device to appear: {self._device_path}")

    def _open_capture(self) -> cv2.VideoCapture:
        deadline = time.time() + _ELGATO_RESET_SETTLE_TIMEOUT_S
        while True:
            cap = cv2.VideoCapture(self._device_path, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()
            if time.time() >= deadline:
                break
            time.sleep(_ELGATO_OPEN_RETRY_INTERVAL_S)
        raise RuntimeError(f"Failed to open UVC camera: {self._device_path}")

    def _warmup_capture(self) -> None:
        deadline = time.time() + _ELGATO_RESET_SETTLE_TIMEOUT_S
        last_error: Optional[RuntimeError] = None
        while time.time() < deadline:
            try:
                self._capture_one()
                return
            except RuntimeError as exc:
                last_error = exc
                time.sleep(_ELGATO_OPEN_RETRY_INTERVAL_S)
        if last_error is not None:
            raise last_error

    def _capture_one(self) -> None:
        if not self._cap.grab():
            raise RuntimeError(f"Failed to grab frame from {self._device_path}")

        ok, frame = self._cap.retrieve()
        t_recv = time.time()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to retrieve frame from {self._device_path}")

        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._flip:
            color = np.flip(color, axis=(0, 1)).copy()

        self._ring_buffer.put(
            {
                "color": color,
                "timestamp": t_recv - self._receive_latency,
                "camera_receive_timestamp": t_recv,
            }
        )
