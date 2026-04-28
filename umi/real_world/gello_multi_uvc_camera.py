"""Drop-in replacement for ``MultiUvcCamera`` backed by GELLO's threaded
``gello.cameras.uvc_camera.UvcCamera``.

UMI's original ``MultiUvcCamera`` runs each camera in an ``mp.Process`` and
preconfigures the device via ``v4l2-ctl`` before opening it with OpenCV. On
some Elgato firmware (e.g. ``/dev/video38`` here), the gap between
``v4l2-ctl`` closing its fd and OpenCV opening lets PipeWire/wireplumber
probe the device, leaving ``VIDIOC_DQBUF`` wedged on the first
``cap.grab()``. GELLO's threaded, single-fd design avoids the race entirely
-- it is the same camera class the data-collection pipeline runs against
this exact hardware every day.
"""
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from umi.real_world._gello.uvc_camera import UvcCamera as GelloUvcCamera


class GelloMultiUvcCamera:
    def __init__(
        self,
        dev_video_paths: Sequence[str],
        resolution,
        capture_fps,
        fourcc,
        get_max_k: int = 30,
        receive_latency: float = 0.125,
        transform=None,
        vis_transform=None,
        # Accepted for MultiUvcCamera signature compatibility, ignored:
        shm_manager=None,
        put_downsample: bool = False,
        cap_buffer_size=None,
        video_recorder=None,
        verbose: bool = False,
    ):
        self.dev_video_paths: List[str] = list(dev_video_paths)
        self.resolution: List[tuple] = [tuple(r) for r in resolution]
        self.capture_fps: List[int] = [int(f) for f in capture_fps]
        self.fourcc: List[Optional[str]] = list(fourcc)
        self.receive_latency: float = float(receive_latency)
        self.get_max_k: int = int(get_max_k)
        n = len(self.dev_video_paths)
        self.transform = list(transform) if transform is not None else [None] * n
        self.vis_transform = list(vis_transform) if vis_transform is not None else [None] * n
        self.n_cameras: int = n
        self._cameras: List[Optional[GelloUvcCamera]] = [None] * n
        self._started: bool = False
        self._verbose: bool = bool(verbose)
        self._video_writers: List[Optional[cv2.VideoWriter]] = []
        self._last_written_ts: List[float] = [0.0] * n

    @property
    def is_ready(self) -> bool:
        return self._started and all(c is not None for c in self._cameras)

    def start(self, wait: bool = False) -> None:
        if self._started:
            return
        # GELLO's UvcCamera.__init__ resets the Elgato (once per process),
        # opens the device, configures FOURCC/W/H/FPS on the same fd, runs a
        # synchronous warmup, and starts the capture thread. By the time the
        # constructor returns, the camera is producing frames into its ring
        # buffer.
        ring_capacity = max(self.get_max_k * 4, 60)
        for i, path in enumerate(self.dev_video_paths):
            fourcc = self.fourcc[i] if self.fourcc[i] is not None else "NV12"
            cam = GelloUvcCamera(
                device_path=path,
                resolution=self.resolution[i],
                fps=self.capture_fps[i],
                fourcc=fourcc,
                receive_latency=self.receive_latency,
                buffer_size=ring_capacity,
                # Module-level guard inside GELLO ensures additional cameras
                # in the same process skip the redundant USB reset.
                reset_elgato=(i == 0),
            )
            self._cameras[i] = cam
        self._started = True

    def stop(self, wait: bool = False) -> None:
        if not self._started:
            return
        for cam in self._cameras:
            if cam is not None:
                cam.close()
        self._cameras = [None] * self.n_cameras
        self._started = False

    def start_wait(self) -> None:
        # GELLO UvcCamera.__init__ already blocks until the first frame is
        # captured, so by the time start() returns the camera is ready.
        return

    def stop_wait(self) -> None:
        return

    def _read_buffer(self, idx: int, k: int) -> Dict[str, np.ndarray]:
        cam = self._cameras[idx]
        if cam is None:
            raise RuntimeError(f"Camera {idx} not started")
        return cam.get_buffer(k=k)

    def get(
        self,
        k: int = 1,
        out: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        if not self._started:
            raise RuntimeError("GelloMultiUvcCamera.get() called before start()")

        result: Dict[int, Dict[str, np.ndarray]] = {}
        for idx in range(self.n_cameras):
            buf = self._read_buffer(idx, k)
            color_rgb = buf["color"]            # (n, H, W, 3) uint8, RGB
            ts = np.asarray(buf["timestamp"])    # (n,)
            recv = np.asarray(
                buf.get("camera_receive_timestamp", buf["timestamp"])
            )
            n = color_rgb.shape[0]

            tf = self.transform[idx]
            transformed: List[np.ndarray] = []
            for j in range(n):
                # GELLO stores RGB; convert once to BGR for both VideoWriter
                # and UMI's transform (which calls get_image_transform with
                # bgr_to_rgb=True, so its input must be BGR).
                bgr_frame = cv2.cvtColor(color_rgb[j], cv2.COLOR_RGB2BGR)
                frame_ts = float(ts[j])

                # Write raw frame to video if recording is active
                if (self._video_writers and idx < len(self._video_writers)
                        and self._video_writers[idx] is not None
                        and self._video_writers[idx].isOpened()
                        and frame_ts > self._last_written_ts[idx]):
                    self._video_writers[idx].write(bgr_frame)
                    self._last_written_ts[idx] = frame_ts
                data = {
                    "color": bgr_frame,
                    "timestamp": frame_ts,
                    "camera_capture_timestamp": float(ts[j]),
                    "camera_receive_timestamp": float(recv[j]),
                }
                if tf is not None:
                    data = tf(data)
                transformed.append(data["color"])
            color_out = np.stack(transformed, axis=0)

            result[idx] = {
                "color": color_out,
                "timestamp": ts,
                "camera_capture_timestamp": ts,
                "camera_receive_timestamp": recv,
            }
        return result

    def get_vis(
        self,
        out: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        if not self._started:
            raise RuntimeError("GelloMultiUvcCamera.get_vis() called before start()")
        per_cam: List[np.ndarray] = []
        for idx in range(self.n_cameras):
            buf = self._read_buffer(idx, k=1)
            frame_rgb = buf["color"][-1]
            tf = self.vis_transform[idx]
            data = {
                "color": cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            }
            if tf is not None:
                data = tf(data)
            per_cam.append(data["color"])
        return {"color": np.stack(per_cam, axis=0)}

    # --- recording via cv2.VideoWriter ---

    def start_recording(self, video_path: List[str], start_time: Optional[float] = None) -> None:
        self._video_writers = []
        self._last_written_ts = [0.0] * self.n_cameras
        for idx in range(self.n_cameras):
            cam = self._cameras[idx]
            res_w, res_h = cam._resolution
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cam._fps
            writer = cv2.VideoWriter(video_path[idx], fourcc, fps, (res_w, res_h))
            if not writer.isOpened():
                print(f"[GelloMultiUvcCamera] WARNING: failed to open VideoWriter "
                      f"for camera {idx} at {video_path[idx]}")
            self._video_writers.append(writer)

    def stop_recording(self) -> None:
        for writer in self._video_writers:
            if writer is not None and writer.isOpened():
                writer.release()
        self._video_writers.clear()

    def restart_put(self, start_time: Optional[float] = None) -> None:
        self._last_written_ts = [0.0] * self.n_cameras
