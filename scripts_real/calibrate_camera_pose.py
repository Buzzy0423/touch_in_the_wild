"""
Camera pose calibration tool for xArm single-arm deploy.

Shows a live camera feed overlaid with a reference frame extracted from
training data, so you can physically adjust the camera until the scene
matches the training viewpoint.

Usage:
    python scripts_real/calibrate_camera_pose.py \
        --train_dir /home/zinan/Documents/zinan/data/gello_raw/session_20260430_164834 \
        --camera_path /dev/video-hdmi

Controls:
    o / p          adjust overlay opacity down/up (0.1 steps)
    m              cycle display mode: overlay -> side-by-side -> diff -> live
    n              cycle to next reference frame
    f              freeze current live frame as new reference
    s              save current live frame to output directory
    q / ESC        quit
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import glob
from typing import Dict, List, Optional, Tuple

import click
import cv2
import numpy as np

from umi.real_world.gello_multi_uvc_camera import GelloMultiUvcCamera


def _resize_to_match(frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize frame to match target (H, W)."""
    if frame.shape[:2] != target_shape:
        return cv2.resize(frame, (target_shape[1], target_shape[0]))
    return frame


def extract_reference_frames(train_dir: str, max_frames: int = 5) -> List[Tuple[np.ndarray, str]]:
    """Extract reference frames from training data videos.

    Returns list of (frame_bgr, label) tuples.
    """
    demos = sorted(glob.glob(os.path.join(train_dir, "demo_*")))
    if not demos:
        raise FileNotFoundError(f"No demo_* directories found in {train_dir}")

    frames: List[Tuple[np.ndarray, str]] = []

    for demo_dir in demos:
        video_path = os.path.join(demo_dir, "video_scene_rgb.mp4")
        if not os.path.exists(video_path):
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            continue

        # Extract first frame (robot typically starts in home position)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        demo_name = os.path.basename(demo_dir)
        frames.append((frame, f"{demo_name}"))

        if len(frames) >= max_frames:
            break

    if not frames:
        raise RuntimeError(f"No usable video files found in {train_dir}")
    return frames


def apply_colormap_to_diff(diff_gray: np.ndarray) -> np.ndarray:
    """Convert grayscale difference to heatmap visualization."""
    diff_gray = np.clip(diff_gray, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)


def build_display(
    live: np.ndarray,
    ref: np.ndarray,
    mode: str,
    opacity: float,
) -> np.ndarray:
    """Build the display image based on current mode."""
    ref = _resize_to_match(ref, live.shape[:2])

    if mode == "overlay":
        blended = cv2.addWeighted(live, 1.0 - opacity, ref, opacity, 0)
        # Draw mode indicator
        cv2.putText(blended, f"OVERLAY  opacity={opacity:.1f}  [o/p adjust]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(blended, "Cyan = reference (training) scene",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return blended

    elif mode == "side-by-side":
        h, w = live.shape[:2]
        ref_resized = cv2.resize(ref, (w, h))
        side = np.hstack([live, ref_resized])
        cv2.putText(side, "LIVE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(side, "TRAINING REFERENCE", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return side

    elif mode == "diff":
        gray_live = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_live, gray_ref)

        # Normalize for visibility
        diff_vis = apply_colormap_to_diff(diff * 3)
        cv2.putText(diff_vis, "DIFF (red = mismatch)  align until mostly blue",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return diff_vis

    elif mode == "diff_split":
        # Top half: overlay, bottom half: diff
        h, w = live.shape[:2]
        half = h // 2

        overlay = cv2.addWeighted(live[:half], 1.0 - opacity, ref[:half], opacity, 0)
        gray_live = cv2.cvtColor(live[half:], cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref[half:], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_live, gray_ref)
        diff_vis = apply_colormap_to_diff(diff * 3)

        top = overlay
        bottom = cv2.resize(diff_vis, (w, half))
        split = np.vstack([top, bottom])
        cv2.putText(split, "OVERLAY", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(split, "DIFF", (10, half + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return split

    else:  # live only
        cv2.putText(live, "LIVE ONLY", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return live


@click.command()
@click.option("--train_dir", "-t", type=str, required=True,
              help="Path to GELLO training session directory (e.g., .../session_20260430_164834).")
@click.option("--camera_path", type=str, default="/dev/video38",
              help="Camera device path.")
@click.option("--capture_fps", type=int, default=60,
              help="Camera capture FPS.")
@click.option("--camera_resolution", type=(int, int), default=(1920, 1080),
              help="Camera resolution (width, height).")
@click.option("--ref_frame_idx", type=int, default=0,
              help="Use Nth frame from each training video (default: first frame).")
@click.option("--init_mode", type=click.Choice(["overlay", "side-by-side", "diff", "diff_split", "live"]),
              default="overlay", help="Initial display mode.")
@click.option("--init_opacity", type=float, default=0.5,
              help="Initial overlay opacity (0.0 to 1.0).")
@click.option("--save_dir", "-o", type=str, default=None,
              help="Directory to save captured frames.")
def main(
    train_dir: str,
    camera_path: str,
    capture_fps: int,
    camera_resolution: Tuple[int, int],
    ref_frame_idx: int,
    init_mode: str,
    init_opacity: float,
    save_dir: Optional[str],
):
    print("=" * 60)
    print("Camera Pose Calibration Tool")
    print("=" * 60)
    print(f"Training data: {train_dir}")
    print(f"Camera path: {camera_path}")
    print(f"Resolution: {camera_resolution[0]}x{camera_resolution[1]} @ {capture_fps}fps")
    print()

    # Extract reference frames
    print("Extracting reference frames from training data...")
    ref_frames = extract_reference_frames(train_dir)
    print(f"  Found {len(ref_frames)} reference frame(s)")
    print()

    # Open camera
    print("Opening camera...")
    cam = GelloMultiUvcCamera(
        dev_video_paths=[camera_path],
        resolution=[camera_resolution],
        capture_fps=[capture_fps],
        fourcc=["NV12"],
        get_max_k=30,
    )
    cam.start()
    print("  Camera ready.")
    print()

    # State
    ref_idx = 0
    mode = init_mode
    opacity = init_opacity
    frozen_ref: Optional[np.ndarray] = None
    cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Calibration", 1280, 720)

    print("Controls:")
    print("  o / p        opacity down / up (0.1 steps)")
    print("  m            cycle mode: overlay -> side-by-side -> diff -> diff_split -> live")
    print("  n            next reference frame")
    print("  f            freeze current live frame as reference")
    print("  r            reset to training reference")
    print("  s            save current frame")
    print("  q / ESC      quit")
    print()

    try:
        while True:
            vis = cam.get_vis()
            live_frame = vis["color"][0]  # (H, W, 3) BGR

            # Current reference: frozen override takes priority
            if frozen_ref is not None:
                ref_frame = frozen_ref
                ref_label = "FROZEN"
            else:
                ref_frame, ref_label = ref_frames[ref_idx]

            display = build_display(live_frame, ref_frame, mode, opacity)

            # Show reference label
            cv2.putText(display, f"Ref: {ref_label} [{ref_idx + 1}/{len(ref_frames)}]",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if frozen_ref is not None:
                cv2.putText(display, "FROZEN REF  [r to unfreeze]",
                            (10, display.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            cv2.imshow("Camera Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord('o'):
                opacity = max(0.0, opacity - 0.1)
                print(f"Opacity: {opacity:.1f}")
            elif key == ord('p'):
                opacity = min(1.0, opacity + 0.1)
                print(f"Opacity: {opacity:.1f}")
            elif key == ord('m'):
                modes = ["overlay", "side-by-side", "diff", "diff_split", "live"]
                cur = modes.index(mode)
                mode = modes[(cur + 1) % len(modes)]
                print(f"Mode: {mode}")
            elif key == ord('n'):
                ref_idx = (ref_idx + 1) % len(ref_frames)
                print(f"Reference: {ref_idx + 1}/{len(ref_frames)}  ({ref_frames[ref_idx][1]})")
            elif key == ord('f'):
                frozen_ref = live_frame.copy()
                print("Froze current live frame as reference. Press 'r' to restore training ref.")
            elif key == ord('r'):
                frozen_ref = None
                print("Restored training reference.")
            elif key == ord('s'):
                out_dir = save_dir or os.path.join(ROOT_DIR, "data_local", "calib_frames")
                os.makedirs(out_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(out_dir, f"calib_{ts}.png")
                cv2.imwrite(path, live_frame)
                print(f"Saved: {path}")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    import time
    main()
