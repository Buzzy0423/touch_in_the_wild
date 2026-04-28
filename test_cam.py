import time
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.multi_uvc_camera import MultiUvcCamera

def main():
    print("starting")
    with SharedMemoryManager() as shm_manager:
        print("shm ready")
        camera = MultiUvcCamera(
            dev_video_paths=["/dev/video38"],
            shm_manager=shm_manager,
            resolution=(1920, 1080),
            capture_fps=30,
            fourcc="NV12",
            put_fps=30,
            put_downsample=False,
            get_max_k=30,
            receive_latency=0.0,
            cap_buffer_size=1,
            verbose=True
        )
        print("camera created")
        camera.start(wait=True)
        print("camera ready!")
        time.sleep(2)
        camera.stop()

if __name__ == "__main__":
    main()