from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
import cv2
from PIL import Image
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.xarm_interpolation_controller import XArmInterpolationController
from umi.real_world.xarm_gripper_controller import XArmGripperController
from umi.real_world.xarm_gello_util import (
    XARM7_GELLO_START_GRIPPER,
    XARM7_GELLO_START_JOINTS,
)
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from umi.real_world.gello_multi_uvc_camera import GelloMultiUvcCamera
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util import (
    draw_predefined_mask, 
    get_mirror_crop_slices,
    inpaint_tag,
)
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import optimal_row_cols
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util import pose_to_pos_rot
from umi.common.interpolation_util import get_interp1d, PoseInterpolator


def _get_training_image_transform(input_res, output_res):
    iw, ih = input_res
    ow, oh = output_res
    crop_h = ih
    crop_w = round(ih / oh * ow)

    w_slice_start = (iw - crop_w) // 2
    w_slice = slice(w_slice_start, w_slice_start + crop_w)
    h_slice_start = (ih - crop_h) // 2
    h_slice = slice(h_slice_start, h_slice_start + crop_h)

    def transform(img):
        if img.shape != (ih, iw, 3):
            raise ValueError(f"Expected RGB frame shape {(ih, iw, 3)}, got {img.shape}")
        img = img[h_slice, w_slice]
        return cv2.resize(img, output_res, interpolation=cv2.INTER_AREA)

    return transform


def _get_aruco_corner_detector(aruco_dict_name="DICT_4X4_50"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    params = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        def detect(gray):
            corners, _, _ = detector.detectMarkers(gray)
            return corners

        return detect

    def detect(gray):
        corners, _, _ = cv2.aruco.detectMarkers(
            image=gray,
            dictionary=aruco_dict,
            parameters=params,
        )
        return corners

    return detect


def _apply_training_rgb_processing(
        img_bgr,
        resize_tf,
        detect_aruco_corners,
        mask_w=320,
        mask_h=160):
    img = np.ascontiguousarray(img_bgr[..., ::-1])
    if mask_w > 0 and mask_h > 0:
        img[:mask_h, :mask_w] = 0
        img[:mask_h, -mask_w:] = 0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for corners in detect_aruco_corners(gray):
        img = inpaint_tag(img, np.asarray(corners).reshape(-1, 2))
    return resize_tf(img)


class UmiEnv:
    def __init__(self,
            # required params
            output_dir,
            robot_ip,
            gripper_ip,
            gripper_port=1000,
            # env params
            frequency=20,
            robot_type='ur5',
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            camera_paths=None,
            no_mirror=False,
            fisheye_converter=None,
            mirror_crop=False,
            mirror_swap=False,
            use_converter=False,
            # timing
            align_camera_idx=0,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            tactile_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            tactile_obs_horizon=None,
            tactile_buffer_fps=150,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # robot
            tcp_offset=0.21,
            init_joints=False,
            xarm_start_joints=None,
            xarm_start_gripper=None,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            tactile_controller_left=None,
            tactile_controller_right=None,
            # shared memory
            shm_manager=None
            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        self.xarm_start_joints = None
        self.xarm_start_gripper = None
        if xarm_start_joints is not None:
            self.xarm_start_joints = np.asarray(xarm_start_joints, dtype=np.float64).copy()
        if xarm_start_gripper is not None:
            self.xarm_start_gripper = float(xarm_start_gripper)

        # Find and reset all Elgato capture cards.
        # Required to workaround a firmware bug: USBDEVFS_RESET ioctl is
        # equivalent to a physical USB replug, clearing wedged state from
        # a prior crashed process. Mirrors gello.cameras.uvc_camera behavior.
        reset_all_elgato_devices()

        # Wait for all v4l cameras to be back online after re-enumeration
        time.sleep(2.5)
        if camera_paths is None:
            v4l_paths = get_sorted_v4l_paths()
            if camera_reorder is not None:
                paths = [v4l_paths[i] for i in camera_reorder]
                v4l_paths = paths
        else:
            v4l_paths = list(camera_paths)

        # compute resolution for vis
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(v4l_paths),
            in_wh_ratio=4/3,
            max_resolution=multi_cam_vis_resolution
        )

        # HACK: Separate video setting for each camera
        # Elagto Cam Link 4k records at 4k 30fps
        # Other capture card records at 720p 60fps
        resolution = list()
        capture_fps = list()
        capture_fourcc = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        vis_transform = list()
        detect_aruco_corners = _get_aruco_corner_detector()
        for idx, path in enumerate(v4l_paths):
            if 'Cam_Link_4K' in path:
                res = (3840, 2160)
                fps = 30
                fourcc = 'MJPG'
                buf = 3
                bit_rate = 6000*1000
                resize_tf = _get_training_image_transform(
                    input_res=res,
                    output_res=obs_image_resolution,
                )
                def tf4k(data, resize_tf=resize_tf, use_converter=use_converter):
                    img = data['color']
                    if use_converter:
                        img = img[..., ::-1]  # BGR to RGB
                        img = np.array(
                            Image.fromarray(img).resize(obs_image_resolution, Image.BILINEAR),
                            dtype=np.uint8)
                    else:
                        img = _apply_training_rgb_processing(
                            img,
                            resize_tf=resize_tf,
                            detect_aruco_corners=detect_aruco_corners,
                        )
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf4k)
            else:
                res = (1920, 1080)
                fps = 60
                fourcc = 'NV12'
                buf = 1
                bit_rate = 3000*1000
                stack_crop = (idx==0) and mirror_crop
                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0,0,0), mirror=False, gripper=False, finger=False)
                    is_mirror = (mirror_mask[...,0] == 0)
                resize_tf = _get_training_image_transform(
                    input_res=res,
                    output_res=obs_image_resolution,
                )

                def tf(data, stack_crop=stack_crop, is_mirror=is_mirror,
                       resize_tf=resize_tf, use_converter=use_converter):
                    img = data['color']
                    if use_converter:
                        # Legacy path: BGR->RGB, full-frame PIL BILINEAR resize, no mask, no crop.
                        img = img[..., ::-1]  # BGR to RGB
                        img = np.array(
                            Image.fromarray(img).resize(obs_image_resolution, Image.BILINEAR),
                            dtype=np.uint8)
                    elif fisheye_converter is None:
                        crop_img = None
                        if stack_crop:
                            slices = get_mirror_crop_slices(img.shape[:2], left=False)
                            crop = img[slices]
                            crop_img = cv2.resize(crop, obs_image_resolution)
                            crop_img = crop_img[:,::-1,::-1] # bgr to rgb
                        img = _apply_training_rgb_processing(
                            img,
                            resize_tf=resize_tf,
                            detect_aruco_corners=detect_aruco_corners,
                        )
                        img = np.ascontiguousarray(img)
                        if is_mirror is not None:
                            img[is_mirror] = img[:,::-1,:][is_mirror]
                        if crop_img is not None:
                            img = np.concatenate([img, crop_img], axis=-1)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[...,::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            capture_fourcc.append(fourcc)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

            def vis_tf(data, max_width=rw):
                img = data['color']
                h, w = img.shape[:2]
                if w > max_width:
                    scale = max_width / float(w)
                    img = cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                data['color'] = img
                return data
            vis_transform.append(vis_tf)

        # Use GELLO's threaded UvcCamera under the hood. UMI's mp.Process-based
        # MultiUvcCamera wedged on this Elgato hardware due to a V4L2/PipeWire
        # race; GELLO's single-fd threaded design is the same code path that
        # the data-collection pipeline runs against this exact camera daily.
        # Recording (HEVC NVENC) is disabled by the adapter; the policy still
        # runs, only post-hoc rollout video is unavailable.
        camera = GelloMultiUvcCamera(
            dev_video_paths=v4l_paths,
            resolution=resolution,
            capture_fps=capture_fps,
            fourcc=capture_fourcc,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            transform=transform,
            vis_transform=vis_transform,
            verbose=False,
        )

        # MultiCameraVisualizer is its own mp.Process and would re-introduce
        # cross-process camera access against this Elgato. The eval scripts
        # already render frames with cv2.imshow in the main loop, so skip it.
        multi_cam_vis = None

        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
        if not init_joints:
            j_init = None

        if robot_type.startswith('ur5'):
            robot = RTDEInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=500, # UR5 CB3 RTDE
                lookahead_time=0.1,
                gain=300,
                max_pos_speed=max_pos_speed*cube_diag,
                max_rot_speed=max_rot_speed*cube_diag,
                launch_timeout=3,
                tcp_offset_pose=[0,0,tcp_offset,0,0,0],
                payload_mass=None,
                payload_cog=None,
                joints_init=j_init,
                joints_init_speed=1.05,
                soft_real_time=False,
                verbose=False,
                receive_keys=None,
                receive_latency=robot_obs_latency
                )
        elif robot_type.startswith('franka'):
            robot = FrankaInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=200,
                Kx_scale=1.0,
                Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
                verbose=False,
                receive_latency=robot_obs_latency
            )
        elif robot_type.startswith('xarm'):
            xarm_j_init = None
            if init_joints:
                if self.xarm_start_joints is None:
                    xarm_j_init = XARM7_GELLO_START_JOINTS.copy()
                else:
                    xarm_j_init = self.xarm_start_joints.copy()
            robot = XArmInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=200,
                lookahead_time=0.1,
                gain=300,
                max_pos_speed=max_pos_speed * cube_diag,
                max_rot_speed=max_rot_speed * cube_diag,
                launch_timeout=3,
                tcp_offset_pose=None,
                payload_mass=None,
                payload_cog=None,
                joints_init=xarm_j_init,
                joints_init_speed=1.0,
                soft_real_time=False,
                verbose=False,
                receive_latency=robot_obs_latency
            )
        else:
            raise NotImplementedError(f"Unsupported robot_type: {robot_type}")

        if robot_type.startswith('xarm'):
            xarm_gripper_init = None
            if init_joints:
                if self.xarm_start_gripper is None:
                    xarm_gripper_init = XARM7_GELLO_START_GRIPPER
                else:
                    xarm_gripper_init = self.xarm_start_gripper
            gripper = XArmGripperController(
                shm_manager=shm_manager,
                hostname=gripper_ip,
                port=gripper_port,
                startup_pos=xarm_gripper_init,
                receive_latency=gripper_obs_latency
            )
        else:
            gripper = WSGController(
                shm_manager=shm_manager,
                hostname=gripper_ip,
                port=gripper_port,
                receive_latency=gripper_obs_latency,
                use_meters=True
            )

        self.camera = camera
        self.robot = robot
        self.gripper = gripper
        self.tactile_left = tactile_controller_left
        self.tactile_right = tactile_controller_right
        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.mirror_crop = mirror_crop
        # timing
        self.align_camera_idx = align_camera_idx
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.robot_action_latency = robot_action_latency
        self.gripper_action_latency = gripper_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.tactile_down_sample_steps = tactile_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.tactile_obs_horizon = (
            camera_obs_horizon if tactile_obs_horizon is None else tactile_obs_horizon
        )
        self.tactile_buffer_fps = tactile_buffer_fps
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.camera.is_ready and self.robot.is_ready and self.gripper.is_ready
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        self.gripper.start(wait=False)
        self.robot.start(wait=False)
        if self.tactile_left is not None:
            self.tactile_left.start()
        if self.tactile_right is not None:
            self.tactile_right.start()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if self.tactile_left is not None:
            self.tactile_left.stop()
        if self.tactile_right is not None:
            self.tactile_right.stop()
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        self.gripper.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.gripper.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        'current' time is the last timestamp of align_camera_idx
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        assert self.is_ready

        # get data
        # 60 Hz, camera_calibrated_timestamp
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency))
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        # 125/500 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # 30 hz, gripper_receive_timestamp
        last_gripper_data = self.gripper.get_all_state()

        last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                this_idxs.append(nn_idx)
            # remap key
            if camera_idx == 0 and self.mirror_crop:
                camera_obs['camera0_rgb'] = value['color'][...,:3][this_idxs]
                camera_obs['camera0_rgb_mirror_crop'] = value['color'][...,3:][this_idxs]
            else:
                camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'], 
            x=last_robot_data['ActualTCPPose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }

        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        gripper_interpolator = get_interp1d(
            t=last_gripper_data['gripper_timestamp'],
            x=last_gripper_data['gripper_position'][...,None]
        )
        gripper_obs = {
            'robot0_gripper_width': gripper_interpolator(gripper_obs_timestamps)
        }

        tactile_obs = dict()
        if (self.tactile_left is not None) and (self.tactile_right is not None):
            tactile_obs_timestamps = last_timestamp - (
                np.arange(self.tactile_obs_horizon)[::-1]
                * self.tactile_down_sample_steps
                * dt
            )
            k_tactile = int(np.ceil(
                self.tactile_obs_horizon
                * self.tactile_down_sample_steps
                * (self.tactile_buffer_fps / self.frequency)
            )) + 2

            left_count = self.tactile_left.ring_buffer.count
            right_count = self.tactile_right.ring_buffer.count
            if left_count < k_tactile or right_count < k_tactile:
                raise RuntimeError(
                    "Not enough tactile samples for aligned observation: "
                    f"left={left_count}, right={right_count}, need={k_tactile}"
                )

            data_left = self.tactile_left.get(k=k_tactile)
            data_right = self.tactile_right.get(k=k_tactile)
            ring_ts_left = data_left['timestamp']
            ring_ts_right = data_right['timestamp']
            ring_fr_left = data_left['frame']
            ring_fr_right = data_right['frame']

            tactile_frames = list()
            for desired_t in tactile_obs_timestamps:
                idx_l = np.argmin(np.abs(ring_ts_left - desired_t))
                idx_r = np.argmin(np.abs(ring_ts_right - desired_t))
                left_frame = ring_fr_left[idx_l]
                right_frame = ring_fr_right[idx_r]
                left_frame = left_frame[::-1, :]
                right_frame = right_frame[::-1, :]
                tactile_frames.append(
                    np.concatenate([left_frame, right_frame], axis=-1)
                )
            tactile_obs['camera0_tactile'] = np.stack(tactile_frames, axis=0)

            # accumulate raw tactile frames for episode recording
            if self.obs_accumulator is not None:
                self.obs_accumulator.put(
                    data={'camera0_tactile': tactile_obs['camera0_tactile']},
                    timestamps=tactile_obs_timestamps,
                )

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            self.obs_accumulator.put(
                data={
                    'robot0_gripper_width': last_gripper_data['gripper_position'][...,None]
                },
                timestamps=last_gripper_data['gripper_timestamp']
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data.update(tactile_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        r_latency = self.robot_action_latency if compensate_latency else 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0

        # schedule waypoints
        for i in range(len(new_actions)):
            r_actions = new_actions[i,:6]
            g_actions = new_actions[i,6:]
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )
            self.gripper.schedule_waypoint(
                pos=g_actions,
                target_time=new_timestamps[i]-g_latency
            )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    def set_recording_dir(self, output_dir):
        if self.obs_accumulator is not None:
            raise RuntimeError("Cannot switch recording directory during an active episode.")
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer

    def move_to_start_pose(self, duration=2.0, joint_tolerance=0.03):
        assert self.is_ready
        if self.xarm_start_joints is None:
            raise RuntimeError("xArm start joints are not configured.")
        gripper_start = (
            XARM7_GELLO_START_GRIPPER
            if self.xarm_start_gripper is None
            else self.xarm_start_gripper
        )
        target_time = time.time() + float(duration)
        self.gripper.schedule_waypoint(pos=gripper_start, target_time=target_time)
        self.robot.moveJ(self.xarm_start_joints)
        deadline = time.monotonic() + float(duration) + 10.0
        while time.monotonic() < deadline:
            q = self.robot.get_state()['ActualQ'][:self.xarm_start_joints.shape[0]]
            if np.max(np.abs(q - self.xarm_start_joints)) <= joint_tolerance:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out moving xArm to GELLO start pose.")

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.camera.stop_recording()

        # TODO
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)
                episode['robot0_eef_pos'] = robot_pose[:,:3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

                gripper_interpolator = get_interp1d(
                    t=np.array(self.obs_accumulator.timestamps['robot0_gripper_width']),
                    x=np.array(self.obs_accumulator.data['robot0_gripper_width'])
                )
                episode['robot0_gripper_width'] = gripper_interpolator(timestamps)

                if 'camera0_tactile' in self.obs_accumulator.timestamps:
                    tactile_ts = np.array(
                        self.obs_accumulator.timestamps['camera0_tactile'])
                    tactile_data = np.array(
                        self.obs_accumulator.data['camera0_tactile'])
                    tactile_frames = list()
                    for t in timestamps:
                        idx = np.argmin(np.abs(tactile_ts - t))
                        tactile_frames.append(tactile_data[idx])
                    episode['camera0_tactile'] = np.stack(
                        tactile_frames, axis=0)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
