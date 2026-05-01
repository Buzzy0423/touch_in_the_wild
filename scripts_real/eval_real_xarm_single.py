"""
Usage:
(umi): python scripts_real/eval_real_xarm_single.py \
    -i /path/to/checkpoint.ckpt

Single-arm xArm deployment without SpaceMouse.

Controls:
- Press "C" to start policy rollout.
- Press "S" to stop the current rollout.
- Press "Q" to quit.
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from umi.common.cv_util import parse_fisheye_intrinsics, FisheyeRectConverter
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode
from umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_action,
    get_real_umi_obs_dict,
)
from umi.real_world.xarm_gello_util import (
    XARM7_GELLO_START_GRIPPER,
    XARM7_GELLO_START_JOINTS,
)
from umi.real_world.tactile_controller_left import TactileControllerLeft
from umi.real_world.tactile_controller_right import TactileControllerRight
from umi.real_world.umi_env import UmiEnv
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

XARM_START_JOINTS = np.array(
    [0.0, -0.6981, 0.0, 0.8727, 0.0, 1.5708, 0.0],
    dtype=np.float64,
)
XARM_START_GRIPPER = 0.0
DEFAULT_OUTPUT_ROOT = "/home/zinan/Documents/zinan/data/titw_eval"


def make_deploy_output_dir(output_root):
    os.makedirs(output_root, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"deploy_{timestamp}")
    suffix = 1
    while os.path.exists(output_dir):
        output_dir = os.path.join(output_root, f"deploy_{timestamp}_{suffix}")
        suffix += 1
    os.makedirs(output_dir)
    return output_dir


def render_obs(obs, vis_camera_idx, mirror_crop):
    if mirror_crop:
        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
        crop_img = obs['camera0_rgb_mirror_crop'][-1]
        vis_img = np.concatenate([vis_img, crop_img], axis=1)
    else:
        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
    return vis_img.copy()


def render_tactile(tactile_left, tactile_right):
    if tactile_left is None or tactile_right is None:
        return None
    left_data = tactile_left.get(k=1)
    right_data = tactile_right.get(k=1)
    if left_data is None or right_data is None:
        return None
    left_frame = left_data['frame'][-1]
    right_frame = right_data['frame'][-1]
    scale = 10
    left_vis_u8 = np.clip(left_frame * 255.0, 0, 255).astype(np.uint8)
    right_vis_u8 = np.clip(right_frame * 255.0, 0, 255).astype(np.uint8)
    left_color = cv2.applyColorMap(left_vis_u8, cv2.COLORMAP_VIRIDIS)
    right_color = cv2.applyColorMap(right_vis_u8, cv2.COLORMAP_VIRIDIS)
    h, w = left_color.shape[:2]
    left_big = cv2.resize(left_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    right_big = cv2.resize(right_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    cv2.putText(left_big, "LEFT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(right_big, "RIGHT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    gap = np.zeros((left_big.shape[0], 20, 3), dtype=np.uint8)
    return np.concatenate([left_big, gap, right_big], axis=1)


def recalibrate_tactile(tactile_left, tactile_right, timeout):
    if tactile_left is None or tactile_right is None:
        return True

    print("[Tactile] Resetting baseline before rollout...")
    tactile_left.reset_baseline()
    tactile_right.reset_baseline()

    print(f"[Tactile] Waiting for baseline calibration (up to {timeout:.1f}s)...")
    deadline = time.monotonic() + timeout
    left_ok = tactile_left.wait_until_calibrated(timeout=max(0.0, deadline - time.monotonic()))
    right_ok = tactile_right.wait_until_calibrated(timeout=max(0.0, deadline - time.monotonic()))
    if left_ok and right_ok:
        print("[Tactile] Baseline calibration complete.")
        return True

    print(f"[Tactile] Baseline calibration timeout. Status: left={left_ok}, right={right_ok}")
    return False


def render_preview(env, obs, vis_camera_idx, mirror_crop):
    vis = env.camera.get_vis()['color'][vis_camera_idx]
    if mirror_crop:
        crop_img = obs['camera0_rgb_mirror_crop'][-1]
        crop_img = (crop_img * 255).astype(np.uint8) if crop_img.dtype.kind == 'f' else crop_img
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
        crop_img = cv2.resize(crop_img, (vis.shape[1], vis.shape[0]))
        vis = np.concatenate([vis, crop_img], axis=1)
    return vis.copy()


def draw_text(vis_img, lines):
    y = 20
    for line in lines:
        cv2.putText(
            vis_img,
            line,
            (10, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            lineType=cv2.LINE_AA,
            thickness=3,
            color=(0, 0, 0),
        )
        cv2.putText(
            vis_img,
            line,
            (10, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=1,
            color=(255, 255, 255),
        )
        y += 20


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', default=DEFAULT_OUTPUT_ROOT, show_default=True,
    help='Root directory for deploy outputs. A deploy_{timestamp} subdirectory is created inside it.')
@click.option('--robot_ip', default='192.168.1.239', show_default=True)
@click.option('--gripper_ip', default=None, help='Defaults to robot_ip for xArm')
@click.option('--camera_path', type=str, default=None,
    help='Explicit camera device path, e.g. /dev/video12')
@click.option('--camera_reorder', '-cr', default='0', show_default=True)
@click.option('--vis_camera_idx', default=0, type=int, help='Which camera to visualize.')
@click.option('--init_joints/--no_init_joints', default=True, show_default=True,
    help='Move xArm to the GELLO start_joints before enabling rollout.')
@click.option('--steps_per_inference', '-si', default=6, type=int, show_default=True)
@click.option('--max_duration', '-md', default=120.0, type=float, show_default=True)
@click.option('--frequency', '-f', default=10.0, type=float, show_default=True)
@click.option('--start_delay', default=1.0, type=float, show_default=True)
@click.option('--auto_start', is_flag=True, default=False, help='Start one rollout immediately.')
@click.option('--reset_before_rollout/--no_reset_before_rollout', default=True, show_default=True,
    help='Move xArm to the GELLO start pose before each rollout.')
@click.option('--reset_duration', default=2.0, type=float, show_default=True,
    help='Settling duration after commanding the GELLO start pose before rollout.')
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--max_pos_speed', default=0.25, type=float, show_default=True)
@click.option('--max_rot_speed', default=0.6, type=float, show_default=True)
@click.option('-uc', '--use_converter', is_flag=True, default=False,
    help='Use legacy converter image processing (full-frame PIL BILINEAR resize, no mask, no center crop).')
@click.option('--enable_tactile', is_flag=True, default=False,
    help='Enable left/right tactile serial controllers and provide camera0_tactile to the policy.')
@click.option('--tactile_left_port', default='/dev/LeftTactile', show_default=True)
@click.option('--tactile_right_port', default='/dev/RightTactile', show_default=True)
@click.option('--tactile_latency', default=0.06, type=float, show_default=True,
    help='Seconds subtracted from tactile receive time for timestamp alignment.')
@click.option('--tactile_median_samples', default=30, type=int, show_default=True)
@click.option('--tactile_buffer_size', default=300, type=int, show_default=True)
@click.option('--tactile_buffer_fps', default=150.0, type=float, show_default=True,
    help='Expected tactile producer frequency used to choose how many recent samples to read.')
@click.option('--flip_tactile_columns/--no_flip_tactile_columns', default=True, show_default=True,
    help='Flip each 12x32 tactile side along columns before concatenating, matching GELLO zarr conversion.')
@click.option('--tactile_recalibration_timeout', default=20.0, type=float, show_default=True,
    help='Seconds to wait for tactile baseline recalibration before each rollout.')
def main(
    input,
    output,
    robot_ip,
    gripper_ip,
    camera_path,
    camera_reorder,
    vis_camera_idx,
    init_joints,
    steps_per_inference,
    max_duration,
    frequency,
    start_delay,
    auto_start,
    reset_before_rollout,
    reset_duration,
    no_mirror,
    sim_fov,
    camera_intrinsics,
    mirror_crop,
    mirror_swap,
    max_pos_speed,
    max_rot_speed,
    use_converter,
    enable_tactile,
    tactile_left_port,
    tactile_right_port,
    tactile_latency,
    tactile_median_samples,
    tactile_buffer_size,
    tactile_buffer_fps,
    flip_tactile_columns,
    tactile_recalibration_timeout,
):
    if gripper_ip is None:
        gripper_ip = robot_ip

    output_root = output
    output = make_deploy_output_dir(output_root)
    print("Saving deploy output to:", output)

    if init_joints:
        assert np.allclose(XARM_START_JOINTS, XARM7_GELLO_START_JOINTS)
        assert XARM_START_GRIPPER == XARM7_GELLO_START_GRIPPER
        print("xArm start joints:", XARM_START_JOINTS.tolist())
        print("xArm start gripper:", XARM_START_GRIPPER)

    camera_paths = None
    camera_reorder_list = None
    if camera_path is not None:
        camera_paths = [camera_path]
    else:
        camera_reorder_list = [int(x) for x in camera_reorder]

    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')

    with open(ckpt_path, 'rb') as f:
        payload = torch.load(f, map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)
    tactile_keys = [
        key for key, attr in cfg.task.shape_meta.obs.items()
        if attr.get('type', 'low_dim') == 'tactile'
    ]
    if tactile_keys and tactile_keys != ['camera0_tactile']:
        raise RuntimeError(
            "xArm tactile deploy currently produces only camera0_tactile, "
            f"but checkpoint expects {tactile_keys}."
        )
    if tactile_keys and not enable_tactile:
        raise RuntimeError(
            "Checkpoint expects tactile observations "
            f"{tactile_keys}, but --enable_tactile was not set."
        )
    if enable_tactile and not tactile_keys:
        print("Tactile enabled, but checkpoint shape_meta has no tactile observation key.")

    # Derive deployment timing from training config to match the effective
    # observation spacing: obs_down_sample_steps / dataset_frequency.
    # Training: 3 / 59.94 ≈ 0.05s. Deployment: camera_down_sample_steps / frequency.
    train_obs_down = cfg.task.obs_down_sample_steps  # e.g. 3
    train_dataset_freq = cfg.task.dataset_frequeny
    if train_dataset_freq == 0:
        train_dataset_freq = 59.94
    train_effective_freq = train_dataset_freq / train_obs_down
    # Derive camera_obs_latency from training config (or keep CLI override)
    train_camera_latency = cfg.task.camera_obs_latency  # e.g. 0.125
    train_robot_latency = cfg.task.robot_obs_latency
    train_gripper_latency = cfg.task.gripper_obs_latency

    dt = 1 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    print(f"Training: dataset_freq={train_dataset_freq}, "
          f"obs_down_sample_steps={train_obs_down}, "
          f"effective_freq={train_effective_freq:.1f} Hz")
    print(f"Deployment: frequency={frequency} Hz, dt={dt:.4f}s")
    print(f"  camera_obs_latency={train_camera_latency}, "
          f"robot_obs_latency={train_robot_latency}, "
          f"gripper_obs_latency={train_gripper_latency}")

    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        with open(camera_intrinsics, 'r') as f:
            opencv_intr_dict = parse_fisheye_intrinsics(json.load(f))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )



    tactile_obs_horizon = cfg.task.shape_meta.obs.camera0_rgb.horizon
    tactile_down_sample_steps = cfg.task.obs_down_sample_steps
    if tactile_keys:
        tactile_meta = cfg.task.shape_meta.obs[tactile_keys[0]]
        tactile_obs_horizon = tactile_meta.horizon
        tactile_down_sample_steps = tactile_meta.get(
            'down_sample_steps',
            cfg.task.obs_down_sample_steps,
        )

    with SharedMemoryManager() as shm_manager:
        tactile_left = None
        tactile_right = None
        if enable_tactile:
            print("Starting tactile controllers:")
            print("  left:", tactile_left_port)
            print("  right:", tactile_right_port)
            print("  flip columns:", flip_tactile_columns)
            tactile_left = TactileControllerLeft(
                shm_manager=shm_manager,
                port_left=tactile_left_port,
                median_samples=tactile_median_samples,
                ring_buffer_size=tactile_buffer_size,
                receive_latency=tactile_latency,
            )
            tactile_right = TactileControllerRight(
                shm_manager=shm_manager,
                port_right=tactile_right_port,
                median_samples=tactile_median_samples,
                ring_buffer_size=tactile_buffer_size,
                receive_latency=tactile_latency,
            )
        # import pdb; pdb.set_trace()
        with KeystrokeCounter() as key_counter, UmiEnv(
            output_dir=output,
            robot_ip=robot_ip,
            gripper_ip=gripper_ip,
            frequency=frequency,
            robot_type='xarm',
            obs_image_resolution=obs_res,
            obs_float32=True,
            camera_reorder=camera_reorder_list,
            camera_paths=camera_paths,
            init_joints=init_joints,
            xarm_start_joints=XARM_START_JOINTS,
            xarm_start_gripper=XARM_START_GRIPPER,
            enable_multi_cam_vis=True,
            camera_obs_latency=train_camera_latency,
            robot_obs_latency=train_robot_latency,
            gripper_obs_latency=train_gripper_latency,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
            robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
            gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
            tactile_obs_horizon=tactile_obs_horizon,
            tactile_down_sample_steps=tactile_down_sample_steps,
            tactile_buffer_fps=tactile_buffer_fps,
            flip_tactile_columns=flip_tactile_columns,
            no_mirror=no_mirror,
            fisheye_converter=fisheye_converter,
            mirror_crop=mirror_crop,
            mirror_swap=mirror_swap,
            use_converter=use_converter,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            tactile_controller_left=tactile_left,
            tactile_controller_right=tactile_right,
            shm_manager=shm_manager,
        ) as env:
            

            # import pdb; pdb.set_trace()


            cv2.setNumThreads(2)
            cv2.namedWindow('default', cv2.WINDOW_NORMAL)
            if enable_tactile:
                cv2.namedWindow('tactile', cv2.WINDOW_NORMAL)
            print("Waiting for camera/tactile" if enable_tactile else "Waiting for camera")
            time.sleep(2.0 if enable_tactile else 1.0)

            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy: BaseImagePolicy
            policy.num_inference_steps = 16
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)

            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()
            # `robot0_eef_rot_axis_angle_wrt_start` (and any other *_wrt_start
            # keys in the policy's shape_meta) are only computed when
            # episode_start_pose is provided. For warmup we use the current
            # pose so wrt_start ~ identity; the real start pose is recaptured
            # at each rollout start below.
            episode_start_pose = [np.concatenate([
                obs['robot0_eef_pos'],
                obs['robot0_eef_rot_axis_angle'],
            ], axis=-1)[-1]]
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs,
                    shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose,
                )
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                )
                result = policy.predict_action(obs_dict)
                warmup_action = result['action_pred'][0].detach().to('cpu').numpy()
                warmup_action = get_real_umi_action(
                    warmup_action,
                    obs,
                    action_pose_repr,
                )
                assert warmup_action.shape[-1] == 7
                del result

            print('Ready!')
            should_quit = False
            ran_auto_episode = False

            while not should_quit:
                print('Idle. Press "c" to start rollout, "s" to stop (during rollout), "q" to quit.')
                t_idle_start = time.monotonic()
                iter_idx = 0
                while True:
                    t_cycle_end = t_idle_start + (iter_idx + 1) * dt
                    obs = env.get_obs()
                    vis_img = render_preview(env, obs, vis_camera_idx, mirror_crop)
                    episode_id = env.replay_buffer.n_episodes
                    draw_text(vis_img, [
                        f'Episode: {episode_id}',
                        'Mode: idle',
                        'Keys: c=start, s=stop (during rollout), q=quit',
                    ])
                    cv2.imshow('default', vis_img)
                    if enable_tactile:
                        tactile_vis = render_tactile(env.tactile_left, env.tactile_right)
                        if tactile_vis is not None:
                            cv2.imshow('tactile', tactile_vis)
                    _ = cv2.pollKey()

                    start_policy = auto_start and not ran_auto_episode
                    for key_stroke in key_counter.get_press_events():
                        if key_stroke == KeyCode(char='q'):
                            should_quit = True
                        elif key_stroke == KeyCode(char='c'):
                            start_policy = True

                    if should_quit or start_policy:
                        break

                    precise_wait(t_cycle_end)
                    iter_idx += 1

                if should_quit:
                    break

                try:
                    policy.reset()
                    if reset_before_rollout:
                        print("Moving to GELLO start pose before rollout.")
                        env.move_to_start_pose(duration=reset_duration)
                    if enable_tactile:
                        recalibrate_tactile(
                            env.tactile_left,
                            env.tactile_right,
                            tactile_recalibration_timeout,
                        )
                    eval_t_start = time.time() + start_delay
                    rollout_t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    # Capture the canonical start pose for wrt_start observations.
                    rollout_start_obs = env.get_obs()
                    episode_start_pose = [np.concatenate([
                        rollout_start_obs['robot0_eef_pos'],
                        rollout_start_obs['robot0_eef_rot_axis_angle'],
                    ], axis=-1)[-1]]
                    print("Started rollout.")
                    iter_idx = 0

                    while True:
                        t_cycle_end = rollout_t_start + (iter_idx + steps_per_inference) * dt
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        with torch.no_grad():
                            infer_start = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs,
                                shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose,
                            )
                            obs_dict = dict_apply(
                                obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                            )
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(
                                raw_action,
                                obs,
                                action_pose_repr,
                            )
                            print('Inference latency:', time.time() - infer_start)

                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64) * dt + obs_timestamps[-1]
                        )
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            action = action[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                            print('Over budget', action_timestamps[0] - curr_time)
                        else:
                            action = action[is_new]
                            action_timestamps = action_timestamps[is_new]

                        env.exec_actions(
                            actions=action,
                            timestamps=action_timestamps,
                            compensate_latency=True,
                        )
                        print(f"Submitted {len(action)} steps of actions.")

                        vis_img = render_preview(env, obs, vis_camera_idx, mirror_crop)
                        draw_text(vis_img, [
                            f'Episode: {env.replay_buffer.n_episodes}',
                            f'Mode: rollout {time.monotonic() - rollout_t_start:.1f}s',
                            'Keys: s=stop, q=quit',
                        ])
                        cv2.imshow('default', vis_img)
                        if enable_tactile:
                            tactile_vis = render_tactile(env.tactile_left, env.tactile_right)
                            if tactile_vis is not None:
                                cv2.imshow('tactile', tactile_vis)
                        _ = cv2.pollKey()

                        stop_episode = False
                        for key_stroke in key_counter.get_press_events():
                            if key_stroke == KeyCode(char='s'):
                                stop_episode = True
                            elif key_stroke == KeyCode(char='q'):
                                stop_episode = True
                                should_quit = True

                        if (time.time() - eval_t_start) > max_duration:
                            print("Max duration reached.")
                            stop_episode = True

                        if stop_episode:
                            env.end_episode()
                            print("Stopped rollout.")
                            break

                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    env.end_episode()
                    should_quit = True
                finally:
                    ran_auto_episode = True

                if auto_start:
                    break


if __name__ == '__main__':
    main()
