"""
Usage:
(umi): python scripts_real/eval_real_xarm_single.py \
    -i /path/to/checkpoint.ckpt \
    -o data_local/xarm_eval

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
from umi.real_world.umi_env import UmiEnv
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def render_obs(obs, vis_camera_idx, mirror_crop):
    if mirror_crop:
        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
        crop_img = obs['camera0_rgb_mirror_crop'][-1]
        vis_img = np.concatenate([vis_img, crop_img], axis=1)
    else:
        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
    return vis_img.copy()


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
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', default='192.168.0.9', show_default=True)
@click.option('--gripper_ip', default=None, help='Defaults to robot_ip for xArm')
@click.option('--camera_reorder', '-cr', default='021', show_default=True)
@click.option('--vis_camera_idx', default=0, type=int, help='Which camera to visualize.')
@click.option('--init_joints/--no_init_joints', default=True, show_default=True,
    help='Move xArm to the GELLO start_joints before enabling rollout.')
@click.option('--steps_per_inference', '-si', default=6, type=int, show_default=True)
@click.option('--max_duration', '-md', default=60.0, type=float, show_default=True)
@click.option('--frequency', '-f', default=10.0, type=float, show_default=True)
@click.option('--start_delay', default=1.0, type=float, show_default=True)
@click.option('--auto_start', is_flag=True, default=False, help='Start one rollout immediately.')
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--max_pos_speed', default=0.25, type=float, show_default=True)
@click.option('--max_rot_speed', default=0.6, type=float, show_default=True)
def main(
    input,
    output,
    robot_ip,
    gripper_ip,
    camera_reorder,
    vis_camera_idx,
    init_joints,
    steps_per_inference,
    max_duration,
    frequency,
    start_delay,
    auto_start,
    no_mirror,
    sim_fov,
    camera_intrinsics,
    mirror_crop,
    mirror_swap,
    max_pos_speed,
    max_rot_speed,
):
    if gripper_ip is None:
        gripper_ip = robot_ip

    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')

    with open(ckpt_path, 'rb') as f:
        payload = torch.load(f, map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    dt = 1 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)

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

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, UmiEnv(
            output_dir=output,
            robot_ip=robot_ip,
            gripper_ip=gripper_ip,
            frequency=frequency,
            robot_type='xarm',
            obs_image_resolution=obs_res,
            obs_float32=True,
            camera_reorder=[int(x) for x in camera_reorder],
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            camera_obs_latency=0.17,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
            robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
            gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
            no_mirror=no_mirror,
            fisheye_converter=fisheye_converter,
            mirror_crop=mirror_crop,
            mirror_swap=mirror_swap,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(2)
            cv2.namedWindow('default', cv2.WINDOW_NORMAL)
            print("Waiting for camera")
            time.sleep(1.0)

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
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs,
                    shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
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
                print('Idle. Press "c" to start rollout, "q" to quit.')
                t_idle_start = time.monotonic()
                iter_idx = 0
                while True:
                    t_cycle_end = t_idle_start + (iter_idx + 1) * dt
                    obs = env.get_obs()
                    vis_img = render_obs(obs, vis_camera_idx, mirror_crop)
                    episode_id = env.replay_buffer.n_episodes
                    draw_text(vis_img, [
                        f'Episode: {episode_id}',
                        'Mode: idle',
                        'Keys: c=start, q=quit',
                    ])
                    cv2.imshow('default', vis_img[..., ::-1])
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
                    eval_t_start = time.time() + start_delay
                    rollout_t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
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

                        vis_img = render_obs(obs, vis_camera_idx, mirror_crop)
                        draw_text(vis_img, [
                            f'Episode: {env.replay_buffer.n_episodes}',
                            f'Mode: rollout {time.monotonic() - rollout_t_start:.1f}s',
                            'Keys: s=stop, q=quit',
                        ])
                        cv2.imshow('default', vis_img[..., ::-1])
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
