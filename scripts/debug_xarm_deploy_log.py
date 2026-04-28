"""
Debug script for xArm deployment issues.
Analyzes the deploy replay buffer to diagnose why the arm jiggles.

Usage:
    python scripts/debug_xarm_deploy_log.py \
        -d /home/zinan/Documents/zinan/data/titw_eval/deploy_20260427_155828 \
        -c /home/zinan/Documents/zinan/data/titw_ckpts/checkpoints/latest.ckpt
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import argparse
import dill
import json
import numpy as np
import torch
import zarr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy_dir', '-d', required=True,
                        help='Path to deploy output directory')
    parser.add_argument('--ckpt', '-c', default=None,
                        help='Path to checkpoint.')
    return parser.parse_args()


def load_checkpoint(ckpt_path):
    payload = torch.load(ckpt_path, map_location='cpu', pickle_module=dill)
    return payload


def main():
    args = parse_args()
    deploy_dir = args.deploy_dir
    zarr_path = os.path.join(deploy_dir, 'replay_buffer.zarr')
    if not os.path.exists(zarr_path):
        print(f"ERROR: {zarr_path} not found")
        sys.exit(1)

    root = zarr.open(zarr_path, mode='r')
    data = root['data']
    meta = root['meta']

    # ---- Basic stats ----
    print("=" * 70)
    print("1. 基本信息")
    print("=" * 70)
    n_episodes = meta['episode_ends'].shape[0]
    ep_lens = meta['episode_ends'][:]
    print(f"  Episodes 数量: {n_episodes}")
    print(f"  Episodes 长度: {ep_lens.tolist()}")

    action = data['action'][:]
    ts = data['timestamp'][:]
    if len(ts) > 1:
        duration = ts[-1] - ts[0]
        freq = len(ts) / duration if duration > 0 else float('nan')
        print(f"  总步数: {len(action)}")
        print(f"  持续时间: {duration:.2f} s")
        print(f"  实际频率: {freq:.1f} Hz")

    # ---- EEF trajectory ----
    print()
    print("=" * 70)
    print("2. EEF 轨迹分析")
    print("=" * 70)
    eef_pos = data['robot0_eef_pos'][:]
    eef_rot = data['robot0_eef_rot_axis_angle'][:]
    print(f"  起始位置 (m): {eef_pos[0].tolist()}")
    print(f"  结束位置 (m): {eef_pos[-1].tolist()}")
    displacement = np.linalg.norm(eef_pos[-1] - eef_pos[0])
    print(f"  总位移: {displacement*1000:.1f} mm")
    print(f"  起始旋转 (axis-angle): {eef_rot[0].tolist()}")
    print(f"  结束旋转 (axis-angle): {eef_rot[-1].tolist()}")

    # ---- Action analysis ----
    print()
    print("=" * 70)
    print("3. Action 分析 (发送给 xArm 的绝对位姿)")
    print("=" * 70)
    print(f"  Action 形状: {action.shape}")
    n_dead = np.sum(np.all(np.abs(action) < 1e-10, axis=1))
    print(f"  零 action 步数: {n_dead} / {len(action)}")
    print(f"  Action 位置范围 (m): x=[{action[:,0].min():.4f}, {action[:,0].max():.4f}], "
          f"y=[{action[:,1].min():.4f}, {action[:,1].max():.4f}], "
          f"z=[{action[:,2].min():.4f}, {action[:,2].max():.4f}]")
    print(f"  Action 旋转范围 (rad): "
          f"rx=[{action[:,3].min():.4f}, {action[:,3].max():.4f}], "
          f"ry=[{action[:,4].min():.4f}, {action[:,4].max():.4f}], "
          f"rz=[{action[:,5].min():.4f}, {action[:,5].max():.4f}]")

    action_diff = np.diff(action[:, :3], axis=0)
    pos_changes = np.linalg.norm(action_diff, axis=1)
    print(f"\n  Action 相邻步位置变化 (mm): 均值={pos_changes.mean()*1000:.2f}, "
          f"标准差={pos_changes.std()*1000:.2f}, 最大={pos_changes.max()*1000:.2f}")
    n_large = np.sum(pos_changes > 0.05)
    print(f"  >5cm 变化的步数: {n_large} / {len(pos_changes)}")

    # ---- Oscillation check ----
    print()
    print("=" * 70)
    print("4. 震荡检测 (方向变化比例)")
    print("=" * 70)
    print("  (方向变化率 > 30% 表示明显的来回抖动)")
    for axis, label in enumerate(['x', 'y', 'z']):
        signs = np.sign(np.diff(action[:, axis]))
        n_flips = np.sum(np.diff(signs) != 0)
        ratio = n_flips / len(signs) * 100
        bar = '!' * max(0, int(ratio / 5))
        print(f"  {label}: {n_flips}/{len(signs)} ({ratio:.1f}%) {bar}")

    # ---- Gripper analysis ----
    print()
    print("=" * 70)
    print("5. Gripper 分析")
    print("=" * 70)
    gripper = data['robot0_gripper_width'][:].flatten()
    print(f"  起始={gripper[0]:.4f}, 结束={gripper[-1]:.4f}, "
          f"最小={gripper.min():.4f}, 最大={gripper.max():.4f}")

    from scipy.ndimage import uniform_filter1d
    grip_smooth = uniform_filter1d(gripper, size=5)
    peaks = max(0, (np.diff(np.sign(np.diff(grip_smooth))) < 0).sum())
    print(f"  大约开合次数 (平滑后): {peaks} 次")
    if peaks > 3:
        print(f"  *** 夹爪反复开合 > 3 次，说明模型不确定何时抓取")

    print()
    print(f"  注: GELLO gripper 约定: 0=完全张开(raw 800), 1=完全闭合(raw 0)")
    print(f"  训练数据使用同样的约定，映射应该正确")

    # ---- EEF tracking quality ----
    print()
    print("=" * 70)
    print("6. EEF 追踪质量")
    print("=" * 70)
    n = min(len(action), len(eef_pos))
    errors = np.linalg.norm(action[:n, :3] - eef_pos[:n], axis=1)
    print(f"  平均追踪误差 (m): {errors.mean():.4f}")
    print(f"  最大追踪误差 (m): {errors.max():.4f}")
    print(f"  误差 > 5cm 步数: {(errors > 0.05).sum()} / {n}")
    if errors.mean() < 0.03:
        print(f"  => 追踪良好，xArm 控制器正常。问题根源在模型输出本身。")

    # ---- Checkpoint config analysis ----
    payload = None
    cfg = None
    if args.ckpt:
        print()
        print("=" * 70)
        print("7. Checkpoint 配置分析")
        print("=" * 70)
        try:
            payload = load_checkpoint(args.ckpt)
            cfg = payload['cfg']
            print(f"  obs_pose_repr: {cfg.task.pose_repr.obs_pose_repr}")
            print(f"  action_pose_repr: {cfg.task.pose_repr.action_pose_repr}")
            sm = cfg.task.shape_meta
            print(f"  观测 horizon: camera={sm['obs']['camera0_rgb']['horizon']}, "
                  f"robot={sm['obs']['robot0_eef_pos']['horizon']}")
            print(f"  观测 down_sample: {sm['obs']['camera0_rgb']['down_sample_steps']}")
            print(f"  Action horizon: {sm['action']['horizon']}, "
                  f"down_sample: {sm['action']['down_sample_steps']}")

            ds = sm['action']['down_sample_steps']
            raw_freq_guess = 59.94
            train_eff_freq = raw_freq_guess / ds
            if len(ts) > 1:
                deploy_freq = len(action) / (ts[-1] - ts[0])
            else:
                deploy_freq = 0

            print(f"\n  训练有效频率: ~{train_eff_freq:.1f} Hz "
                  f"(raw={raw_freq_guess}Hz / down_sample={ds})")
            print(f"  部署实际频率: {deploy_freq:.1f} Hz")
            if abs(deploy_freq - train_eff_freq) < 3:
                print(f"  => 频率匹配，排除时序错位")
            else:
                print(f"  *** 频率不匹配! 建议 --frequency {train_eff_freq:.0f}")

            print(f"\n  ImageNet 归一化: {cfg.policy.obs_encoder.get('imagenet_norm', '?')}")
            print(f"  模型: {cfg.policy.obs_encoder.model_name}")
            print(f"  训练数据: {cfg.task.dataset.dataset_path}")

            # Check training augmentations
            transforms = cfg.policy.obs_encoder.get('transforms', [])
            print(f"  训练 augmentations: {[t.get('type', t.get('_target_', '?')) for t in transforms]}")
            print(f"  (部署时不用 augmentation，这是正确的)")

        except Exception as e:
            print(f"  加载 checkpoint 失败: {e}")
            import traceback
            traceback.print_exc()

    # ---- Reconstruct relative actions ----
    print()
    print("=" * 70)
    print("8. 重建模型输出 (相对动作)")
    print("=" * 70)
    try:
        from scipy.spatial.transform import Rotation as R
        def pose_to_mat(pose):
            pos, rot_aa = pose[..., :3], pose[..., 3:]
            rot = R.from_rotvec(rot_aa)
            mat = np.zeros(pose.shape[:-1] + (4, 4))
            mat[..., :3, :3] = rot.as_matrix()
            mat[..., :3, 3] = pos
            mat[..., 3, 3] = 1
            return mat

        def mat_to_rot6d(mat):
            return mat[:3, :3][:2, :].reshape(6)

        rel_actions_9d = []
        for i in range(len(action)):
            eef_pose = np.concatenate([eef_pos[i], eef_rot[i]])
            eef_mat = pose_to_mat(eef_pose)
            act_pose_mat = pose_to_mat(action[i, :6])
            rel_mat = np.linalg.inv(eef_mat) @ act_pose_mat
            rel_actions_9d.append(
                np.concatenate([rel_mat[:3, 3], mat_to_rot6d(rel_mat)]))
        rel_actions_9d = np.array(rel_actions_9d)

        valid = rel_actions_9d[3:]
        pos_norms = np.linalg.norm(valid[:, :3], axis=1)
        print(f"  预热后每步位移: 均值={pos_norms.mean()*1000:.2f}mm, "
              f"最大={pos_norms.max()*1000:.2f}mm, 最小={pos_norms.min()*1000:.2f}mm")

        rel_diff = np.diff(valid[:, :3], axis=0)
        for a, label in enumerate(['x', 'y', 'z']):
            n_flips = np.sum(np.diff(np.sign(rel_diff[:, a])) != 0)
            print(f"  相对位移方向变化 ({label}): {n_flips}/{len(rel_diff)} "
                  f"({n_flips/len(rel_diff)*100:.1f}%)")
    except ImportError:
        print("  scipy 未安装，跳过")

    # ---- Video check ----
    print()
    print("=" * 70)
    print("9. 相机数据检查")
    print("=" * 70)
    videos_dir = os.path.join(deploy_dir, 'videos')
    video_files = []
    if os.path.exists(videos_dir):
        for root_dir, dirs, files in os.walk(videos_dir):
            for f in files:
                video_files.append(os.path.join(root_dir, f))
    if video_files:
        print(f"  找到 {len(video_files)} 个视频文件: {video_files}")
    else:
        print("  无视频文件！")
        print("  可能原因: GelloMultiUvcCamera 的 NVENC 录制未启用，")
        print("  或者视频录制失败了。无法视觉验证相机输入。")

    # ---- Summary & recommendations ----
    print()
    print("=" * 70)
    print("诊断总结")
    print("=" * 70)

    # Check 1: oscillation
    n_flips_x = np.sum(np.diff(np.sign(np.diff(action[:, 0]))) != 0)
    osc_ratio = n_flips_x / (len(action) - 1) * 100
    if osc_ratio > 30:
        print("[异常] Action 方向变化率 > 30%，手臂严重抖动")
    else:
        print("[正常] Action 方向变化率在可接受范围")

    # Check 2: gripper oscillation
    if peaks > 3:
        print("[异常] 夹爪反复开合，抓取策略不稳定")
    else:
        print("[正常] 夹爪行为合理")

    # Check 3: tracking
    if errors.mean() < 0.03:
        print("[正常] EEF 追踪良好，xArm 控制器工作正常")
    else:
        print("[异常] EEF 追踪误差大，检查控制器")

    # Check 4: freq match
    if args.ckpt and cfg is not None and len(ts) > 1:
        ds = cfg.task.shape_meta['action']['down_sample_steps']
        train_freq = 59.94 / ds
        deploy_freq = len(action) / (ts[-1] - ts[0])
        if abs(deploy_freq - train_freq) < 3:
            print("[正常] 部署频率与训练频率匹配")
        else:
            print(f"[异常] 频率不匹配 (训练~{train_freq:.0f}Hz, 部署~{deploy_freq:.0f}Hz)")

    print()
    print("可能的根因:")
    print("  1. 相机图像与训练数据差异太大 (光照、视角、背景)")
    print("  2. 模型过拟合训练数据，泛化能力不足")
    print("  3. 图像预处理配置与训练不一致 (mask, crop, mirror)")
    print()
    print("建议下一步:")
    print("  1. 在部署时保存相机图像，与训练数据对比")
    print("  2. 检查 eval 脚本的 --no_mirror, --mirror_crop 参数是否正确")
    print("  3. 尝试用训练时的 camera 设置采集新数据测试")
    print("  4. 尝试在部署时设置 -nm (no mirror mask) 查看效果")
    print("  5. 在部署时调整 --camera_reorder 参数确保相机顺序正确")


if __name__ == '__main__':
    main()
