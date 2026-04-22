# Training Touch-in-the-Wild Vision Diffusion Policy on xArm Gello Data

This guide covers converting xArm gello demonstrations into the format expected by `UmiDataset`, and training the vision-only diffusion policy (ViT-Base CLIP + UNet).

## 1. Data Conversion

### Input: gello raw session

```
session_20260409_123442/
├── demo_20260409_123806/
│   ├── video_gopro_rgb.mp4      # wrist camera, 1920x1080 @ 30fps
│   └── robot_state.jsonl        # one JSON line per frame
├── demo_20260409_123842/
│   └── ...
└── ...  (20 demos)
```

Each `robot_state.jsonl` line contains:

```json
{
  "frame_index": 0,
  "ee_pos_quat": [x, y, z, qx, qy, qz, qw],
  "gripper_position": 0.0,
  "joint_positions": [7 floats],
  ...
}
```

- `ee_pos_quat[0:3]`: EE position in meters (xArm base_link frame)
- `ee_pos_quat[3:7]`: quaternion in **XYZW** order (produced by gello's `_quat_from_aa`)
- `gripper_position`: gripper value, range ~[0, 0.6]

### Run the conversion script

```bash
python /path/to/gello_software/scripts/convert_gello_to_umi_zarr.py \
    --session_dir /path/to/session_20260409_123442 \
    --output /path/to/xarm_pick_apple.zarr.zip \
    --img_size 224
```

### Output: Zarr ReplayBuffer

| Zarr key | Shape | Dtype | Description |
|---|---|---|---|
| `data/robot0_eef_pos` | `(T, 3)` | float32 | EE position, meters, absolute |
| `data/robot0_eef_rot_axis_angle` | `(T, 3)` | float32 | EE rotation as axis-angle (rotvec), radians, absolute |
| `data/robot0_gripper_width` | `(T, 1)` | float32 | Gripper width |
| `data/robot0_demo_start_pose` | `(T, 6)` | float32 | `[pos(3), rotvec(3)]` of first frame, repeated T times |
| `data/robot0_demo_end_pose` | `(T, 6)` | float32 | `[pos(3), rotvec(3)]` of last frame, repeated T times |
| `data/camera0_rgb` | `(T, 224, 224, 3)` | uint8 | RGB frames |
| `data/action` | `(T, 7)` | float32 | `[pos(3), rotvec(3), gripper(1)]`, absolute |
| `meta/episode_ends` | `(N_eps,)` | int | Cumulative episode boundaries |

All poses are stored **absolute**. The `UmiDataset` converts them to relative (w.r.t. last observation timestep) on-the-fly during training.

### Rotation convention note

The quaternion in `ee_pos_quat` is **XYZW** (as output by gello's `_quat_from_aa`). The conversion script reads it correctly and converts to scipy `rotvec`. For sessions recorded after 2026-04-10, the script uses `ee_rpy_sdk` instead (xArm SDK RPY, no quaternion ambiguity).

Previous pipelines (e.g. `convert_xarm_clip_data.py`) misread this as WXYZ, causing a ~180 degree roll error that required a deploy-time `R_flip` correction. That bug does not apply here.

## 2. Task Config

A task config is provided at `diffusion_policy/config/task/xarm_pick_apple.yaml`. Key differences from the default `umi.yaml`:

| Setting | `umi.yaml` (default) | `xarm_pick_apple.yaml` |
|---|---|---|
| `obs_down_sample_steps` | 3 (for ~60Hz GoPro) | 1 (data is ~30Hz) |
| `camera0_tactile` | present | removed (no tactile sensors) |
| `dataset_frequeny` | 0 | 0 |
| All latency steps | computed from GoPro-robot offset | 0 (single-source recording) |
| `n_train_episodes` | 100 | null (use all) |
| `dataset_path` | example_demo_session/... | `/home/zinan/Documents/zinan/data/xarm_pick_apple.zarr.zip` |

### Observation shapes at training time

After `UmiDataset.__getitem__` processes the raw data:

```
obs:
  camera0_rgb:                        (2, 3, 224, 224)  float32   # 2-step obs horizon
  robot0_eef_pos:                     (2, 3)            float32   # relative to last obs step
  robot0_eef_rot_axis_angle:          (2, 6)            float32   # rot6d, relative
  robot0_gripper_width:               (2, 1)            float32
  robot0_eef_rot_axis_angle_wrt_start:(2, 6)            float32   # rot6d, relative to episode start

action:                               (16, 10)          float32   # [pos(3), rot6d(6), gripper(1)]
```

The dataset stores 3D axis-angle but the config specifies `rotation_rep: rotation_6d`, so `UmiDataset.__getitem__` converts the 3D rotvec to 6D rotation representation (first two rows of rotation matrix) via `mat_to_pose10d`.

## 3. Training

### Single-GPU

```bash
conda activate touchwild

python train.py \
    --config-name train_diffusion_unet_timm_umi_workspace \
    task=xarm_pick_apple \
    policy.obs_encoder.use_tactile=false \
    training.n_train_episodes=null
```

### Multi-GPU

```bash
accelerate --num_processes <NGPUS> train.py \
    --config-name train_diffusion_unet_timm_umi_workspace \
    task=xarm_pick_apple \
    policy.obs_encoder.use_tactile=false \
    training.n_train_episodes=null
```

### Commonly adjusted parameters

```bash
# Change dataset path
task.dataset_path=/path/to/other_dataset.zarr.zip

# Reduce batch size if GPU memory is limited
dataloader.batch_size=32

# Train longer (default 200 epochs)
training.num_epochs=500

# Change learning rate
optimizer.lr=1e-4

# Disable wandb logging
logging.mode=disabled

# Use a specific GPU
training.device=cuda:1
```

## 4. Architecture Summary

```
                    ┌─────────────────────────┐
 camera0_rgb ──────►│ ViT-Base (CLIP pretrained)│──► attention_pool_2d ──┐
 (2, 3, 224, 224)   └─────────────────────────┘    per frame → (768,)   │
                                                    flatten → (1536,)    │
                                                                         ├── cat ──► global_cond
 robot0_eef_pos ────────────────┐                                        │    (1568,)
 robot0_eef_rot_axis_angle ─────┤ flatten over time ──► (32,) ──────────┘        │
 robot0_gripper_width ──────────┤                                                │
 robot0_eef_rot_axis_angle_wrt_start ┘                                           │
                                                                                 ▼
                                                              ┌──────────────────────────┐
                         noisy actions ──────────────────────►│  Conditional UNet1D       │
                         (16, 10)                              │  FiLM conditioning       │
                                                              │  16 DDIM denoising steps  │
                                                              └──────────┬───────────────┘
                                                                         │
                                                                         ▼
                                                              action_pred (16, 10)
                                                              [pos(3), rot6d(6), grip(1)]
```

- The ViT encoder is fine-tuned (not frozen) with a lower learning rate (`pretrain_clip_lr: 2e-5`)
- The UNet uses FiLM conditioning: at every residual block, the global_cond vector produces per-channel scale and bias
- 8 of the 16 predicted action steps are actually executed (`n_action_steps: 8`)

## 5. Checkpoints and Evaluation

Checkpoints are saved to `data/outputs/<date>/<time>_train_diffusion_unet_timm_xarm_pick_apple/checkpoints/`.

Top-k checkpoints are kept based on `train_loss` (lower is better). The last checkpoint is also saved.

To evaluate a checkpoint on the real robot, see the main README's "Real-World Deployment" section. The key command:

```bash
python eval_real.py \
    --robot_config example/eval_robots_config.yaml \
    -i /path/to/checkpoint.ckpt \
    -o /path/to/eval_output
```

## 6. Adding More Demos

To add more demonstrations to the dataset:

1. Record new demos in a new gello session
2. Convert the new session to a separate zarr
3. Merge zarr files, or re-run conversion pointing at multiple session directories

The conversion script currently processes one session at a time. To combine sessions, convert each to a `.zarr` directory (not `.zarr.zip`), then write a short script that calls `replay_buffer.add_episode` from each source into a single output buffer.
