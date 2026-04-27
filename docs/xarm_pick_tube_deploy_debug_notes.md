# xArm Pick Tube Deploy Debug Notes

This note summarizes the training output under:

`data/outputs/2026.04.22/10.44.49_train_diffusion_unet_timm_xarm_pick_tube`

The goal is to help debug a deploy-time symptom where the robot arm jitters and does not complete the task.

## Training Run Summary

- Complete run: `10.44.49_train_diffusion_unet_timm_xarm_pick_tube`
- Dataset: `/home/zinan/workspace/data/processed/xarm_pick_tube.zarr.zip`
- Total steps: `410800`
- Total epochs: `200`
- Model checkpoint used by user: epoch `110`
- Tactile in this run: deploy does not use tactile, and the saved training config also has `policy.obs_encoder.use_tactile: false`

Loss trend:

| Epoch | train_loss | train_loss_all | val_loss_all |
| --- | ---: | ---: | ---: |
| 0 | 0.120021 | 0.023753 | 0.024160 |
| 5 | 0.019045 | 0.015982 | 0.018213 |
| 10 | 0.016664 | 0.014532 | 0.019070 |
| 20 | 0.014962 | 0.013365 | 0.021861 |
| 50 | 0.013167 | 0.012107 | 0.027541 |
| 100 | 0.011846 | 0.011227 | 0.033981 |
| 110 | 0.011594 | 0.011110 | 0.035073 |
| 150 | 0.010920 | 0.010739 | 0.043931 |
| 199 | 0.010574 | n/a | 0.051419 |

Interpretation:

- The model fits the training set well.
- Validation loss is best around epoch `5`, then trends upward.
- Epoch `110` is already in the overfitting region by validation loss.
- This alone may hurt success rate, but hard robot jitter is more likely caused by deploy-time mismatch: units, action scaling, pose representation, timing, or command execution.

## Dataset Ranges

These are from the processed zarr dataset used for training.

Dataset size:

- Episodes: `91`
- Total frames: `152266`
- Episode length min / mean / max: `1131 / 1673.25 / 2388`
- Validation split with `val_ratio=0.01`, `seed=42`: episode `[8]`

Raw low-dimensional data:

| Field | Min | Max | Mean | Std |
| --- | --- | --- | --- | --- |
| `robot0_eef_pos` | `[0.23561, -0.35077, 0.29272]` | `[0.59503, 0.24073, 0.59503]` | `[0.38960, -0.01737, 0.43873]` | `[0.07442, 0.10957, 0.05975]` |
| `robot0_eef_rot_axis_angle` | `[-3.14121, -1.19891, -0.98642]` | `[3.14137, 1.19876, 1.31222]` | `[1.58270, -0.09857, 0.20044]` | `[2.51592, 0.37667, 0.39866]` |
| `robot0_gripper_width` | `[-0.00250]` | `[0.93750]` | `[0.43547]` | `[0.38435]` |

Important observation:

- `robot0_eef_pos` is in the `0.2 - 0.6` range, so the training data position unit looks like meters.
- `robot0_gripper_width` is in approximately `[0, 0.94]`, not `[0, 0.09]`.

These are raw values from the zarr replay buffer. They are not the final values seen by the policy. The policy sees poses after the dataset converts them to the configured pose representation.

## Training Data Transform Path

The task config says:

```yaml
pose_repr:
  obs_pose_repr: relative
  action_pose_repr: relative
```

The training data path is:

1. `SequenceSampler` samples raw observation and action sequences.
2. If the zarr has no explicit `data/action`, `SequenceSampler` constructs action by concatenating:

```text
robot0_eef_pos + robot0_eef_rot_axis_angle + robot0_gripper_width
```

3. `UmiDataset.__getitem__()` converts raw pose observations and raw pose actions into the configured pose representation.
4. The normalizer is fit by iterating over `UmiDataset.__getitem__()`, so the normalizer statistics are computed after the relative-pose conversion.

Relevant code:

```python
# diffusion_policy/common/sampler.py
if 'action' in replay_buffer:
    self.replay_buffer['action'] = replay_buffer['action'][:]
else:
    actions = list()
    for robot_idx in range(self.num_robot):
        for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
            key = f'robot{robot_idx}_{cat}'
            if key in self.replay_buffer:
                actions.append(self.replay_buffer[key])
    self.replay_buffer['action'] = np.concatenate(actions, axis=-1)
```

```python
# diffusion_policy/dataset/umi_dataset.py
obs_pose_mat = convert_pose_mat_rep(
    pose_mat,
    base_pose_mat=pose_mat[-1],
    pose_rep=self.obs_pose_repr,
    backward=False)

action_pose_mat = convert_pose_mat_rep(
    action_mat,
    base_pose_mat=pose_mat[-1],
    pose_rep=self.obs_pose_repr,
    backward=False)
```

For `pose_rep: relative`, the transform is:

```python
# diffusion_policy/common/pose_repr_util.py
out = np.linalg.inv(base_pose_mat) @ pose_mat
```

This means both observation pose and action pose are represented relative to the current observation pose, specifically `pose_mat[-1]`.

Important distinction:

- Raw zarr `robot0_eef_pos` should look like meters: roughly `[0.2, 0.6]`.
- Policy input `robot0_eef_pos` after relative conversion should be near zero.
- Action position after relative conversion is a short-horizon relative target, with normalizer range roughly `[-0.105, 0.158]` meters.

Do not compare deploy `obs["robot0_eef_pos"]` directly against the normalizer's `robot0_eef_pos` range unless it has already passed through `get_real_umi_obs_dict()` / relative conversion.

## Normalizer Ranges

The saved `normalizer.pkl` uses these ranges.

Action normalizer input stats, action dim `10`:

| Slice | Meaning | Min | Max | Std |
| --- | --- | --- | --- | --- |
| `0:3` | relative target position | `[-0.084871, -0.090001, -0.105742]` | `[0.086163, 0.158440, 0.061255]` | `[0.009968, 0.012853, 0.012904]` |
| `3:9` | rotation 6D | `[0.915561, -0.401626, -0.300475, -0.317898, 0.913778, -0.303317]` | `[1.000000, 0.314427, 0.221656, 0.402162, 1.000000, 0.201228]` | `[0.002673, 0.030637, 0.030315, 0.030687, 0.002396, 0.021631]` |
| `9:10` | gripper | `[-0.002500]` | `[0.937500]` | `[0.384298]` |

Observation normalizer input stats:

| Field | Min | Max | Std |
| --- | --- | --- | --- |
| `robot0_eef_pos` | `[-0.022998, -0.027541, -0.021189]` | `[0.018396, 0.018946, 0.020215]` | `[0.001017, 0.001310, 0.001354]` |
| `robot0_gripper_width` | `[-0.002500]` | `[0.937500]` | `[0.384733]` |

Why `robot0_eef_pos` obs range is small:

- The task config uses `pose_repr.obs_pose_repr: relative`.
- During training, observations are converted relative to the current pose, so position obs are near zero.
- Raw robot state is still expected to be in the same physical unit as training before this relative transform is applied.
- The normalizer is fit after this conversion, not on the raw zarr pose.

## Deploy Checks On The Other Machine

### 1. Confirm robot pose units before any manual scaling

Print the raw deploy observation immediately after `env.get_obs()`:

```python
obs = env.get_obs()
print("raw robot0_eef_pos last:", obs["robot0_eef_pos"][-1])
print("raw robot0_eef_rot_axis_angle last:", obs["robot0_eef_rot_axis_angle"][-1])
print("raw robot0_gripper_width last:", obs["robot0_gripper_width"][-1])
```

Expected if deploy matches training:

- `robot0_eef_pos` should look like `[0.3, 0.0, 0.5]`, i.e. meters.
- If it looks like `[300, 0, 500]`, the robot API is returning millimeters and conversion to meters is needed.
- If it already looks like `[0.3, 0.0, 0.5]`, do not divide it by `1000`.

The current `eval_real.py` has manual unit conversion:

```python
obs["robot0_eef_pos"][:3] /= 1000.0
raw_action[:, :3] *= 1000
```

This is correct only if raw xArm observations/actions are in millimeters. If the environment already outputs meters, this will introduce a 1000x mismatch and can cause severe jitter.

### 2. Confirm final command units

Before `env.exec_actions(...)`, print:

```python
print("target poses min/max:", this_target_poses[:, :3].min(axis=0), this_target_poses[:, :3].max(axis=0))
print("first target pose:", this_target_poses[0])
print("current obs pos:", obs["robot0_eef_pos"][-1])
```

Expected command position scale depends on the controller:

- If `xArmAPI.set_position_aa` or equivalent expects millimeters, command positions may need to be around `[300, 0, 500]`.
- If the wrapper converts to meters internally, command positions should stay around `[0.3, 0.0, 0.5]`.
- The observation unit and action command unit must be handled consistently.

### 3. Confirm pose representation

Training config:

```yaml
pose_repr:
  obs_pose_repr: relative
  action_pose_repr: relative
```

Deploy must use the checkpoint config values, not a local default. Confirm logs print:

```text
obs_pose_rep relative
action_pose_repr relative
```

If deploy accidentally uses `abs`, `rel`, or `delta`, action decoding will be wrong.

### 4. Confirm checkpoint path

If using epoch `110`, confirm the input path is explicitly:

```text
data/outputs/2026.04.22/10.44.49_train_diffusion_unet_timm_xarm_pick_tube/checkpoints/epoch=0110-train_loss=0.012.ckpt
```

Do not pass only the run directory unless you intend to use `checkpoints/latest.ckpt`.

### 5. Confirm gripper scale

Training gripper width range:

```text
min -0.0025
max 0.9375
mean 0.4355
```

Deploy code should print current and target gripper values:

```python
print("obs gripper:", obs["robot0_gripper_width"][-1])
print("target gripper:", this_target_poses[:, 6].min(), this_target_poses[:, 6].max())
```

If deploy uses `[0, 0.09]` meters but training uses `[0, 0.94]`, gripper behavior will be off. This may not explain arm jitter directly, but it can break task completion.

### 6. Confirm action timing

Current deploy logic timestamps actions as:

```python
action_timestamps = np.arange(len(action)) * dt + obs_timestamps[-1]
```

Check:

- `frequency` matches the training/control assumption. Current default is `20 Hz`.
- `steps_per_inference` is reasonable. Current default is `6`; policy predicts `16` future actions.
- Most submitted actions should be in the future. If the logs often say `Over budget`, control may collapse to one late command repeatedly, which can look like jitter.

Print:

```python
print("num future actions:", np.sum(is_new), "of", len(action))
print("first timestamp delta:", action_timestamps[0] - time.time())
print("last timestamp delta:", action_timestamps[-1] - time.time())
```

### 7. Confirm observation values after preprocessing

Right before `policy.predict_action(obs_dict)`, print:

```python
for k, v in obs_dict_np.items():
    if k != "camera0_rgb":
        print(k, v.shape, np.nanmin(v), np.nanmax(v), v[-1] if v.ndim >= 2 else v)
```

Expected rough checks:

- `robot0_eef_pos` after relative conversion should be near zero, roughly within `[-0.03, 0.03]`.
- `robot0_eef_rot_axis_angle` is rotation 6D, not raw axis-angle; values should be near identity-like columns, not arbitrary huge values.
- `robot0_gripper_width` should be in the same range as training, roughly `[0, 0.94]`.

Recommended two-stage print:

```python
obs = env.get_obs()
print("RAW obs robot0_eef_pos:", obs["robot0_eef_pos"][-1])
print("RAW obs robot0_eef_rot_axis_angle:", obs["robot0_eef_rot_axis_angle"][-1])
print("RAW obs robot0_gripper_width:", obs["robot0_gripper_width"][-1])

obs_dict_np = get_real_umi_obs_dict(
    env_obs=obs,
    shape_meta=cfg.task.shape_meta,
    obs_pose_repr=obs_pose_rep,
    tx_robot1_robot0=tx_robot1_robot0,
    episode_start_pose=episode_start_pose)

print("POLICY obs robot0_eef_pos:", obs_dict_np["robot0_eef_pos"])
print("POLICY obs robot0_eef_rot_axis_angle:", obs_dict_np["robot0_eef_rot_axis_angle"])
print("POLICY obs robot0_gripper_width:", obs_dict_np["robot0_gripper_width"])
```

Expected:

- Raw `robot0_eef_pos`: same physical unit as training, likely meters, e.g. `[0.3, 0.0, 0.5]`.
- Policy `robot0_eef_pos`: relative pose values near zero.
- If raw position already looks like meters, applying `/1000` before `get_real_umi_obs_dict()` is likely wrong.
- If raw position looks like millimeters, e.g. `[300, 0, 500]`, convert to meters before relative conversion.

### 7b. Confirm action decode path

Training action is relative to the current observation pose. Deploy should invert that transform with the current raw robot pose:

```python
raw_action = result["action_pred"][0].detach().to("cpu").numpy()
action = get_real_umi_action(raw_action, obs, action_pose_repr)
```

For `action_pose_repr: relative`, `get_real_umi_action()` uses:

```python
action_mat = base_pose_mat @ action_pose_mat
```

where `base_pose_mat` is built from:

```python
obs[f"robot{robot_idx}_eef_pos"][-1]
obs[f"robot{robot_idx}_eef_rot_axis_angle"][-1]
```

Therefore the `obs` passed to `get_real_umi_action()` must use the same physical unit as the action decoder expects. If `obs` was divided by `1000` for policy input and then multiplied back before action decode, confirm that the resulting value is exactly in the controller's expected unit.

### 8. Confirm no tactile input is required

This run's saved config has:

```yaml
use_tactile: false
```

So deploy should not need `camera0_tactile`. If deploy code still does:

```python
obs_dict["camera0_tactile"] = obs_dict["camera0_tactile"].squeeze(1)
```

make sure this line is guarded or removed for this checkpoint. Otherwise it may crash or silently reflect a code path from a different tactile model.

## Most Likely Failure Modes

1. Unit mismatch between xArm API, `env.get_obs()`, and `env.exec_actions()`.
   - Especially suspicious: manual `/1000` and `*1000` in deploy.

2. Using epoch `110` despite validation loss already worsening.
   - This may reduce success, but usually does not by itself create violent jitter.

3. Gripper scale mismatch.
   - Training range is around `[0, 0.94]`; some scripts use `max_gripper_width = 0.09`.

4. Action timing/latency issue.
   - If many actions are late, deploy may repeatedly execute stale or single-step commands.

5. Pose representation mismatch.
   - Must stay `relative` for both obs and action.

## Minimal Field Test

On the deploy machine, before running a full rollout:

1. Start policy.
2. Get one observation.
3. Run one `predict_action`.
4. Print raw obs, processed obs, raw predicted action, decoded target action.
5. Do not send action to robot yet.

The decoded target position should be close to current pose plus a plausible short-horizon movement. If current pose is around `[0.3, 0.0, 0.5]` meters, decoded target should not suddenly jump to `[300, 0, 500]` unless the controller expects millimeters.
