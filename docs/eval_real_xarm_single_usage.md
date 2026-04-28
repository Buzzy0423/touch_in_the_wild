# Single-Arm xArm Deploy Usage

This document explains how to use:

- `scripts_real/eval_real_xarm_single.py`

for single-arm xArm policy deployment in `touch_in_the_wild`.

## Scope

This document covers the current implementation only.

Facts:

- the script is intended for a single-arm xArm checkpoint
- it does not require SpaceMouse
- it uses the single-arm `UmiEnv` path
- xArm pose/gripper semantics have been aligned to the current GELLO training convention

Current limitations:

- tactile observation is not included yet
- the script has passed syntax checks, but hardware behavior still needs real-robot validation

## Expected Data Semantics

The deploy path now assumes the same external semantics as the current GELLO xArm training data:

- end-effector position: meters
- end-effector rotation: radians
- gripper scalar: GELLO-normalized
  - `0 = open`
  - `1 = closed`

The xArm SDK still uses its own native units internally. Conversion is handled in:

- `umi/real_world/xarm_gello_util.py`
- `umi/real_world/xarm_interpolation_controller.py`
- `umi/real_world/xarm_gripper_controller.py`

The default xArm initialization joint target is taken from the GELLO config:

- `/home/zinan/Documents/zinan/gello_software/configs/xarm7_titw_hdmi.yaml`
- `agent.start_joints = [0.0, -0.6981, 0.0, 0.8727, 0.0, 1.5708, 0.0, 0.0]`

In the current deploy code:

- the first 7 values are used as xArm arm joints
- the last value is treated as the GELLO gripper entry and is not passed to the arm joint controller

## Prerequisites

You need all of the following before running the script:

- a working Python environment for this repository
- `cv2`, `torch`, `hydra`, `click`, `dill`, and xArm SDK dependencies installed
- a reachable xArm controller IP
- camera devices visible to `MultiUvcCamera`
- a checkpoint trained with the single-arm xArm observation/action convention

You should also confirm:

- the checkpoint expects vision-only observations
- the checkpoint action dimension is `7`
  - `[pos(3), rotvec(3), gripper(1)]`
- the checkpoint was trained on GELLO-style xArm data

## Script Entry Point

Run:

```bash
python scripts_real/eval_real_xarm_single.py -i /home/zinan/Documents/zinan/data/titw_ckpts/checkpoints/epoch=0110-train_loss=0.012.ckpt -o /home/zinan/Documents/zinan/data/titw_eval --camera_path /dev/video38
    # --camera_path /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB519234637-video-index0
```

If your gripper is controlled through the same xArm controller, you can omit `--gripper_ip`.

The script will default to:

- `gripper_ip = robot_ip`
- `robot_type = xarm`
- keyboard-only start / stop flow
- `camera_reorder = 0`

For HDMI capture devices, using an explicit `--camera_path` is recommended.

## Common Example

```bash
python scripts_real/eval_real_xarm_single.py \
    -i data/outputs/2026.04.22/your_run/checkpoints/latest.ckpt \
    -o data_local/xarm_eval_demo \
    --robot_ip 192.168.0.9 \
    --camera_path /dev/video12 \
    --frequency 10 \
    --steps_per_inference 6
```

If your setup uses multiple capture devices:

```bash
python scripts_real/eval_real_xarm_single.py \
    -i /path/to/checkpoint.ckpt \
    -o data_local/xarm_eval_demo \
    --robot_ip 192.168.0.9 \
    --camera_reorder 021 \
    --vis_camera_idx 0
```

## Main Options

Important arguments:

- `--input`, `-i`
  - checkpoint path
  - if you pass a run directory instead of a `.ckpt`, the script will use `checkpoints/latest.ckpt`
- `--output`, `-o`
  - output directory for replay buffer and recorded videos
- `--robot_ip`
  - xArm controller IP
- `--gripper_ip`
  - optional
  - defaults to `robot_ip`
- `--camera_reorder`
  - camera order string, for example `0` or `021`
- `--camera_path`
  - explicit camera device path, for example `/dev/video12`
  - when provided, it takes precedence over automatic camera discovery
- `--vis_camera_idx`
  - which camera stream to show in the OpenCV window
- `--init_joints / --no_init_joints`
  - whether to move to the GELLO default start joint position before rollout
  - default is enabled
- `--frequency`
  - control frequency in Hz
- `--steps_per_inference`
  - number of policy steps submitted after each inference
- `--max_duration`
  - maximum rollout duration in seconds
- `--start_delay`
  - delay between pressing start and actual rollout
- `--auto_start`
  - start one rollout immediately after initialization
- `--max_pos_speed`
  - max Cartesian position speed limit exposed to the controller
- `--max_rot_speed`
  - max Cartesian rotation speed limit exposed to the controller

Camera-related options:

- `--mirror_crop`
- `--mirror_swap`
- `--no_mirror`
- `--sim_fov`
- `--camera_intrinsics`

These follow the same general meaning as in the existing real-world deploy scripts.

## Runtime Flow

The script runs in this order:

1. load checkpoint
2. infer observation resolution from `shape_meta`
3. create `UmiEnv` with `robot_type='xarm'`
4. initialize cameras, robot controller, and gripper controller
5. warm up one policy inference pass
6. enter idle mode
7. wait for keyboard start or `--auto_start`
8. start an episode and run rollout
9. stop on user command or max duration

## Keyboard Controls

Idle mode:

- `c`: start rollout
- `q`: quit program

Rollout mode:

- `s`: stop the current rollout
- `q`: stop the rollout and quit

There is no teleoperation mode in this script.

## Output

The script writes data under the directory passed to `--output`.

Current outputs include:

- `replay_buffer.zarr`
- `videos/<episode_id>/...`

Each rollout started through the script becomes one recorded episode.

## What Changed Relative to Older xArm Deploy Code

This script is designed to avoid the older semantic mismatch between GELLO data and deploy controllers.

Now the deploy-facing contract is:

- robot observations are exposed in meters + radians
- robot actions are accepted in meters + radians
- gripper observations/actions use GELLO normalization

This means the deploy script does not need extra ad-hoc conversions such as:

- dividing Cartesian position by `1000`
- flipping gripper direction in the eval loop

## Recommended First Dry Run

For the first real test, keep the run conservative:

```bash
python scripts_real/eval_real_xarm_single.py \
    -i /path/to/checkpoint.ckpt \
    -o data_local/xarm_eval_dryrun \
    --robot_ip 192.168.0.9 \
    --camera_path /dev/video12 \
    --frequency 5 \
    --steps_per_inference 4 \
    --max_pos_speed 0.10 \
    --max_rot_speed 0.30
```

Before pressing `c`, verify:

- camera image is correct
- observation stream is stable
- robot is in a safe starting pose
- gripper is not obstructed
- emergency stop is reachable

During the first rollout, specifically verify:

- xArm translation scale is correct
- orientation direction is correct
- gripper command direction is correct
  - predicted smaller value should open
  - predicted larger value should close

## Known Caveats

Known facts:

- this path is currently vision-only
- tactile-enabled checkpoints are not wired into this script yet
- xArm default init joints come from the GELLO HDMI config listed above

Assumptions that still need validation on hardware:

- current speed defaults are appropriate for your robot setup
- current latency settings are acceptable for your cameras and controller
- the checkpoint action range is physically safe for your workspace

## Related Files

- `scripts_real/eval_real_xarm_single.py`
- `umi/real_world/umi_env.py`
- `umi/real_world/xarm_gello_util.py`
- `umi/real_world/xarm_interpolation_controller.py`
- `umi/real_world/xarm_gripper_controller.py`
- `docs/gello_data_vs_deploy.md`
- `docs/plan_single_arm_xarm_deploy.md`
