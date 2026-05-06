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
- tactile can be enabled for checkpoints that expect `camera0_tactile`
- tactile can alternatively be predicted from RGB via Tactile_Gen (`--tacgen_ckpt_path`), which is mutually exclusive with `--enable_tactile`
- the visualization window shows real-time loop frequency (moving average of the last 10 cycles) in both idle and rollout modes

Current limitations:

- the script has passed syntax checks, but hardware behavior still needs real-robot validation
- tactile is provided to the policy online, but the current deploy replay buffer writer does not persist `camera0_tactile`

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

- the checkpoint expects either vision-only observations or `camera0_rgb` plus `camera0_tactile`
- the checkpoint action dimension is `7`
  - `[pos(3), rotvec(3), gripper(1)]`
- the checkpoint was trained on GELLO-style xArm data
- if using tactile, both tactile serial devices are present and match the selected ports
- if using tactile, the checkpoint tactile key is `camera0_tactile`

The deploy script will refuse to run a tactile checkpoint unless `--enable_tactile` is set. It currently produces only `camera0_tactile`; other tactile key names are not supported by this script.

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
    --camera_path /dev/video-hdmi \
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

For a vision+tactile checkpoint:

```bash
python scripts_real/eval_real_xarm_single.py \
    -i /path/to/vision_tactile_checkpoint.ckpt \
    -o data_local/xarm_eval_tactile \
    --robot_ip 192.168.1.239 \
    --camera_path /dev/video-hdmi \
    --enable_tactile \
    --tactile_left_port /dev/LeftTactile \
    --tactile_right_port /dev/RightTactile
```

For a vision+tactile checkpoint using Tactile_Gen (simulated tactile from RGB, no hardware required):

```bash
python scripts_real/eval_real_xarm_single.py \
    -i /path/to/vision_tactile_checkpoint.ckpt \
    -o data_local/xarm_eval_tacgen \
    --robot_ip 192.168.1.239 \
    --camera_path /dev/video38 \
    --tacgen_ckpt_path /home/zinan/Documents/zinan/Tactile_Gen/outputs/your_run/checkpoints/latest.ckpt
```

Tactile_Gen inference only runs during rollout, not during idle. The predicted tactile is visualized in the `'tactile'` window alongside the rollout.

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

	- `--headless`
	  - pure CLI mode: skip all OpenCV visualization windows
	  - logging (print statements) and video/replay buffer saving run normally
	  - keyboard controls (`s`/`q`) still work via pynput
Tactile-related options (real hardware):

- `--enable_tactile`
  - starts the left and right tactile serial controllers
  - required if the checkpoint `shape_meta.obs` contains `camera0_tactile`
  - if the checkpoint is vision-only, this can be omitted
  - mutually exclusive with `--tacgen_ckpt_path`
- `--tactile_left_port`
  - serial device path for the left tactile sensor
  - default: `/dev/LeftTactile`
- `--tactile_right_port`
  - serial device path for the right tactile sensor
  - default: `/dev/RightTactile`
- `--tactile_latency`
  - latency compensation in seconds
  - this value is subtracted from tactile receive time before timestamp alignment
  - default: `0.06`
- `--tactile_median_samples`
  - number of initial tactile frames used to estimate the median baseline
  - default: `30`
  - keep the fingers unloaded during this baseline period
- `--tactile_buffer_size`
  - maximum number of tactile samples that can be requested from each tactile ring buffer
  - default: `300`
- `--tactile_buffer_fps`
  - expected tactile producer frequency used to compute how many recent samples to read for alignment
  - default: `150.0`
  - increase this only if the tactile producer is actually faster and the buffer size is large enough

Tactile_Gen options (simulated tactile, no hardware required):

- `--tacgen_ckpt_path`
  - path to a Tactile_Gen DETR checkpoint
  - when set, predicted tactile from RGB replaces real tactile input
  - mutually exclusive with `--enable_tactile`
  - the checkpoint must match the tactile key name expected by the policy (`camera0_tactile`)
  - Tactile_Gen inference only runs during rollout (skipped in idle mode to save GPU time)
- `--tacgen_depth_encoder`
  - DepthAnything encoder variant used when the Tactile_Gen checkpoint expects depth
  - choices: `vits`, `vitb`, `vitl` (default: `vitl`)
  - only relevant if the Tactile_Gen model has `uses_depth=True`
- `--tacgen_depth_ckpt`
  - path to a DepthAnything-V2 checkpoint
  - default: `<Tactile_Gen>/third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth`
  - only relevant if the Tactile_Gen model has `uses_depth=True`
- `--tacgen_mask_path`
  - path to a fisheye mask PNG used for border suppression in Tactile_Gen's DETR attention
  - overrides the `mask_path` stored in the checkpoint config (useful when loading a checkpoint trained on another machine)
  - default: `<Tactile_Gen>/assets/mask.png`

Tactile recalibration:

- `--tactile_recalibration_timeout`
  - seconds to wait for tactile baseline recalibration before each rollout
  - default: `20.0`
  - applies to both real tactile hardware and Tactile_Gen paths

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
3. detect whether `shape_meta` expects tactile
4. optionally create tactile controllers if `--enable_tactile` is set
5. create `UmiEnv` with `robot_type='xarm'`
6. initialize cameras, robot controller, gripper controller, and optional tactile controllers
7. warm up one policy inference pass
8. enter idle mode (real-time frequency shown in overlay)
9. wait for keyboard start or `--auto_start`
10. start an episode and run rollout (real-time frequency shown in overlay)
11. stop on user command or max duration

When tactile is enabled, `UmiEnv.get_obs()` aligns tactile to the camera reference timeline:

- the latest camera timestamp defines the reference time
- tactile target timestamps are built backwards from that reference time
- left and right tactile frames are selected by nearest-neighbor timestamp
- each side is flipped along rows (first row becomes last row, matching the offline GELLO converter)
- the two sides are concatenated into `camera0_tactile` with shape `(T, 12, 64)`
- the deploy script adds the batch dimension before policy inference, producing `(B, T, 12, 64)`

## Keyboard Controls

Idle mode:

- `s`: start rollout
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

Current note for tactile:

- tactile is used for online policy inference when enabled
- `camera0_tactile` is not currently written into the deploy replay buffer by `UmiEnv.end_episode()`

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

Before pressing `s`, verify:

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

- this path supports vision-only checkpoints and checkpoints expecting `camera0_tactile`
- tactile support requires both left and right tactile serial devices
- tactile is aligned online using nearest-neighbor timestamps against the camera reference timeline
- xArm default init joints come from the GELLO HDMI config listed above

Assumptions that still need validation on hardware:

- current speed defaults are appropriate for your robot setup
- current latency settings are acceptable for your cameras and controller
- `--tactile_latency 0.06` is acceptable for your tactile hardware
- tactile row flip matches the offline converter and your training data
- the checkpoint action range is physically safe for your workspace

## Related Files

- `scripts_real/eval_real_xarm_single.py`
- `umi/real_world/umi_env.py`
- `umi/real_world/tactile_controller_left.py`
- `umi/real_world/tactile_controller_right.py`
- `umi/real_world/xarm_gello_util.py`
- `umi/real_world/xarm_interpolation_controller.py`
- `umi/real_world/xarm_gripper_controller.py`
- `/home/zinan/Documents/zinan/gello_software/scripts/convert_gello_to_umi_zarr.py`
- `scripts_real/calibrate_camera_pose.py`
- `docs/gello_data_vs_deploy.md`
- `docs/plan_single_arm_xarm_deploy.md`

## Camera Pose Calibration

If you suspect the deploy camera pose differs from the training camera pose, use the calibration script to align them:

```bash
python scripts_real/calibrate_camera_pose.py \
    -t /home/zinan/Documents/zinan/data/gello_raw/session_20260430_164834 \
    --camera_path /dev/video38
```

This overlays a training reference frame on the live camera feed so you can physically adjust the camera until the scene matches.

Controls:

| Key | Action |
|-----|--------|
| `o` / `p` | decrease / increase overlay opacity (0.1 steps) |
| `m` | cycle display mode: `overlay` -> `side-by-side` -> `diff` -> `diff_split` -> `live` |
| `n` | cycle to next reference frame (from different demos) |
| `f` | freeze current live frame as reference |
| `r` | restore training reference (unfreeze) |
| `s` | save current frame to disk |
| `q` / `ESC` | quit |

Display modes:

- **overlay**: training reference frame is semi-transparent on top of live feed — align edges and features until they coincide
- **side-by-side**: live feed on left, training reference on right — compare scene composition
- **diff**: pixel-wise absolute difference as a heatmap — red/yellow means large difference, blue means good alignment
- **diff_split**: top half is overlay, bottom half is diff — best for fine-tuning alignment
- **live**: live feed only, no reference

Recommended workflow:

1. Start in **overlay** mode at 0.5 opacity
2. Roughly align the camera so major scene features overlap
3. Switch to **diff** mode for precise alignment (minimize red/yellow areas)
4. Use **diff_split** for final fine-tuning
5. Press `s` to save a snapshot of the calibrated position for future reference


vision
'''bash
python scripts_real/eval_real_xarm_single.py -i /home/zinan/Documents/zinan/data/titw_ckpts/checkpoints_0502_v/checkpoints/epoch=0030-val_loss_all=0.0208-train_loss=0.0131.ckpt -o /home/zinan/Documents/zinan/data/titw_eval --camera_path /dev/video-hdmi --headless
'''

vision-tac
'''bash
python scripts_real/eval_real_xarm_single.py -i /home/zinan/Documents/zinan/data/titw_ckpts/checkpoints_0501_vt/epoch=0030-train_loss=0.013.ckpt -o /home/zinan/Documents/zinan/data/titw_eval --camera_path /dev/video-hdmi --enable_tactile --headless
'''


vision-sim-tac
'''bash
python scripts_real/eval_real_xarm_single.py -i /home/zinan/Documents/zinan/data/titw_ckpts/checkpoints_0501_vt/epoch=0030-train_loss=0.013.ckpt -o /home/zinan/Documents/zinan/data/titw_eval --camera_path /dev/video-hdmi --tacgen_ckpt_path /home/zinan/Documents/zinan/data/tacgen_ckpts/0502_v16/latest.pth  --headless
'''