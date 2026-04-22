# Plan: Single-Arm xArm Deploy in `touch_in_the_wild`

This document captures the current implementation plan for deploying a
single-arm xArm policy in `touch_in_the_wild`, using GELLO-style training data
semantics.

The immediate goal is:

- deploy a single-arm xArm checkpoint using the `touch_in_the_wild` stack
- avoid mixing this path with the current bimanual+tactile deploy script
- keep GELLO-specific data handling isolated and explicit

## Short Answer

Yes:

- GELLO-related data handling should be split into one or a few dedicated modules
- a new deploy script under `scripts_real/` should be created for single-arm xArm

This is preferable to continuing to patch the current root-level `eval_real.py`,
which is bimanual, tactile-specific, and currently contains leftover logic that
is not a clean base for single-arm xArm deployment.

## Why Separate GELLO Handling

GELLO data introduces a few conventions that are not generic real-robot deploy
assumptions:

- end-effector pose is expected in the xArm SDK frame
- position is in meters at the training-data level
- gripper uses GELLO-normalized semantics:
  - `0 = open`
  - `1 = closed`

These are not properties that should be scattered through deploy code as
ad-hoc conversions.

Separating GELLO handling into dedicated modules has three benefits:

- makes all xArm/GELLO assumptions explicit
- keeps `UmiEnv` and deploy logic easier to reason about
- makes future tactile support easier to add without duplicating conversions

## Recommended Structure

### 1. New Single-Arm Deploy Script

Create a new script under `scripts_real/`, for example:

- `scripts_real/eval_real_xarm_single.py`

This script should become the dedicated entrypoint for:

- single-arm xArm
- GELLO-trained checkpoints
- vision-only first
- optional tactile later

It should not inherit the current human-control + SpaceMouse flow by default.

Recommended behavior for the first version:

- load checkpoint
- initialize xArm single-arm environment
- warm up policy
- optionally move to a configured initial pose
- wait for a keyboard start command or `--auto_start`
- run policy rollout
- support stop / exit keys

### 2. GELLO-Specific Utility Module(s)

Recommended new module(s) under a reusable location such as:

- `umi/real_world/xarm_gello_util.py`
- optionally `umi/real_world/xarm_gello_obs.py`

These modules should centralize:

- pose unit conversion
- pose frame conversion if needed
- gripper conversion between:
  - xArm SDK raw value
  - GELLO-normalized value
- validation helpers for expected ranges

The first module can be minimal. It does not need to be over-engineered.

Example responsibilities:

- `gripper_raw_to_gello_norm(raw)`
- `gripper_gello_norm_to_raw(norm)`
- `xarm_pose_mm_to_m(pose)`
- `xarm_pose_m_to_mm(pose)`
- simple range checks for deploy safety

### 3. Single-Arm xArm Support in `UmiEnv`

The single-arm path should use `UmiEnv`, not the current bimanual env.

`UmiEnv` needs to expose observations and actions in the same semantic space as
the training data:

- `robot0_eef_pos`: meters
- `robot0_eef_rot_axis_angle`: radians
- `robot0_gripper_width`: GELLO-normalized scalar

This means xArm support in `UmiEnv` should be updated so that:

- xArm robot state returned to deploy code matches the training convention
- xArm gripper state returned to deploy code matches GELLO convention
- `exec_actions` accepts the same semantic convention

## Recommended Implementation Phases

### Phase 1: Clean Entry Point

Create the new script:

- `scripts_real/eval_real_xarm_single.py`

Base it on:

- `scripts_real/eval_real_umi.py`

Do not base it on:

- root `eval_real.py`

Reason:

- `eval_real_umi.py` is already single-arm
- root `eval_real.py` is bimanual+tactile specific

### Phase 2: Normalize xArm Semantics

Update xArm controller-facing semantics so deploy code sees:

- pose translation in meters
- rotation in radians
- gripper in GELLO-normalized space

This should be done in one place, not in scattered deploy-script patches.

Preferred location:

- xArm controller wrappers and/or GELLO utility module

### Phase 3: Connect xArm to `UmiEnv`

Extend the single-arm `UmiEnv` path so it can use:

- `XArmInterpolationController`
- `XArmGripperController`

and still expose the same observation/action contract expected by:

- `get_real_umi_obs_dict`
- `get_real_umi_action`
- the checkpoint `shape_meta`

### Phase 4: Remove SpaceMouse Requirement

The new single-arm xArm script should not depend on SpaceMouse.

Replace the current human-control loop with a minimal workflow:

- optional init pose
- keyboard start
- rollout loop
- keyboard stop

This should be the default path.

### Phase 5: Dry-Run Validation

Before any real motion, add a dry-run mode that checks:

- observation keys match checkpoint expectations
- pose units are correct
- gripper direction is correct
- decoded action ranges are reasonable
- inference runs end-to-end without shape mismatch

### Phase 6: Add Tactile Later

Do not mix tactile into the first xArm single-arm deploy patch.

After vision-only deploy is stable:

- add tactile controller support to the single-arm env
- add timestamp-aligned tactile obs handling
- extend the new xArm deploy script to accept tactile-enabled checkpoints

## Proposed File-Level Work

### New files

- `docs/plan_single_arm_xarm_deploy.md`
- `scripts_real/eval_real_xarm_single.py`
- `umi/real_world/xarm_gello_util.py`

### Existing files likely to change

- `umi/real_world/umi_env.py`
- `umi/real_world/xarm_interpolation_controller.py`
- `umi/real_world/xarm_gripper_controller.py`
- optionally `umi/real_world/real_inference_util.py` if a small xArm-specific
  helper is needed

## Design Rules

To keep this path maintainable:

- keep GELLO-specific conversions in dedicated helpers
- keep the new deploy script single-arm specific
- do not expand the first patch to include tactile unless required
- do not rely on root `eval_real.py` as the base implementation
- prefer one clean semantic interface over repeated deploy-time fixes

## Immediate Next Steps

Recommended next implementation order:

1. Create `umi/real_world/xarm_gello_util.py`
2. Fix xArm pose/gripper external semantics to match GELLO-trained data
3. Extend `UmiEnv` single-arm path for xArm
4. Add `scripts_real/eval_real_xarm_single.py`
5. Add dry-run checks
6. Test vision-only deploy
7. Add tactile support later

## Non-Goals for the First Patch

The following should not be included in the first implementation unless they
block deploy:

- tactile fusion
- bimanual support
- broad refactors of replay buffer or training code
- generalized multi-robot abstractions beyond what xArm single-arm needs

