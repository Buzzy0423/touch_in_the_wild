# GELLO Data vs Deploy Semantics

This note documents the current semantic mismatch between:

- xArm data collected through the GELLO pipeline
- xArm deploy controllers currently used in this repository

The main goal is to make pose units and gripper meaning explicit before
single-arm xArm deployment.

## Scope

This document focuses on:

- end-effector pose units and conventions
- gripper scalar meaning and scaling
- what is already confirmed from code
- what must be unified before reliable deployment

It does not describe the full deployment pipeline.

## Short Summary

Confirmed from code:

- GELLO xArm data logs end-effector position in meters.
- GELLO xArm data logs orientation in xArm SDK RPY radians.
- GELLO xArm data logs gripper as a normalized scalar where:
  - `0` is approximately open
  - `1` is approximately closed
- The current xArm deploy gripper controller in this repository uses a
  different external convention:
  - larger values mean more open
  - values are interpreted on a roughly `0.0 .. 0.8` scale
- The current xArm deploy pose controller appears to expose Cartesian
  position in millimeters, not meters.

As a result, GELLO-trained xArm checkpoints should not be deployed against the
current xArm controllers without an explicit unit / semantic alignment step.

## GELLO Data Semantics

### Pose

In the GELLO xArm robot implementation:

- `get_position(is_radian=True)` returns SDK pose in millimeters + radians
- the logger converts position from millimeters to meters before recording
- the logger stores:
  - `ee_pos_sdk`: meters
  - `ee_rpy_sdk`: radians

Relevant code:

- `gello_software/gello/robots/xarm_robot.py`
  - `ee_pos_sdk = pose_rpy[:3] / 1000.0`
  - `ee_rpy_sdk = pose_rpy[3:6]`

So the canonical GELLO pose convention is:

- translation: meters
- rotation: radians
- frame: xArm SDK frame

### Gripper

In the GELLO xArm robot implementation:

- SDK gripper open is defined as `800`
- SDK gripper close is defined as `0`
- logged gripper value is normalized as:

```text
gello_gripper = (sdk_gripper - 800) / (0 - 800)
               = 1 - sdk_gripper / 800
```

This means:

- `sdk_gripper = 800` -> `gello_gripper = 0` -> open
- `sdk_gripper = 0` -> `gello_gripper = 1` -> closed

So the canonical GELLO gripper convention is:

- scalar range: approximately `0 .. 1`
- direction:
  - small value = open
  - large value = closed

Relevant code:

- `gello_software/gello/robots/xarm_robot.py`
  - `GRIPPER_OPEN = 800`
  - `GRIPPER_CLOSE = 0`
  - normalized readback in `_get_gripper_pos`

## Training Data Semantics

The current GELLO-to-UMI conversion writes:

- `robot0_eef_pos` from `ee_pos_sdk`
- `robot0_eef_rot_axis_angle` from `ee_rpy_sdk`
- `robot0_gripper_width` directly from logged `gripper_position`

The converter does not invert or rescale the logged gripper into a different
 convention. It preserves the GELLO scalar meaning.

So for current xArm GELLO datasets in this workflow:

- `robot0_eef_pos`: meters
- `robot0_eef_rot_axis_angle`: radians
- `robot0_gripper_width`: GELLO normalized scalar
  - `0` open
  - `1` closed

Relevant code:

- `gello_software/scripts/convert_gello_to_umi_zarr.py`
  - `_load_robot_series`
  - `gripper = np.array([... r["gripper_position"] ...])`
  - episode field `robot0_gripper_width = aligned_gripper`

## Current Deploy Semantics in This Repository

### xArm Pose Controller

The current deploy-side xArm pose controller is:

- `umi/real_world/xarm_interpolation_controller.py`

It reads xArm pose using `get_position_aa(is_radian=True)` and stores
`ActualTCPPose` directly.

There is a commented-out line:

```python
#actual_pose_aa[:3] = actual_pose_aa[:3]/1000
```

This strongly indicates the controller currently exposes Cartesian translation
in millimeters rather than meters.

Implication:

- GELLO / training data pose translation uses meters
- current deploy controller likely exposes millimeters

So deploy code must currently compensate for this mismatch manually, or the
controller must be normalized to meters.

### xArm Gripper Controller

The current deploy-side xArm gripper controller is:

- `umi/real_world/xarm_gripper_controller.py`

It uses:

```text
controller_value = sdk_gripper / 1000
sdk_gripper = controller_value * 1000
```

This means its external convention is approximately:

- `0.0` closed
- `0.8` open

So its direction is opposite to GELLO data:

- GELLO data:
  - `0` open
  - `1` closed
- current deploy controller:
  - small value closed
  - large value open

This is not just a scale mismatch. It is also a direction mismatch.

## Confirmed Mismatch Table

| Quantity | GELLO / training data | Current deploy controller | Status |
|---|---|---|---|
| EE position | meters | likely millimeters | mismatch |
| EE rotation | radians | radians | mostly aligned |
| Gripper scalar direction | `0=open`, `1=closed` | small=closed, large=open | mismatch |
| Gripper scalar scale | about `0..1` | about `0..0.8` | mismatch |

## Practical Deployment Risk

If a checkpoint trained on current GELLO xArm data is deployed directly against
the current xArm controllers:

- end-effector translation may be off by a factor of `1000` unless corrected
- gripper behavior may be inverted
- gripper amplitude may also be wrong even if direction is fixed

This can produce rollout behavior that looks "reasonable" in software while
being physically incorrect on hardware.

## Recommended Canonical Semantics

Suggested canonical interface for xArm deployment:

- end-effector translation: meters
- end-effector rotation: radians / rotvec internally as needed
- gripper scalar:
  - range approximately `0 .. 1`
  - `0 = open`
  - `1 = closed`

Reason:

- this matches GELLO logging
- this matches current GELLO-derived training data
- this minimizes deploy-time special cases

Under that convention, deploy-side conversion back to SDK command should be:

```text
sdk_gripper = 800 * (1 - gripper_value)
```

and deploy-side pose conversion to SDK should be:

```text
sdk_xyz_mm = xyz_m * 1000
```

## What Is Confirmed vs Inferred

Confirmed from code:

- GELLO logs pose in meters and radians.
- GELLO logs gripper with `0=open`, `1=closed`.
- current GELLO-to-UMI conversion preserves that gripper scalar.
- current deploy xArm gripper controller uses a different convention.
- current deploy xArm pose controller likely exposes millimeter translation.

Inference:

- single-arm xArm deploy should standardize controller-facing semantics before
  policy rollout, instead of scattering one-off conversions through eval code.

## Relevant Files

- `gello_software/gello/robots/xarm_robot.py`
- `gello_software/scripts/convert_gello_to_umi_zarr.py`
- `docs/train_xarm_gello_data.md`
- `umi/real_world/xarm_interpolation_controller.py`
- `umi/real_world/xarm_gripper_controller.py`
