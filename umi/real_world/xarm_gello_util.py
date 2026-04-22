import numpy as np


XARM_SDK_GRIPPER_OPEN = 800.0
XARM_SDK_GRIPPER_CLOSE = 0.0
XARM7_GELLO_START_JOINTS = np.array(
    [0.0, -0.6981, 0.0, 0.8727, 0.0, 1.5708, 0.0],
    dtype=np.float64,
)
XARM7_GELLO_START_GRIPPER = 0.0


def xarm_pose_mm_to_m(pose):
    pose = np.array(pose, dtype=np.float64, copy=True)
    pose[..., :3] /= 1000.0
    return pose


def xarm_pose_m_to_mm(pose):
    pose = np.array(pose, dtype=np.float64, copy=True)
    pose[..., :3] *= 1000.0
    return pose


def gripper_raw_to_gello_norm(raw):
    raw = np.asarray(raw, dtype=np.float64)
    return (raw - XARM_SDK_GRIPPER_OPEN) / (
        XARM_SDK_GRIPPER_CLOSE - XARM_SDK_GRIPPER_OPEN
    )


def gripper_gello_norm_to_raw(norm):
    norm = np.asarray(norm, dtype=np.float64)
    return XARM_SDK_GRIPPER_OPEN + norm * (
        XARM_SDK_GRIPPER_CLOSE - XARM_SDK_GRIPPER_OPEN
    )
