#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuJoCo 四足机器人：V9 稳定同步起立控制器
=========================================

本版本针对前几版的主要问题，改成“纯关节空间同步站立”：

1. 按 6 时绝不做 IK / 大规模 mj_forward
   - 站立目标在 viewer 启动前只计算一次。
   - 按 6 只切换状态，因此不会再出现点击后 MuJoCo 长时间卡顿。

2. 四条腿共用同一个同步时间相位
   - 前左 / 前右 / 后右 / 后左同时执行站立动作。
   - 不再使用足端 XY 拉回控制，不人为把脚往里/往前拖。
   - 不再使用上一版强 J^T F 六维支撑控制，避免后腿突然腾空和侧翻。

3. hip 外展/内收角固定到 0 rad
   - 让整条腿的主要运动平面保持竖直。
   - 左右、前后使用严格对称的站立目标。

4. 大腿 / 小腿使用接近直角的站立目标
   - 目标姿态接近“大小腿垂直”。
   - 站立高度由该机械姿态对应的脚端高度自动计算。

5. 电机控制
   - 站立：高刚度 PD + MuJoCo bias force（重力/科氏补偿）。
   - 不增加额外水平足端力，因此尽量避免脚在地面上打滑。
   - 增加 torque rate limit，避免电机力矩瞬间跳变。

6. 阻尼模式
   - 默认状态为低阻尼模式，让机器人启动后快速趴下。
   - 按 7 返回阻尼模式。

按键：
    6 = 连续、同步、平稳站立
    7 = 低阻尼模式

U / I 不使用，避免和 MuJoCo Viewer 的默认快捷键冲突。

运行：
    python dog_stand_controller.py "black_description.xml"

依赖：
    pip install mujoco numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# 文件 / 按键
# ============================================================

DEFAULT_XML = "black_description.xml"

STAND_KEY = "6"
DAMPING_KEY = "7"


# ============================================================
# 站立参数
# ============================================================

# 站立动作持续时间。
# 4 秒可以保证从趴卧状态平滑进入目标姿态。
STAND_TIME = 4.0

# 足端球半径约 0.02 m；球心保持约 22 mm 高于地面。
GROUND_FOOT_Z = 0.022


# ============================================================
# 站立目标姿态
# ============================================================
#
# 这里不再做运行时 IK，也不再追踪“趴下时脚的位置”。
#
# 核心要求：
#   hip = 0 -> 腿的运动平面尽量保持竖直
#   thigh / calf -> 接近“大小腿垂直”的站姿
#
# 四条腿严格镜像。
MANUAL_STAND_POSE = {
    "FL_hip_joint": 0.00,
    "FL_thigh_joint": 0.60,
    "FL_calf_joint": -1.20,

    "FR_hip_joint": 0.00,
    "FR_thigh_joint": -0.60,
    "FR_calf_joint": 1.20,

    "RR_hip_joint": 0.00,
    "RR_thigh_joint": -0.60,
    "RR_calf_joint": 1.20,

    "RL_hip_joint": 0.00,
    "RL_thigh_joint": 0.60,
    "RL_calf_joint": -1.20,
}


# ============================================================
# 起立轨迹
# ============================================================

# hip 先回到 0，再开始主要的大小腿伸展。
# 这样在腿还比较低的时候，就先把腿平面调整到竖直，
# 后面的主要抬身动作不会继续依靠 hip 把腿向内/向外甩。
HIP_SETTLE_END = 0.25

# 大腿 / 小腿从 8% 开始进入主要站立轨迹。
LEG_MOTION_START = 0.08


# ============================================================
# 电机 PD
# ============================================================

# XML 中 motor ctrlrange = [-20, 20] Nm。
#
# 这一版明显提高刚度，解决上一版：
#   hip_err ≈ 0.27 rad
#   rear_tau 逐渐下降
# 的问题。
KP_HIP = 72.0
KP_THIGH = 125.0
KP_CALF = 105.0

KD_HIP = 4.8
KD_THIGH = 7.2
KD_CALF = 5.8

# 不再给前后腿设置很大的额外比例，
# 四条腿尽可能使用同一套动力学强度。
FRONT_SCALE = 1.0
REAR_SCALE = 1.0

# 默认低阻尼：让机器人启动后较快趴下。
DAMPING_HIP = 0.10
DAMPING_THIGH = 0.16
DAMPING_CALF = 0.13

# 扭矩变化率限制。
MOTOR_TORQUE_RATE = 1400.0


# ============================================================
# 最终站姿辅助
# ============================================================

# 站立后只保留很小的速度阻尼，不再额外施加足端水平力。
# 这样避免靠“拉脚”来稳定身体。
STAND_EXTRA_VEL_DAMP = 0.15

# 当站立目标已经接近完成时，提高少量速度阻尼，减小抖动。
FINAL_DAMPING_BLEND_START = 0.72

REALTIME_SLEEP = True


# ============================================================
# 数学工具
# ============================================================

def quintic_scurve(x: float) -> float:
    """五次 S 曲线：起点/终点速度、加速度均为 0。"""
    x = float(np.clip(x, 0.0, 1.0))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(
        mat,
        np.asarray(quat_wxyz, dtype=np.float64),
    )
    return mat.reshape(3, 3)


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    R = quat_to_mat(quat_wxyz)
    return float(
        np.arctan2(
            R[1, 0],
            R[0, 0],
        )
    )


def yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array(
        [
            np.cos(half),
            0.0,
            0.0,
            np.sin(half),
        ],
        dtype=np.float64,
    )


def upright_orientation_error(
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """
    机身 z 轴与世界 z 轴之间的叉积。
    仅用于状态显示，不直接施加大姿态外力。
    """
    R = quat_to_mat(quat_wxyz)

    body_z = R[:, 2]
    world_z = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    err = np.cross(
        body_z,
        world_z,
    )

    err[2] = 0.0

    return err


# ============================================================
# 模型映射
# ============================================================

LEG_ORDER = (
    "FL",
    "FR",
    "RR",
    "RL",
)

JOINT_SUFFIX = (
    "hip",
    "thigh",
    "calf",
)


def find_joint(
    model: mujoco.MjModel,
    name: str,
) -> int:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"XML 中找不到关节：{name}"
        )

    if (
        model.jnt_type[jid]
        != mujoco.mjtJoint.mjJNT_HINGE
    ):
        raise RuntimeError(
            f"{name} 不是 hinge joint。"
        )

    return jid


def build_joint_info(
    model: mujoco.MjModel,
):

    infos = []

    for leg in LEG_ORDER:
        for suffix in JOINT_SUFFIX:

            joint_name = (
                f"{leg}_{suffix}_joint"
            )

            jid = find_joint(
                model,
                joint_name,
            )

            actuator_name = (
                f"{joint_name}_motor"
            )

            actuator_id = (
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator_name,
                )
            )

            if actuator_id < 0:
                raise RuntimeError(
                    f"XML 中找不到执行器："
                    f"{actuator_name}"
                )

            if suffix == "hip":
                kp = KP_HIP
                kd = KD_HIP
                damp = DAMPING_HIP

            elif suffix == "thigh":
                kp = KP_THIGH
                kd = KD_THIGH
                damp = DAMPING_THIGH

            else:
                kp = KP_CALF
                kd = KD_CALF
                damp = DAMPING_CALF

            if leg in ("FL", "FR"):
                scale = FRONT_SCALE
            else:
                scale = REAR_SCALE

            kp *= scale
            kd *= scale

            if joint_name not in MANUAL_STAND_POSE:
                raise RuntimeError(
                    f"站立目标缺少："
                    f"{joint_name}"
                )

            q_lo = float(
                model.jnt_range[
                    jid,
                    0,
                ]
            )

            q_hi = float(
                model.jnt_range[
                    jid,
                    1,
                ]
            )

            ctrl_lo = float(
                model.actuator_ctrlrange[
                    actuator_id,
                    0,
                ]
            )

            ctrl_hi = float(
                model.actuator_ctrlrange[
                    actuator_id,
                    1,
                ]
            )

            q_target = float(
                np.clip(
                    MANUAL_STAND_POSE[
                        joint_name
                    ],
                    q_lo,
                    q_hi,
                )
            )

            infos.append(
                {
                    "leg": leg,
                    "suffix": suffix,
                    "joint_name": joint_name,
                    "joint_id": jid,
                    "qpos_adr": int(
                        model.jnt_qposadr[jid]
                    ),
                    "dof_adr": int(
                        model.jnt_dofadr[jid]
                    ),
                    "actuator_id": actuator_id,
                    "q_lo": q_lo,
                    "q_hi": q_hi,
                    "ctrl_lo": ctrl_lo,
                    "ctrl_hi": ctrl_hi,
                    "q_target": q_target,
                    "kp": kp,
                    "kd": kd,
                    "damping": damp,
                }
            )

    return infos


def build_foot_info(
    model: mujoco.MjModel,
):

    feet = []

    for leg in LEG_ORDER:

        body_name = (
            f"{leg}_foot"
        )

        bid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if bid < 0:
            raise RuntimeError(
                f"XML 中找不到足端："
                f"{body_name}"
            )

        feet.append(
            {
                "leg": leg,
                "body_name": body_name,
                "body_id": bid,
            }
        )

    return feet


# ============================================================
# 启动时计算“纯几何站姿”
# ============================================================

def find_freejoint(
    model: mujoco.MjModel,
) -> tuple[int, int]:

    for jid in range(model.njnt):

        if (
            model.jnt_type[jid]
            == mujoco.mjtJoint.mjJNT_FREE
        ):

            return (
                int(
                    model.jnt_qposadr[jid]
                ),
                int(
                    model.jnt_dofadr[jid]
                ),
            )

    raise RuntimeError(
        "模型中没有 freejoint。"
    )


def prepare_stand_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints,
    feet,
):
    """
    只做一次几何计算。

    给出固定的四足机械站姿，然后把 trunk 上下移动，
    让四个脚球心平均落在地面上方 GROUND_FOOT_Z。

    因为最终目标角度本身是对称的，所以不需要运行时 IK。
    """

    free_qpos_adr, _ = find_freejoint(
        model
    )

    base_x = float(
        data.qpos[
            free_qpos_adr
        ]
    )

    base_y = float(
        data.qpos[
            free_qpos_adr + 1
        ]
    )

    base_quat = np.asarray(
        data.qpos[
            free_qpos_adr + 3:
            free_qpos_adr + 7
        ],
        dtype=np.float64,
    )

    yaw = yaw_from_quat(
        base_quat
    )

    planner = mujoco.MjData(model)

    planner.qpos[:] = data.qpos
    planner.qvel[:] = 0.0

    # 先在任意参考高度设置最终关节角。
    trial_z = 0.50

    planner.qpos[
        free_qpos_adr:
        free_qpos_adr + 3
    ] = np.array(
        [
            base_x,
            base_y,
            trial_z,
        ],
        dtype=np.float64,
    )

    # 保持当前 yaw，roll / pitch 设为 0。
    planner.qpos[
        free_qpos_adr + 3:
        free_qpos_adr + 7
    ] = yaw_quat(
        yaw
    )

    for item in joints:

        planner.qpos[
            item["qpos_adr"]
        ] = item["q_target"]

    mujoco.mj_forward(
        model,
        planner
    )

    foot_z = np.array(
        [
            planner.xpos[
                foot["body_id"],
                2,
            ]
            for foot in feet
        ],
        dtype=np.float64,
    )

    # 整体平移 trunk，使平均足端刚好离地。
    body_z = (
        trial_z
        + GROUND_FOOT_Z
        - float(
            np.mean(foot_z)
        )
    )

    # 不允许出现不合理的过低值。
    body_z = float(
        max(
            body_z,
            0.40,
        )
    )

    planner.qpos[
        free_qpos_adr:
        free_qpos_adr + 3
    ] = np.array(
        [
            base_x,
            base_y,
            body_z,
        ],
        dtype=np.float64,
    )

    mujoco.mj_forward(
        model,
        planner
    )

    target_feet = np.array(
        [
            planner.xpos[
                foot["body_id"]
            ].copy()
            for foot in feet
        ],
        dtype=np.float64,
    )

    target_q = np.array(
        [
            item["q_target"]
            for item in joints
        ],
        dtype=np.float64,
    )

    foot_height_error = float(
        np.mean(
            np.abs(
                target_feet[:, 2]
                - GROUND_FOOT_Z
            )
        )
    )

    front_span = float(
        np.linalg.norm(
            target_feet[0, :2]
            - target_feet[1, :2]
        )
    )

    rear_span = float(
        np.linalg.norm(
            target_feet[2, :2]
            - target_feet[3, :2]
        )
    )

    return (
        float(body_z),
        target_q,
        foot_height_error,
        target_feet,
        front_span,
        rear_span,
    )


# ============================================================
# 控制器
# ============================================================

class DogController:

    DAMPING = 0
    STANDING = 1

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ):

        self.model = model
        self.data = data

        self.joints = build_joint_info(
            model
        )

        self.feet = build_foot_info(
            model
        )

        self.nj = len(
            self.joints
        )

        if self.nj != 12:
            raise RuntimeError(
                f"预期 12 个腿部关节，"
                f"实际为 {self.nj}"
            )

        (
            self.free_qpos_adr,
            self.free_dof_adr,
        ) = find_freejoint(
            model
        )

        (
            self.body_target_z,
            self.stand_target_q,
            self.foot_height_error,
            self.target_feet,
            self.target_front_span,
            self.target_rear_span,
        ) = prepare_stand_target(
            model,
            data,
            self.joints,
            self.feet,
        )

        self.mode = (
            self.DAMPING
        )

        self.stand_start_q = (
            self.read_q()
        )

        self.stand_start_time = 0.0

        self.torque_prev = np.zeros(
            self.nj,
            dtype=np.float64,
        )

        self.last_print_wall = (
            time.perf_counter()
        )

        self.initial_base_z = float(
            self.read_base_pos()[2]
        )

        self.initial_base_quat = (
            self.read_base_quat()
        )

    # --------------------------------------------------------
    # state
    # --------------------------------------------------------

    def read_q(self):
        return np.array(
            [
                self.data.qpos[
                    item["qpos_adr"]
                ]
                for item in self.joints
            ],
            dtype=np.float64,
        )

    def read_dq(self):
        return np.array(
            [
                self.data.qvel[
                    item["dof_adr"]
                ]
                for item in self.joints
            ],
            dtype=np.float64,
        )

    def read_base_pos(self):
        return np.asarray(
            self.data.qpos[
                self.free_qpos_adr:
                self.free_qpos_adr + 3
            ],
            dtype=np.float64,
        ).copy()

    def read_base_quat(self):
        return np.asarray(
            self.data.qpos[
                self.free_qpos_adr + 3:
                self.free_qpos_adr + 7
            ],
            dtype=np.float64,
        ).copy()

    def read_base_linvel(self):
        return np.asarray(
            self.data.qvel[
                self.free_dof_adr:
                self.free_dof_adr + 3
            ],
            dtype=np.float64,
        ).copy()

    def read_base_angvel(self):
        return np.asarray(
            self.data.qvel[
                self.free_dof_adr + 3:
                self.free_dof_adr + 6
            ],
            dtype=np.float64,
        ).copy()

    def read_foot_positions(self):
        feet = np.zeros(
            (4, 3),
            dtype=np.float64,
        )

        for i, foot in enumerate(
            self.feet
        ):
            feet[i] = self.data.xpos[
                foot["body_id"]
            ]

        return feet

    # --------------------------------------------------------
    # keyboard
    # --------------------------------------------------------

    def enter_standing(self):
        """
        只做常量/状态赋值。
        不执行 IK，不调用大规模 mj_forward。
        """

        self.stand_start_q[:] = (
            self.read_q()
        )

        self.stand_start_time = (
            float(self.data.time)
        )

        self.mode = (
            self.STANDING
        )

        for i, item in enumerate(
            self.joints
        ):
            self.torque_prev[i] = (
                self.data.ctrl[
                    item["actuator_id"]
                ]
            )

        pass

    def enter_damping(self):
        self.mode = (
            self.DAMPING
        )

        pass

    def keyboard_callback(
        self,
        keycode: int,
    ):
        try:
            key = chr(
                keycode
            )
        except (
            ValueError,
            TypeError,
        ):
            return

        key = key.lower()

        if key == STAND_KEY:
            self.enter_standing()

        elif key == DAMPING_KEY:
            self.enter_damping()

    # --------------------------------------------------------
    # synchronized trajectory
    # --------------------------------------------------------

    def stand_phase(self):
        elapsed = max(
            0.0,
            float(
                self.data.time
            )
            - self.stand_start_time,
        )

        return float(
            np.clip(
                elapsed / STAND_TIME,
                0.0,
                1.0,
            )
        )

    def desired_q(self):
        phase = self.stand_phase()

        # ----------------------------------------------------
        # Phase A: 先让 hip 同步回 0，建立竖直腿平面
        # ----------------------------------------------------
        hip_phase = quintic_scurve(
            min(
                1.0,
                phase / HIP_SETTLE_END,
            )
        )

        # ----------------------------------------------------
        # Phase B: 四条腿同时完成 thigh/calf 主站立运动
        # ----------------------------------------------------
        if phase <= LEG_MOTION_START:
            leg_phase = 0.0
        else:
            leg_phase = quintic_scurve(
                (
                    phase
                    - LEG_MOTION_START
                )
                / (
                    1.0
                    - LEG_MOTION_START
                )
            )

        q_des = self.stand_start_q.copy()

        for i, item in enumerate(
            self.joints
        ):

            if item["suffix"] == "hip":
                s = hip_phase
            else:
                s = leg_phase

            q_des[i] = (
                self.stand_start_q[i]
                + s
                * (
                    self.stand_target_q[i]
                    - self.stand_start_q[i]
                )
            )

        return q_des

    # --------------------------------------------------------
    # control
    # --------------------------------------------------------

    def apply_torque_rate_limit(
        self,
        torque,
    ):
        dt = max(
            float(
                self.model.opt.timestep
            ),
            1e-6,
        )

        max_delta = (
            MOTOR_TORQUE_RATE
            * dt
        )

        delta = (
            torque
            - self.torque_prev
        )

        limited = (
            self.torque_prev
            + np.clip(
                delta,
                -max_delta,
                max_delta,
            )
        )

        self.torque_prev[:] = (
            limited
        )

        return limited

    def compute_control(self):

        q = self.read_q()
        dq = self.read_dq()

        torque = np.zeros(
            self.nj,
            dtype=np.float64,
        )

        if self.mode == self.DAMPING:

            for i, item in enumerate(
                self.joints
            ):

                torque[i] = (
                    -item["damping"]
                    * dq[i]
                )

        else:

            q_des = self.desired_q()
            phase = self.stand_phase()

            # 最终阶段增加一点速度阻尼，
            # 但不增加水平足端力。
            final_blend = quintic_scurve(
                max(
                    0.0,
                    (
                        phase
                        - FINAL_DAMPING_BLEND_START
                    )
                    / (
                        1.0
                        - FINAL_DAMPING_BLEND_START
                    ),
                )
            )

            for i, item in enumerate(
                self.joints
            ):

                pos_err = (
                    q_des[i]
                    - q[i]
                )

                kd = (
                    item["kd"]
                    + STAND_EXTRA_VEL_DAMP
                    * final_blend
                )

                torque[i] = (
                    item["kp"]
                    * pos_err
                    - kd
                    * dq[i]
                )

                # 标准 MuJoCo bias compensation。
                # 不给后腿额外放大，避免后腿突然弹起。
                torque[i] += float(
                    self.data.qfrc_bias[
                        item["dof_adr"]
                    ]
                )

        torque = (
            self.apply_torque_rate_limit(
                torque
            )
        )

        for i, item in enumerate(
            self.joints
        ):

            self.data.ctrl[
                item["actuator_id"]
            ] = float(
                np.clip(
                    torque[i],
                    item["ctrl_lo"],
                    item["ctrl_hi"],
                )
            )

    # --------------------------------------------------------
    # status
    # --------------------------------------------------------

    def print_status(self):

        if self.mode != self.STANDING:
            return

        phase = self.stand_phase()

        q = self.read_q()
        dq = self.read_dq()

        base_pos = (
            self.read_base_pos()
        )

        base_quat = (
            self.read_base_quat()
        )

        torques = np.array(
            [
                self.data.ctrl[
                    item["actuator_id"]
                ]
                for item in self.joints
            ],
            dtype=np.float64,
        )

        front_ids = [
            i
            for i, item in enumerate(
                self.joints
            )
            if item["leg"] in (
                "FL",
                "FR",
            )
        ]

        rear_ids = [
            i
            for i, item in enumerate(
                self.joints
            )
            if item["leg"] in (
                "RR",
                "RL",
            )
        ]

        hip_ids = [
            i
            for i, item in enumerate(
                self.joints
            )
            if item["suffix"] == "hip"
        ]

        front_tau = float(
            np.mean(
                np.abs(
                    torques[
                        front_ids
                    ]
                )
            )
        )

        rear_tau = float(
            np.mean(
                np.abs(
                    torques[
                        rear_ids
                    ]
                )
            )
        )

        hip_err = float(
            np.max(
                np.abs(
                    q[hip_ids]
                    - np.array(
                        [
                            self.stand_target_q[i]
                            for i in hip_ids
                        ],
                        dtype=np.float64,
                    )
                )
            )
        )

        max_dq = float(
            np.max(
                np.abs(dq)
            )
        )

        level_err = float(
            np.linalg.norm(
                upright_orientation_error(
                    base_quat
                )
            )
        )

        feet = (
            self.read_foot_positions()
        )

        front_span = float(
            np.linalg.norm(
                feet[0, :2]
                - feet[1, :2]
            )
        )

        rear_span = float(
            np.linalg.norm(
                feet[2, :2]
                - feet[3, :2]
            )
        )

        foot_z_mean = float(
            np.mean(
                feet[:, 2]
            )
        )

        foot_z_span = float(
            np.max(
                feet[:, 2]
            )
            - np.min(
                feet[:, 2]
            )
        )

        max_xy_drift = float(
            np.max(
                np.linalg.norm(
                    feet[:, :2]
                    - self.target_feet[:, :2],
                    axis=1,
                )
            )
        )

        sat = int(
            np.count_nonzero(
                np.abs(torques)
                > 19.5
            )
        )

        pass


# ============================================================
# 主程序
# ============================================================

def print_model_summary(
    model: mujoco.MjModel,
    controller: DogController,
    xml_path: Path,
):

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass

    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MuJoCo 四足机器人 "
            "V9 同步站立控制器"
        )
    )

    parser.add_argument(
        "xml",
        nargs="?",
        default=DEFAULT_XML,
        help=(
            f"MuJoCo XML 路径，"
            f"默认：{DEFAULT_XML}"
        ),
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    xml_path = (
        Path(args.xml)
        .expanduser()
        .resolve()
    )

    if not xml_path.exists():

        pass

        return 1

    try:

        model = (
            mujoco.MjModel
            .from_xml_path(
                str(xml_path)
            )
        )

        data = mujoco.MjData(
            model
        )

    except Exception as exc:

        pass

        pass

        return 2

    try:

        controller = DogController(
            model,
            data,
        )

    except Exception as exc:

        pass

        pass

        return 3

    print_model_summary(
        model,
        controller,
        xml_path,
    )

    pass

    try:

        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=(
                controller.keyboard_callback
            ),
        ) as viewer:

            while viewer.is_running():

                loop_start = (
                    time.perf_counter()
                )

                controller.compute_control()

                mujoco.mj_step(
                    model,
                    data,
                )

                viewer.sync()

                now = (
                    time.perf_counter()
                )

                if (
                    now
                    - controller.last_print_wall
                    >= 1.0
                ):

                    controller.print_status()

                    controller.last_print_wall = (
                        now
                    )

                if REALTIME_SLEEP:

                    elapsed = (
                        time.perf_counter()
                        - loop_start
                    )

                    remaining = (
                        float(
                            model.opt.timestep
                        )
                        - elapsed
                    )

                    if remaining > 0.0:

                        time.sleep(
                            remaining
                        )

    except KeyboardInterrupt:

        pass

        return 0

    except Exception as exc:

        pass

        return 4

    pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
