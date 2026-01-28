#!/usr/bin/env python3
import sys

from sympy import group
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt
import numpy as np

from load_traj import read_xz_theta_samples_from_txt, xz_build_waypoints_from_aligned_samples, yz_build_waypoints_from_aligned_samples


# =========================
# 可调参数
# =========================
GROUP_NAME = "arm"

# 回零（初始）关节角（按 MoveIt group 的关节顺序）
XZ_HOME_JOINTS = [0.0, 0.866, -0.960, 0.0, 0.182, 1.571]
YZ_HOME_JOINTS = [0.0, 0.866, -0.960, 0.0, 0.182, 0.00]
YZ_HOME_JOINTS_HIGH = [0.0, 0.928, -1.195, 0.0, 0.355, 0.0]
YZ_HOME_JOINTS_HIGHER = [0.0, 1.2125674978230603, -2.1935472571989933, 0.0, 1.0688221806288074, 0.0]

YZ_HOME_JOINTS_Z_630 = [0.0, 1.017247701232375, -1.4614514491574517, 0.0, 0.5315923835724329, 0.0]
ZERO_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# HOME
HOME_VEL_SCALE = 0.4
HOME_ACC_SCALE = 0.4

# TASK
TASK_VEL_SCALE = 0.08
TASK_ACC_SCALE = 0.08

# 额外整体放慢倍数（>1 更慢）
SLOW_DOWN_SCALE = 2.0

# refine is the path to the txt file
TXT_PATH = "t3c1_15.txt"
# TXT_PATH = "t1c4_refine.txt"

# compute_cartesian_path 的插值步长（建议小一点更平滑）
EEF_STEP = 0.01
JUMP_THRESHOLD = 0.0

# ===== 夹爪参数 =====
GRIPPER_GROUP = "gripper"

# 微微打开（根据你夹爪实际调）
GRIPPER_OPEN = [0.01]    # 例：张开一点
GRIPPER_CLOSE = [0.0]   # 合上

GRIPPER_VEL_SCALE = 0.2
GRIPPER_ACC_SCALE = 0.2
# =========================


def slow_down_plan(plan, scale):
    """整体把轨迹放慢 scale 倍（scale>1 更慢）"""
    if scale is None or scale <= 1.0:
        return plan
    jt = plan.joint_trajectory
    for p in jt.points:
        p.time_from_start *= scale
        if p.velocities:
            p.velocities = [v / scale for v in p.velocities]
        if p.accelerations:
            p.accelerations = [a / (scale * scale) for a in p.accelerations]
    return plan


def move_to_home_pose(group):
    """每次执行前回到固定初始关节角，并且慢慢过去"""
    group.set_max_velocity_scaling_factor(HOME_VEL_SCALE)
    group.set_max_acceleration_scaling_factor(HOME_ACC_SCALE)

    group.set_joint_value_target(YZ_HOME_JOINTS_Z_630)

    plan = group.plan()
    # 兼容不同 MoveIt 版本：plan() 可能返回 tuple 或直接返回 plan
    if isinstance(plan, tuple):
        plan = plan[1]

    # plan 为空就直接报错返回
    if not plan or not hasattr(plan, "joint_trajectory") or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Planning to HOME failed. Check joint limits / controllers / group joint order.")
        return False

    # plan = slow_down_plan(plan, SLOW_DOWN_SCALE)

    rospy.loginfo("Going to HOME slowly...")
    group.execute(plan, wait=True)
    group.stop()
    return True

def move_to_zero_pose(group):
    """每次执行前回到固定初始位姿，并且慢慢过去"""
    group.set_max_velocity_scaling_factor(HOME_VEL_SCALE)
    group.set_max_acceleration_scaling_factor(HOME_ACC_SCALE)

    group.set_joint_value_target(ZERO_JOINTS)

    plan = group.plan()
    # 兼容不同 MoveIt 版本：plan() 可能返回 tuple 或直接返回 plan
    if isinstance(plan, tuple):
        plan = plan[1]

    # plan 为空就直接报错返回
    if not plan or not hasattr(plan, "joint_trajectory") or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Planning to ZERO pose failed. Check joint limits / controllers / group joint order.")
        return False

    # plan = slow_down_plan(plan, SLOW_DOWN_SCALE)

    rospy.loginfo("Going to ZERO pose slowly...")
    group.execute(plan, wait=True)
    group.stop()
    return True

def move_gripper(group, joint_values, vel=0.2, acc=0.2):
    group.set_max_velocity_scaling_factor(vel)
    group.set_max_acceleration_scaling_factor(acc)

    group.set_joint_value_target(joint_values)
    plan = group.plan()
    if isinstance(plan, tuple):
        plan = plan[1]

    if not plan or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Gripper planning failed")
        return False

    group.execute(plan, wait=True)
    group.stop()
    return True


def wait_enter_or_quit(prompt):
    rospy.loginfo(prompt)
    cmd = input().strip().lower()
    if cmd == 'q':
        rospy.loginfo("User quit.")
        return False
    return True

def visualize_waypoints_yz(waypoints, title="Waypoints (Y-Z)"):
    """
    可视化 Cartesian waypoints 的 Y-Z 投影
    同时计算相邻路径点的距离统计
    """
    ys = np.array([p.position.y for p in waypoints])
    zs = np.array([p.position.z for p in waypoints])

    # ===== 计算相邻点之间的距离 =====
    if len(ys) >= 2:
        dists = np.sqrt(np.diff(ys)**2 + np.diff(zs)**2)
        mean_dist = np.mean(dists)
        min_dist = np.min(dists)
        max_dist = np.max(dists)
    else:
        dists = np.array([])
        mean_dist = min_dist = max_dist = 0.0

    # ===== 可视化 =====
    plt.figure()
    plt.plot(ys, zs, 'b.-', label='Waypoints')
    plt.scatter(ys[0], zs[0], c='g', s=80, label='Start')
    plt.scatter(ys[-1], zs[-1], c='r', s=80, label='End')

    plt.axis('equal')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.grid(True)
    plt.legend()

    plt.title(
        f"{title}\n"
        f"mean Δ={mean_dist:.5f}, min Δ={min_dist:.5f}, max Δ={max_dist:.5f}"
    )

    plt.show()

    # ===== 同时返回数值，方便你程序里用 =====
    return mean_dist, min_dist, max_dist, dists

def filter_waypoints_yz(waypoints, min_dist=1e-3):
    """
    只在 Y-Z 平面过滤 waypoint：
    - 相邻点距离 < min_dist 的会被删除
    - 不生成新点
    - orientation 完全不修改，随 waypoint 原样保留
    """
    if len(waypoints) < 2:
        return waypoints

    filtered = [waypoints[0]]

    for p in waypoints[1:]:
        last = filtered[-1]

        dy = p.position.y - last.position.y
        dz = p.position.z - last.position.z

        # print('ori:', p.orientation)

        if (dy * dy + dz * dz) ** 0.5 >= min_dist:
            filtered.append(p)   # 原 waypoint，orientation 不动

    return filtered

def check_waypoints_ik(group, waypoints):
    """
    用 MoveGroupCommander 本身来检查每个 waypoint 是否有 IK 解
    返回：
      valid_idx   : 有解的 waypoint index
      invalid_idx : 无解的 waypoint index
    """
    valid = []
    invalid = []

    # 先存一下当前状态，避免污染
    group.clear_pose_targets()

    for i, p in enumerate(waypoints):
        group.set_pose_target(p)
        plan = group.plan()

        # 兼容不同 MoveIt 版本
        if isinstance(plan, tuple):
            plan = plan[1]

        if plan and hasattr(plan, "joint_trajectory") and len(plan.joint_trajectory.points) > 0:
            valid.append(i)
        else:
            invalid.append(i)

        group.clear_pose_targets()

    return valid, invalid

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("piper_cartesian_from_txt", anonymous=True)

    group = moveit_commander.MoveGroupCommander(GROUP_NAME)
    group.set_max_velocity_scaling_factor(HOME_VEL_SCALE)
    group.set_max_acceleration_scaling_factor(HOME_ACC_SCALE)

    gripper_group = moveit_commander.MoveGroupCommander(GRIPPER_GROUP)

    # ===== 功能 1：每次执行前回 HOME（慢慢过去） =====
    # if not move_to_home_pose(group):
    #     return
    # if not move_to_zero_pose(group):
    #     return

    # 微微打开夹爪
    rospy.loginfo("Opening gripper slightly...")
    move_gripper(
        gripper_group,
        GRIPPER_OPEN,
        vel=GRIPPER_VEL_SCALE,
        acc=GRIPPER_ACC_SCALE
    )

    # 等待人工确认
    if not wait_enter_or_quit("Press ENTER to close gripper, 'q'+ENTER to quit."):
        return

    # 合上夹爪
    rospy.loginfo("Closing gripper...")
    move_gripper(
        gripper_group,
        GRIPPER_CLOSE,
        vel=GRIPPER_VEL_SCALE,
        acc=GRIPPER_ACC_SCALE
    )

    # 在回到 HOME 后，再取一次当前 pose 作为“姿态锁定”的参考
    start_pose = group.get_current_pose().pose
    rospy.loginfo("Start pose after HOME: %s", str(start_pose))

    # ===== 功能 2：从 txt 读轨迹点 =====
    samples = read_xz_theta_samples_from_txt(TXT_PATH)
    if not samples:
        rospy.logerr("No valid samples loaded from txt: %s", TXT_PATH)
        return

    waypoints = yz_build_waypoints_from_aligned_samples(samples, start_pose)

    # mean_d, min_d, max_d, dists = visualize_waypoints_yz(waypoints)

    # print("Average spacing:", mean_d)
    # print("Min spacing:", min_d)
    # print("Max spacing:", max_d)

    # exit()

    waypoints = filter_waypoints_yz(waypoints, min_dist=1e-3)

    # mean_d, min_d, max_d, dists = visualize_waypoints_yz(waypoints)

    # print("Average spacing:", mean_d)
    # print("Min spacing:", min_d)
    # print("Max spacing:", max_d)

    # # exit()
    # print("Before IK check, total waypoints:", len(waypoints))
    # # robot = moveit_commander.RobotCommander()
    # valid_idx, invalid_idx = check_waypoints_ik(group, waypoints)
    # waypoints = [waypoints[i] for i in valid_idx]
    # print("After IK check, valid waypoints:", len(waypoints))
    # print('percentage of valid waypoints:', len(waypoints) / (len(valid_idx) + len(invalid_idx)) * 100)

    (plan, fraction) = group.compute_cartesian_path(
        waypoints,
        EEF_STEP,
        JUMP_THRESHOLD
    )

    rospy.loginfo("Cartesian path fraction: %.3f", fraction)
    if fraction < 0.5:
        rospy.logwarn("Path not fully achievable (fraction<0.99). Try smaller EEF_STEP or check collisions.")
        return

    # # 再额外整体放慢
    # plan = slow_down_plan(plan, SLOW_DOWN_SCALE)
    # 规划完 plan 后：
    plan = group.retime_trajectory(
        group.get_current_state(),
        plan,
        velocity_scaling_factor=TASK_VEL_SCALE,
        acceleration_scaling_factor=TASK_ACC_SCALE,
        algorithm="time_optimal_trajectory_generation"  # 有的版本可用
)

    # ===== 轨迹已规划，RViz 中可见 =====
    rospy.loginfo("Trajectory planned and visualized in RViz.")
    rospy.loginfo("Press ENTER to execute, or type 'q' then ENTER to quit.")

    user_input = input().strip().lower()

    if user_input == 'q':
        rospy.loginfo("Quit without executing.")
        return

    # 只有按 ENTER（空输入）才会继续执行
    rospy.loginfo("Executing trajectory...")
    group.execute(plan, wait=True)
    group.stop()

    # rospy.loginfo("Executing cartesian trajectory from txt slowly...")
    # group.execute(plan, wait=True)
    # group.stop()


if __name__ == "__main__":
    main()
