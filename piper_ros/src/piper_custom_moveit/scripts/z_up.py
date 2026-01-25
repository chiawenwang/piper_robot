#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose


# =========================
# 可调参数
# =========================
GROUP_NAME = "arm"

# 回零（初始）关节角（按 MoveIt group 的关节顺序）
HOME_JOINTS = [0.0, 0.866, -0.960, 0.0, 0.182, 1.571]
ZERO_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# HOME
HOME_VEL_SCALE = 0.4
HOME_ACC_SCALE = 0.4

# TASK
TASK_VEL_SCALE = 0.03
TASK_ACC_SCALE = 0.03

# 额外整体放慢倍数（>1 更慢）
SLOW_DOWN_SCALE = 2.0

# txt 文件路径（你运行时可改成自己的绝对路径）
# 你上传到本对话环境的文件在 /mnt/data/case_1_refine.txt
TXT_PATH = "case_1_refine.txt"

# compute_cartesian_path 的插值步长（建议小一点更平滑）
EEF_STEP = 0.002
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

    group.set_joint_value_target(HOME_JOINTS)

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

def read_xz_samples_from_txt(txt_path):
    """
    读取 txt：
    - 忽略空行、# 注释行
    - 每行至少两列：a b
    返回: list of (a,b) 作为采样序列（后续只用相邻差分）
    """
    samples = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0])
                b = float(parts[1])
                samples.append((a, b))
            except ValueError:
                continue
    return samples


from geometry_msgs.msg import Pose

def build_waypoints_from_relative_deltas(samples, start_pose):
    """
    samples: [(sx0, sz0), (sx1, sz1), ...] 只是形状采样值
    只使用相邻差分 (dx, dz) 来累积生成路径：
      dx = sx[i]-sx[i-1]
      dz = sz[i]-sz[i-1]
      x_next = x_prev - dx   # x 方向反
      z_next = z_prev + dz
      y 固定为 start_pose.y
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to compute relative deltas.")

    waypoints = [start_pose]

    x = start_pose.position.x
    y = start_pose.position.y  # 固定
    z = start_pose.position.z

    prev_xs, prev_zs = samples[0]

    for i in range(1, len(samples)):
        xs, zs = samples[i]
        dx = xs - prev_xs
        dz = zs - prev_zs

        # 关键：x 方向反
        x = x - dx
        z = z + dz

        p = Pose()
        p.position.x = x
        p.position.y = y
        p.position.z = z
        p.orientation = start_pose.orientation
        waypoints.append(p)

        prev_xs, prev_zs = xs, zs

    return waypoints


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("piper_cartesian_from_txt", anonymous=True)

    group = moveit_commander.MoveGroupCommander(GROUP_NAME)
    group.set_max_velocity_scaling_factor(HOME_VEL_SCALE)
    group.set_max_acceleration_scaling_factor(HOME_ACC_SCALE)

    gripper_group = moveit_commander.MoveGroupCommander(GRIPPER_GROUP)

    # ===== 功能 1：每次执行前回 HOME（慢慢过去） =====
    if not move_to_home_pose(group):
        return
    # if not move_to_zero_pose(group):
    #     return

    # # 微微打开夹爪
    # rospy.loginfo("Opening gripper slightly...")
    # move_gripper(
    #     gripper_group,
    #     GRIPPER_OPEN,
    #     vel=GRIPPER_VEL_SCALE,
    #     acc=GRIPPER_ACC_SCALE
    # )

    # # 等待人工确认
    # if not wait_enter_or_quit("Press ENTER to close gripper, 'q'+ENTER to quit."):
    #     return

    # # 合上夹爪
    # rospy.loginfo("Closing gripper...")
    # move_gripper(
    #     gripper_group,
    #     GRIPPER_CLOSE,
    #     vel=GRIPPER_VEL_SCALE,
    #     acc=GRIPPER_ACC_SCALE
    # )

    # 在回到 HOME 后，再取一次当前 pose 作为“姿态锁定”的参考
    start_pose = group.get_current_pose().pose
    rospy.loginfo("Start pose after HOME: %s", str(start_pose))

    # ===== 功能 2：从 txt 读轨迹点 =====
    samples = read_xz_samples_from_txt(TXT_PATH)
    if not samples:
        rospy.logerr("No valid samples loaded from txt: %s", TXT_PATH)
        return

    waypoints = build_waypoints_from_relative_deltas(samples, start_pose)

    (plan, fraction) = group.compute_cartesian_path(
        waypoints,
        EEF_STEP,
        JUMP_THRESHOLD
    )

    rospy.loginfo("Cartesian path fraction: %.3f", fraction)
    if fraction < 0.99:
        rospy.logwarn("Path not fully achievable (fraction<0.99). Try smaller EEF_STEP or check collisions.")
        return

    # 再额外整体放慢
    plan = slow_down_plan(plan, SLOW_DOWN_SCALE)

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

    rospy.loginfo("Executing cartesian trajectory from txt slowly...")
    group.execute(plan, wait=True)
    group.stop()


if __name__ == "__main__":
    main()
