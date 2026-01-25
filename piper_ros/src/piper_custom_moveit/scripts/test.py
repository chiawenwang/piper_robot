#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import moveit_commander

from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import sys
sys.path.append("/home/sci-lab/Documents/piper_robot/piper_sdk/piper_sdk/custom_controller")

from piper_control import PiperArmController

GROUP_NAME = "arm"   # 改成你的 moveit planning group 名字（你截图里就是 arm）


class HybridController:
    """
    同一脚本里：
    - self.sdk: 直接用 SDK 做便捷动作
    - self.group: 用 MoveIt 规划
    - self.exec_client: 把轨迹交给 MoveIt 的 /execute_trajectory 去执行
    """
    def __init__(self):
        # ---- SDK（便捷功能）----
        can_name = rospy.get_param("~can_name", "can0")
        self.sdk = PiperArmController(can_name=can_name)
        self.sdk.connect()
        self.sdk.enable()
        # 可选
        try:
            self.sdk.switch_to_can_control()
        except Exception:
            pass

        # ---- MoveIt（规划）----
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander(GROUP_NAME)

        # ---- MoveIt（执行）----
        # 这是 MoveIt 自己的 ExecuteTrajectory action（你 rostopic 里就有 /execute_trajectory/*）
        self.exec_client = actionlib.SimpleActionClient("/execute_trajectory", ExecuteTrajectoryAction)
        rospy.loginfo("Waiting for /execute_trajectory action...")
        self.exec_client.wait_for_server()
        rospy.loginfo("Connected to /execute_trajectory")

    # -----------------------
    # 1) 直接用 SDK 做便捷动作
    # -----------------------
    def quick_home(self):
        # 例如：直接回零（根据你 SDK 里是否有 goto_home）
        self.sdk.goto_home(speed=30, gripper_m=None)

    def quick_gripper(self, open_=True):
        if open_:
            self.sdk.open_gripper(wait=True)
        else:
            self.sdk.close_gripper(wait=True)

    # -----------------------
    # 2) 用 MoveIt 规划并执行（MoveIt 负责执行链路）
    # -----------------------
    def plan_and_execute_joint_goal(self, joint_goal_rad):
        """
        joint_goal_rad: list[6] in rad
        """
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(joint_goal_rad)
        plan = self.group.plan()

        traj = self._extract_robot_trajectory(plan)
        if traj is None or len(traj.joint_trajectory.points) == 0:
            rospy.logerr("Planning failed / empty trajectory")
            return False

        return self.send_trajectory_to_moveit_execute(traj)

    # -----------------------
    # 3) 你“自己算出来的轨迹” -> 交给 MoveIt 执行
    # -----------------------
    def send_trajectory_to_moveit_execute(self, robot_traj: RobotTrajectory):
        """
        robot_traj: moveit_msgs/RobotTrajectory
        MoveIt 会把它继续转发给底层 controller 去执行
        """
        goal = ExecuteTrajectoryGoal()
        goal.trajectory = robot_traj

        self.exec_client.send_goal(goal)
        self.exec_client.wait_for_result()
        res = self.exec_client.get_result()

        # ExecuteTrajectoryResult.error_code 里有执行结果
        ok = (res is not None and res.error_code.val == 1)  # 1 通常是 SUCCESS
        rospy.loginfo("Execute result: %s", "SUCCESS" if ok else f"FAILED code={res.error_code.val if res else 'None'}")
        return ok

    # -----------------------
    # 4) 如果你的“自己算的轨迹”是 JointTrajectory（trajectory_msgs），也可包装成 RobotTrajectory
    # -----------------------
    def wrap_joint_traj(self, joint_traj: JointTrajectory) -> RobotTrajectory:
        rt = RobotTrajectory()
        rt.joint_trajectory = joint_traj
        return rt

    # -----------------------
    # utils：兼容不同 moveit_commander plan() 返回
    # -----------------------
    @staticmethod
    def _extract_robot_trajectory(plan):
        """
        Noetic 上 moveit_commander 的 plan() 可能返回:
        - RobotTrajectory
        - tuple(success, RobotTrajectory, planning_time, error_code)
        """
        if plan is None:
            return None
        if isinstance(plan, tuple):
            success, traj, _, _ = plan
            return traj if success else None
        return plan


def build_my_computed_joint_trajectory(joint_names):
    """
    示例：你“自己算出来的轨迹”如果是离散点，可以这样构造 JointTrajectory
    注意：time_from_start 要递增
    """
    jt = JointTrajectory()
    jt.joint_names = joint_names

    # 例子：3 个点
    points = [
        ([0.0, 0.2, -0.3, 0.0, 0.1, 0.0], 0.5),
        ([0.1, 0.3, -0.2, 0.1, 0.2, 0.1], 1.0),
        ([0.2, 0.4, -0.1, 0.2, 0.3, 0.2], 1.5),
    ]
    for pos, t in points:
        p = JointTrajectoryPoint()
        p.positions = pos
        p.time_from_start = rospy.Duration.from_sec(t)
        jt.points.append(p)
    return jt


def main():
    rospy.init_node("hybrid_sdk_moveit_control")

    ctl = HybridController()

    # ---- A) 直接用 SDK 做便捷动作 ----
    # ctl.quick_home()
    # ctl.quick_gripper(open_=True)

    # ---- B) MoveIt 规划 + MoveIt 执行 ----
    # joint_goal = [0.0, 0.6, -0.8, 0.0, 0.3, 0.0]
    # ctl.plan_and_execute_joint_goal(joint_goal)

    # ---- C) 你自己算的轨迹 -> 交给 MoveIt 执行 ----
    joint_names = ctl.group.get_active_joints()  # 用 MoveIt 的 joint 顺序最稳
    jt = build_my_computed_joint_trajectory(joint_names)
    rt = ctl.wrap_joint_traj(jt)
    ctl.send_trajectory_to_moveit_execute(rt)

    rospy.spin()


if __name__ == "__main__":
    main()
