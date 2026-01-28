from geometry_msgs.msg import Pose
import math
from geometry_msgs.msg import Pose
import rospy
import numpy as np
import matplotlib.pyplot as plt


from tf.transformations import quaternion_from_euler, quaternion_multiply
def read_xz_theta_samples_from_txt(txt_path):
    """
    读取 txt 文件，统一输出 (x, z, theta)

    支持格式：
      - 2 列:  x z        -> theta = 0
      - 3 列:  x z theta  -> 正常读取

    规则：
      - 第 1 列 -> x
      - 第 2 列 -> z
      - 第 3 列 -> theta（若不存在则为 0）
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
                x = float(parts[0])
                z = float(parts[1])

                if len(parts) >= 3:
                    theta = float(parts[2])
                else:
                    theta = 0.0

                samples.append((x, z, theta))

            except ValueError:
                # 跳过非数字行
                continue

    samples = samples[::-1] 
    return samples


def xz_build_waypoints_from_aligned_samples(
    samples,
    start_pose,
    theta_in_degrees=False,
    theta_axis='x'
):
    """
    samples: [(sx, sz, theta), ...]
      - 位置：只做“相对首点”的平移（不累加）
      - x 方向反，z 正方向，y 固定
      - theta：直接使用每行给的值（不做相对/不累加）

    theta_axis:
      - 'y'：XOZ 平面常见（推荐）
      - 'z' / 'x'：按你的工具坐标系需要
    """
    if not samples:
        raise ValueError("No samples provided.")

    # txt 首点
    sx0, sz0, _ = samples[0]

    start_x = start_pose.position.x
    start_y = start_pose.position.y
    start_z = start_pose.position.z

    # 基准姿态
    q0 = [
        start_pose.orientation.x,
        start_pose.orientation.y,
        start_pose.orientation.z,
        start_pose.orientation.w
    ]

    waypoints = []
    # 可选：把当前位姿作为第一个 waypoint
    waypoints.append(start_pose)

    for (sx, sz, theta) in samples:
        dx = sx - sx0
        dz = sz - sz0

        # 角度单位
        if theta_in_degrees:
            theta = math.radians(theta)

        # 角度 → 四元数
        if theta_axis == 'y':
            q_theta = quaternion_from_euler(0.0, theta, 0.0)
        elif theta_axis == 'z':
            q_theta = quaternion_from_euler(0.0, 0.0, theta)
        elif theta_axis == 'x':
            q_theta = quaternion_from_euler(theta, 0.0, 0.0)
        else:
            raise ValueError("theta_axis must be 'x', 'y', or 'z'")

        # 最终姿态 = 起始姿态 ⊗ 角度旋转
        q = quaternion_multiply(q0, q_theta)

        p = Pose()
        p.position.x = start_x - dx 
        p.position.y = start_y           # y 固定
        p.position.z = start_z + dz      # z 正向

        p.orientation.x = q[0]
        p.orientation.y = q[1]
        p.orientation.z = q[2]
        p.orientation.w = q[3]

        waypoints.append(p)

    return waypoints


def yz_build_waypoints_from_aligned_samples(
    samples,
    start_pose,
    theta_in_degrees=False,
    theta_axis='z'
):

    if not samples:
        raise ValueError("No samples provided.")

    # txt 首点
    sy0, sz0, theta0 = samples[0]

    start_x = start_pose.position.x
    start_y = start_pose.position.y
    start_z = start_pose.position.z

    # 基准姿态
    q0 = [
        start_pose.orientation.x,
        start_pose.orientation.y,
        start_pose.orientation.z,
        start_pose.orientation.w
    ]

    waypoints = []
    # 可选：把当前位姿作为第一个 waypoint
    waypoints.append(start_pose)

    for (sy, sz, theta) in samples:
        dy = sy - sy0
        dz = sz - sz0
        dtheta = theta - theta0

        # 角度单位
        if theta_in_degrees:
            dtheta = math.radians(dtheta)
        # theta = 0

        # 角度 → 四元数
        if theta_axis == 'y':
            q_theta = quaternion_from_euler(0.0, dtheta, 0.0)
        elif theta_axis == 'z':
            q_theta = quaternion_from_euler(0.0, 0.0, dtheta)
        elif theta_axis == 'x':
            q_theta = quaternion_from_euler(dtheta, 0.0, 0.0)
        else:
            raise ValueError("theta_axis must be 'x', 'y', or 'z'")

        # 最终姿态 = 起始姿态 ⊗ 角度旋转
        # q = quaternion_multiply(q0, 0)
        q = quaternion_multiply(q0, q_theta)


        p = Pose()
        p.position.x = start_x
        p.position.y = start_y + dy
        p.position.z = start_z + dz

        p.orientation.x = q[0]
        p.orientation.y = q[1]
        p.orientation.z = q[2]
        p.orientation.w = q[3]

        waypoints.append(p)

    # waypoints = waypoints[::-1]

    return waypoints

def plot_beam_shape(
    txt_path: str,
    *,
    scale: float = 1.0,
    center: bool = False,
    show_projection: bool = True,
    equal_aspect_3d: bool = True,
    title: str = "Beam shape (x, y, z)"
):
    """
    读取三列 (x y z) 的 txt 并绘制 beam 形状。
    
    参数
    - txt_path: 文件路径
    - scale: 对坐标整体缩放（比如单位换算）
    - center: 是否将点云平移到以均值为中心
    - show_projection: 是否额外画 XY / XZ / YZ 投影
    - equal_aspect_3d: 3D 中是否尽量保持各轴等比例显示
    - title: 图标题
    """
    # 读取：支持空格/制表符分隔；自动跳过空行；如果有注释行也能处理（以 # 开头）
    data = np.loadtxt(txt_path, comments="#")

    if data.ndim == 1:
        if data.size != 3:
            raise ValueError("文件里似乎不是三列数据（x y z）。")
        data = data.reshape(1, 3)

    if data.shape[1] < 3:
        raise ValueError(f"读取到的列数为 {data.shape[1]}，需要至少 3 列（x y z）。")

    xyz = data[:, :3].astype(float) * scale
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if center:
        x = x - x.mean()
        y = y - y.mean()
        z = z - z.mean()

    if show_projection:
        fig = plt.figure(figsize=(12, 9))

        # 3D
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax3d.plot(x, y, z, marker="o", markersize=3, linewidth=1)
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title(title)

        # 尽量等比例（matplotlib 新版支持 set_box_aspect）
        if equal_aspect_3d and hasattr(ax3d, "set_box_aspect"):
            xr = np.ptp(x) if np.ptp(x) > 0 else 1.0
            yr = np.ptp(y) if np.ptp(y) > 0 else 1.0
            zr = np.ptp(z) if np.ptp(z) > 0 else 1.0
            ax3d.set_box_aspect((xr, yr, zr))

        # XY
        ax_xy = fig.add_subplot(2, 2, 2)
        ax_xy.plot(x, y, marker="o", markersize=3, linewidth=1)
        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.set_title("Projection: XY")
        ax_xy.axis("equal")

        # XZ
        ax_xz = fig.add_subplot(2, 2, 3)
        ax_xz.plot(x, z, marker="o", markersize=3, linewidth=1)
        ax_xz.set_xlabel("x")
        ax_xz.set_ylabel("z")
        ax_xz.set_title("Projection: XZ")
        ax_xz.axis("equal")

        # YZ
        ax_yz = fig.add_subplot(2, 2, 4)
        ax_yz.plot(y, z, marker="o", markersize=3, linewidth=1)
        ax_yz.set_xlabel("y")
        ax_yz.set_ylabel("z")
        ax_yz.set_title("Projection: YZ")
        ax_yz.axis("equal")

        plt.tight_layout()
        plt.show()
        return fig

    else:
        fig = plt.figure(figsize=(8, 6))
        ax3d = fig.add_subplot(1, 1, 1, projection="3d")
        ax3d.plot(x, y, z, marker="o", markersize=3, linewidth=1)
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title(title)
        plt.tight_layout()
        plt.show()
        return fig
    
if __name__ == "__main__":
    plot_beam_shape("exp_initial_4.txt")