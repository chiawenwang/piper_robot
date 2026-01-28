import numpy as np
from matplotlib import pyplot as plt

# load npz file
data = np.load("exp_case_3_C_1.npz")

verts_history = data["verts_hist"]  # (T+1, N, 2)
target = data["target"]             # (N, 2)

T = verts_history.shape[0]

traj = []
ee_pose = []   # 用来存 [x, y, theta]

for i in range(T):
    # 末端位置
    p_end = verts_history[i, -1]
    p_prev = verts_history[i, -2]

    x, y = p_end

    # 末端朝向 theta
    dx = p_end[0] - p_prev[0]
    dy = p_end[1] - p_prev[1]
    theta = np.arctan2(dy, dx)

    traj.append([x, y])
    ee_pose.append([x, y, theta])

traj = np.array(traj)
ee_pose = np.array(ee_pose)

# ===============================
# 写入 txt 文件
# ===============================
np.savetxt(
    "end_effector_pose.txt",
    ee_pose,
    header="x y theta(rad)",
    comments=""
)

print("End-effector pose saved to end_effector_pose.txt")

# ===============================
# 可视化
# ===============================
for i in range(T):
    vertices = verts_history[i]

    plt.clf()
    plt.plot(vertices[:,0], vertices[:,1], 'o-', label='Step '+str(i))
    plt.plot(vertices[50, 0], vertices[50, 1], 'ro', label='Middle Node')
    plt.plot(target[:,0], target[:,1], 'r--', label='Target')
    plt.plot(traj[:,0], traj[:,1], 'k--', label='End Effector Traj')

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Deformation of the Beam')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.pause(0.1)

plt.show()