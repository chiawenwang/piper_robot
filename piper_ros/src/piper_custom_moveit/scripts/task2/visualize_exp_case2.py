import numpy as np
from matplotlib import pyplot as plt


# load npz file
# data = np.load("exp_case_2_circular_results.npz")
# data = np.load("exp_case_2_sine_results.npz")
data = np.load("exp_case_3_C_1.npz")

print(data.files)  # ['vertices_hist', 'target']

print(data['verts_hist'].shape)  # (T+1, N, 2)
print(data['target'].shape)          # (T+1, 2)

print(data['verts_hist'][0])  # initial vertices

exit()

vertices_history = data["verts_hist"]  # shape (T+1, N, 2)
target_curve = data["target"]          # shape (T+1, 2)

vertices_history = vertices_history.reshape(-1, 202)

ee_xy_traj = []

for i in range(vertices_history.shape[0]):
    vertices = vertices_history[i].reshape(-1, 2)
    ee_xy = vertices[-1]   # 末端点 (x, y)
    ee_xy_traj.append(ee_xy)

ee_xy_traj = np.array(ee_xy_traj)  # shape (T, 2)

# 保存为 txt
np.savetxt(
    "end_effector_xy_traj.txt",
    ee_xy_traj,
    fmt="%.8f",
    header="x y",
    comments=""
)

print("已保存为 end_effector_xy_traj.txt，shape =", ee_xy_traj.shape)

for i in range(vertices_history.shape[0]):
    vertices = vertices_history[i].reshape(-1, 2)
    plt.clf()
    plt.plot(vertices[:,0], vertices[:,1], 'o-', label='Step '+str(i))
    plt.plot(vertices[50, 0], vertices[50, 1], 'ro', label='Middle Node')
    plt.plot(target_curve[:,0], target_curve[:,1], 'r--', label='Target')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Deformation of the Beam')
    plt.grid() 
    plt.legend()
    plt.draw()
    plt.pause(0.1)
plt.show()