import numpy as np

def IK(a1, a2, px, py, pz):
    # px, py, pz are the coordinates of the end effector
    # a1 and a2 are the lengths of the two links.

    theta_1 = np.arctan2(py, px)  # Fix this to match the geometry
    r = (px ** 2 + py ** 2) ** 0.5  # Horizontal distance from origin
    s = pz - a1  # Vertical distance, considering a1 as link 1 height
    theta_2 = np.arctan2(s, r)  # Correct theta_2 based on geometry

    d3 = (r ** 2 + s ** 2) ** 0.5 - a2  # Extension along prismatic joint

    return theta_1, theta_2, d3

a1 = 5  # Length of the first link
a2 = 3  # Length of the second link
px = 6  # x-coordinate of the end effector
py = 4  # y-coordinate of the end effector
pz = 7  # z-coordinate of the end effector

q1, q2, d3 = IK(a1, a2, px, py, pz)
print(f"Joint variables are: {q1:.2f}, {q2:.2f}, {d3:.2f}")

def FK(a1, a2, q1, q2, d3):
    cos_q1 = np.cos(q1)
    sin_q1 = np.sin(q1)
    cos_q2 = np.cos(q2)
    sin_q2 = np.sin(q2)

    # Homogeneous transformation from base to first joint (H01)
    R01 = np.array([[cos_q1, -sin_q1, 0], [sin_q1, cos_q1, 0], [0, 0, 1]])
    d01 = np.array([0, 0, 0])
    H01 = np.vstack([np.hstack([R01, d01.reshape(3, 1)]), np.array([0, 0, 0, 1])])

    # Homogeneous transformation from first joint to second joint (H12)
    R12 = np.array([[cos_q2, 0, -sin_q2], [0, 1, 0], [sin_q2, 0, cos_q2]])
    d12 = np.array([0, 0, a1])
    H12 = np.vstack([np.hstack([R12, d12.reshape(3, 1)]), np.array([0, 0, 0, 1])])

    # Position of the second joint
    p2 = np.array([a2 + d3, 0, 0])  # Taking into account d3 as prismatic joint extension
    p2 = np.hstack([p2, 1])

    # Compute the final transformation and position of the end effector
    H02 = np.dot(H01, H12)
    p0 = np.dot(H02, p2)[:3]
    
    return np.reshape(p0, (3, 1))

end_effector_position = FK(a1, a2, q1, q2, d3)
print(f"End effector position is: {end_effector_position}")
