import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import numpy as np
import sympy as sp


def create_jacobian(df):
    
    n=df.shape[0]# Number of Joints
    Jv=sp.zeros(3,n)
    Jw=sp.zeros(3,n)
    A =[]
    for i, row in df.iterrows():
        alpha_i = row['alpha']
        d_i = row['d']
        a_i = row['a']
        q_i = row['q']
    
        cos_qi = sp.cos(q_i)
        sin_qi = sp.sin(q_i)
        cos_alpha = sp.cos(alpha_i)
        sin_alpha = sp.sin(alpha_i)
        T_i = np.array([
        [cos_qi, -sin_qi * cos_alpha, sin_qi * sin_alpha, a_i * cos_qi],
        [sin_qi, cos_qi * cos_alpha, -cos_qi * sin_alpha, a_i * sin_qi],
        [0, sin_alpha, cos_alpha, d_i],
        [0, 0, 0, 1]
        ])

        A.append(T_i)
    
    
    # A contains the list of transforation matrices

    # For each joint we have to calculate Jv and Jw
    # First of all we need the origin of the end effector wrt base frame

    A=np.array(A)
    T_n=np.eye(4,4)# Transformation matrix from base to end effector
    for i in range(n):
        T_n=T_n@A[i]

    

    # We can find the end effector coordinates in base frame
    p_n=T_n[:3,3] # End effector position in base frame

    # We need to find the Coordinates of each joint wrt base frame
    # Create an array of joint poitions
    p_joints=[]
    R_joints=[]# Rotation matrices for the joints
    for i in range(0,n):
        T=np.eye(4,4)# Iniialsing Transformation matrix
        for j in range(i):
            T=T@A[j]
        p_j=T[:3,3]
        r_j=T[:3,:3]

        p_joints.append(p_j)
        R_joints.append(r_j)
    p_joints=np.array(p_joints)
    R_joints=np.array(R_joints)

    # Ensure R_joints and p_joints have appropriate dimensions
    assert len(R_joints) == n, "R_joints length must match the number of joints"
    assert len(p_joints) == n, "p_joints length must match the number of joints"

# Check if the 'Type' column exists; if not, assume all are rotational ('R')
    if 'Type' not in df.columns:
        df['Type'] = 'R'  # Assign 'R' to all joints

    
    for i in range(n):
        row = df.iloc[i]  # Access the row using iloc, which starts indexing from 0
        if row['Type'] == 'R':  # Rotational joint
            zi = np.dot(R_joints[i], np.array([0, 0, 1]))
            pi = p_joints[i]
            Jv[:, i] = np.cross(zi, p_n - pi)
            Jw[:, i] = zi
        else:  # Prismatic joint
            zi = np.dot(R_joints[i], np.array([0, 0, 1]))
            Jv[:, i] = zi
            Jw[:, i] = np.array([0, 0, 0])
    J = np.vstack((Jv, Jw))
    
    return(p_n,J)

def joint_velocity(J_v,v_e):
    return np.dot(np.linalg.pinv(J_v),v_e)
    
# Define the system of equations
def equations(vars, a1, a2, px, py):
    q1, q2 = vars
    eq1 = a1 * np.cos(q1) + a2 * np.cos(q1 + q2) - px
    eq2 = a1 * np.sin(q1) + a2 * np.sin(q1 + q2) - py
    return [eq1, eq2]

# Link lengths (a1, a2) and end-effector position (px, py)
a1 = 5  # Length of first link
a2 = 3  # Length of second link
px = 6  # x-coordinate of the end effector
py = 4  # y-coordinate of the end effector
pz= -1

# Initial guesses for q1 and q2 in radians (both positive)
initial_guess = [np.radians(45), np.radians(45)]  # Adjusted to ensure positive angles

# Solve for q1 and q2
solution = fsolve(equations, initial_guess, args=(a1, a2, px, py))

# Extract the solutions
q1, q2 = solution

# Convert the angles to degrees for readability (optional)
q1_deg = np.degrees(q1)
q2_deg = np.degrees(q2)

# Ensure q2 is positive (if negative, adjust by adding 360 degrees)
if q2_deg < 0:
    q2_deg += 360

print(f"q1 = {q1_deg:.2f} degrees")
print(f"q2 = {q2_deg:.2f} degrees")

a3=0
q3=0
d3=1
alpha_1=0
alpha_2=np.pi
alpha_3=0
data = [[1, a1, alpha_1, 0, q1, 'R'], [2, a2, alpha_2, 0, q2, 'R'], [3, 0, alpha_3, d3, 0, 'P']]
df = pd.DataFrame(data, columns=['Link', 'a', 'alpha', 'd', 'q', 'Type'])
df.set_index('Link', inplace=True)
print(f"D-H Parameters = {df}")
end_effector_coordinates,Jacobian=create_jacobian(df)
# velocity,angular_velocity=calculate_velocity(Jacobian,q_dot)

# print(f"End effector coordinates wrt base frame = {end_effector_coordinates}")
print(f"Jacobian Matrix = {Jacobian}")
J_v=Jacobian[:3,:]
v_e= np.array(list(map(float, input("Enter the 3D end-effector velocity (vx, vy, vz) as space-separated values: ").split())))
q_dots=joint_velocity(J_v,v_e)
