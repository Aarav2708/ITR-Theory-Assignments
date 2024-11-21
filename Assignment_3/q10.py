import numpy as np
import pandas as pd
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

def calculate_velocity(J,q_dot):
    Jv=J[:3,:]
    Jw=J[3:,:]
    v=Jv@q_dot
    w=Jw@q_dot
    return(v,w)

a1 = 0
a2 =sp.Symbol('l2')
a3=sp.Symbol('l3')
a4=0
a5=0
a6=0

d1=sp.Symbol('l1')
d2=0
d3 = 0
d4=0
d5=0
d6=sp.Symbol('l6')

q1 = sp.Symbol('q1')
q2 =sp.Symbol('q2')
q3=sp.Symbol('q3')
q4=sp.Symbol('q4')
q5=sp.Symbol('q5')+np.pi/2
q6=sp.Symbol('q6')

alpha_1=np.pi/2
alpha_2=0
alpha_3=0
alpha_4=-np.pi/2
alpha_5=np.pi/2
alpha_6=0
q_dot=[sp.Symbol('q1_dot'),sp.Symbol('q2_dot'),sp.Symbol('q3_dot'),sp.Symbol('q4_dot'),sp.Symbol('q5_dot'),sp.Symbol('q6_dot')]
data=np.array([[1,a1,alpha_1,d1,q1,'R']
               ,[2,a2,alpha_2,d2,q2,'R']
               ,[3,a3,alpha_3,d3,q3,'R']
               ,[4,a4,alpha_4,d4,q4,'R']
               ,[5,a5,alpha_5,d5,q5,'R']
               ,[6,a6,alpha_6,d6,q6,'R']])
df = pd.DataFrame(data, columns=['Link', 'a', 'alpha', 'd', 'q', 'Type'])
df.set_index('Link', inplace=True)
print(f"D-H Parameters = {df}")
end_effector_coordinates,Jacobian=create_jacobian(df)
linear_velocity,angular_velocity=calculate_velocity(Jacobian,q_dot)

print(f"End effector coordinates wrt base frame = {end_effector_coordinates}")
print(f"Jacobian Matrix = {Jacobian}")
print(f"Velocity = {linear_velocity}")
print(f"Angular Velocity = {angular_velocity}")
