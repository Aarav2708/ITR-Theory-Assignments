import numpy as np
import sympy as sp
import pandas as pd

## Consider that we are given R and d as the orientation and end-effector coordinates of the robot. 

## Also, we have been given the DH parameters of the robot



def IK_solver_1(df,R,d):# Finds Joint variables q1,d2,d3

    d6=df.loc[6,'d']
    pc=d-d6*R*np.array([0,0,1])# Wrist Center
    q1=np.arctan2(pc[1],pc[0])
    d2=pc[2]-df.loc[0,'d']
    d3=(pc[0]**2+pc[1]**2)

    return(q1,d2,d3)
    

def IK_solver_2(q1,R,d):# Finds End-Effector angles q4,q5, and q6

    # Uisng the DH Parameters, we can write 
    R03=np.array([[np.cos(q1),0,-np.sin(q1)],[np.sin(q1),0,np.cos(q1)],[0,-1,0]]) # Rotation matrix from frame 0 to frame 3
    R36=R03.T@R # Rotation matrix from frame 3 to frame 6, this is know now
    u=R36 
    q4=np.arctan2(u[0,2],u[1,2])
    q5=np.arctan2(u[2,2],-(1-u[2,2])**0.5)
    q6=np.arctan2(-u[2,0],u[2,1])

    return q4,q5,q6


## Inputs: Joint variables are q1,d2,d3,q4,q5,q6
a1 = 0
a2 = 0
a3=0
a4=0
a5=0
a6=0
d1=sp.Symbol('d1')
d2=sp.Symbol('d2')
d3 = sp.Symbol('d3')
d4=0
d5=0
d6=sp.Symbol('d6')
theta_1 = sp.Symbol('q1')
theta_2 = 0
theta_3=0
theta_4=sp.Symbol('q4')
theta_5=sp.Symbol('q5')
theta_6=sp.Symbol('q6')
data = [[1, a1, 0, d1, theta_1, 'R'], [2, a2, -(np.pi)/2, d2, theta_2, 'P'], [3, a3, 0, d3, theta_3, 'P'],[4, a4, -(np.pi)/2, d4, theta_4,'R'],[5, a5, (np.pi)/2, d5, theta_5, 'R'],[6, a6, 0, d6, theta_6, 'R']]
df = pd.DataFrame(data, columns=['Link', 'a', 'alpha', 'd', 'q', 'Type'])
df.set_index('Link', inplace=True)
print(f"D-H Parameters = {df}")
R = np.array([list(map(float, input(f"Enter row {i+1} of the 3x3 rotation matrix R (space-separated): ").split())) for i in range(3)])
d = np.array(list(map(float, input("Enter the 3x1 end-effector position vector d (space-separated): ").split())))
q1,d2,d3=IK_solver_1(df,R,d)
q4,q5,q6=IK_solver_2(q1,R,d)
print(f"q1 = {q1}, d2 = {d2}, d3 = {d3}, q4 = {q4}, q5 = {q5}, q6 = {q6}")

