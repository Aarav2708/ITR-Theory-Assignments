import numpy as np
import sympy as sp

## q1, q2, d are the symbols for the angles and distances between the frames

def calculate_transformed_coordinates(q1,q2,d,l1,l2):
        
    cos_q1=sp.cos(q1)
    sin_q1=sp.sin(q1)

    cos_q2=sp.cos(q2)
    sin_q2=sp.sin(q2)

    R01=np.array([[cos_q1,-sin_q1,0],[sin_q1,cos_q1,0],[0,0,1]])
    d01=np.array([0,0,0])
    H01=np.vstack([np.hstack([R01,d01.reshape(3,1)]),np.array([0,0,0,1])])
    print(f"The Matrix H01 is {H01}")

    R12=np.array([[1,0,0],[0,cos_q2,-sin_q2],[0,sin_q2,cos_q2]])
    d12=np.array([0,0,l1])
    H12=np.vstack([np.hstack([R12,d12.reshape(3,1)]),np.array([0,0,0,1])])
    print(f"The Matrix H12 is {H12}")

    p2=np.array([l2,0,-d])
    p2=np.hstack([p2,1])
    print(f"The vector p2 is {p2.reshape(4,1)}")

    H02=np.dot(H01,H12)
    print(f"The Matrix H02 is {H02}")
    p0=np.dot(H02,p2)[:3]

    return np.reshape(p0,(3,1))



q1=sp.symbols('q1')
q2=sp.symbols('q2')
d=sp.symbols('d')

## l1 and l2 are the lengths of the links

l1=sp.symbols('l1')
l2=sp.symbols('l2')

## p2 is the position of the end effector in the non-inertial frame 2
print(f"Positon of point in the inertial frame ={calculate_transformed_coordinates(q1,q2,d,l1,l2)}")