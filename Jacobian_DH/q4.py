
import sympy as sp
import numpy as np

def Jacobian(l1,l2,l3,q1,q2,q3):

    ## q1,q2 and q3 are the angles between the frames.

    cos_q1=sp.cos(q1)
    sin_q1=sp.sin(q1)

    cos_q2=sp.cos(q2)
    sin_q2=sp.sin(q2)

    sin_q3=sp.sin(q3)
    cos_q3=sp.cos(q3)

    ## The Jacobian matrix is going to be a 6X3 Matrix here

    Jv=sp.zeros(3,3)
    Jw=sp.zeros(3,3)
    R01=np.array([[cos_q1,-sin_q1,0],[sin_q1,cos_q1,0],[0,0,1]])
    R12=np.array([[cos_q2,-sin_q2,0],[sin_q2,cos_q2,0],[0,0,1]])
    R23=np.array([[cos_q3,-sin_q3,0],[sin_q3,cos_q3,0],[0,0,1]])

    d01=np.array([0,0,0])# Distance between the frames
    d12=np.array([l1,0,0])
    d23=np.array([l2,0,0])
    pe_3=np.array([l3,0,0])# Position of the end effector in the frame 3


    ## For the 1st joint

    z0=np.array([0,0,1])
    H01=np.vstack([np.hstack([R01,d01.reshape(3,1)]),np.array([0,0,0,1])])
    H12=np.vstack([np.hstack([R12,d12.reshape(3,1)]),np.array([0,0,0,1])])
    H23=np.vstack([np.hstack([R23,d23.reshape(3,1)]),np.array([0,0,0,1])])

    H03=np.dot(H01,np.dot(H12,H23))
    pe_3=np.hstack([pe_3,1])
    pe_0=np.dot(H03,pe_3)[:3]# Position of the end effector in the frame 0
    Jv[:,0] = np.cross(z0, pe_0)
    Jw[:,0]=z0
    

    ## For the 2nd Joint

    z1=np.array([0,0,1])
    # To find position of joint 2 wrt base frame
    H02=np.dot(H01,H12)
    p2_1=d12# Position of the joint 2 in the frame 1
    p2_0=np.dot(H02,np.hstack([p2_1,1]))[:3]# Position of the joint 2 in the frame 0
    Jv[:,1] = np.cross(z1, pe_0-p2_0)
    Jw[:,1]=z1

    ## For the 3rd Joint
    
    z2=np.array([0,0,1])
    # To find position of joint 3 wrt base frame
    H03=np.dot(H02,H23)
    p3_2=d23# Position of the joint 3 in the frame 2
    p3_0=np.dot(H03,np.hstack([p3_2,1]))[:3]# Position of the joint 3 in the frame 0
    Jv[:,2] = np.cross(z2, pe_0-p3_0)
    Jw[:,2]=z2

    J=np.vstack([Jv,Jw])
    return(J)

q1=sp.symbols('q1')
q2=sp.symbols('q2')
q3=sp.symbols('q3')

## l1 and l2 are the lengths of the links

l1=sp.symbols('l1')
l2=sp.symbols('l2')
l3=sp.symbols('l3')

## p2 is the position of the end effector in the non-inertial frame 2
print(f"Value of the Jacobian Matrix ={Jacobian(l1,l2,l3,q1,q2,q3)}")