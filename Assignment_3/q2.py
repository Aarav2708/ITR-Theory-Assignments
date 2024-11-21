import numpy as np
import sympy as sp

def Jacobian(q1,q2,z,h,l1,l2):
    cos_q1=sp.cos(q1)
    sin_q1=sp.sin(q1)

    cos_q2=sp.cos(q2)
    sin_q2=sp.sin(q2)

    ## The Jacobian matrix is going to be a 6X3 Matrix here
    Jv=sp.zeros(3,3)
    Jw=sp.zeros(3,3)
    R01=np.array([[cos_q1,-sin_q1,0],[sin_q1,cos_q1,0],[0,0,1]])
    R12=np.array([[cos_q2,-sin_q2,0],[sin_q2,cos_q2,0],[0,0,1]])
    R23=np.array([[1,0,0],[0,1,0],[0,0,1]])

    d01=np.array([0,0,h])
    d12=np.array([l1,0,0])
    d23=np.array([l2,0,0])

    H01=np.vstack([np.hstack([R01,d01.reshape(3,1)]),np.array([0,0,0,1])])
    H12=np.vstack([np.hstack([R12,d12.reshape(3,1)]),np.array([0,0,0,1])])
    
    H23=np.vstack([np.hstack([R23,d23.reshape(3,1)]),np.array([0,0,0,1])])
    H03=np.dot(H01,np.dot(H12,H23))

    ## For the 1st joint
    z0=np.array([0,0,1])
    pe_3=np.array([0,0,-z])# Position of the end effector in the frame 3
    pe_3=np.hstack([pe_3,1])
    pe_0=np.dot(H03,pe_3)[:3]# Position of the end effector in the frame 0
    p0_1=d01# Position of the joint 1 in the frame 0
    Jv[:,0] = np.cross(z0, pe_0-p0_1)
    Jw[:,0]=z0
    #print(f"Jv1={Jv[:,0]}")

    ## For the 2nd Joint
    z1=np.array([0,0,1])
    H02=np.dot(H01,H12)
    #print(f"H02: {H02}")
    p1_2=d12# Position of the joint 2 in the frame 1
    p0_2=np.dot(H02,np.hstack([p1_2,1]))[:3]# Position of the joint 2 in the frame 0
    
    Jv[:,1] = np.cross(z1, pe_0-p0_2)
    Jw[:,1]=z1

    ## For the 3rd Joint
    z2=np.array([0,0,1])
    H03=np.dot(H02,H23)
    #print(f"H03: {H03}")
    p2_3=d23# Position of the joint 3 in the frame 2
    p0_3=np.dot(H03,np.hstack([p2_3,1]))[:3]# Position of the joint 3 in the frame 0
    Jv[:,2]=z2
    Jw[:,2]=np.array([0,0,0])

    J=np.vstack([Jv,Jw])
    return Jv[:,1]


q1=sp.symbols('q1')
q2=sp.symbols('q2')
z=sp.symbols('z')

## l1 and l2 are the lengths of the links

h=sp.symbols('h')
l1=sp.symbols('l1')
l2=sp.symbols('l2')

print(Jacobian(q1,q2,z,h,l1,l2)[:,0])