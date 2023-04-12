
from spatialmath import SO3
from spatialmath import SE3

from math import pi
import numpy as np
import matplotlib.pyplot as plt

def planar_error(P,a,b,c):
    # Input
    # P: points n by 3
    # a,b,c: 3d plane equation z = ax + by +c
    # Output
    # e_p: planar error
    normvec = np.array([a, b, -1])
    e_ps = (np.matmul(normvec, P.transpose()) + c) / np.linalg.norm(normvec)  # distance to plane
    e_p = np.sqrt(np.mean(e_ps ** 2))
    return e_p

def radial_error(P_plane,G):
    # Input
    # P_plane: points rotated to the horizontal plane 3 by n
    # G: [u, v, r]: (u,v) center of rotation, r radius of rotation
    # Output
    # Es: radial error of each point
    # eps: square sum of Es
    Es = np.linalg.norm(P_plane[0:2,:].transpose() - G[0:2],axis=1) - G[2]
    eps = np.sum(Es**2)
    return Es, eps

def e_jacobian(P_plane,G,E):
    J1 = -(P_plane[0,:] - G[0])/(E + G[2])
    J2 = -(P_plane[1,:] - G[1])/(E + G[2])
    J3 = -np.ones(J1.shape)
    J = np.stack((J1,J2,J3),axis=0).transpose()
    return J

def joint_axis_estm(P, center, radius):
    # Input
    # P: points n by 3
    # center: initial guess of center
    # radius: initial guess of radius
    # Output
    # Q_opt: joint rotation center
    # V_opt: joint rotation axis
    # e_p: planar error
    # e_r: radial error

    # regression plane
    M = np.concatenate((P[:, 0:2], np.ones((P.shape[0], 1))), axis=1)  # [x y 1]
    Z = P[:, 2]
    K = np.matmul(np.linalg.pinv(M), Z)
    a = K[0]
    b = K[1]
    c = K[2]
    normvec = np.array([a, b, -1])

    # plannar error
    e_p = planar_error(P, a, b, c)

    # geometric rotation
    zeta = np.arctan2(b, a)
    gamma = np.arccos(1 / np.linalg.norm(normvec))
    P_plane = SO3.Ry(gamma) * SO3.Rz(-zeta) * P.transpose()  # 3 by n

    # in-plane circle fitting
    # initialize G_old
    G_old = SO3.Ry(gamma) * SO3.Rz(-zeta) * center
    G_old = np.append(G_old[0:2], radius)
    plt.plot(P_plane[0, :], P_plane[1, :], marker='.')
    plt.plot(G_old[0], G_old[1], '.')

    # calculate E, eps, E Jacobian
    E_old, eps_old = radial_error(P_plane, G_old)
    J_old = e_jacobian(P_plane, G_old, E_old)

    # iterations for estimating center
    X = 10
    max_itr = 200
    for ii in range(max_itr):
        damping = 1e-5
        for jj in range(max_itr):
            damped_pinv = np.matmul(np.linalg.inv(np.matmul(J_old.transpose(), J_old) + damping * np.eye(3)),
                                    J_old.transpose())
            G_new = G_old - np.matmul(damped_pinv, E_old)
            E_new, eps_new = radial_error(P_plane, G_new)
            damping = X * damping
            if eps_new < eps_old:
                break
        if eps_new < eps_old:
            G_old = G_new
            plt.plot(G_old[0], G_old[1], '.')
            E_old, eps_old = radial_error(P_plane, G_old)
            J_old = e_jacobian(P_plane, G_old, E_old)
    G_opt = G_old
    E_opt = E_old
    e_r = np.sqrt(np.mean(E_opt ** 2))
    plt.plot(G_opt[0], G_opt[1], '*')
    # plt.show()

    # estimation of rotary joint axis
    Q_opt_plane = np.append(G_opt[0:2], np.mean(P_plane[2, :]))
    Q_opt = SO3.Rz(zeta) * SO3.Ry(-gamma) * Q_opt_plane
    Q_opt = np.squeeze(Q_opt)
    V_opt = normvec/np.linalg.norm(normvec)

    # plot
    mag = 10
    ax = plt.subplot(projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker='.')
    ax.scatter(Q_opt[0], Q_opt[1], Q_opt[2], marker='*')
    ax.quiver(Q_opt[0], Q_opt[1], Q_opt[2], mag * normvec[0], mag * normvec[1], mag * normvec[2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # plt.show()

    return Q_opt, V_opt, e_p, e_r

def frame_setup(Q1,Q2,V1,V2,L1):
    # Input
    # Q, V: center and axis of rotation
    # L: known kinematic information, for base L1 = -d1, for ee L6 = d6
    X1 = np.cross(V2,V1)
    X1 = X1/np.linalg.norm(X1)
    Y1 = np.cross(V1, X1)
    Y1 = Y1 / np.linalg.norm(Y1)

    H = np.stack((V1,-V2,X1),axis=1) # Q1 + t1V1 + t3X1 = Q2 + t2V2
    t = np.matmul(np.linalg.inv(H),Q2-Q1)  # t = [t1 t2 t3]
    R = np.stack((X1,Y1,V1),axis=1)
    P = Q1 + (t[0]+L1)*V1
    T = SE3.Rt(R,P)

    return T


if __name__ == '__main__':

    # load data
    data_base = np.loadtxt('data_setup/Basesetup.csv', delimiter=",")
    data_tool = np.loadtxt('data_setup/Toolsetup.csv', delimiter=",")

    # divide data into each joint
    P1 = data_base[1:32,6:9]
    P2 = data_base[32:,6:9]
    P5 = data_tool[1:30,6:9]
    P6 = data_tool[30:-3,6:9]

    # zero position data for tool setup
    theta_zero = data_tool[-1,5]
    P_faro_smr = np.mean(data_tool[-3:,6:9],axis=0)
    P_faro_smr = np.append(P_faro_smr,1)

    # initial guess of center and radius of rotation
    C1 = np.array([1250, 600, 200])
    C2 = np.array([1400, 700, 220])
    C5 = np.array([1200, 550, 150])
    C6 = np.array([1100, 530, 150])
    r1 = 30
    r2 = 255
    r5 = 200
    r6 = 30

    Q1, V1, e_p1, e_r1 = joint_axis_estm(P1, C1, r1)
    Q2, V2, e_p2, e_r2 = joint_axis_estm(P2, C2, r2)
    Q5, V5, e_p5, e_r5 = joint_axis_estm(P5, C5, r5)
    Q6, V6, e_p6, e_r6 = joint_axis_estm(P6, C6, r6)

    V1 = -V1

    T_base = frame_setup(Q1,Q2,V1,V2,-135)
    T_faro_flange = frame_setup(Q6,Q5,V6,V5,70)
    P_flange_smr = np.matmul(np.linalg.inv(T_faro_flange),P_faro_smr)[:3]
    P_flange_smr = SO3.Rz(-theta_zero*pi/180)*P_flange_smr
    T_tool = SE3.Rt(np.eye(3), P_flange_smr)

    print(T_base)
    print(T_tool)