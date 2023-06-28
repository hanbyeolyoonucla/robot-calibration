
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath import base

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import myrobot

def generate(robot, mu, sigma, box_l, angle, N, M):

    # box center configuration
    T_nominal = nrobot.fkine(mu * pi / 180)
    P_nominal = T_nominal.t
    Z_nominal = T_nominal.a

    # generate random points
    random_joints = np.random.normal(mu, sigma, (N, 6))
    Ts = nrobot.fkine(random_joints * pi / 180)
    Ps = Ts.t
    Zs = Ts.R[:, :, 2]

    # define position and orientation constraints
    box_c = P_nominal
    box = np.stack((box_c - box_l, box_c + box_l))

    # check constraints
    lb = np.all(Ps >= box[0], axis=1)
    ub = np.all(Ps <= box[1], axis=1)
    dev = np.arccos(np.dot(Zs, Z_nominal)) <= angle
    constraints = np.stack((lb, ub, dev))

    # points satisfying constraints
    P_cal = Ps[np.all(constraints, axis=0)]
    joints_cal = random_joints[np.all(constraints, axis=0)]

    # M number of sample
    P_cal = P_cal[:M,:]
    joints_cal = joints_cal[:M,:]


    return P_cal, joints_cal


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # nominal DH: alpha a theta d
    nominal = np.array([[0, 0, 0, 0.135],
                        [-pi / 2, 0, -pi / 2, 0],
                        [0, 0.135, 0, 0],
                        [-pi / 2, 0.038, 0, 0.120],
                        [pi / 2, 0, 0, 0],
                        [-pi / 2, 0, pi, 0.070]])
    nominal[:, [1, 3]] = 1e3 * nominal[:, [1, 3]]

    # robot base and tool information
    TCP = np.array([-0.7041735503, -0.3506146737, 158.0632991, -153.5539579, 85.22858361, -26.93566677])
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    T_base = SE3()

    # define robot
    nrobot = myrobot.SerialLink(mdh=nominal, T_base=T_base, T_tool=T_tool)

    # nominal configuration
    mu = np.array([4.333911931,	13.46457946,	39.44606267,	-89.63181963,	-88.44431986,	76.71547582])
    sigma = np.array([20, 20, 20, 20, 20, 90])
    box_l = 80 / 2
    angle = pi / 2
    N = 1000
    M = 80

    P_cal, joints_cal = generate(nrobot, mu, sigma, box_l, angle, N, M)

    # np.savetxt('cal_points/test.txt', joints_cal)

    # plot joint distribution
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].hist(joints_cal[:, 0])
    axs[0, 1].hist(joints_cal[:, 1])
    axs[0, 2].hist(joints_cal[:, 2])
    axs[1, 0].hist(joints_cal[:, 3])
    axs[1, 1].hist(joints_cal[:, 4])
    axs[1, 2].hist(joints_cal[:, 5])
    plt.show()

    # plot cartesian distribution
    ax = plt.subplot(projection='3d')
    ax.scatter(P_cal[:, 0], P_cal[:, 1], P_cal[:, 2], marker='.')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()