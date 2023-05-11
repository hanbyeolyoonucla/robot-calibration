# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath import base

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import myrobot


def filter (nmdh, cmdh, T_base, T_tool, trajectory, q0):

    # define robot
    nrobot = myrobot.SerialLink(mdh=nmdh, T_base=SE3(T_base), T_tool=SE3(T_tool))
    crobot = myrobot.SerialLink(mdh=cmdh, T_base=SE3(T_base), T_tool=SE3(T_tool))

    # calibrated IK
    q_init = q0
    T_filt = []
    traj_filt =[]
    q_filt = []
    for idx, via_point in enumerate(trajectory):

        Te = SE3.Trans(via_point[0:3]) * SE3.RPY(np.flip(via_point[3:6]), unit='deg', order='xyz')
        q = crobot.ik_lm_chan(Te, q0=q_init)[0]
        q = q + [0, 0, 0, 0, 0, 2*pi]
        q_filt.append(q)

        T = nrobot.fkine(q)
        T = np.array(T)
        T_filt.append(T)

        # base.tr2rpy(T,unit='deg',order='xyz')
        traj = base.tr2x(T)
        traj[3:6] = np.flip(traj[3:6]*180/pi)
        traj_filt.append(traj)

        # nrobot.plot(np.array(q_filt),backend='pyplot')

    return np.array(T_filt), np.array(traj_filt), np.array(q_filt)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # nominal DH: alpha a theta d
    nominal = np.array([[0, 0, 0, 135],
                    [-pi / 2, 0, -pi / 2, 0],
                    [0, 135, 0, 0],
                    [-pi / 2, 38, 0, 120],
                    [pi / 2, 0, 0, 0],
                    [-pi / 2, 0, pi, 70]])

    # calibrated DH: alpha a theta d
    calibrated = np.array([[0,	0,	0, 135],
                           [-1.567671326, -0.09418696603, -1.567522278, 0.6327129301],
                           [0.00232117105, 134.99947, -0.001921290774, 0],
                           [-1.57539267, 37.78467258, 0.004128683254, 119.8986164],
                           [1.572228153, -0.04966838003, -0.0007294401206, -0.003682920773],
                           [-1.567004188, 0.0002605447373, 3.141592654,	70]])
    # calibrated = np.transpose(calibrated)

    # robot base and tool information
    TCP = np.array([-0.666253614,	-0.1984292699,	158.1194595,	166.8310916,	85.81981908,	12.75822422])
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    T_base = SE3()

    # load pre-filtered trajectory
    traj_prefilt = np.loadtxt('data_prefilter/typo_1.2F-8/Unfiltered_axial_04.txt')

    # filter trajectory
    q_init = np.array([4.632414,35.811466,10.487845,-98.90431,-98.297586,70.859483])*pi/180
    T_filt, traj_filt, q_filt = filter(nmdh=nominal, cmdh=calibrated, T_base=T_base, T_tool=T_tool,trajectory=traj_prefilt,q0=q_init)

    # check deviation due to filtering
    print(np.mean(traj_filt[:,0:3]-traj_prefilt[:,0:3],axis=0))

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.plot(traj_prefilt[:,0], traj_prefilt[:,1], traj_prefilt[:,2], marker='.')
    ax.plot(traj_filt[:, 0], traj_filt[:, 1], traj_filt[:, 2], marker='.')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # save result
    np.savetxt('data_postfilter/typo_1.2F-8/axial_04.txt',traj_filt)
