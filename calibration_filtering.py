# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import roboticstoolbox as rtb
from spatialmath import SE3, base
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import myrobot


def filter(nmdh, cmdh, T_base, T_tool, trajectory, q0):
    # define robot
    nrobot = myrobot.SerialLink(mdh=nmdh, T_base=SE3(T_base), T_tool=SE3(T_tool))
    crobot = myrobot.SerialLink(mdh=cmdh, T_base=SE3(T_base), T_tool=SE3(T_tool))

    # calibrated IK
    q_init = q0
    T_filt = []
    traj_filt = []
    q_filt = []
    for idx, via_point in enumerate(trajectory):
        Te = SE3.Trans(via_point[0:3]) * SE3.RPY(np.flip(via_point[3:6]), unit='deg', order='xyz')
        q = crobot.ik_lm_chan(Te, q0=q_init)[0]
        # q = q + [0, 0, 0, 0, 0, 2 * pi]
        q_filt.append(q)

        T = nrobot.fkine(q)
        T = np.array(T)
        T_filt.append(T)

        # base.tr2rpy(T,unit='deg',order='xyz')
        traj = base.tr2x(T)
        traj[3:6] = np.flip(traj[3:6] * 180 / pi)
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
    calibrated = np.array([[0,	0,	0, 135],  # Athena DEMO
                           [-1.567671326, -0.09418696603, -1.567522278, 0.6327129301],
                           [0.00232117105, 134.99947, -0.001921290774, 0],
                           [-1.57539267, 37.78467258, 0.004128683254, 119.8986164],
                           [1.572228153, -0.04966838003, -0.0007294401206, -0.003682920773],
                           [-1.567004188, 0.0002605447373, 3.141592654,	70]])

    # calibrated = np.array([[0, 0, 0, 135],  # Hecate all
    #                        [-1.569505826, -0.0835910953, -1.567817367, 0.20853925],
    #                        [0.0002239402671, 134.9639272, -7.57E-04, 0],
    #                        [-1.571312979, 38.04809468, -0.001173209742, 119.9796628],
    #                        [1.571217148, 0.06784178805, 0.003849618677, -0.02988467229],
    #                        [-1.572022644, -0.003656865739, 3.141592654, 70]])

    # robot base and tool information
    T_base = SE3()
    TCP = np.array([-0.6594253352,	-0.1539743677,	152.412665,	123.2434375,	82.39040453,	56.49168609])  # Athena DEMO
    # TCP = np.array([-8.750342005, 0.234656936, 147.472938, -125.5313525, 87.57287802, -53.99604566])  # Hecate Bur
    # TCP = np.array([24.2737551,	-10.13695709,	58.75298778,	0,	0,	0])  # Hecate SMR
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    # T_tool = SE3(base.trnorm(np.array([[0.07638	,-0.006016, 0.9971, -6.022],
    #                                    [0.003141, -1, -0.006274, -0.2272],
    #                                    [0.9971, 0.003611, -0.07636, 145.9],
    #                                    [0, 0, 0, 1]])))

    # forward kinematics
    # crobot = myrobot.SerialLink(mdh=calibrated, T_base=T_base, T_tool=T_tool)
    # debug = crobot.fkine(np.array([2.651121,40.404569,-2.491293,-103.08569,-102.340345,84.152586]) * pi / 180)

    # load pre-filtered trajectory
    traj_prefilt = np.loadtxt('data_prefilter/unfiltered_test_points_dh.txt')
    # traj_prefilt = np.loadtxt('data_prefilter/unfiltered_cut_path/Unfiltered_occ_01.txt')
    # traj_prefilt = np.loadtxt('data_prefilter/unfiltered_cut_path/Unfiltered_axial_01.txt')

    # filter trajectory
    q_init = np.array([2.642586,29.025776,21.941379,-97.147241,-95.93431,76.152586]) * pi / 180
    T_filt, traj_filt, q_filt = filter(nmdh=nominal, cmdh=calibrated, T_base=T_base, T_tool=T_tool,
                                       trajectory=traj_prefilt, q0=q_init)

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.plot(traj_prefilt[:, 0], traj_prefilt[:, 1], traj_prefilt[:, 2], marker='.')
    ax.plot(traj_filt[:, 0], traj_filt[:, 1], traj_filt[:, 2], marker='.')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # save result
    np.savetxt('data_postfilter/filtered_test_points_dh.txt', traj_filt)
    # np.savetxt('data_postfilter/filtered_cut_path/occ_01.txt', traj_filt)
    # np.savetxt('data_postfilter/filtered_cut_path/axial_01.txt', traj_filt)
