# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath import base

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import Meca


def filter (nmdh, cmdh, T_base, T_tool, trajectory, q0):


    # define robot
    nrobot = Meca.Meca(mdh=nmdh, T_base=SE3(T_base), T_tool=SE3(T_tool))
    crobot = Meca.Meca(mdh=cmdh, T_base=SE3(T_base), T_tool=SE3(T_tool))

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
    nominal = np.array([[0, 0, 0, 0.135],
                    [-pi / 2, 0, -pi / 2, 0],
                    [0, 0.135, 0, 0],
                    [-pi / 2, 0.038, 0, 0.120],
                    [pi / 2, 0, 0, 0],
                    [-pi / 2, 0, pi, 0.070]])
    nominal[:,[1,3]] = 1e3*nominal[:,[1,3]]
    # print(nominal)

    # calibrated DH: alpha a theta d
    calibrated = np.array([[0.0000000,-1.5685302,-0.0015844,-1.5710132,1.5713967,-1.5703714],
                           [0.0000000,0.3760858,134.9888038,37.8811929,0.0460869,0.0019857],
                           [0.0000000,-1.5714982,0.0019691,0.0000571,0.0018524,3.1415927],
                           [135.0000000,0.1216121,0.0000000,120.0658861,-0.0415388,70.0000000]])
    calibrated = np.transpose(calibrated)
    # calibrated[:,[1,3]] = 0.001*calibrated[:,[1,3]]

    # robot base and tool information
    TCP = np.array([-0.7041735503,	-0.3506146737,	158.0632991,	-153.5539579,	85.22858361,	-26.93566677])
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    T_base = SE3()
    # print(T_tool)
    # print(T_base)

    # load pre-filtered trajectory
    traj_prefilt = np.loadtxt('data_prefilter/typo_1.2A-8/Unfiltered_occ_01.txt')
    # print(traj_prefilt[0])

    # filter trajectory
    q_init = np.array([5.583879,14.467759,37.938621,-88.654655,-87.679138,25.230172])*pi/180
    T_filt, traj_filt, q_filt = filter(nmdh=nominal, cmdh=calibrated, T_base=T_base, T_tool=T_tool,trajectory=traj_prefilt,q0=q_init)
    # check deviation due to filtering
    print(np.mean(traj_filt[:,0:3]-traj_prefilt[:,0:3],axis=0))

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.plot(traj_prefilt[:,0], traj_prefilt[:,1], traj_prefilt[:,2], marker='.')
    ax.plot(traj_filt[:, 0], traj_filt[:, 1], traj_filt[:, 2], marker='x')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # save result
    np.savetxt('data_postfilter/typo_1.2A-8/occ_01.txt',traj_filt)

    # print(T_filt.shape)
    # print(traj_prefilt.shape)
    # print(traj_filt.shape)
    # print(q_filt.shape)

    # print(crobot)
    # qt = rtb.jtraj(crobot.qr, crobot.qz, 50)
    # Te = crobot.fkine(qt.q)
    # sol = crobot.ik_lm_chan(Te)
    # print(crobot.ik_lm_chan(Te[5]))

    # Te = nrobot.fkine(nrobot.qr)  # forward kinematics
    # print(Te)

    # Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    # sol = nrobot.ik_lm_chan(Tep)  # solve IK
    # print(sol)

    # q_pickup = sol[0]
    # print(robot.fkine(q_pickup))  # FK shows that desired end-effector pose was achieved

    #
    # nrobot.plot(qt.q, backend='pyplot')

    # qt = rtb.jtraj(crobot.qr, crobot.qz, 50)
    # print(qt.q.shape)
    # crobot.plot(qt.q, backend='pyplot')

    # robot.plot(qt.q)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
