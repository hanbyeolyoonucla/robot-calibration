# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import roboticstoolbox as rtb
from spatialmath import SE3, base
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import myrobot
import pandas as pd
import argparse
from poe_local_filtering import ecat_format


def dh_filter(nmdh, cmdh, T_base, T_tool, trajectory, q0):
    # define robot
    nrobot = myrobot.SerialLink(mdh=nmdh, T_base=SE3(T_base), T_tool=SE3(T_tool))
    crobot = myrobot.SerialLink(mdh=cmdh, T_base=SE3(T_base), T_tool=SE3(T_tool))

    # calibrated IK
    q_init = q0
    T_filt = []
    traj_filt = []
    q_filt = []
    success_filt = []

    for idx, via_point in enumerate(trajectory):
        Te = SE3.Trans(via_point[0:3]) * SE3.RPY(np.flip(via_point[3:6]), unit='deg', order='xyz')
        sol = crobot.ik_lm_chan(Te, q0=q_init)
        q = sol[0]
        success = sol[1]
        assert success != 0, 'IK does not exist!'

        q_filt.append(q)
        success_filt.append(success)

        T = nrobot.fkine(q)
        T = np.array(T)
        T_filt.append(T)

        # base.tr2rpy(T,unit='deg',order='xyz')
        traj = base.tr2x(T)
        traj[3:6] = np.flip(traj[3:6] * 180 / pi)
        traj_filt.append(traj)

    return np.array(T_filt), np.array(traj_filt), np.array(q_filt), np.array(success_filt)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ###################################
    # ARGUMENTS FOR FILTERING PROCESS #
    ###################################

    # arg parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_fname", help="input file name: ex) Unfiltered_axial_01.txt")
    argParser.add_argument("-o", "--output_fname", help="output file name: ex) axial_01.txt")
    argParser.add_argument("-oe", "--output_eCAT_fname", help="output eCAT file name: ex) cutpath_a1.txt")
    argParser.add_argument("-c", "--calibration", help="calibration model: ex) CAL00001.csv")
    argParser.add_argument("-t", "--tcp", help="tcp model: ex) TCP0001.csv")
    argParser.add_argument("-q", "--q_init", help="initial joint angle: ex) 0,0,0,0,0,0")
    args = argParser.parse_args()

    # unfiltered cut path file name : Unfiltered_occ_01.txt / Unfiltered_axial_01.txt
    fname_pre_filter = "".join(args.input_fname) if args.input_fname else 'Unfiltered_axial_04.txt'

    # filtered cut path file name : occ_01.txt / axial_01.txt
    fname_post_filter = "".join(args.output_fname) if args.output_fname else 'axial_04.txt'

    # filtered cut path file name [EtherCAT] : cutpath_o1.csv / cutpath_a1.csv
    fname_post_filter_ecat = "".join(args.output_eCAT_fname) if args.output_eCAT_fname else 'cutpath_a4.csv'

    # calibrated DH: alpha a theta d
    cal = "".join(args.calibration) if args.calibration else 'CAL00004.csv'

    # tool center point
    tcp = "".join(args.tcp) if args.tcp else 'TCP00008.csv'

    # q_init forward kinematics
    q_init = np.fromstring(args.q_init, count=6, sep=',') * pi / 180 if args.q_init else np.array(
        [-106.65362, 48.68948, -34.35517, -94.12345, -85.86414, 70.94483]) * pi / 180

    ##########################
    # MAIN FILTERING PROCESS #
    ##########################

    # load calibration and tcp model
    calibrated = np.loadtxt('result_calibration/data/%s' % cal, delimiter=',')
    TCP = np.loadtxt('data_tcp_calibration/data/%s' % tcp, delimiter=',')

    # robot base and tool information
    T_base = SE3()
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    # T_tool = SE3()

    # load pre-filtered trajectory
    traj_prefilt = np.loadtxt('data_filtering/unfiltered_cut_path/%s' % fname_pre_filter)

    # debug forward kinematics for initial joint angles
    crobot = myrobot.SerialLink(mdh=calibrated, T_base=T_base, T_tool=T_tool)
    debug = crobot.fkine(q_init)
    np.savetxt('data_filtering/debug_fk_dh.txt', debug, fmt='%.18f')

    # nominal DH: alpha a theta d
    # nominal = np.array([[0, 0, 0, 135],
    #                     [-pi / 2, 0, -pi / 2, 0],
    #                     [0, 135, 0, 0],
    #                     [-pi / 2, 38, 0, 120],
    #                     [pi / 2, 0, 0, 0],
    #                     [-pi / 2, 0, pi, 70]])
    nominal = np.loadtxt('result_calibration/dh_nominal.csv', delimiter=',')

    # filter trajectory
    T_filt, traj_filt, q_filt, success = dh_filter(nmdh=nominal, cmdh=calibrated, T_base=T_base, T_tool=T_tool,
                                                trajectory=traj_prefilt, q0=q_init)

    # EtherCAT format
    file_ecat = ecat_format(traj_filt, TCP)

    # check joint angles
    # print(q_filt * 180 / pi)

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.plot(traj_prefilt[:, 0], traj_prefilt[:, 1], traj_prefilt[:, 2], marker='.', label='pre_filtered')
    ax.plot(traj_filt[:, 0], traj_filt[:, 1], traj_filt[:, 2], marker='.', label='filtered_dh')
    ax.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # save result
    np.savetxt('data_filtering/filtered_cut_path/%s' % fname_post_filter, traj_filt, fmt='%.18f')

    # save result as EtherCAT format
    file_ecat.to_csv('data_filtering/filtered_cut_path_ecat/%s' % fname_post_filter_ecat, header=False, index=False)
