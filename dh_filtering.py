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


def filter(nmdh, cmdh, T_base, T_tool, trajectory, q0):
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


def ecat_format(traj_filt, TCP):
    # trajectory command info: 3 = MoveLine
    meca_id = 3 * np.ones(traj_filt.shape[0], dtype=np.uintp)
    check_pt = 1 * np.ones(traj_filt.shape[0], dtype=np.uintp)
    file_ecat = pd.DataFrame(traj_filt)
    file_ecat.insert(0, 'id', meca_id.copy().tolist())
    file_ecat.insert(1, 'check', check_pt.copy().tolist())
    file_ecat.columns = pd.RangeIndex(file_ecat.columns.size)

    # Header: setTRF, setWRF, id info
    settrf = np.array([13, 0] + TCP.copy().tolist(), dtype=object)
    setwrf = [14, 0, 0, 0, 0, 0, 0, 0]
    id_config = ['123456ABCDER', 'DC2300', '', '', '', '', '', '']
    file_ecat = pd.concat([pd.DataFrame(settrf).transpose(), file_ecat.loc[:]]).reset_index(drop=True)
    file_ecat = pd.concat([pd.DataFrame(setwrf).transpose(), file_ecat.loc[:]]).reset_index(drop=True)
    file_ecat = pd.concat([pd.DataFrame(id_config).transpose(), file_ecat.loc[:]]).reset_index(drop=True)

    return file_ecat


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
    cal = "".join(args.calibration) if args.calibration else 'CAL00002.csv'

    # tool center point
    tcp = "".join(args.tcp) if args.tcp else 'TCP00008.csv'

    # q_init forward kinematics
    q_init = np.fromstring(args.q_init, count=6, sep=',') * pi / 180 if args.q_init else np.array([11.263707, -1.7625, 32.841983, -96.902586, -71.128966, 98.931034]) * pi / 180

    ##########################
    # MAIN FILTERING PROCESS #
    ##########################

    # load calibration and tcp model
    calibrated = np.loadtxt('result_calibration/data/%s' % cal, delimiter=',')
    TCP = np.loadtxt('data_tcp_calibration/data/%s' % tcp, delimiter=',')

    # robot base and tool information
    T_base = SE3()
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')

    # load pre-filtered trajectory
    traj_prefilt = np.loadtxt('data_prefilter/unfiltered_cut_path/%s' % fname_pre_filter)

    # debug forward kinematics for initial joint angles
    crobot = myrobot.SerialLink(mdh=calibrated, T_base=T_base, T_tool=T_tool)
    debug = crobot.fkine(q_init)
    # np.savetxt('data_postfilter/debug_fk_dh.txt', debug, fmt='%.18f')

    # nominal DH: alpha a theta d
    nominal = np.array([[0, 0, 0, 135],
                        [-pi / 2, 0, -pi / 2, 0],
                        [0, 135, 0, 0],
                        [-pi / 2, 38, 0, 120],
                        [pi / 2, 0, 0, 0],
                        [-pi / 2, 0, pi, 70]])

    # filter trajectory
    T_filt, traj_filt, q_filt, success = filter(nmdh=nominal, cmdh=calibrated, T_base=T_base, T_tool=T_tool,
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
    np.savetxt('data_postfilter/filtered_cut_path/%s' % fname_post_filter, traj_filt, fmt='%.18f')

    # save result as EtherCAT format
    file_ecat.to_csv('data_postfilter/filtered_cut_path_ecat/%s' % fname_post_filter_ecat, header=False, index=False)