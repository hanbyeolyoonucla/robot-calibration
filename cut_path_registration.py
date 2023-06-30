# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import roboticstoolbox as rtb
from spatialmath import SE3, base
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import myrobot
import argparse
from poe_local_filtering import poe_forward_kinematics

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ######################################
    # ARGUMENTS FOR REGISTRATION PROCESS #
    ######################################

    # arg parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_fname", help="input file name: ex) new_axial_path_1.txt")
    argParser.add_argument("-o", "--output_fname", help="output file name: ex) Unfiltered_axial_01.txt")
    argParser.add_argument("-c", "--calibration", help="calibration model: ex) CAL00001.csv")
    argParser.add_argument("-t", "--tcp", help="tcp model: ex) TCP0001.csv")
    argParser.add_argument("-q", "--q_init", help="initial joint angle: ex) 0,0,0,0,0,0")
    args = argParser.parse_args()

    # unregistered cut path file name : new_axial_path_1.txt / new_occlusal_path.txt
    fname_unreg = "".join(args.input_fname) if args.input_fname else 'new_axial_path_1.csv'

    # registered (=unfiltered) cut path file name : Unfiltered_axial_01.txt / Unfiltered_occ_01.txt
    fname_reg = "".join(args.output_fname) if args.output_fname else 'Unfiltered_axial_011.csv'

    # calibrated DH: alpha a theta d
    cal = "".join(args.calibration) if args.calibration else 'CAL00001.csv'

    # tool center point
    tcp = "".join(args.tcp) if args.tcp else 'TCP00009.csv'

    # q_init forward kinematics
    q_init = np.fromstring(args.q_init, count=6, sep=',') * pi / 180 if args.q_init else np.array(
        [8.680086, 20.944397, -5.848966, -112.971724, -89.514828, 113.590517]) * pi / 180

    #############################
    # MAIN REGISTRATION PROCESS #
    #############################

    # load unregistered cut path
    unreg_cut_path_pos = np.loadtxt('data_registration/unregistered_cut_path/%s' % fname_unreg, delimiter=',')
    unreg_cut_path_ori = np.tile(np.array([0, pi, -15 * pi / 180]), (unreg_cut_path_pos.shape[0], 1))
    unreg_cut_path = np.column_stack((unreg_cut_path_pos, unreg_cut_path_ori))

    # load registration matrix
    T_tb_tip = np.loadtxt('data_registration/registration_matrix/T_tb_tip.csv', delimiter=',')
    T_tb_tip = T_tb_tip[1:, :]  # skip header
    T_tb_tip = SE3(T_tb_tip)  # np array to SE3

    # load robot information
    CAL = np.loadtxt('result_calibration/data/%s' % cal, delimiter=',')
    TCP = np.loadtxt('data_tcp_calibration/data/%s' % tcp, delimiter=',')
    T_base = SE3()
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')

    # compute forward kinematics
    if CAL.shape[0] == 6:  # DH
        crobot = myrobot.SerialLink(mdh=CAL, T_base=T_base, T_tool=T_tool)
        T_r_tip = SE3(crobot.fkine(q_init))
    else:  # POE
        Tc = []
        for idx in range(6):
            Tc.append(SE3(CAL[4 * idx:4 * idx + 4, :]))
        Tc.append(T_tool)
        s_local = np.array([0, 0, 0, 0, 0, 1])
        T_r_tip = SE3(poe_forward_kinematics(Tc, s_local, q_init))

    T_r_tb = T_r_tip * T_tb_tip.inv()

    traj_registered = []
    for idx, via_point in enumerate(unreg_cut_path):
        unreg_T = SE3.Trans(via_point[0:3]) * SE3.RPY(np.flip(via_point[3:6]), unit='deg', order='xyz')
        reg_T = T_r_tb * unreg_T
        traj_registered.append(np.append(reg_T.t, np.flip(reg_T.rpy(unit='deg', order='xyz'))))

    np.savetxt('data_filtering/unfiltered_cut_path/%s' % fname_reg, np.array(traj_registered), fmt='%.18f')



