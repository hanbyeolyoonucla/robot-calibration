# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from spatialmath import SE3
from math import pi
import numpy as np
import myrobot
import matplotlib.pyplot as plt
import argparse
from poe_local_filtering import poe_filter, ecat_format
from cut_path_registration import registration
from dh_filtering import dh_filter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ######################################
    # ARGUMENTS FOR REGISTRATION PROCESS #
    ######################################

    # arg parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_fname", help="input file name: ex) new_axial_path_1.txt")
    argParser.add_argument("-o", "--output_fname", help="output file name: ex) Unfiltered_axial_01.txt")
    argParser.add_argument("-oe", "--output_eCAT_fname", help="output eCAT file name: ex) cutpath_a1.csv")
    argParser.add_argument("-c", "--calibration", help="calibration model: ex) CAL00001.csv")
    argParser.add_argument("-t", "--tcp", help="tcp model: ex) TCP0001.csv")
    argParser.add_argument("-q", "--q_init", help="initial joint angle: ex) 0,0,0,0,0,0")
    args = argParser.parse_args()

    # pre-registered cut path file name : new_axial_path_1.txt / new_occlusal_path.txt
    fname_unreg = "".join(args.input_fname) if args.input_fname else 'new_axial_path_4.csv'

    # filtered cut path file name : occ_01.txt / axial_01.txt
    fname_post_filter = "".join(args.output_fname) if args.output_fname else 'axial_04.txt'

    # filtered cut path file name [EtherCAT] : cutpath_o1.csv / cutpath_a1.csv
    fname_post_filter_ecat = "".join(args.output_eCAT_fname) if args.output_eCAT_fname else 'cutpath_a4.csv'

    # calibrated DH: alpha a theta d
    cal = "".join(args.calibration) if args.calibration else 'CAL00002.csv'

    # tool center point
    tcp = "".join(args.tcp) if args.tcp else 'TCP00008.csv'

    # q_init forward kinematics
    q_init = np.fromstring(args.q_init, count=6, sep=',') * pi / 180 if args.q_init else np.array(
        [16.555345, -12.135259, 38.147328, -95.963793, -61.101207, 116.599138]) * pi / 180

    #########################################
    # MAIN REGISTRATION & FILTERING PROCESS #
    #########################################

    # load unregistered cut path
    unreg_cut_path_pos = np.loadtxt('data_registration/unregistered_cut_path/%s' % fname_unreg, delimiter=',')
    unreg_cut_path_ori = np.tile(np.array([180, 0, (180 - 10)]), (unreg_cut_path_pos.shape[0], 1))
    unreg_cut_path = np.column_stack((unreg_cut_path_pos, unreg_cut_path_ori))

    # load registration matrix
    T_tb_tip = np.loadtxt('data_registration/registration_matrix/T_tb_tip.csv', delimiter=',')
    T_tb_tip = T_tb_tip[1:, :]  # skip header

    # load robot information
    CAL = np.loadtxt('result_calibration/data/%s' % cal, delimiter=',')
    TCP = np.loadtxt('data_tcp_calibration/data/%s' % tcp, delimiter=',')

    # register cut path
    post_registered_cut_path, T_r_tb = registration(T_tb_tip, CAL, TCP, q_init, unreg_cut_path)

    # filter cut path
    if CAL.shape[0] == 6:  # DH
        nominal = np.loadtxt('result_calibration/dh_nominal.csv', delimiter=',')
        T_filt, traj_filt, q_filt, success = dh_filter(nmdh=nominal, cmdh=CAL, TCP=TCP,
                                                       trajectory=post_registered_cut_path, q_init=q_init)
    else:  # POE
        Tn_array = np.loadtxt('result_calibration/poe_local_nominal.csv', delimiter=',')
        traj_filt, q_filt = poe_filter(Tn_array=Tn_array, Tc_array=CAL, TCP=TCP, traj_prefilt=post_registered_cut_path,
                                       q_init=q_init)

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.plot(post_registered_cut_path[:, 0], post_registered_cut_path[:, 1], post_registered_cut_path[:, 2], marker='.', label='pre_filtered')
    ax.plot(traj_filt[:, 0], traj_filt[:, 1], traj_filt[:, 2], marker='.', label='filtered')
    ax.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig('data_filtering/filtered_cut_path/%s.png' % fname_post_filter[:-4])
    plt.show()

    # EtherCAT format
    file_ecat = ecat_format(traj_filt, TCP)

    # save result
    np.savetxt('data_registration/registration_matrix/T_r_tb.csv', np.array(T_r_tb), fmt='%.18f')
    np.savetxt('data_filtering/filtered_cut_path/%s' % fname_post_filter, traj_filt, fmt='%.18f')
    file_ecat.to_csv('data_filtering/filtered_cut_path_ecat/%s' % fname_post_filter_ecat, header=False, index=False)