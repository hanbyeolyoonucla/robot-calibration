from spatialmath import SE3, SO3, base
import numpy as np
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import argparse


# def skew(x):
#     # skew of vector
#     x_hat = np.array([[0, -x[2], x[1]],
#                       [x[2], 0, -x[0]],
#                       [-x[1], x[0], 0]])
#     return x_hat
#
#
# def adjoint_mat(T):
#     T = np.array(T)
#     R = T[:3, :3]
#     P = T[:3, 3]
#     Adj_T1 = np.concatenate((R, np.zeros((3, 3))), axis=1)
#     Adj_T2 = np.concatenate((np.matmul(skew(P), R), R), axis=1)
#     Adj_T = np.concatenate((Adj_T1, Adj_T2), axis=0)
#     return Adj_T


def space_screw(Tc, s_local):
    s_space = []
    T_idx = SE3()
    for idx in range(len(Tc) - 1):
        T_idx = T_idx * Tc[idx]
        s_idx = np.matmul(T_idx.Ad(), s_local)
        s_space.append(s_idx)
    return s_space  # [v w]


def space_jacobian(s_space, q):
    Js = []
    T_idx = SE3()
    for idx, screw in enumerate(s_space):
        Js.append(np.matmul(T_idx.Ad(), screw))
        T_idx = T_idx * SE3.Exp(screw * q[idx])
    return np.array(Js).transpose()


def poe_forward_kinematics(T_0, s_local, qs):
    T = SE3()
    for idx, q in enumerate(qs):
        T_i = T_0[idx] * SE3.Exp(s_local * q)
        T = T * T_i
    T = T * T_0[-1]
    return T


def check_joint_limit(q_check):
    q_ll = np.array([-175, -70, -135, -170, -115, -180])
    q_ul = np.array([175, 90, 70, 170, 115, 360])
    return np.all(np.array([q_check < q_ul, q_check > q_ll]))

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

def poe_filter(Tn_array, Tc_array, TCP, traj_prefilt, q_init):
    # POE nparray to SE(3)
    Tc = []
    Tn = []
    for idx in range(6):
        Tc.append(SE3(Tc_array[4 * idx:4 * idx + 4, :]))
        Tn.append(SE3(Tn_array[4 * idx:4 * idx + 4, :]))
    # tool information
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    Tc.append(T_tool)
    Tn.append(T_tool)

    # filtering iteration
    # nominal local screws for revolute joints [v w]
    s_local = np.array([0, 0, 0, 0, 0, 1])
    s_space = space_screw(Tc, s_local)

    # newton-ralphson inverse kinematics
    eps_w = 1e-7
    eps_v = 1e-7
    step = 0.5
    max_itr = 500
    # debug = poe_forward_kinematics(Tc, s_local, q_init)
    # np.savetxt('data_postfilter/debug_fk_poe.txt', debug, fmt='%.18f')

    q_filt = []
    traj_filt = []
    for idx, xyzrpy in enumerate(traj_prefilt):
        Tsd = SE3.Rt(SO3.RPY(np.flip(xyzrpy[3:6]), unit='deg', order='xyz'), xyzrpy[0:3])
        q = q_init
        # iteration
        itr = 1
        w_norm = 1
        v_norm = 1
        while w_norm > eps_w and v_norm > eps_v:
            Tsb = SE3(poe_forward_kinematics(Tc, s_local, q))
            errT = Tsb.inv() * Tsd
            Vb_mat = base.trlog(errT.data[0], check=False)
            wb = np.array([Vb_mat[2, 1], Vb_mat[0, 2], Vb_mat[1, 0]])
            vb = Vb_mat[:3, 3]
            w_norm = np.linalg.norm(wb)
            v_norm = np.linalg.norm(vb)
            Vb = np.append(vb, wb)
            Js = space_jacobian(s_space, q)
            Jb = np.matmul(Tsb.inv().Ad(), Js)
            q = q + step * np.matmul(np.linalg.pinv(Jb), Vb)
            itr = itr + 1
            assert itr < max_itr, 'IK does not exist!'
        assert check_joint_limit(q), 'Exceed Joint Limit!'
        q_filt.append(q)
        T_filt = SE3(poe_forward_kinematics(Tn, s_local, q))
        traj_filt.append(np.append(T_filt.t, np.flip(T_filt.rpy(unit='deg', order='xyz'))))

    return np.array(traj_filt), np.array(q_filt)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ###################################
    # ARGUMENTS FOR FILTERING PROCESS #
    ###################################

    # arg parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_fname", help="input file name: ex) Unfiltered_axial_01.txt")
    argParser.add_argument("-o", "--output_fname", help="output file name: ex) axial_01.txt")
    argParser.add_argument("-oe", "--output_eCAT_fname", help="output eCAT file name: ex) cutpath_a1.csv")
    argParser.add_argument("-c", "--calibration", help="calibration model: ex) CAL00001.csv")
    argParser.add_argument("-t", "--tcp", help="tcp model: ex) TCP0001.csv")
    argParser.add_argument("-q", "--q_init", help="initial joint angle: ex) 0,0,0,0,0,0")
    args = argParser.parse_args()

    # unfiltered cut path file name : Unfiltered_occ_01.txt / Unfiltered_axial_01.txt
    fname_pre_filter = "".join(args.input_fname) if args.input_fname else 'Unfiltered_axial_01.txt'

    # filtered cut path file name : occ_01.txt / axial_01.txt
    fname_post_filter = "".join(args.output_fname) if args.output_fname else 'axial_01.txt'

    # filtered cut path file name [EtherCAT] : cutpath_o1.csv / cutpath_a1.csv
    fname_post_filter_ecat = "".join(args.output_eCAT_fname) if args.output_eCAT_fname else 'cutpath_a1.csv'

    # calibrated DH: alpha a theta d
    cal = "".join(args.calibration) if args.calibration else 'CAL00003.csv'

    # tool center point
    tcp = "".join(args.tcp) if args.tcp else 'TCP00009.csv'

    # q_init forward kinematics
    q_init = np.fromstring(args.q_init, count=6, sep=',') * pi / 180 if args.q_init else np.array(
        [13.293103,10.183966,0.131379,-118.633448,-80.442414,107.75431]) * pi / 180

    ##########################
    # MAIN FILTERING PROCESS #
    ##########################

    # load pre-filtered trajectory
    traj_prefilt = np.loadtxt('data_filtering/unfiltered_cut_path/%s' % fname_pre_filter)

    # load calibrated / nominal POE model
    Tc_array = np.loadtxt('result_calibration/data/%s' % cal, delimiter=',')
    Tn_array = np.loadtxt('result_calibration/poe_local_nominal.csv', delimiter=',')

    # load tcp information
    TCP = np.loadtxt('data_tcp_calibration/data/%s' % tcp, delimiter=',')

    # filter trajectory
    traj_filt, q_filt = poe_filter(Tn_array, Tc_array, TCP, traj_prefilt, q_init)

    # EtherCAT format
    file_ecat = ecat_format(traj_filt, TCP)

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.plot(traj_prefilt[:, 0], traj_prefilt[:, 1], traj_prefilt[:, 2], marker='.', label='pre_filtered')
    ax.plot(traj_filt[:, 0], traj_filt[:, 1], traj_filt[:, 2], marker='.', label='filtered')
    ax.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # save result
    np.savetxt('data_filtering/filtered_cut_path/%s' % fname_post_filter, traj_filt, fmt='%.18f')
    file_ecat.to_csv('data_filtering/filtered_cut_path_ecat/%s' % fname_post_filter_ecat, header=False, index=False)
