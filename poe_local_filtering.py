from spatialmath import SE3, SO3, base
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm


def skew(x):
    # skew of vector
    x_hat = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])
    return x_hat


def adjoint_mat(T):
    T = np.array(T)
    R = T[:3, :3]
    P = T[:3, 3]
    Adj_T1 = np.concatenate((R, np.zeros((3, 3))), axis=1)
    Adj_T2 = np.concatenate((np.matmul(skew(P), R), R), axis=1)
    Adj_T = np.concatenate((Adj_T1, Adj_T2), axis=0)
    return Adj_T


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


def forward_kinematics(T_0, s_local, qs):
    T = SE3()
    for idx, q in enumerate(qs):
        T_i = T_0[idx] * SE3.Exp(s_local * q)
        T = T * T_i
    T = T * T_0[-1]
    return T


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # load pre-filtered trajectory
    traj_prefilt = np.loadtxt('data_prefilter/unfiltered_test_points_poe.txt')
    # traj_prefilt = np.loadtxt('data_prefilter/unfiltered_cut_path/Unfiltered_occ_01.txt')


    # calibrated / nominal POE
    Tc_array = np.loadtxt('result_calibration/poe_local_calib.csv', delimiter=',')
    Tn_array = np.loadtxt('result_calibration/poe_local_nominal.csv', delimiter=',')
    Tc = []
    Tn = []
    for idx in range(6):
        Tc.append(SE3(Tc_array[4 * idx:4 * idx + 4, :]))
        Tn.append(SE3(Tn_array[4 * idx:4 * idx + 4, :]))

    # tool information
    TCP = np.array([-5.986996235,	-0.1226059,	145.9501981,	174.8557519,	85.66877675,	4.928643451])  # Hecate TCP
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    # T_tool = SE3()
    # T_tool = SE3(base.trnorm(np.array([[0.07692182,	-0.00844348, 0.99700137, -6.00748941],
    #                                    [0.00279028, -0.9999584, -0.0086838, -0.16843419],
    #                                    [0.99703322, 0.00344989, -0.07689506, 146.0180594],
    #                                    [0, 0, 0, 1]])))

    Tc.append(T_tool)
    Tn.append(T_tool)

    # nominal local screws for revolute joints [v w]
    s_local = np.array([0, 0, 0, 0, 0, 1])

    # space screw
    s_space = space_screw(Tc, s_local)

    # newton-ralphson inverse kinematics
    eps_w = 1e-7
    eps_v = 1e-7
    step = 0.5
    q_init = np.array([8.527759,38.374914,3.170948,-99.838448,-97.556897,76.537931]) * pi / 180

    debug = forward_kinematics(Tc,s_local,q_init)

    # q = q_init
    q_ik = []
    traj_filt = []
    for idx, xyzrpy in enumerate(traj_prefilt):
        print('\n\n iteration: ', idx)
        Tsd = SE3.Rt(SO3.RPY(np.flip(xyzrpy[3:6]), unit='deg', order='xyz'), xyzrpy[0:3])
        q = q_init
        # iteration
        w_norm = 1
        v_norm = 1
        while w_norm > eps_w and v_norm > eps_v:
            Tsb = SE3(forward_kinematics(Tc, s_local, q))
            errT = Tsb.inv() * Tsd
            print('check ishom: ', base.ishom(np.array(errT), check=True, tol=100))
            print('check err mat: \n', errT)
            # Vb_mat = errT.log()
            Vb_mat = base.trlog(errT.data[0], check=False)
            wb = np.array([Vb_mat[2, 1], Vb_mat[0, 2], Vb_mat[1, 0]])
            vb = Vb_mat[:3, 3]
            w_norm = np.linalg.norm(wb)
            v_norm = np.linalg.norm(vb)
            Vb = np.append(vb, wb)
            Js = space_jacobian(s_space, q)
            Jb = np.matmul(Tsb.inv().Ad(), Js)
            q = q + step * np.matmul(np.linalg.pinv(Jb), Vb)
        q_ik.append(q)
        T_filt = SE3(forward_kinematics(Tn, s_local, q))
        traj_filt.append(np.append(T_filt.t, np.flip(T_filt.rpy(unit='deg', order='xyz'))))
    q_ik = np.array(q_ik)
    traj_filt = np.array(traj_filt)

    # check deviation due to filtering and result
    # print(np.mean(traj_filt[:, 0:3] - traj_prefilt[:, 0:3], axis=0))
    print(traj_filt[:,:3])
    print(traj_filt[:,3:])
    print(q_ik)

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
    np.savetxt('data_postfilter/filtered_test_points_poe.txt', traj_filt)
    # np.savetxt('data_postfilter/filtered_cut_path/occ_01.txt', traj_filt)

