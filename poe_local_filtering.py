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
    # traj_prefilt = np.loadtxt('data_prefilter/typo_1.2A-8/Unfiltered_axial_04.txt')
    # traj_prefilt = np.array([[115.268524, 215.479918, 269.881699, -168.89405, 28.67713, -92.718579],
    #                          [115.268524, 215.479918, 269.881699, -168.464767, 38.643218, -92.210535],
    #                          [115.268524, 215.479918, 269.881699, -168.092169, 48.611554, -91.721678],
    #                          [115.268524, 215.479918, 269.881699, -169.383924, 18.715117, -93.267138],
    #                          [115.268524, 215.479918, 269.881699, -169.949611, 8.758066, -93.878949],
    #                          [115.268524, 215.479918, 269.881699, -170.614948, -1.191798, -94.581214],
    #                          [115.268524, 215.479918, 269.881699, -171.416318, -11.132485, -95.414893],
    #                          [115.268524, 215.479918, 269.881699, -172.412346, -21.058653, -96.438497],
    #                          [115.268524, 215.479918, 269.881699, -173.702557, -30.963459, -97.753099],
    #                          [115.268524, 215.479918, 269.881699, -175.463549, -40.833719, -99.536204]])
    traj_prefilt = np.array([[117.146989,	136.997104,	158.312049,	-89.452941, -1.016995, 122.752537],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 112.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 142.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 152.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 162.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 172.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, -177.241868],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, -167.241868],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, -157.241868],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, -147.241868],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 112.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 102.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 92.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 82.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 72.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 62.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 52.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 42.758132],
                             [117.146989, 136.997104, 158.312049, -89.453776, -1.016995, 32.758132]])

    # calibration result local T 7*4 x 4
    Tc_array = np.loadtxt('result_calibration/poe_local_calib.txt')
    # nominal local T 7*4 x 4
    Tn_array = np.loadtxt('result_calibration/poe_local_nominal.txt')
    Tc = []
    Tn = []
    for idx in range(7):
        Tc.append(SE3(Tc_array[4 * idx:4 * idx + 4, :]))
        Tn.append(SE3(Tn_array[4 * idx:4 * idx + 4, :]))

    # tool information
    # TCP = np.array([-0.7041735503, -0.3506146737, 158.0632991, -153.5539579, 85.22858361, -26.93566677])
    # TCP = np.array([-0.6876713825, -0.2276940919, 158.1200114, 165.9487774, 85.77018622, 13.60255658])
    # T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    # Tc.append(T_tool)
    # Tn.append(T_tool)

    # nominal local screws for revolute joints [v w]
    s_local = np.array([0, 0, 0, 0, 0, 1])

    # space screw
    s_space = space_screw(Tc, s_local)

    # newton-ralphson inverse kinematics
    eps_w = 1e-7
    eps_v = 1e-7
    step = 0.5
    q_init = np.array([4.038621, 37.282241, 21.044741, -95.307931, -100.296207, 61.309483]) * pi / 180
    q_init = np.array([5.536484, 13.208381, 45.997117, -85.836122, -88.158258, 63.448787]) * pi / 180
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
    print(traj_filt)
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

    # # save result
    # np.savetxt('data_postfilter/typo_1.2A-8/occ_01.txt', traj_filt)
