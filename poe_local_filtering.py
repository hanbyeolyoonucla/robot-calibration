from spatialmath import SE3, SO3
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
    traj_prefilt = np.loadtxt('data_prefilter/typo_1.2A-8/Unfiltered_axial_01.txt')

    # calibration result local T 7*4 x 4
    Tc_array = np.loadtxt('result_calibration/poe_local_calib.txt')
    # nominal local T 7*4 x 4
    Tn_array = np.loadtxt('result_calibration/poe_local_nominal.txt')
    Tc = []
    Tn = []
    for idx in range(6):
        Tc.append(SE3(Tc_array[4 * idx:4 * idx + 4, :]))
        Tn.append(SE3(Tn_array[4 * idx:4 * idx + 4, :]))

    # tool information
    TCP = np.array([-0.7041735503, -0.3506146737, 158.0632991, -153.5539579, 85.22858361, -26.93566677])
    T_tool = SE3.Trans(TCP[0:3]) * SE3.RPY(np.flip(TCP[3:6]), unit='deg', order='xyz')
    Tc.append(T_tool)
    Tn.append(T_tool)

    # nominal local screws for revolute joints [v w]
    s_local = np.array([0, 0, 0, 0, 0, 1])

    # space screw
    s_space = space_screw(Tc, s_local)

    # newton-ralphson inverse kinematics
    eps_w = 1e-6
    eps_v = 1e-6
    q_init = np.array([5.583879, 14.467759, 37.938621, -88.654655, -87.679138, 25.230172]) * pi / 180
    q_ik = []
    traj_filt = []
    for idx, xyzrpy in enumerate(traj_prefilt):
        Tsd = SE3.Rt(SO3.RPY(np.flip(xyzrpy[3:6]), unit='deg', order='xyz'), xyzrpy[0:3])
        # iteration
        q = q_init
        w_norm = 1
        v_norm = 1
        while w_norm > eps_w and v_norm > eps_v:
            Tsb = SE3(forward_kinematics(Tc, s_local, q))
            errT = Tsb.inv() * Tsd
            Vb_mat = errT.log()
            wb = np.array([Vb_mat[2, 1], Vb_mat[0, 2], Vb_mat[1, 0]])
            vb = Vb_mat[:3, 3]
            w_norm = np.linalg.norm(wb)
            v_norm = np.linalg.norm(vb)
            Vb = np.append(vb,wb)
            Js = space_jacobian(s_space, q)
            Jb = np.matmul(Tsb.inv().Ad() , Js)
            q = q + np.matmul(np.linalg.pinv(Jb), Vb)
        q_ik.append(q)
        T_filt = SE3(forward_kinematics(Tn, s_local, q))
        traj_filt.append(np.append(T_filt.t, np.flip(T_filt.rpy(unit='deg', order='xyz'))))
    q_ik = np.array(q_ik)
    traj_filt = np.array(traj_filt)

    # check deviation due to filtering
    print(np.mean(traj_filt[:, 0:3] - traj_prefilt[:, 0:3], axis=0))

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
