from spatialmath import SE3, SO3
import numpy as np
import math
from scipy.linalg import expm, logm


def skew(x):
    # skew of vector
    x_hat = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])
    return x_hat


def skew_mat(screw):
    # skew of matrix
    s_hat1 = np.column_stack((skew(screw[:3]), screw[3:]))
    s_hat2 = np.array([0, 0, 0, 0])
    s_hat = np.row_stack((s_hat1, s_hat2))
    return s_hat


def adjoint_mat(T):
    R = T[:3, :3]
    P = T[:3, 3]
    Adj_T1 = np.concatenate((R, np.zeros((3, 3))), axis=1)
    Adj_T2 = np.concatenate((np.matmul(skew(P), R), R), axis=1)
    Adj_T = np.concatenate((Adj_T1, Adj_T2), axis=0)
    return Adj_T


def adjoint_screw(s):
    adj_s1 = np.concatenate((skew(s[:3]), np.zeros((3, 3))), axis=1)
    adj_s2 = np.concatenate((skew(s[3:]), skew(s[:3])), axis=1)
    adj_s = np.concatenate((adj_s1, adj_s2), axis=0)
    return adj_s


def a_mat(screw, q):
    ohm = adjoint_screw(screw)
    w_abs = np.linalg.norm(screw[:3])
    theta = w_abs * q
    A = theta * np.eye(6) + (4 - theta * math.sin(theta) - 4 * math.cos(theta)) / (2 * w_abs ** 2) * ohm + \
        (4 * theta - 5 * math.sin(theta) + theta * math.cos(theta)) / (2 * w_abs ** 3) * np.linalg.matrix_power(ohm,
                                                                                                                2) + \
        (2 - theta * math.sin(theta) - 2 * math.cos(theta)) / (2 * w_abs ** 4) * np.linalg.matrix_power(ohm, 3) + \
        (2 * theta - 3 * math.sin(theta) + theta * math.cos(theta)) / (2 * w_abs ** 5) * np.linalg.matrix_power(ohm, 4)
    return A


def a_st_mat(screw):
    ohm = adjoint_screw(screw)
    theta = np.linalg.norm(screw[:3])
    A_st = np.eye(6) + (4 - theta * math.sin(theta) - 4 * math.cos(theta)) / (2 * theta ** 2) * ohm + \
           (4 * theta - 5 * math.sin(theta) + theta * math.cos(theta)) / (2 * theta ** 3) * np.linalg.matrix_power(ohm,
                                                                                                                   2) + \
           (2 - theta * math.sin(theta) - 2 * math.cos(theta)) / (2 * theta ** 4) * np.linalg.matrix_power(ohm, 3) + \
           (2 * theta - 3 * math.sin(theta) + theta * math.cos(theta)) / (2 * theta ** 5) * np.linalg.matrix_power(ohm,
                                                                                                                   4)
    return A_st


def id_jacobian_q(s, theta, s_st):
    # identification Jacobian matrix 6 by 7*6+6
    J = np.empty((6, 0))
    T = np.eye(4)
    Adj = adjoint_mat(T)
    for i, screw in enumerate(s):
        J_i = np.matmul(Adj, np.column_stack((a_mat(screw, theta[i]), screw)))
        J = np.append(J, J_i, axis=1)
        T = np.matmul(T, expm(skew_mat(screw) * theta[i]))
        Adj = adjoint_mat(T)
    J_st = np.matmul(Adj, a_st_mat(s_st))
    J = np.append(J, J_st, axis=1)
    return J

def id_jacobian(s, theta, s_st):
    # identification Jacobian matrix 6 by 6*6+6
    J = np.empty((6, 0))
    T = np.eye(4)
    Adj = adjoint_mat(T)
    for i, screw in enumerate(s):
        J_i = np.matmul(Adj, a_mat(screw, theta[i]))
        J = np.append(J, J_i, axis=1)
        T = np.matmul(T, expm(skew_mat(screw) * theta[i]))
        Adj = adjoint_mat(T)
    J_st = np.matmul(Adj, a_st_mat(s_st))
    J = np.append(J, J_st, axis=1)
    return J

def id_jacobians(s, thetas, s_st):
    # identification Jacobian matrix 6*N by 7*6+6
    Js = np.empty((0, 6 * 6 + 6))
    for i, theta in enumerate(thetas):
        J = id_jacobian(s, theta, s_st)
        Js = np.append(Js, J, axis=0)
    return Js


def forward_kinematics(s, theta, s_st):
    T = np.eye(4)
    for i, screw in enumerate(s):
        T = np.matmul(T, expm(skew_mat(screw) * theta[i]))
    T = np.matmul(T, expm(skew_mat(s_st)))
    return T


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # calibration data [q1 q2 ... q6 x y z]
    data = np.loadtxt('data_calibration/Cali_Points_Nornimal76_plus_Normal50.csv', delimiter=',')
    data[:, :6] = data[:, :6] * math.pi / 180  # degree to rad

    # frame setup
    T_base = np.loadtxt('data_setup/T_base.txt')
    T_tool = np.loadtxt('data_setup/T_tool.txt')
    P_data = np.column_stack((data[:, 6:9], np.ones(data.shape[0])))
    P_data = np.matmul(np.linalg.inv(T_base), P_data.transpose()).transpose()
    data[:, 6:9] = P_data[:, :3]

    # nominal screws [s1 s2 ... s6]
    l = np.array([135, 135, 38, 120, 70])
    w = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [1, 0, 0]])
    p = np.array([[0, 0, 0],
                  [0, 0, l[0]],
                  [0, 0, l[0] + l[1]],
                  [0, 0, l[0] + l[1] + l[2]],
                  [l[3], 0, l[0] + l[1] + l[2]],
                  [0, 0, l[0] + l[1] + l[2]]])
    s_init = np.concatenate((w, np.cross(p, w)), axis=1)
    s = s_init

    # initial M
    M = np.array(SE3.Rt(SO3.Ry(90, unit='deg'), np.array([l[3] + l[4], 0, l[0] + l[1] + l[2]])))
    M = np.matmul(M, T_tool)
    s_st_hat = logm(M)
    w_st = np.array([s_st_hat[2, 1], s_st_hat[0, 2], s_st_hat[1, 0]])
    v_st = np.array([s_st_hat[0, 3], s_st_hat[1, 3], s_st_hat[2, 3]])
    s_st_init = np.append(w_st, v_st)
    s_st = s_st_init

    # iteration
    x_norm = 1e3
    q = np.zeros(6)
    for j in range(100):
    # while x_norm > 1e-5:
        Pe_nominal = np.empty((0, 3))
        K = np.empty((0, 6 * 6 + 6))
        for i in range(data.shape[0]):
            # nominal FK
            T_nominal = forward_kinematics(s, data[i, :6] + q, s_st)
            Pe_nominal = np.row_stack((Pe_nominal, T_nominal[:3, 3]))
            # identification matrix J and K
            J = id_jacobian(s, data[i, :6] + q, s_st)
            K_PI = np.concatenate((-skew(T_nominal[:3, 3]), np.eye(3)), axis=1)
            K_i = np.matmul(K_PI, J)
            K = np.concatenate((K, K_i), axis=0)
        Pe_actual = data[:, 6:9]
        z = Pe_actual - Pe_nominal
        z = z.reshape((-1, 1))
        x = np.matmul(np.linalg.pinv(K), z)
        # LM = np.matmul(np.linalg.inv(np.matmul(K.transpose(),K) + 0.1*np.eye(48)), K.transpose())
        # x = np.matmul(LM, z)

        # update
        del_s = x[:6 * 6].reshape((-1, 6))
        # del_q = x[:6 * 6].reshape((-1, 7))[:, 6].squeeze()
        del_st = x[6 * 6:].squeeze()

        s = s + del_s
        s_st = s_st + del_st
        # q = q + del_q

        # conformance
        s = s / np.linalg.norm(s[:,:3], axis=1)[:, np.newaxis]
        for idx, screw in enumerate(s):
            v = screw[3:6]
            w = screw[:3]
            v_proj = np.inner(v,w)/np.inner(w,w)*w
            v_corrected = v-v_proj
            s[idx,3:6] = v_corrected
        s_st = s_st / np.linalg.norm(s_st[:3])
        v = s_st[3:6]
        w = s_st[:3]
        v_proj = np.inner(v, w) / np.inner(w, w) * w
        v_corrected = v - v_proj
        s_st[3:6] = v_corrected

        # evaluation
        x_norm = np.linalg.norm(x)
        err = np.linalg.norm(z)**2/data.shape[0]

        print('itr:', j)
        print('x norm:', x_norm)
        print('err:',np.sqrt(err))

    print('nominal: ', s_init)
    print('nominal: ', s_st_init)

    print('calib: ', s)
    print('calib: ', s_st)