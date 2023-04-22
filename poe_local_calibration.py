from spatialmath import SE3, SO3
import numpy as np
from math import pi
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


def id_jacobian(T_0):
    # identification Jacobian matrix 6 by 6*6+6
    A = np.empty((6, 0))
    T = SE3()
    for T_i in T_0:
        Adj = np.matmul(adjoint_mat(T), adjoint_mat(T_i))
        A = np.concatenate((A, Adj), axis=1)
        T = T * T_i
    return A[:,6:]


def forward_kinematics(T_0, s_local, qs):
    T = SE3()
    for idx, q in enumerate(qs):
        T_i = T_0[idx] * SE3.Exp(s_local * q)
        T = T * T_i
    T = T * T_0[-1]
    return T


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # calibration data [q1 q2 ... q6 x y z]
    data = np.loadtxt('data_calibration/Cali_Points_Nornimal76_plus_Normal50.csv', delimiter=',')
    data[:, :6] = data[:, :6] * pi / 180  # degree to rad

    # frame setup
    T_base = np.loadtxt('data_setup/T_base.txt')
    T_tool = np.loadtxt('data_setup/T_tool.txt')
    P_data = np.column_stack((data[:, 6:9], np.ones(data.shape[0])))
    P_data = np.matmul(np.linalg.inv(T_base), P_data.transpose()).transpose()
    data[:, 6:9] = P_data[:, :3]

    # nominal local frame [T01 T12 ... T56]
    l = np.array([135, 135, 38, 120, 70])
    T_0 = []
    T_0.append(SE3.Rt(np.eye(3), np.zeros(3)))  # T01
    T_0.append(SE3.Rt(SO3.Rx(-pi / 2), np.array([0, 0, l[0]])))  # T12
    T_0.append(SE3.Rt(np.eye(3), np.array([0, -l[1], 0])))  # T23
    T_0.append(SE3.Rt(SO3.Ry(pi / 2), np.array([0, -l[2], 0])))  # T34
    T_0.append(SE3.Rt(SO3.Ry(-pi / 2), np.array([0, 0, l[3]])))  # T45
    T_0.append(SE3.Rt(SO3.RPY(np.array([pi / 2, 0, pi / 2])), np.array([l[4], 0, 0])))  # T56
    T_0.append(SE3(T_tool))

    # calibrated local frame
    Tc_0 = T_0

    # nominal local screws for revolute joints
    s_local = np.array([0, 0, 0, 0, 0, 1])

    # iteration
    x_norm = 1e3
    for j in range(100):
        # while x_norm > 1e-5:
        Pe_nominal = np.empty((0, 3))
        K = np.empty((0, 6 * 6))
        for i in range(data.shape[0]):
            # nominal FK
            T_nominal = forward_kinematics(Tc_0, s_local, data[i, :6])
            Pe_nominal = np.row_stack((Pe_nominal, T_nominal.t))

            # identification matrix A and K
            A = id_jacobian(Tc_0)
            K_PI = np.concatenate((-skew(T_nominal.t), np.eye(3)), axis=1)
            K_i = np.matmul(K_PI, A)
            K = np.concatenate((K, K_i), axis=0)
        Pe_actual = data[:, 6:9]
        z = Pe_actual - Pe_nominal
        z = z.reshape((-1, 1))
        x = np.matmul(np.linalg.pinv(K), z)
        # LM = np.matmul(np.linalg.inv(np.matmul(K.transpose(),K) + 0.1*np.eye(42)), K.transpose())
        # x = np.matmul(LM, z)

        # update
        del_p = x.reshape((-1, 6))
        for idx, del_p_i in enumerate(del_p):
            Tc_0[idx] = Tc_0[idx] * SE3.Exp(del_p_i)

        # evaluation
        x_norm = np.linalg.norm(x)
        err = np.linalg.norm(z)**2/data.shape[0]
        print('itr:', j)
        print('x norm:', x_norm)
        print('err:',np.sqrt(err))

