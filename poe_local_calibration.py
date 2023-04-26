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


def id_jacobian(T_0, s_local, q):
    # identification Jacobian matrix 6 by 6*6+6
    A = np.empty((6, 0))
    T = SE3()
    for idx, q_i in enumerate(q):
        debug = T.Ad()
        debug2 = adjoint_mat(T)
        Adj = np.matmul(adjoint_mat(T), adjoint_mat(T_0[idx]))
        A = np.concatenate((A, Adj), axis=1)
        T = T * T_0[idx] * SE3.Exp(s_local*q_i)
    Adj = np.matmul(adjoint_mat(T), adjoint_mat(T_0[idx+1]))
    A = np.concatenate((A, Adj), axis=1)
    T = T * T_0[idx+1]
    return A


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
    # data_cal = np.loadtxt('data_calibration/Cali_Points_Nornimal76_plus_Normal50.csv', delimiter=',')
    data1 = np.loadtxt('data_calibration/cal_data/uniform_dist_data_0deg_160cube_Output.csv', delimiter=',')
    data2 = np.loadtxt('data_calibration/cal_data/norm_dist_data_0deg_80cube_Output.csv', delimiter=',')
    data3 = np.loadtxt('data_calibration/cal_data/norm_dist_data_0deg_160cube_Output.csv', delimiter=',')
    data4 = np.loadtxt('data_calibration/cal_data/norm_dist_data_90deg_80cube_Output.csv', delimiter=',')
    data5 = np.loadtxt('data_calibration/cal_data/norm_dist_data_90deg_160cube_Output.csv', delimiter=',')
    data6 = np.loadtxt('data_calibration/cal_data/norm_dist_data_n90_80cube_Output.csv', delimiter=',')
    data7 = np.loadtxt('data_calibration/cal_data/norm_dist_data_n90_160cube_Output.csv', delimiter=',')
    data_cal = np.concatenate((data1,data2,data3,data4,data5,data6,data7),axis=0)
    # data_cal = data7
    data_cal[:, 6:9] = data_cal[:, 6:9] * 1000  # m to mm
    data_cal[:, :6] = data_cal[:, :6] * pi / 180  # degree to rad

    # validation data [q1 q2 ... q6 x y z]
    # data_val = np.loadtxt('data_calibration/Test_Points_Normal.csv', delimiter=',')
    data_val = np.loadtxt('data_calibration/val_data/uniform_dist_data_0deg_160cube_Output.csv', delimiter=',')
    # data_val = np.loadtxt('data_calibration/val_data/norm_dist_data_0deg_80cube_Output.csv', delimiter=',')
    # data_val = np.loadtxt('data_calibration/val_data/norm_dist_data_0deg_160cube_Output.csv', delimiter=',')
    # data_val = np.loadtxt('data_calibration/val_data/norm_dist_data_90deg_80cube_Output.csv', delimiter=',')
    # data_val = np.loadtxt('data_calibration/val_data/norm_dist_data_90deg_160cube_Output.csv', delimiter=',')
    # data_val = np.loadtxt('data_calibration/val_data/norm_dist_data_n90_80cube_Output.csv', delimiter=',')
    # data_val = np.loadtxt('data_calibration/val_data/norm_dist_data_n90_160cube_Output.csv', delimiter=',')
    data_val[:, 6:9] = data_val[:, 6:9] * 1000  # m to mm
    data_val[:, :6] = data_val[:, :6] * pi / 180  # degree to rad

    # frame setup
    T_base = np.loadtxt('data_setup/T_base.txt')
    T_tool = np.loadtxt('data_setup/T_tool.txt')

    # transform data to robot base
    P_data = np.column_stack((data_cal[:, 6:9], np.ones(data_cal.shape[0])))
    P_data = np.matmul(np.linalg.inv(T_base), P_data.transpose()).transpose()
    data_cal[:, 6:9] = P_data[:, :3]
    P_data = np.column_stack((data_val[:, 6:9], np.ones(data_val.shape[0])))
    P_data = np.matmul(np.linalg.inv(T_base), P_data.transpose()).transpose()
    data_val[:, 6:9] = P_data[:, :3]

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
    itr = 0
    # for j in range(100):
    while x_norm > 1e-12:
        Pe_nominal = np.empty((0, 3))
        K = np.empty((0, 6 * 6 + 6))
        for i in range(data_cal.shape[0]):
            # nominal FK
            T_nominal = forward_kinematics(Tc_0, s_local, data_cal[i, :6])
            Pe_nominal = np.row_stack((Pe_nominal, T_nominal.t))

            # identification matrix A and K
            A = id_jacobian(Tc_0, s_local, data_cal[i, :6])
            K_PI = np.concatenate((-skew(T_nominal.t), np.eye(3)), axis=1)
            K_i = np.matmul(K_PI, A)
            K = np.concatenate((K, K_i), axis=0)
        Pe_actual = data_cal[:, 6:9]
        z = Pe_actual - Pe_nominal
        z = z.reshape((-1, 1))
        x = np.matmul(np.linalg.pinv(K), z)
        # LM = np.matmul(np.linalg.inv(np.matmul(K.transpose(),K) + 0.01*np.eye(42)), K.transpose())
        # x = np.matmul(LM, z)

        # update
        del_p = x.reshape((-1, 6))
        for idx, del_p_i in enumerate(del_p):
            screw = np.append(del_p_i[3:6],del_p_i[:3])
            Tc_0[idx] = Tc_0[idx] * SE3.Exp(screw)

        # calibration evaluation
        itr += 1
        x_norm = np.linalg.norm(x)
        err = np.linalg.norm(z)**2/data_cal.shape[0]
        print('itr:', itr)
        print('x norm:', x_norm)
        print('err:',np.sqrt(err))

    for i in range(len(Tc_0)):
        print(Tc_0[i])

    # validation
    Pe = np.empty((0,3))
    for i in range(data_val.shape[0]):
        T = forward_kinematics(Tc_0, s_local, data_val[i,:6])
        Pe = np.row_stack((Pe, T.t))
    err_val = data_val[:,6:9] - Pe
    err_val = err_val.reshape((-1,1))
    err_val = np.linalg.norm(err_val)**2/data_val.shape[0]
    print('validataion err :', np.sqrt(err_val))