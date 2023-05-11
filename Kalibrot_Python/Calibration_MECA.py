import numpy as np
from RobotKinematics import Robot_Kinematics
from Calibrate import Calibrate


A = np.loadtxt('./RealRobotsData/MECA/Cali_data_01.csv', delimiter=',').T
m = A.shape[1]  # number data points
P_m = np.zeros((7, m))
P_m[0:3, :] = A[6:9, :] * 0.001
Q = A[0:6, :] * np.pi / 180

T_init = np.array([[-0.799136808187398,	    0.601148299603764,	    0.00189221940791717,	2.35286995562536],
                   [-0.601138360242273,	    -0.799131598604939,	    0.00349011730428195,	-0.260300686010470],
                   [0.00361021402253178,	0.00306757145282890,	0.999992119262404,	    0.0507530942489586],
                   [0,	                    0,	                    0,	                    1]])

T_tool = np.array([[1,	0,	0,	-0.000217677829199658],
                   [0,	1,	0,	0.0500386452999182],
                   [0,	0,	1,	0.0277939548117020],
                   [0,	0,  0,  1]])

n_joints = 6
types = 'rrrrrr'


Robot = Robot_Kinematics(n_joints, types, T_init, T_tool)

DH_nominal = np.array([[0.135, 0, 0, 0, 0],
                       [0, -np.pi/2, 0, -np.pi/2, 0],
                       [0, 0, 0.135, 0, 0],
                       [0.120, 0, 0.038, -np.pi/2, 0],
                       [0, 0, 0, np.pi/2, 0],
                       [0.070, np.pi, 0, -np.pi/2, 0]])

# DH param limits for each link. 1 = min
# in R^njx4 [d1 theta1 a1 alpha1],;...[dn thetan an alphan]
Limits = np.zeros((n_joints, 5, 2))
Limits[:, :, 0] = np.array([[0, -np.pi/2, -0.1, -np.pi/2, -0.05*np.pi/180],
                            [-0.1, -np.pi/2, -0.1, -np.pi/2, -0.05*np.pi/180],
                            [-0.1, -np.pi/2, 0, -np.pi/2, -0.05*np.pi/180],
                            [0, -np.pi/2, 0, -np.pi/2, -0.05*np.pi/180],
                            [-0.1, -np.pi/2, -0.1, -np.pi/2, -0.05*np.pi/180],
                            [0, -np.pi, -0.1, -np.pi/2, -0.05*np.pi/180]])

Limits[:, :, 1] = np.array([[0.2, np.pi/2, 0.1, np.pi/2, 0.05*np.pi/180],
                            [0.1, np.pi, 0.1, np.pi/2, 0.05*np.pi/180],
                            [0.1, np.pi/2, 0.2, np.pi/2, 0.05*np.pi/180],
                            [0.2, np.pi, 0.1, np.pi/2, 0.05*np.pi/180],
                            [0.1, np.pi/2, 0.1, np.pi/2, 0.05*np.pi/180],
                            [0.1, np.pi, 0.1, np.pi/2, 0.05*np.pi/180]])

# initial estimates
# in R^njx4 [d1 theta1 a1 alpha1],;...[dn thetan an alphan]
DH = np.array([[0.135, 0, 0, 0, 0],
               [0, -np.pi/2, 0, -np.pi/2, 0],
               [0, 0, 0.135, 0, 0],
               [0.120, 0, 0.038, -np.pi/2, 0],
               [0, 0, 0, np.pi/2, 0],
               [0.070, np.pi, 0, -np.pi/2, 0]])

w_p = np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 1, 1, 0]])

# motion components to consider: 0 = x, 1 = y, 2 = z, 3 = qx, 4 = qy, 5= qz, 6 = qw
dim = [0, 1, 2]  # motion directions was [1,2,3] in matlab but in pyth use [0,1,2]
# dim = [1, 2, 3, 4, 5, 6, 7]  # motion directions
# weights for each motion direction
W = np.vstack([1*np.ones((1, m)),
              1*np.ones((1, m)),
              1*np.ones((1, m))])


# data to consider
P_m = P_m[dim, :]

options_pinv = {'solver': "pinv", 'damping': 1e-03, 'MaxIter': 1000, 'Visualize': [True, np.array([0, 0, 0, 0, 0, 0])]}


DH_params_pinv, P_pinv, W_pinv = Calibrate(Robot, dim, P_m, Q, DH, W, w_p, Limits, options_pinv)


# options_qp = {'solver': "qp", 'damping': 1e-03, 'MaxIter': 1000}
# # DH_params_qp, P_qp, W_qp = Calibrate(Robot, dim, P_m, Q, DH, W, w_p, Limits, options_qp)

DH_pinv = np.reshape(DH_params_pinv, (6, 5))[:, ::-1]
print(DH_pinv)
# DH_qp = np.reshape(DH_params_qp, (6, 5))[:, ::-1]






