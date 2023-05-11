import numpy as np
import sympy as sp
from sympy import symbols, cos, sin
from sympy import *

from sympy.matrices import Matrix



def DH_matNum(x):
    d = x[0]
    theta = x[1]
    a = x[2]
    alpha = x[3]
    beta = x[4]
    T = np.zeros((4, 4))

    T[3, 3] = 1

    T[0, 0] = np.cos(beta) * np.cos(theta) + np.sin(alpha) * np.sin(beta) * np.sin(theta)
    T[0, 1] = np.sin(alpha) * np.sin(beta) * np.cos(theta) - np.cos(beta) * np.sin(theta)
    T[0, 2] = np.cos(alpha) * np.sin(beta)
    T[0, 3] = d * np.cos(alpha) * np.sin(beta) + a * np.cos(beta)

    T[1, 0] = np.cos(alpha) * np.sin(theta)
    T[1, 1] = np.cos(alpha) * np.cos(theta)
    T[1, 2] = -np.sin(alpha)
    T[1, 3] = -d * np.sin(alpha)

    T[2, 0] = np.cos(beta) * np.sin(alpha) * np.sin(theta) - np.sin(beta) * np.cos(theta)
    T[2, 1] = np.sin(beta) * np.sin(theta) + np.sin(alpha) * np.cos(beta) * np.cos(theta)
    T[2, 2] = np.cos(alpha) * np.cos(beta)
    T[2, 3] = d * np.cos(beta) * np.cos(alpha) - a * np.sin(beta)

    return T


def DH_mat(x):
    d = x[0]
    theta = x[1]
    a = x[2]
    alpha = x[3]
    beta = x[4]
    T = sp.zeros(4, 4)

    T[3, 3] = 1

    T[0, 0] = cos(beta) * cos(theta) + sin(alpha) * sin(beta) * sin(theta)
    T[0, 1] = sin(alpha) * sin(beta) * cos(theta) - cos(beta) * sin(theta)
    T[0, 2] = cos(alpha) * sin(beta)
    T[0, 3] = d * cos(alpha) * sin(beta) + a * cos(beta)

    T[1, 0] = cos(alpha) * sin(theta)
    T[1, 1] = cos(alpha) * cos(theta)
    T[1, 2] = -sin(alpha)
    T[1, 3] = -d * sin(alpha)

    T[2, 0] = cos(beta) * sin(alpha) * sin(theta) - sin(beta) * cos(theta)
    T[2, 1] = sin(beta) * sin(theta) + sin(alpha) * cos(beta) * cos(theta)
    T[2, 2] = cos(alpha) * cos(beta)
    T[2, 3] = d * cos(beta) * cos(alpha) - a * sin(beta)

    # T = Matrix([[cos(beta)*cos(theta) + sin(alpha)*sin(beta)*sin(theta), sin(alpha)*sin(beta)*cos(theta) - cos(beta)*sin(theta), cos(alpha)*sin(beta), d*cos(alpha)*sin(beta) + a*cos(beta)],
    #        [cos(alpha)*sin(theta), cos(alpha)*cos(theta), -sin(alpha), -d*sin(alpha)],
    #        [cos(beta)*sin(alpha)*sin(theta)-sin(beta)*cos(theta), sin(beta)*sin(theta) + sin(alpha)*cos(beta)*cos(theta), cos(alpha)*cos(beta), d*cos(beta)*cos(alpha) - a*sin(beta)],
    #        [0, 0, 0, 1]])

    return T


def Rot2Quat_1(R):
    # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions
    t = R[0, 0] - R[1, 1] - R[2, 2]
    quat_in = np.zeros((4, 1))
    quat_in[0, 0] = 0.5 * sqrt(1 + t)
    s = 1 / (4 * quat_in[0])

    quat_in[1, 0] = s * (R[0, 1] + R[1, 0])
    quat_in[2, 0] = s * (R[0, 2] + R[2, 0])
    quat_in[3, 0] = s * (R[2, 1] - R[1, 2])

    return quat_in


def Rot2Quat_2(R):
    # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions
    t = R[0, 0] + R[1, 1] + R[2, 2]
    quat_in = np.zeros((4, 1))
    quat_in[3, 0] = 0.5 * sqrt(1 + t)
    s = 1 / (4 * quat_in[3])

    quat_in[0, 0] = s * (R[2, 1] - R[1, 2])
    quat_in[1, 0] = s * (R[0, 2] - R[2, 0])
    quat_in[2, 0] = s * (R[1, 0] - R[0, 1])

    return quat_in


def Rot2Quat_3(R):
    # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions
    t = R[1, 1] - R[0, 0] - R[2, 2]
    quat_in = np.zeros((4, 1))
    quat_in[1, 0] = 0.5 * sqrt(1 + t)
    s = 1 / (4 * quat_in[1])

    quat_in[0, 0] = s * (R[0, 1] + R[1, 0])
    quat_in[2, 0] = s * (R[1, 2] + R[2, 1])
    quat_in[3, 0] = s * (R[0, 2] - R[2, 0])

    return quat_in


def Rot2Quat_4(R):
    # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Quaternions
    t = R[2, 2] - R[1, 1] - R[0, 0]
    quat_in = np.zeros((4, 1))
    quat_in[2, 0] = 0.5 * sqrt(1 + t)
    s = 1 / (4 * quat_in[2])

    quat_in[0, 0] = s * (R[0, 2] + R[2, 0])
    quat_in[1, 0] = s * (R[1, 2] + R[2, 1])
    quat_in[3, 0] = s * (R[1, 0] - R[0, 1])

    return quat_in


def Rot2Quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 1e-12:
        quat_in = Rot2Quat_2(R)
        condition = 2
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            quat_in = Rot2Quat_1(R)
            condition = 1
        elif R[1, 1] > R[2, 2]:
            quat_in = Rot2Quat_3(R)
            condition = 3
        else:
            quat_in = Rot2Quat_4(R)
            condition = 4
    return quat_in, condition


def Quat2Rot(quat):
    quat_in = quat.copy()
    x, y, z, w = quat_in[0], quat_in[1], quat_in[2], quat_in[3]

    R = np.zeros((3, 3))

    R[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
    R[0, 1] = 2 * (x * y - z * w)
    R[0, 2] = 2 * (x * z + y * w)

    R[1, 0] = 2 * (x * y + z * w)
    R[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
    R[1, 2] = 2 * (y * z - x * w)

    R[2, 0] = 2 * (x * z - y * w)
    R[2, 1] = 2 * (y * z + x * w)
    R[2, 2] = 1 - x ** 2 - y ** 2

    return R


def Pose_deriv(q_i, DH_params_i, type):
    # Initialize the outputs
    dP = np.zeros((3, 5))
    dR = np.zeros((3, 3, 5))

    if type == 'r':
        DH_params_i[1] = DH_params_i[1] + q_i
    else:
        DH_params_i[0] = DH_params_i[0] + q_i

    d = DH_params_i[0]
    theta = DH_params_i[1]
    a = DH_params_i[2]
    alpha = DH_params_i[3]
    beta = DH_params_i[4]

    # dd, dtheta, da, dalpha, dbeta
    dP[0, :] = [np.cos(alpha) * np.sin(beta), 0, np.cos(beta), -d * np.sin(beta) * np.sin(alpha),
                d * np.cos(alpha) * np.cos(beta) - a * np.sin(beta)]
    dP[1, :] = [-np.sin(alpha), 0, 0, -d * np.cos(alpha), 0]
    dP[2, :] = [np.cos(alpha) * np.cos(beta), 0, -np.sin(beta), -d * np.cos(beta) * np.sin(alpha),
                -a * np.cos(beta) - d * np.cos(alpha) * np.sin(beta)]

    # derivative of first row
    # [dR11/dx; dR12/dx; dR13/dx] 
    # dR with beta parameter
    dR[0, :, :] = np.array([[0, np.sin(alpha) * np.sin(beta) * np.cos(theta) - np.cos(beta) * np.sin(theta), 0,
                             np.cos(alpha) * np.sin(beta) * np.sin(theta),
                             np.cos(beta) * np.sin(alpha) * np.sin(theta) - np.sin(beta) * np.cos(theta)],
                            [0, -np.cos(theta) * np.cos(beta) - np.sin(alpha) * np.sin(beta) * np.sin(theta), 0,
                             np.cos(alpha) * np.sin(beta) * np.cos(theta),
                             np.sin(beta) * np.sin(theta) + np.cos(beta) * np.sin(alpha) * np.cos(theta)],
                            [0, 0, 0, -np.sin(alpha) * np.sin(beta), np.cos(alpha) * np.cos(beta)]],dtype = object)

    # [dR21/dx; dR22/dx; dR23/dx]       
    dR[1, :, :] = np.array([[0, np.cos(theta) * np.cos(alpha), 0, -np.sin(alpha) * np.sin(theta), 0],
                            [0, -np.cos(alpha) * np.sin(theta), 0, -np.cos(theta) * np.sin(alpha), 0],
                            [0, 0, 0, -np.cos(alpha), 0]],dtype = object)

    # [dR31/dx; dR32/dx; dR33/dx]
    dR[2, :, :] = np.array([[0, np.sin(beta) * np.sin(theta) + np.cos(beta) * np.sin(alpha) * np.cos(theta), 0,
                             np.cos(alpha) * np.cos(beta) * np.sin(theta),
                             -np.cos(beta) * np.cos(theta) - np.sin(alpha) * np.sin(beta) * np.sin(theta)],
                            [0, np.cos(theta) * np.sin(beta) - np.cos(beta) * np.sin(alpha) * np.sin(theta), 0,
                             np.cos(alpha) * np.cos(beta) * np.cos(theta),
                             np.cos(beta) * np.sin(theta) - np.sin(alpha) * np.sin(beta) * np.cos(theta)],
                            [0, 0, 0, -np.sin(alpha) * np.cos(beta), -np.cos(alpha) * np.sin(beta)]],dtype = object)

    return dP, dR


def quatDeriv_Rot_1(R):
    # quat_in = quat.copy()
    dq_dR = np.zeros((4, 9))

    t = 1 + R[0, 0] - R[1, 1] - R[2, 2]

    quat_in = np.zeros((4, 1))
    quat_in[0, 0] = 0.5 * sqrt(t)  # qx

    if quat_in[0, 0] < 1e-10:
        quat_in[0, 0] = 1e-10

    dq_dR[0, :] = 1 / 4 * t ** (-1 / 2) * np.array([1, 0, 0, 0, -1, 0, 0, 0, -1])

    dq_dR[1, :] = -1 / (4 * quat_in[0, 0] ** 2) * (R[0, 1] + R[1, 0]) * dq_dR[0, :] + 1 / (4 * quat_in[0, 0]) * np.array(
        [0, 1, 0, 1, 0, 0, 0, 0, 0])

    dq_dR[2, :] = -1 / (4 * quat_in[0, 0] ** 2) * (R[0, 2] + R[2, 0]) * dq_dR[0, :] + 1 / (4 * quat_in[0, 0]) * np.array(
        [0, 0, 1, 0, 0, 0, 1, 0, 0])

    dq_dR[3, :] = -1 / (4 * quat_in[0, 0] ** 2) * (R[2, 1] - R[1, 2]) * dq_dR[0, :] + 1 / (4 * quat_in[0, 0]) * np.array(
        [0, 0, 0, 0, 0, -1, 0, 1, 0])

    return dq_dR


def quatDeriv_Rot_2(R):
    #quat_in = quat.copy()
    dq_dR = np.zeros((4, 9))

    t = 1 + R[0, 0] + R[1, 1] + R[2, 2]

    quat_in = np.zeros((4, 1))
    quat_in[3, 0] = 0.5 * sqrt(t)  # qw

    if quat_in[3, 0] < 1e-10:
        quat_in[3, 0] = 1e-10

    dq_dR[3, :] = 1 / 4 * t ** (-1 / 2) * np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])

    dq_dR[0, :] = -1 / (4 * quat_in[3, 0] ** 2) * (R[2, 1] - R[1, 2]) * dq_dR[3, :] + 1 / (4 * quat_in[3, 0]) * np.array(
        [0, 0, 0, 0, 0, -1, 0, 1, 0])

    dq_dR[1, :] = -1 / (4 * quat_in[3, 0] ** 2) * (R[0, 2] - R[2, 0]) * dq_dR[3, :] + 1 / (4 * quat_in[3, 0]) * np.array(
        [0, 0, 1, 0, 0, 0, -1, 0, 0])

    dq_dR[2, :] = -1 / (4 * quat_in[3, 0] ** 2) * (R[1, 0] - R[0, 1]) * dq_dR[3, :] + 1 / (4 * quat_in[3, 0]) * np.array(
        [0, -1, 0, 1, 0, 0, 0, 0, 0])

    return dq_dR


def quatDeriv_Rot_3(R):
    #quat_in = quat.copy()
    dq_dR = np.zeros((4, 9))

    t = 1 + R[1, 1] - R[0, 0] - R[2, 2]

    quat_in = np.zeros((4, 1))
    quat_in[1, 0] = 0.5 * sqrt(t)  # qw

    if quat_in[1, 0] < 1e-10:
        quat_in[1, 0] = 1e-10

    dq_dR[1, :] = -1 / (4 * quat_in[1, 0] ** 2) * (R[0, 1] + R[1, 0]) * dq_dR[2, :] + 1 / (4 * quat_in[1, 0]) * np.array(
        [0, 1, 0, 1, 0, 0, 0, 0, 0])

    dq_dR[2, :] = -1 / (4 * quat_in[1, 0] ** 2) * (R[1, 2] + R[2, 1]) * dq_dR[2, :] + 1 / (4 * quat_in[1, 0]) * np.array(
        [0, 0, 0, 0, 0, 1, 0, 1, 0])

    dq_dR[3, :] = -1 / (4 * quat_in[1, 0] ** 2) * (R[0, 2] - R[2, 0]) * dq_dR[2, :] + 1 / (4 * quat_in[1, 0]) * np.array(
        [0, 0, 1, 0, 0, 0, -1, 0, 0])

    dq_dR[0, :] = -1 / (4 * quat_in[1, 0] ** 2) * (R[2, 1] - R[1, 2]) * dq_dR[2, :] + 1 / (4 * quat_in[1, 0]) * np.array(
        [0, -1, 0, 1, 0, 0, 0, 0, 0])

    return dq_dR


def quatDeriv_Rot_4(R):
    #quat_in = quat.copy()
    dq_dR = np.zeros((4, 9))

    t = 1 + R[2, 2] - R[0, 0] - R[1, 1]

    quat_in = np.zeros((4, 1))
    quat_in[2, 0] = 0.5 * sqrt(t)

    if quat_in[2, 0] < 1e-10:
        quat_in[2, 0] = 1e-10

    dq_dR[2, :] = 1 / (4 * t ** 0.5) * np.array([-1, 0, 0, 0, -1, 0, 0, 0, 1])

    dq_dR[0, :] = -1 / (4 * quat_in[2, 0] ** 2) * (R[0, 2] + R[2, 0]) * dq_dR[2, :] + 1 / (4 * quat_in[2, 0]) * np.array(
        [0, 0, 1, 0, 0, 0, 1, 0, 0])

    dq_dR[1, :] = -1 / (4 * quat_in[2, 0] ** 2) * (R[1, 2] + R[2, 1]) * dq_dR[2, :] + 1 / (4 * quat_in[2, 0]) * np.array(
        [0, 0, 0, 0, 0, 1, 0, 1, 0])

    dq_dR[3, :] = -1 / (4 * quat_in[2, 0] ** 2) * (R[1, 0] - R[0, 1]) * dq_dR[2, :] + 1 / (4 * quat_in[2, 0]) * np.array(
        [0, -1, 0, 1, 0, 0, 0, 0, 0])

    return dq_dR


def RotMatDeriv(q_i, DH_params_i, type):
    dR = np.zeros((9, 5))  # for each row of R

    if type == 'r':
        DH_params_i[1] = DH_params_i[1] + q_i
    else:
        DH_params_i[0] = DH_params_i[0] + q_i

    d = DH_params_i[0]
    theta = DH_params_i[1]
    a = DH_params_i[2]
    alpha = DH_params_i[3]
    beta = DH_params_i[4]

    # derivative of first row
    # [dR11/dx; dR12/dx; dR13/dx]
    dR_mat_1 = np.array([
        [0, np.sin(alpha) * np.sin(beta) * np.cos(theta) - np.cos(beta) * np.sin(theta), 0,
         np.cos(alpha) * np.sin(beta) * np.sin(theta),
         np.cos(beta) * np.sin(alpha) * np.sin(theta) - np.sin(beta) * np.cos(theta)],
        [0, -np.cos(theta) * np.cos(beta) - np.sin(alpha) * np.sin(beta) * np.sin(theta), 0,
         np.cos(alpha) * np.sin(beta) * np.cos(theta),
         np.sin(beta) * np.sin(theta) + np.cos(beta) * np.sin(alpha) * np.cos(theta)],
        [0, 0, 0, -np.sin(alpha) * np.sin(beta), np.cos(alpha) * np.cos(beta)]
    ],dtype = object)

    # [dR21/dx; dR22/dx; dR23/dx]
    dR_mat_2 = np.array([
        [0, np.cos(theta) * np.cos(alpha), 0, -np.sin(alpha) * np.sin(theta), 0],
        [0, -np.cos(alpha) * np.sin(theta), 0, -np.cos(theta) * np.sin(alpha), 0],
        [0, 0, 0, -np.cos(alpha), 0]
    ],dtype = object)

    # [dR31/dx; dR32/dx; dR33/dx]
    dR_mat_3 = np.array([
        [0, np.sin(beta) * np.sin(theta) + np.cos(beta) * np.sin(alpha) * np.cos(theta), 0,
         np.cos(alpha) * np.cos(beta) * np.sin(theta),
         -np.cos(beta) * np.cos(theta) - np.sin(alpha) * np.sin(beta) * np.sin(theta)],
        [0, np.cos(theta) * np.sin(beta) - np.cos(beta) * np.sin(alpha) * np.sin(theta), 0,
         np.cos(alpha) * np.cos(beta) * np.cos(theta),
         np.cos(beta) * np.sin(theta) - np.sin(alpha) * np.sin(beta) * np.cos(theta)],
        [0, 0, 0, -np.sin(alpha) * np.cos(beta), -np.cos(alpha) * np.sin(beta)]
    ],dtype = object)

    dR = np.concatenate((dR_mat_1, dR_mat_2, dR_mat_3), axis=0)

    return dR


class Robot_Kinematics:

    def __init__(self, n_joints, types, T_init, T_tool=None):

        self.m_n_joints = n_joints
        self.m_joint_types = types
        self.m_T_init = T_init

        if T_tool is not None:
            self.m_tool_added = 1
            self.m_T_tool = T_tool
        else:
            self.m_tool_added = 0
            self.m_T_tool = np.eye(4)

        self.m_DH_params = symbols('m_DH_params:' + str(5 * n_joints))
        self.m_q = symbols('m_q:' + str(n_joints))

        # (self.m_T_sym, self.m_Dp_sym, self.m_Dor1_sym,
        #  self.m_Dor2_sym, self.m_Dor3_sym, self.m_Dor4_sym) = self.getKineSym()

    # def getPose(self, q, DH_params):
    #     T_val = self.m_T_sym.replace([self.m_DH_params, self.m_q], [DH_params, q])
    #     T_val = np.array(T_val).astype(float)
    #
    #     R = T_val[0:3, 0:3]
    #     quat, self.m_orient_cond = Rot2Quat(
    #         R)  # assuming this function returns a quaternion and the orientation condition
    #
    #     P = np.concatenate([T_val[0:3, 3], quat])
    #
    #     return T_val, P
    #
    # def getDerivs(self, q, DH_params):
    #     Dp = self.m_Dor1_sym.replace([self.m_DH_params, self.m_q], [DH_params, q])
    #
    #     if self.m_orient_cond == 1:
    #         Dor = self.m_Dor1_sym.replace([self.m_DH_params, self.m_q], [DH_params, q])
    #     elif self.m_orient_cond == 2:
    #         Dor = self.m_Dor2_sym.replace([self.m_DH_params, self.m_q], [DH_params, q])
    #     elif self.m_orient_cond == 3:
    #         Dor = self.m_Dor3_sym.replace([self.m_DH_params, self.m_q], [DH_params, q])
    #     elif self.m_orient_cond == 4:
    #         Dor = self.m_Dor4_sym.replace([self.m_DH_params, self.m_q], [DH_params, q])
    #
    #     Dp = np.array(Dp).astype(float)
    #     Dor = np.array(Dor).astype(float)
    #
    #     return Dp, Dor

    # def getKineSym(self):
    #     T = self.m_T_init
    #     T = Matrix(T)
    #
    #     for i in range(self.m_n_joints):
    #         v1 = 5 * i
    #         xi = self.m_DH_params[v1:v1 + 5]
    #         xi_list = list(xi)
    #
    #         if self.m_joint_types[i] == 'r':
    #
    #             xi_list[1] += self.m_q[i]
    #         else:
    #
    #             xi_list[0] += self.m_q[i]
    #
    #         Ti = DH_mat(xi_list)
    #
    #         T = T @ Ti
    #
    #     T = T @ self.m_T_tool
    #
    #     P = T[0:3, 3]
    #     R = T[0:3, 0:3]
    #
    #     quat_1 = Rot2Quat_1(R)
    #     quat_2 = Rot2Quat_2(R)
    #     quat_3 = Rot2Quat_3(R)
    #     quat_4 = Rot2Quat_4(R)
    #
    #     Dp = P.jacobian(self.m_DH_params)  # size(3 x 5)
    #     Dor1 = quat_1.jacobian(self.m_DH_params)  # size(4 x 5)
    #     Dor2 = quat_2.jacobian(self.m_DH_params)
    #     Dor3 = quat_3.jacobian(self.m_DH_params)
    #     Dor4 = quat_4.jacobian(self.m_DH_params)
    #
    #     # DP = JACOBIAN(P, SELF.M_DH_PARAMS)
    #     # DOR1 = JACOBIAN(QUAT_1, SELF.M_DH_PARAMS)
    #     # DOR2 = JACOBIAN(QUAT_2, SELF.M_DH_PARAMS)
    #     # DOR3 = JACOBIAN(QUAT_3, SELF.M_DH_PARAMS)
    #     # DOR4 = JACOBIAN(QUAT_4, SELF.M_DH_PARAMS)
    #
    #     return T, Dp, Dor1, Dor2, Dor3, Dor4

    def getPoseNum(self, q, DH_params):
        T_val = self.m_T_init
        DH_copy = DH_params.copy()
        for i in range(self.m_n_joints):
            v1 = 5 * i
            xi = DH_copy[v1:v1 + 5]
            xi_list = list(xi)

            if self.m_joint_types[i] == 'r':
                xi_list[1] += q[i]
            else:
                xi_list[0] += q[i]

            Ti = DH_matNum(xi_list)
            T_val = T_val @ Ti

        T_val = T_val @ self.m_T_tool

        R = T_val[0:3, 0:3]
        quat, _ = Rot2Quat(R)

        P = np.vstack([T_val[0, 3], T_val[1, 3], T_val[2, 3], quat[0], quat[1], quat[2], quat[3]])

        return T_val, P

    def getPDerivNum(self, q, DH_params, P_e):
        n_var = len(DH_params)
        D = np.zeros((3, n_var))

        DH_copy = DH_params.copy()

        if P_e.size == 0:
            pass
        else:
            _, P_e = self.getPoseNum(q, DH_params)
            P_e = P_e[0:3, 0]

        T = self.m_T_init



        for i in range(self.m_n_joints):
            R_i_1_0 = T[:3, :3]  # R_i-1_0

            v1 = 5 * i
            xi = DH_copy[v1:v1 + 5]

            type = self.m_joint_types[i]
            DP_i, dR_i = Pose_deriv(q[i], xi, type)  # of i wrt i-1

            # if type == 'r':
            #     xi[1] += q[i]
            # else:
            #     xi[0] += q[i]

            Ti = DH_matNum(xi)

            T = T @ Ti  # T_i_0
            R_i_0 = T[0:3, 0:3]  # R_i_0

            P_i = T[0:3, 3]
            P_e_i = R_i_0.T @ (P_e - P_i)
            DR_i = np.zeros((3, 5))

            for dim in range(3):
                  DR_i[dim, :] = P_e_i.T @ dR_i[dim, :, :]

            Di = R_i_1_0 @ (DP_i + DR_i)

            D[:, v1:v1 + 5] = Di

        return D

    def getQuatDerivNum(self, q, DH_params):
        n_var = len(DH_params)
        T = self.m_T_init
        R_i = T[0:3, 0:3]
        quat, _ = Rot2Quat(R_i)
        Dquat = np.zeros((4, n_var))

        DH_copy = DH_params.copy()

        for i in range(self.m_n_joints):
            v1 = 5 * i
            xi = DH_copy[v1:v1 + 5]
            # xi_list = list(xi)
            type = self.m_joint_types[i]
            dR_i = RotMatDeriv(q[i], xi, type)
            dR = np.zeros((9, n_var))
            dR[:, v1:v1 + 5] = dR_i
            if type == 'r':
                xi[1] += q[i]
            else:
                xi[0] += q[i]
            Ti = DH_matNum(xi)
            R_i = Ti[0:3, 0:3]
            quat_i, cond = Rot2Quat(R_i)

            if cond == 1:
                dq_dR_i = quatDeriv_Rot_1(R_i)
            elif cond == 2:
                dq_dR_i = quatDeriv_Rot_2(R_i)
            elif cond == 3:
                dq_dR_i = quatDeriv_Rot_3(R_i)
            elif cond == 4:
                dq_dR_i = quatDeriv_Rot_4(R_i)

            n = quat[3, 0]
            e = quat[0:3, 0]
            n_i = quat_i[3, 0]
            e_i = quat_i[0:3, 0]
            quat[0:3, 0] = n * e_i + n_i * e + np.cross(e, e_i)
            quat[3, 0] = n * n_i - e.T @ e_i

            Dn = Dquat[3, :]
            De = Dquat[0:3, :]
            Dn_i = dq_dR_i[3, :] @ dR
            De_i = dq_dR_i[0:3, :] @ dR

            # derivative of qw
            Dquat[3, :] = Dn * n_i + n * Dn_i
            der = np.zeros((1, n_var))
            for dim in range(3):
                der = der + De[dim, :] * e_i[dim] + e[dim] * De_i[dim, :]
            Dquat[3, :] = Dquat[3, :] - der

            # derivative of qx, qy, qz
            Dquat[0:3, :] = n * De_i + n_i * De
            der = np.zeros((3, n_var))
            for dim in range(3):
                der[dim, :] = Dn * e_i[dim] + Dn_i * e[dim]
            Dquat[0:3, :] = Dquat[0:3, :] + der

            # cross product deriv
            Dcp = np.zeros((3, n_var))
            Dcp[0, :] = -De[2, :] * e_i[1] + De[1, :] * e_i[2]
            Dcp[1, :] = De[2, :] * e_i[0] - De[0, :] * e_i[2]
            Dcp[2, :] = -De[1, :] * e_i[0] + De[0, :] * e_i[1]

            Dquat[0:3, :] = Dquat[0:3, :] + Dcp

            Dcp[0, :] = -De_i[2, :] * e[1] + De_i[1, :] * e[2]
            Dcp[1, :] = De_i[2, :] * e[0] - De_i[0, :] * e[2]
            Dcp[2, :] = -De_i[1, :] * e[0] + De_i[0, :] * e[1]

            Dquat[0:3, :] = Dquat[0:3, :] - Dcp

        if self.m_tool_added == 1:
            # Tool added
            R_tool = self.m_T_tool[0:3, 0:3]
            quat_tool = Rot2Quat(R_tool)

            n = quat[3, 0]
            e = quat[0:3, 0]

            n_i = quat_tool[3, 0]
            e_i = quat_tool[0:3, 0]

            quat[0:3, 0] = n * e_i + n_i * e + np.cross(e, e_i)
            quat[3, 0] = n * n_i - e.T @ e_i

            Dn = Dquat[3, :]
            De = Dquat[0:3, :]

            Dquat[3, :] = Dn * n_i - e_i.T @ De
            Dquat[0:3, :] = n_i * De
            der = np.zeros((3, n_var))
            for dim in range(3):
                der[dim, :] = Dn * e_i[dim]
            Dquat[0:3, :] = Dquat[0:3, :] + der

            # cross product deriv
            Dcp = np.zeros((3, n_var))
            Dcp[0, :] = -De[2, :] * e_i[1] + De[1, :] * e_i[2]
            Dcp[1, :] = De[2, :] * e_i[0] - De[0, :] * e_i[2]
            Dcp[2, :] = -De[1, :] * e_i[0] + De[0, :] * e_i[1]

            Dquat[0:3, :] = Dquat[0:3, :] + Dcp

        return Dquat, quat

    def getQuat(self, R):
        quat, _ = Rot2Quat(R)
        return quat

    def getRotmat(self, quat):
        R = Quat2Rot(quat)
        return R

    def getKineDeriv_Ana(self, q, DH_params):
        n_var = len(DH_params)
        T = self.m_T_init
        R_i = T[0:3, 0:3]
        quat, _ = Rot2Quat(R_i)
        Dquat = np.zeros((4, n_var))

        DH_copy = DH_params.copy()

        for i in range(self.m_n_joints):
            v1 = 5 * i
            xi = DH_copy[v1:v1 + 5]
            # xi_list = list(xi)
            type = self.m_joint_types[i]
            dR_i = RotMatDeriv(q[i], xi, type)
            dR = np.zeros((9, n_var))
            dR[:, v1:v1 + 5] = dR_i.copy()
            # if type == 'r':
            #     xi[1] += q[i]
            # else:
            #     xi[0] += q[i]
            Ti = DH_matNum(xi)

            T = T @ Ti

            R_i = Ti[0:3, 0:3]
            quat_i, cond = Rot2Quat(R_i)

            if cond == 1:
                dq_dR_i = quatDeriv_Rot_1(R_i)
            elif cond == 2:
                dq_dR_i = quatDeriv_Rot_2(R_i)
            elif cond == 3:
                dq_dR_i = quatDeriv_Rot_3(R_i)
            elif cond == 4:
                dq_dR_i = quatDeriv_Rot_4(R_i)

            n = quat.copy()[3, 0]
            e = quat.copy()[0:3, 0]
            n_i = quat_i.copy()[3, 0]
            e_i = quat_i.copy()[0:3, 0]
            quat[0:3, 0] = n * e_i + n_i * e + np.cross(e, e_i)
            quat[3, 0] = n * n_i - e.T @ e_i

            Dn = Dquat.copy()[3, :]
            De = Dquat.copy()[0:3, :]
            Dn_i = dq_dR_i[3, :] @ dR
            De_i = dq_dR_i[0:3, :] @ dR

            # derivative of qw
            Dquat[3, :] = Dn * n_i + n * Dn_i
            der = np.zeros((1, n_var))
            for dim in range(3):
                der = der + De[dim, :] * e_i[dim] + e[dim] * De_i[dim, :]
            Dquat[3, :] = Dquat[3, :] - der

            # derivative of qx, qy, qz
            Dquat[0:3, :] = n * De_i + n_i * De
            der = np.zeros((3, n_var))
            for dim in range(3):
                der[dim, :] = Dn * e_i[dim] + Dn_i * e[dim]
            Dquat[0:3, :] = Dquat[0:3, :] + der

            # cross product deriv
            Dcp = np.zeros((3, n_var))
            Dcp[0, :] = -De[2, :] * e_i[1] + De[1, :] * e_i[2]
            Dcp[1, :] = De[2, :] * e_i[0] - De[0, :] * e_i[2]
            Dcp[2, :] = -De[1, :] * e_i[0] + De[0, :] * e_i[1]

            Dquat[0:3, :] = Dquat[0:3, :] + Dcp

            Dcp[0, :] = -De_i[2, :] * e[1] + De_i[1, :] * e[2]
            Dcp[1, :] = De_i[2, :] * e[0] - De_i[0, :] * e[2]
            Dcp[2, :] = -De_i[1, :] * e[0] + De_i[0, :] * e[1]

            Dquat[0:3, :] = Dquat[0:3, :] - Dcp

        if self.m_tool_added == 1:
            # Tool added
            R_tool = self.m_T_tool[0:3, 0:3]
            quat_tool, _ = Rot2Quat(R_tool)

            n = quat[3, 0]
            e = quat[0:3, 0]

            n_i = quat_tool[3, 0]
            e_i = quat_tool[0:3, 0]

            quat[0:3, 0] = n * e_i + n_i * e + np.cross(e, e_i)
            quat[3, 0] = n * n_i - e.T @ e_i

            Dn = Dquat[3, :]
            De = Dquat[0:3, :]

            Dquat[3, :] = Dn * n_i - e_i.T @ De
            Dquat[0:3, :] = n_i * De
            der = np.zeros((3, n_var))
            for dim in range(3):
                der[dim, :] = Dn * e_i[dim]
            Dquat[0:3, :] = Dquat[0:3, :] + der

            # cross product deriv
            Dcp = np.zeros((3, n_var))
            Dcp[0, :] = -De[2, :] * e_i[1] + De[1, :] * e_i[2]
            Dcp[1, :] = De[2, :] * e_i[0] - De[0, :] * e_i[2]
            Dcp[2, :] = -De[1, :] * e_i[0] + De[0, :] * e_i[1]

            Dquat[0:3, :] = Dquat[0:3, :] + Dcp
        P = np.zeros((7, 1))
        T = T @ self.m_T_tool
        P[0:3, 0] = T[0:3, 3]
        P[3:7, :] = quat

        P_e = P[0:3, 0]
        Dp = self.getPDerivNum(q, DH_params, P_e)

        return self, T, P, Dp, Dquat
