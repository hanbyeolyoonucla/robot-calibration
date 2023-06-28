from spatialmath import SE3, base
from math import pi
import numpy as np
from cv2 import cv2
import myrobot

def forward_kinematics(T_0, s_local, qs):
    T = SE3()
    for idx, q in enumerate(qs):
        T_i = T_0[idx] * SE3.Exp(s_local * q)
        T = T * T_i
    return T


if __name__ == '__main__':

    # load data
    poses = np.loadtxt('data_tcp_calibration/poses.txt')
    joints = np.loadtxt('data_tcp_calibration/joints.txt') * pi / 180
    pose_num = 5

    # poses from tb to tip
    T_tip_tb = []
    for idx in range(pose_num):
        T_tip_tb.append(base.trnorm(poses[4 * idx:4 * idx + 4, :]))
    T_tip_tb = SE3(T_tip_tb)
    T_tip_tb = T_tip_tb.inv()

    # [POE] calibration result local T 7*4 x 4
    Tc_array = np.loadtxt('result_calibration/poe_local_calib.csv',delimiter=',')
    Tc = []
    for idx in range(6):
        Tc.append(SE3(Tc_array[4 * idx:4 * idx + 4, :]))

    # poses from rb to flange
    T_rb_flange = []
    for idx in range(pose_num):
        T_rb_flange.append(forward_kinematics(Tc, s_local=np.array([0, 0, 0, 0, 0, 1]), qs=joints[idx]))
    T_rb_flange = SE3(T_rb_flange)

    # [DH] calibration result
    cmdh = np.array([[0, 0, 0, 135],
                     [-1.569505826, -0.0835910953, -1.567817367, 0.20853925],
                     [0.0002239402671, 134.9639272, -7.57E-04, 0],
                     [-1.571312979, 38.04809468, -0.001173209742, 119.9796628],
                     [1.571217148, 0.06784178805, 0.003849618677, -0.02988467229],
                     [-1.572022644, -0.003656865739, 3.141592654, 70]])
    crobot = myrobot.SerialLink(mdh=cmdh)
    T_rb_flange_DH = crobot.fkine(joints)

    # [POE] calibrate TCP
    R_tcp, t_tcp = cv2.calibrateHandEye(T_rb_flange.R, T_rb_flange.t, T_tip_tb.R, T_tip_tb.t,
                                        method=cv2.CALIB_HAND_EYE_PARK)  #cv2.CALIB_HAND_EYE_TSAI cv2.CALIB_HAND_EYE_PARK CALIB_HAND_EYE_HORAUD
    T_tcp_POE = SE3.Rt(base.trnorm(R_tcp), t_tcp)
    T_tcp_POE_XYZRPW = np.append(T_tcp_POE.t, np.flip(T_tcp_POE.rpy(unit='deg',order='xyz')))
    print(T_tcp_POE)
    print(T_tcp_POE_XYZRPW)

    # [DH] calibrate TCP
    R_tcp, t_tcp = cv2.calibrateHandEye(T_rb_flange_DH.R, T_rb_flange_DH.t, T_tip_tb.R, T_tip_tb.t,
                                        method=cv2.CALIB_HAND_EYE_PARK)
    T_tcp_DH = SE3.Rt(base.trnorm(R_tcp), t_tcp)
    T_tcp_DH_XYZRPW = np.append(T_tcp_DH.t, np.flip(T_tcp_DH.rpy(unit='deg',order='xyz')))
    print(T_tcp_DH)
    print(T_tcp_DH_XYZRPW)

    # save result
    np.savetxt('data_tcp_calibration/TCP_POE.csv', T_tcp_POE_XYZRPW, newline=" ")
    np.savetxt('data_tcp_calibration/TCP_DH.csv', T_tcp_DH_XYZRPW, newline=" ")