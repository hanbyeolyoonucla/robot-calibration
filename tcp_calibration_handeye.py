

from spatialmath import SE3, base
from math import pi
import numpy as np
from cv2 import cv2

def forward_kinematics(T_0, s_local, qs):
    T = SE3()
    for idx, q in enumerate(qs):
        T_i = T_0[idx] * SE3.Exp(s_local * q)
        T = T * T_i
    return T

if __name__ == '__main__':

    # load data
    poses = np.loadtxt('data_tcp_calibration/poses.txt')
    joints = np.loadtxt('data_tcp_calibration/joints.txt')*pi/180
    pose_num = 4

    # poses from tb to tip
    T_tb_tip = []
    for idx in range(pose_num):
        T_tb_tip.append(base.trnorm(poses[4*idx:4*idx+4, :]))
    T_tb_tip = SE3(T_tb_tip)
    T_tip_tb = T_tb_tip.inv()

    # calibration result local T 7*4 x 4
    Tc_array = np.loadtxt('result_calibration/poe_local_calib.txt')
    Tc = []
    for idx in range(7):
        Tc.append(SE3(Tc_array[4 * idx:4 * idx + 4, :]))

    # poses from rb to flange
    T_rb_flange = []
    for idx in range(pose_num):
        T_rb_flange.append(forward_kinematics(Tc,s_local=np.array([0,0,0,0,0,1]),qs=joints[idx]))
    T_rb_flange = SE3(T_rb_flange)
    print(forward_kinematics(Tc,s_local=np.array([0,0,0,0,0,1]),qs=np.zeros(6)))

    # calibrate TCP
    R_tcp, t_tcp = cv2.calibrateHandEye(T_rb_flange.R, T_rb_flange.t, T_tip_tb.inv().R, T_tip_tb.inv().t, method= cv2.CALIB_HAND_EYE_TSAI)
    T_tcp = SE3.Rt(R_tcp, t_tcp)
    print(T_tcp)

