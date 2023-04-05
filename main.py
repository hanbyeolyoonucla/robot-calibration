# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import roboticstoolbox as rtb
from spatialmath import SE3

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    robot = rtb.models.Panda()
    print(robot)

    Te = robot.fkine(robot.qr)  # forward kinematics
    print(Te)

    Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ik_lm_chan(Tep)  # solve IK
    print(sol)

    q_pickup = sol[0]
    print(robot.fkine(q_pickup))  # FK shows that desired end-effector pose was achieved

    qt = rtb.jtraj(robot.qr, q_pickup, 50)
    # robot.plot(qt.q, backend='pyplot', movie='panda1.gif')

    robot.plot(qt.q)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
