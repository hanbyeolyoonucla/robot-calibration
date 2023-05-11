import numpy as np
from roboticstoolbox import RevoluteMDH, SerialLink
import matplotlib.pyplot as plt

def VisualizeResults(Robot, DH_params_init, DH_params, DH_params_lims, options):
    joint_types = Robot.m_joint_types
    n_jnts = len(joint_types)

    # ROBOT PLOTTING
    sigma = np.zeros(len(joint_types))
    f_prism = np.where(joint_types == 'p')[0]
    f_rev = np.where(joint_types == 'r')[0]
    sigma[f_prism] = 1

    DH_tab = np.reshape(DH_params, (n_jnts, 5))
    eps = 1e-15

    # Convert to Robotic Toolbox convention
    theta = eps * np.ones(n_jnts)
    d = eps * np.ones(n_jnts)
    offset = np.zeros(n_jnts)

    offset[f_rev] = DH_tab[f_rev, 1]
    offset[f_prism] = DH_tab[f_prism, 0]
    theta[f_prism] = DH_tab[f_prism, 1]
    d = DH_tab[:, 0]
    a = DH_tab[:, 2]
    alpha = DH_tab[:, 3]

    beta = DH_tab[:, 4]

    # Combine the DH parameters
    dh = np.column_stack((theta, d, a, alpha, sigma, offset))

    L = [RevoluteMDH(d=d[i], a=a[i], alpha=alpha[i]) for i in range(6)]

    robot_RT = SerialLink(L, name="Final Robot")
    if len(f_prism) > 0:
        robot_RT.qlim[f_prism, :] = np.column_stack((np.zeros(len(f_prism)), 0.11 * np.ones(len(f_prism))))

    q = options['Visualize'][1]
    n_r = q.shape[0]
    if n_r > 1:
        q = q.T
    if q.size == 0:
        q = np.zeros((1, n_jnts))

    print("PLOTTING ROBOT STRUCTURE")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    robot_RT.plot(q)
    #plt.show(block=True)

    print("PLOTTING DH PARAMETERS")
    n_vars = len(DH_params)
    fig, ax = plt.subplots()
    fig.suptitle("DH Parameters")
    if options['solver'] == "qp":
        ax.plot(np.arange(1, n_vars + 1), DH_params, 'ob', markersize=10, linewidth=3)
        center = (DH_params_lims[:, 0] + DH_params_lims[:, 1]) / 2
        delta = (DH_params_lims[:, 1] - DH_params_lims[:, 0]) / 2
        ax.errorbar(np.arange(1, n_vars + 1), center, delta, delta, '.r', linewidth=1, markersize=0.1)
        ax.plot(np.arange(1, n_vars + 1), DH_params_init, '.k', markersize=10)
        ax.legend(["Final DH params", "limits", "Initial DH Params"])
    else:
        ax.plot(np.arange(1, n_vars + 1), DH_params, 'ob', markersize=8, linewidth=3)
        ax.plot(np.arange(1, n_vars + 1), DH_params_init, '.r', markersize=4)
        ax.legend(["Final DH params", "Initial DH Params"])

    #ax.set_box(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid(True)
    labels = [None] * n_vars
    labels[0::5] = ["d"] * (n_vars // 5)
    labels[1::5] = ["theta"] * (n_vars // 5)
    labels[2::5] = ["a"] * (n_vars // 5)
    labels[3::5] = ["alpha"] * (n_vars // 5)
    labels[4::5] = ["beta"] * (n_vars // 5)
    ax.set_xticks(np.arange(1, n_vars + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim([0, n_vars + 1])
    ax.set_xlabel("DH Params")
    ax.set_ylabel("DH Params Values")
    ax.set_xticklabels(labels, fontname="Times New Roman")
    plt.show(block=True)

