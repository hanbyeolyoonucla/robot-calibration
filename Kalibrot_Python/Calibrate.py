import numpy as np
from getModel import get_Model
from visualization import VisualizeResults
def Calibrate(Robot, dim, P_m, Q, DH, W, w_p, Limits, options):

    print("*********CALIBRATING********")

    n_joints = Q.shape[0]
    DH_params = DH.reshape(5*n_joints, 1)
    DH_params_init = DH_params.copy() #for visualizing results
    DH_param_lims = np.zeros((5*n_joints, 2))
    DH_param_lims[:, 0] = Limits[:, :, 0].reshape(-1)
    DH_param_lims[:, 1] = Limits[:, :, 1].reshape(-1)

    if options['damping'] is None:
        options.damping = 1e-3

    if options['solver'] == "qp":
        f = np.where(DH_params < DH_param_lims[:, 0])[0]

        if len(f) > 0:
            print("VALUES OUT OF BOUNDS " + str(f))
            print(str(DH_params[f].T) + " < " + str(DH_param_lims[f, 0].T))
        f = np.where(DH_params > DH_param_lims[:, 1])[0]

        if len(f) > 0:
            print("VALUES OUT OF BOUNDS " + str(f))
            print(str(DH_params[f].T) + " > " + str(DH_param_lims[f, 1].T))

    # print(dir(Robot))

    DH_params, P, W = get_Model(Robot, dim, P_m, Q, DH_params, W, w_p, DH_param_lims, options)

    if options['Visualize'][0] is True:
        VisualizeResults(Robot, DH_params_init, DH_params, DH_param_lims, options)

    return DH_params, P, W