import numpy as np

#import sympy as sp
#from sympy import symbols, subs, cos, sin
#from sympy import symbols, subs, cos, sin

#from sympy.matrices import Matrix

from scipy.linalg import svd
from itertools import combinations

from qpsolvers import solve_qp

from sklearn.metrics import mean_squared_error

import time

from temp import Robot_Kinematics



def rsqrd(Y, Y_est):
    n_out, n_samples = Y.shape
    R_2 = np.zeros(n_out)

    y_ave = np.mean(Y, axis=1)
    for i in range(n_out):
        ss_tot = np.sum((Y[i, :] - y_ave[i]) ** 2)

        if ss_tot < 1e-10:
            ss_tot = 1e-10

        ss_res = np.sum((Y[i, :] - Y_est[i, :]) ** 2)
        R_2[i] = 1 - ss_res / ss_tot

    return R_2


def initializeSol(H, f, lb, ub):
    x0_qp = -np.linalg.pinv(H) @ f
    for i in range(len(x0_qp)):
        if x0_qp[i] > lb[i] and x0_qp[i] < ub[i]:
            pass
        else:
            x0_qp[i] = (ub[i] + lb[i]) / 2

    return x0_qp


def getPermutationMat(H, ran):
    U, S, V = svd(H)
    V2 = V[:, ran + 1:]

    n = V2.shape[0]  # number of variables
    dim = n - ran

    Comb = np.array(list(combinations(range(n), dim)))[:, ::-1]

    for i in range(Comb.shape[0]):
        comb = Comb[i, :]

        V22 = V2[comb, :]

        H_base = H.copy()
        H_base[:, comb] = np.zeros((H.shape[0], dim))

        det_V22 = np.linalg.det(V22)
        if (abs(det_V22) > 1e-06):
            ran_V22 = np.linalg.matrix_rank(V22)
        else:
            ran_V22 = np.linalg.matrix_rank(V22) - 1

        if ran_V22 == dim and np.linalg.matrix_rank(H_base) == ran:
            break

    V21 = np.delete(V2, comb, axis=0)

    return V21, V22, comb


def getEstimate(Robot, Q, DH_params):
    _, n_points = Q.shape
    P = np.zeros((7, n_points))

    for i in range(n_points):
        q = Q[:, i]
        _, P[:, i] = Robot.getPoseNum(q, DH_params)
        # _, _, P[:, i] = Robot.getPose(q, DH_params)

    return P


# class get_Model:
#     def __init__(self, Robot, dim, P_m, Q, x0, W, w_p, DH_param_lims, options):
#         self.Robot = Robot
#         self.dim = dim
#         self.P_m = P_m
#         self.Q = Q
#         self.x0 = x0
#         self.W = W
#         self.w_p = w_p
#         self.DH_param_lims = DH_param_lims
#         self.options = options



def get_Model(Robot, dim, P_m, Q, x0, W, w_p, DH_param_lims, options):
    damping = options['damping']
    solver = options['solver']

    DH_params = x0.copy()
    n_var = len(DH_params) # 4*nj
    n_joints, _ = Q.shape

    W_p = w_p.T.flatten()
    f = np.where(W_p == 0)[0]
    f_optimize = np.where(W_p == 1)[0]
    W_p = np.diag(W_p)
    f_notOptimize = f

    [n_dim, m] = P_m.shape

    resid = np.zeros((n_dim, m))

    Err = 1e-15
    dErr = 1e-20
    Err_rel = 1e-03
    Iter = options['MaxIter']

    Inliers_Idx = np.arange(m)

    n_points = len(Inliers_Idx)
    iter = 0

    err = 0

    options = {
        #"maxiter": 1000,  # maximum number of iterations
        #"ftol": 1e-9,     # stopping tolerance for the objective function
        "verbose": False  # suppress solver output
    }

    # x, fval, exitflag, _ = solve_qp(H, f, A=None, b=None, C=None, d=None, lb=lb, ub=ub, initvals=x0_qp, solver="quadprog", options=options)


    for i in range(m):
        q = Q[:,i]
        _, P_expect = Robot.getPoseNum(q, DH_params) # replaced DH_params with x0
        P_expect = P_expect[dim, 0]
    
        resid[:,i] = P_m[:,i] - P_expect
        err_i = (np.linalg.norm(W[:,i] * resid[:,i]))**2
        err += err_i

    err = err/m
    print("Initial err = " + str(err))

    err_opt = err
    DH_params_opt = DH_params

    t = 0

    Lambda = damping
    err_old = err
    err_prev = err_old


    while err > Err and iter < Iter:
        start = time.time()

        dP = np.zeros(n_dim*n_points)
        D = np.zeros((n_dim*n_points,n_var))
        x_tot = np.zeros(n_var)
        x_base_tot = np.zeros(n_var)


        # tempDH = DH_params[1::5]
        # tempA = np.sin(DH_params[1::5])
        # tempB = np.cos(DH_params[1::5])
        # tempC = np.arctan2(tempA, tempB)

        DH_params[1::5] = np.arctan2(np.sin(DH_params[1::5]), np.cos(DH_params[1::5]))
        DH_params[3::5] = np.arctan2(np.sin(DH_params[3::5]), np.cos(DH_params[3::5]))
        DH_params[4::5] = np.arctan2(np.sin(DH_params[4::5]), np.cos(DH_params[4::5]))

        lb = (DH_param_lims[:, 0] - DH_params) - 1e-06
        ub = (DH_param_lims[:, 1] - DH_params) + 1e-06

        lb = np.delete(lb, f_notOptimize)
        ub = np.delete(ub, f_notOptimize)
        
        for i in range(n_points):
        
            j = Inliers_Idx[i]
        
            q = Q[:, j]
        
            Robot, _, P_expect, Dp, Dor = Robot.getKineDeriv_Ana(q, DH_params)
            P_expect = P_expect[dim, 0]
        
            v1 = n_dim*i-n_dim
            v2 = v1+n_dim
        
            dP[v1:v2, 0] = W[:, j]*(P_m[:, j]-P_expect)
        
            Der = np.vstack((Dp, Dor))
            D[v1:v2, :] = W[:, j]*Der[dim, :].dot(W_p)
        
        D = np.delete(D, f_notOptimize, axis=1)
        H = D.T.dot(D)
        U, S, V = np.linalg.svd(H)
        ran = np.linalg.matrix_rank(H)

        if ran < len(f_optimize):
    
            V21, V22, comb = getPermutationMat(H, ran) # comb = parameters that are in combination with others
    
            comp = list(range(len(f_optimize)))
            comp_base = comp.copy()
            for i in comb: # parameters that can be optimized independently
                comp_base.remove(i)
        else:
            comp = list(range(len(f_optimize)))
            comp_base = comp.copy()

        I = np.eye(len(f_optimize), len(f_optimize))
        H = np.dot(D.T, D) + Lambda * I
        f = -np.dot(D.T, dP)

        print("*****SOLVE OPTIMIZATION")
        print("iter = " + str(iter))

        if solver == "pinv":
            x = -np.linalg.pinv(H) @ f
        elif solver == "qp":
            x0_qp = initializeSol(H, f, lb, ub)
            x = solve_qp(H, f, None, None, None, None, lb, ub, x0_qp, solver="quadprog", options=options)
            if x is None:
                print("*******INFEASIBLE******")
                break
        # elif solver == "gd":

        if ran < len(f_optimize):
            Perm_comp_base = np.zeros((len(comp_base), len(x)))
            for n_cb in range(len(comp_base)):
                Perm_comp_base[n_cb, comp_base[n_cb]] = 1
            Perm_comb = np.zeros((len(comb), len(x)))
            for n_c in range(len(comb)):
                Perm_comb[n_c, comb[n_c]] = 1
            x_base = np.vstack((x[comp_base, :] - V21 @ np.linalg.inv(V22) @ x[comb, :], np.zeros((len(comb), 1))))
            K = np.vstack((Perm_comp_base - V21 @ np.linalg.inv(V22) @ Perm_comb, np.zeros((len(comb), len(x)))))
        else:
            x_base = x
            K = np.eye(len(f_optimize))

        K_tot = np.zeros((n_var, n_var))
        K_tot[f_optimize, f_optimize] = K

        x_tot = np.zeros((n_var, 1))
        x_tot[f_optimize, 0] = x
        x_base_tot = np.zeros((n_var, 1))
        x_base_tot[f_optimize, 0] = x_base
        DH_params_old = DH_params
        DH_params_base = DH_params + x_base_tot
        DH_params = DH_params + x_tot

        err = 0

        #  "MSE"
        for i in range(m):
            q = Q[:,i]
            _, P_expect = Robot.getPoseNum(q, DH_params)
            P_expect = P_expect[dim, 0]
    
            resid[:,i] = P_m[:,i] - P_expect
            err_i = (np.linalg.norm(W[:,i] * resid[:,i]))**2
            err += err_i

        err = err/m
        print("err = " + str(err))
        print("lambda = " + str(Lambda))

        if solver != "gd":
            if err > err_old:
                Lambda = Lambda * 10
                Lambda = min(Lambda, 1e03)
                DH_params = DH_params_old
                derr = abs(err - err_prev)
                err_prev = err
                if Lambda >= 1e06 and derr < dErr:
                    print("ERROR NOT DECREASING. Error change:", derr)
                    break
            elif err <= err_old:
                Lambda = Lambda / 5
                Lambda = max(Lambda, 1e-06)
                derr = abs(err - err_old)
                err_old = err
                err_rel = derr / err
                if Lambda <= 1e-06:
                    if derr < dErr:
                        print("ERROR NOT DECREASING. Error change:", derr, "<", dErr)
                        break

        if err < err_opt:
            err_opt = err
            print("err opt = "+str(err_opt))
            DH_params_opt = DH_params
            x_opt = x
            dP_opt = dP
            D_opt = D

        end = time.time()
        time_iter = end - start

        t = t + time_iter

        if np.linalg.norm(x) <= 1e-12:
            print("PARAMETERS NOT CHANGING. Parameters change: {} < {}".format(np.linalg.norm(x), 1e-12))
            break
    print("MAX ITER REACHED")
    DH_params = DH_params_opt

    P = getEstimate(Robot, Q, DH_params)
    R2 = rsqrd(P_m,P[dim,:])

    err = np.zeros(len(dim))

    for i in range(len(dim)):
        err[i] = mean_squared_error(P_m[i, :], P[dim[i], :])

    return DH_params, P, W


        #Info = {
        #    "err_iter": [],
        #    "dx": [],
        #    "dx_base": [],
        #    "base_params": [],
        #    "base_params_values": [],
        #    "PermMat": [],
        #    "Observability": []
        #}

        #Info["err_iter"].append(err_opt)
        #Info["dx"].append

        #Info['err_iter'][iter-1] = err_opt
        #Info['dx'][:,iter-1] = x_tot
        #Info['dx_base'][:,iter-1] = x_base_tot
        #Info['base_params'][iter-1] = f_optimize[comp_base]
        #Info['base_params_values'][:,iter-1] = DH_params_base
        #Info['PermMat'][:,:,iter-1] = K_tot

        #[obs, _] = AnalyzeData(dP, D[:,comp_base], x[comp_base,:], n_pars, m)
        #Info['Observability'][:,iter-1] = obs




