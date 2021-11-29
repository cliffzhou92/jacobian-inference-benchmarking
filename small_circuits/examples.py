import numpy as np
import pandas as pd
import os.path

import matplotlib.pyplot as plt

import small_circ_func as scf
import plot_small_circuits as psc

def double_positive(S0X=1, S0Y = 1., n = 2., g = 0.1, k0 = 0., kx = 1, ky = 1, hill_X='pos', hill_Y='pos', T=200., dt=0.1, seed=100, T_pert = 10, dt_pert = 0.1, return_traj=True):
    if not os.path.isdir('results'):
        os.mkdir('results')

    np.random.seed(seed)
    pars = [n, g, kx, ky, k0, S0X, S0Y, hill_X, hill_Y]
    
    # pick up steady state values starting from random initial conditions
    print('Finding stable fixed point...')
    Ux0, Uy0, Sx0, Sy0 = 1, 1, 1, 1
    Ux, Uy, Sx, Sy = scf.time_course_twogenes(T, dt, Ux0, Uy0, Sx0, Sy0, n = n, g = g, kx = kx, ky = ky, k0 = k0, S0X = S0X, S0Y = S0Y)
    UXss, UYss, SXss, SYss = Ux[-1], Uy[-1], Sx[-1], Sy[-1]
    x0 = UXss, UYss, SXss, SYss

    # true jacobian
    Jxy = kx*scf.pos_hill_der(SYss, S0Y, n)
    Jyx = ky*scf.pos_hill_der(SXss, S0X, n)
    true_jac = np.array([[0, Jxy], [Jyx, 0]])

    print('Running regression...')

    B, G, K, U_data, S_data = scf.run_regression(x0, pars, circ='double_pos', n_traj=2000, n_sim=10, noise_scale_init=1e-1,
                                                          noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert, return_traj=return_traj)
    psc.plot_double_positive(B, true_jac, x0, U_data)

    return [x0], B


def double_negative(S0X=100, S0Y = 100., n = 4., g = 0.1, k0 = 5, kx = 20, ky = 20, hill_X='neg',hill_Y='neg', T=200., dt=0.1, seed=100, T_pert = 10, dt_pert = 0.1, return_traj=True):
    if not os.path.isdir('results'):
        os.mkdir('results')

    np.random.seed(seed)

    print('Finding stable fixed points...')
    Ux, Uy, Sx, Sy = scf.time_course_twogenes(T, dt, 10,0,10.,0., n = n, g = g, kx = kx, ky = ky, k0 = k0, S0X = S0X, S0Y = S0Y, hill_X='neg', hill_Y='neg')
    UXss1, UYss1, SXss1, SYss1 = Ux[-1], Uy[-1], Sx[-1], Sy[-1]
    x01 = UXss1, UYss1, SXss1, SYss1

    Ux, Uy, Sx, Sy = scf.time_course_twogenes(T, dt, 0, 10, 0., 10., n = n, g = g, kx = kx, ky = ky, k0 = k0, S0X = S0X, S0Y = S0Y, hill_X='neg',hill_Y='neg')
    UXss2, UYss2, SXss2, SYss2 = Ux[-1], Uy[-1], Sx[-1], Sy[-1]
    x02 = UXss2, UYss2, SXss2, SYss2

    # run regression on both fixed points
    pars = [n, g, kx, ky, k0, S0X, S0Y, hill_X, hill_Y]

    print('Running regression...')
    B1, G1, K1, U1_data, S1_data = scf.run_regression(x01, pars, circ='double_neg', n_traj=2000, n_sim=10, noise_scale_init=1e-1,
                                                          noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert, return_traj=return_traj)
    B2, G2, K2, U2_data, S2_data = scf.run_regression(x02, pars, circ='double_neg', n_traj=2000, n_sim=10, noise_scale_init=1e-1,
                                                          noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert, return_traj=return_traj)

    # true jacobians
    mat1 = np.array([[0, kx*scf.neg_hill_der(SYss1, S0Y, n)], [ky*scf.neg_hill_der(SXss1, S0X, n), 0]])
    mat2 = np.array([[0, kx*scf.neg_hill_der(SYss2, S0Y, n)], [ky*scf.neg_hill_der(SXss2, S0X, n), 0]])

    psc.plot_double_negative(x01, x02, U1_data, U2_data, B1, B2, mat1, mat2)

    return [x01, x02], [B1, B2]


def repressilator(S0=10., n=2, g=0.1, k=10., T=500., dt=0.1, seed=100, T_pert = 10, dt_pert = 0.1, return_traj=True):
    if not os.path.isdir('results'):
        os.mkdir('results')

    np.random.seed(seed)

    # pick up steady state values starting from random initial conditions
    print('Finding fixed point...')
    Ux, Uy, Uz, Sx, Sy, Sz = scf.time_course_repr(T, dt, 1., 2., 2., 1., 2., 1., S0=S0, n=n, g=g, k=k, sigma=0)
    UX1, UY1, UZ1, SX1, SY1, SZ1 = Ux[-1], Uy[-1], Uz[-1], Sx[-1], Sy[-1], Sz[-1]
    x0 = [UX1, UY1, UZ1, SX1, SY1, SZ1]

    Jxy = k * scf.pos_hill_der(SX1, S0, n)
    Jyz = k * scf.pos_hill_der(SY1, S0, n)
    Jzx = k * scf.pos_hill_der(SZ1, S0, n)
    true_jac = np.array([[0, Jxy, 0], [0, 0, Jyz], [Jzx, 0, 0]])

    print('Running regression...')
    pars = [S0, n, g, k]
    B, G, K, U_data, S_data = scf.run_regression(x0, pars, circ='repressilator', n_traj=2000, n_sim=10, noise_scale_init=1e-1,
                       noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert, return_traj=return_traj)

    psc.plot_repr(B, true_jac, x0, U_data)
    return [x0], B


def tristable(k=4, g=0.1, nxy=1, nyx=1, nxx=3, nyy=3, lxy=0.1, lyx=0.1, lxx=10, lyy=10, Scross=150,
              Sself=100, T=500., dt=0.1, seed=100, T_pert = 10, dt_pert = 0.1, return_traj=True):
    if not os.path.isdir('results'):
        os.mkdir('results')

    np.random.seed(seed)

    print('Finding fixed point...')

    # fixed point 1
    Ux, Uy, Sx, Sy = scf.time_course_tristable(T, dt, 1., 10., 1., 100., k=k, g=g, nxy=nxy, nyx=nyx, nxx=nxx, nyy=nyy,
                                               lxy=lxy, lyx=lyx, lxx=lxx, lyy=lyy, Scross=Scross, Sself=Sself, sigma=0)
    x1 = Ux[-1], Uy[-1], Sx[-1], Sy[-1]
    J1 = scf.jacobian_tristable(x1)

    # fixed point 2
    Ux, Uy, Sx, Sy = scf.time_course_tristable(T, dt, 10., 1., 100., 1., k=k, g=g, nxy=nxy, nyx=nyx, nxx=nxx, nyy=nyy,
                                               lxy=lxy, lyx=lyx, lxx=lxx, lyy=lyy, Scross=Scross, Sself=Sself, sigma=0)
    x2 = Ux[-1], Uy[-1], Sx[-1], Sy[-1]
    J2 = scf.jacobian_tristable(x2)

    # fixed point 3
    Ux, Uy, Sx, Sy = scf.time_course_tristable(T, dt, 9., 10., 99., 100.,k=k, g=g, nxy=nxy, nyx=nyx, nxx=nxx, nyy=nyy,
                                               lxy=lxy, lyx=lyx, lxx=lxx, lyy=lyy, Scross=Scross, Sself=Sself, sigma=0)
    x3 = Ux[-1], Uy[-1], Sx[-1], Sy[-1]
    J3 = scf.jacobian_tristable(x3)

    print('Running regression...')
    pars = [k, g, nxy, nyx, nxx, nyy, lxy, lyx, lxx, lyy, Scross, Sself]

    B1, G1, K1, U1, S1 = scf.run_regression(x1, pars, circ='tristable', n_traj=2000, n_sim=10, noise_scale_init=1e-1,
                       noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert, return_traj=return_traj)
    B2, G2, K2, U2, S2 = scf.run_regression(x2, pars, circ='tristable', n_traj=2000, n_sim=10, noise_scale_init=1e-1,
                       noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert, return_traj=return_traj)
    B3, G3, K3, U3, S3 = scf.run_regression(x3, pars, circ='tristable', n_traj=2000, n_sim=10, noise_scale_init=1e-1,
                       noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert, return_traj=return_traj)

    psc.plot_tristable(x1, x2, x3, B1, B2, B3, J1, J2, J3, U1, U2, U3)

    return [x1, x2, x3], [B1, B2, B3]



def sensitivity(x0, tag='double_pos_', n_traj_vec=np.arange(1000, 5001, 500), n_iter_vec=np.arange(5, 16, 1),
                verb=False, seed=100, T_pert = 10, dt_pert = 0.1, return_traj=True, print_status=False):

    if not os.path.isdir('data'):
        os.mkdir('data')

    np.random.seed(seed)

    # true value
    if tag=='double_pos_':
        UXss, UYss, SXss, SYss = x0
        S0X, S0Y, n, g, k0, kx, ky = 1, 1, 2, 0.1, 0, 1, 1
        Jxy = kx * scf.pos_hill_der(SYss, S0Y, n)
        hill_X, hill_Y = 'pos', 'pos'
    elif tag=='double_neg_':
        UXss, UYss, SXss, SYss = x0
        S0X, S0Y, n, g, k0, kx, ky = 100, 100, 4, 0.1, 5, 20, 20
        Jxy = ky * scf.neg_hill_der(SXss, S0X, n)
        hill_X, hill_Y = 'neg', 'neg'
    elif tag=='repressilator_':
        UX1, UY1, UZ1, SX1, SY1, SZ1 = x0
        S0, n, g, k = 10, 2, 0.1, 10
        Jxy = k * scf.pos_hill_der(SX1, S0, n)
    elif tag=='tristable_':
        UXss, UYss, SXss, SYss = x0
        k, g, nxy, nyx, nxx, nyy, lxy, lyx, lxx, lyy, Scross, Sself = 4, 0.1, 1, 1, 3, 3, 0.1, 0.1, 10, 10, 150, 100
        Jxy = k * scf.shifted_hill(SXss, Sself, lxx, nxx) * scf.der_shifted(SYss, Scross, lyx, nyx)

    if not os.path.exists('data/'+tag+'n_traj_sens.txt'):
        print('Testing sensitivity to number of trajectories...')
        est_ntraj = np.zeros(n_traj_vec.size)
        for i in range(n_traj_vec.size):
            if print_status:
                print('n_traj=' + str(n_traj_vec[i]))
            if tag=='repressilator_':
                B, G, K, U_data, S_data = scf.run_regression_repr(x0, n_traj=n_traj_vec[i], n_sim=10, noise_scale_init=1e-1,
                                                                  noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert,
                                                                  S0=S0, n=n, g=g, k=k, verb=verb,
                                                                  return_traj=return_traj)
            elif tag=='tristable_':
                B, G, K, U, S = scf.run_regression_tristable(x0, n_traj=n_traj_vec[i], n_sim=10, noise_scale_init=1e-1,
                                                                  noise_scale_traj=1e-2, T_pert=T_pert,
                                                                  dt_pert=dt_pert, k=k, g=g, nxy=nxy, nyx=nyx, nxx=nxx,
                                                                  nyy=nyy,
                                                                  lxy=lxy, lyx=lyx, lxx=lxx, lyy=lyy, Scross=Scross,
                                                                  Sself=Sself,
                                                                  verb=verb, return_traj=return_traj)
            else:
                B, G, K, U_data, S_data = scf.run_regression_twogenes(x0, n_traj=n_traj_vec[i], n_sim=10,
                                                                      noise_scale_init=1e-1, noise_scale_traj=1e-2,
                                                                      T_pert=T_pert, dt_pert=dt_pert, n=n, g=g, kx=kx,
                                                                      ky=ky, k0=k0, S0X=S0X, S0Y=S0Y, hill_X=hill_X,
                                                                      hill_Y=hill_Y, verb=verb, return_traj=return_traj)
            est_ntraj[i] = B[1][0]
        data = np.array([n_traj_vec, est_ntraj])
        np.savetxt('data/'+tag+'n_traj_sens.txt', data)
    else:
        est_ntraj = np.loadtxt('data/'+tag+'n_traj_sens.txt')[1]

    if not os.path.exists('data/'+tag+'n_iter_sens.txt'):
        print('Testing sensitivity to number of iterations...')
        est_niter = np.zeros(n_iter_vec.size)
        for i in range(n_iter_vec.size):
            if print_status:
                print('n_iter=' + str(n_iter_vec[i]))
            if tag=='repressilator_':
                B, G, K, U_data, S_data = scf.run_regression_repr(x0, n_traj=2000, n_sim=n_iter_vec[i],
                                                                  noise_scale_init=1e-1,
                                                                  noise_scale_traj=1e-2, T_pert=T_pert, dt_pert=dt_pert,
                                                                  S0=S0, n=n, g=g, k=k, verb=verb,
                                                                  return_traj=return_traj)
            elif tag=='tristable_':
                B, G, K, U, S = scf.run_regression_tristable(x0, n_traj=2000, n_sim=n_iter_vec[i], noise_scale_init=1e-1,
                                                             noise_scale_traj=1e-2, T_pert=T_pert,
                                                             dt_pert=dt_pert, k=k, g=g, nxy=nxy, nyx=nyx, nxx=nxx,
                                                             nyy=nyy,
                                                             lxy=lxy, lyx=lyx, lxx=lxx, lyy=lyy, Scross=Scross,
                                                             Sself=Sself,
                                                             verb=verb, return_traj=return_traj)
            else:
                B, G, K, U_data, S_data = scf.run_regression_twogenes(x0, n_traj=2000, n_sim=n_iter_vec[i],
                                                                      noise_scale_init=1e-1, noise_scale_traj=1e-2,
                                                                      T_pert=T_pert, dt_pert=dt_pert, n=n, g=g, kx=kx,
                                                                      ky=ky, k0=k0, S0X=S0X, S0Y=S0Y, hill_X=hill_X,
                                                                      hill_Y=hill_Y, verb=verb, return_traj=return_traj)
            est_niter[i] = B[1][0]
        data = np.array([n_iter_vec, est_niter])
        np.savetxt('data/'+tag+'n_iter_sens.txt', data)
    else:
        est_niter = np.loadtxt('data/'+tag+'n_iter_sens.txt')[1]

    # vary both n_iter and n_traj
    if not os.path.exists('data/'+tag+'ntraj_niter_sens.txt'):
        print('Testing sensitivity to number of iterations and trajectories...')
        est = np.zeros((n_traj_vec.size, n_iter_vec.size))
        for i in range(n_traj_vec.size):
            for j in range(n_iter_vec.size):
                if print_status:
                    print('n_traj, n_iter=' + str(n_traj_vec[i]) + ', ' + str(n_iter_vec[j]))
                if tag=='repressilator_':
                    B, G, K, U_data, S_data = scf.run_regression_repr(x0, n_traj=n_traj_vec[i], n_sim=n_iter_vec[j],
                                                                      noise_scale_init=1e-1,
                                                                      noise_scale_traj=1e-2, T_pert=T_pert,
                                                                      dt_pert=dt_pert,
                                                                      S0=S0, n=n, g=g, k=k, verb=verb,
                                                                      return_traj=return_traj)
                elif tag=='tristable_':
                    B, G, K, U, S = scf.run_regression_tristable(x0, n_traj=n_traj_vec[i], n_sim=n_iter_vec[j],
                                                                 noise_scale_init=1e-1,
                                                                 noise_scale_traj=1e-2, T_pert=T_pert,
                                                                 dt_pert=dt_pert, k=k, g=g, nxy=nxy, nyx=nyx, nxx=nxx,
                                                                 nyy=nyy,
                                                                 lxy=lxy, lyx=lyx, lxx=lxx, lyy=lyy, Scross=Scross,
                                                                 Sself=Sself,
                                                                 verb=verb, return_traj=return_traj)
                else:
                    B, G, K, U_data, S_data = scf.run_regression_twogenes(x0, n_traj=n_traj_vec[i], n_sim=n_iter_vec[j],
                                                                          noise_scale_init=1e-1, noise_scale_traj=1e-2,
                                                                          T_pert=T_pert, dt_pert=dt_pert, n=n, g=g,
                                                                          kx=kx,
                                                                          ky=ky, k0=k0, S0X=S0X, S0Y=S0Y, hill_X=hill_X,
                                                                          hill_Y=hill_Y, verb=verb,
                                                                          return_traj=return_traj)
                est[i][j] = B[1][0]
        np.savetxt('data/'+tag+'ntraj_niter_sens.txt', est)
    else:
        est = np.loadtxt('data/'+tag+'ntraj_niter_sens.txt')

    psc.plot_sens(Jxy, n_traj_vec, n_iter_vec, est_ntraj, est_niter, est, figname='results/'+tag+'sens.pdf', format='pdf', showfig=False)





