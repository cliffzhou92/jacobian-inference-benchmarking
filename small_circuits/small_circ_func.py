import numpy as np
import time
from numba import jit_module

from sklearn.linear_model import Ridge, LinearRegression, Lasso

def pos_hill(x, x0, n):
    '''
    positive hill fucntion
    :param x: regulator level
    :param x0: threshold
    :param n: hill coefficient
    :return: hill function value at point x
    '''
    return ((x/x0)**n)/( 1 + (x/x0)**n )

def pos_hill_der(x, x0, n):
    '''
    derivative of positive hill function
    :param x: regulator level
    :param x0: threshold
    :param n: hill coefficient
    :return: hill function value at point x
    '''
    return (n*((x/x0)**n))/( x*(1 + (x/x0)**n)**2 )

def neg_hill(x, x0, n):
    '''
    negative hill function
    :param x: regulator level
    :param x0: threshold
    :param n: hill coefficient
    :return: hill function value at point x
    '''
    return 1./( 1 + (x/x0)**n )

def neg_hill_der(x, x0, n):
    '''
    derivative of negative hill function
    :param x: regulator level
    :param x0: threshold
    :param n: hill coefficient
    :return: hill function value at point x
    '''
    return -(n*((x/x0)**n))/( x*(1 + (x/x0)**n)**2 )

def shifted_hill(x, x0, l, n):
    '''
    shiften hill function
    :param x: regulator level
    :param x0: threshold
    :param l: fold-change
    :param n: hill coefficient
    :return: hill function value at point x
    '''
    return (1 + l*((x/x0)**n))/(1 + ((x/x0)**n))

def der_shifted(x, x0, l, n):
    '''
    derivative if shiften hill function
    :param x: regulator level
    :param x0: threshold
    :param l: fold-change
    :param n: hill coefficient
    :return: derivative value at point x
    '''
    return (l-1)*n*((x/x0)**n)/( x*(( (x/x0)**n + 1 )**2) )


def jacobian_tristable(x0, k=4, g=0.1, nxy=1, nyx=1, nxx=3, nyy=3, lxy=0.1, lyx=0.1, lxx=10, lyy=10, Scross=150, Sself=100):
    Ux, Uy, Sx, Sy = x0
    Jxx = k * der_shifted(Sx, Sself, lxx, nxx) * shifted_hill(Sy, Scross, lyx, nyx)
    Jxy = k * shifted_hill(Sx, Sself, lxx, nxx) * der_shifted(Sy, Scross, lyx, nyx)
    Jyx = k * shifted_hill(Sy, Sself, lyy, nyy) * der_shifted(Sx, Scross, lxy, nxy)
    Jyy = k * der_shifted(Sy, Sself, lyy, nyy) * shifted_hill(Sx, Scross, lxy, nxy)
    jac = np.array([[Jxx, Jxy], [Jyx, Jyy]])
    return jac


### time course functions

def time_course_twogenes(T, dt, Ux0, Uy0, Sx0, Sy0, n = 2, g = 0.1, kx = 1, ky = 1, k0 = 0., S0X = 1., S0Y = 1., sigma=0.002, hill_X='pos', hill_Y='pos'):

    npoints = int(T/dt)
    Ux, Uy, Sx, Sy = np.zeros(npoints+1),np.zeros(npoints+1),np.zeros(npoints+1),np.zeros(npoints+1)
    Ux[0], Uy[0], Sx[0], Sy[0] = Ux0, Uy0, Sx0, Sy0
    gaussian_noise = sigma*np.random.normal(loc = 0., scale = np.sqrt(dt), size=(npoints,4))
    for i in range(npoints):
        fX=pos_hill(Sy[i], S0Y, n) if hill_X=='pos' else neg_hill(Sy[i], S0Y, n)
        fY=pos_hill(Sx[i], S0X, n) if hill_Y=='pos' else neg_hill(Sx[i], S0X, n)

        Ux[i+1] = Ux[i] + dt*( k0 + kx*fX - Ux[i] )+ gaussian_noise[i][0]
        Uy[i+1] = Uy[i] + dt*( k0 + ky*fY - Uy[i] )+ gaussian_noise[i][1]
        Sx[i+1] = Sx[i] + dt*( Ux[i] - g*Sx[i] ) + gaussian_noise[i][2]
        Sy[i+1] = Sy[i] + dt*( Uy[i] - g*Sy[i] ) + gaussian_noise[i][3]
    return Ux, Uy, Sx, Sy


def time_course_repr(T, dt, Ux0, Uy0, Uz0, Sx0, Sy0, Sz0, S0=10., n=2, g=0.1, k=10., sigma=0.002):
    '''
    time course starting from given initial conditions
    T, dt: time interval and timestep
    Ux0, Sx0, Uy0, Sy0: initial conditions
    S0X, S0Y, n, gm, kx, ky: controllable parameters of hill function
    sigma: the white noise amplitude
    returns full vectorts of time course
    '''
    npoints = int(T/dt)
    Ux, Uy, Uz, Sx, Sy, Sz = np.zeros(npoints+1),np.zeros(npoints+1),np.zeros(npoints+1),np.zeros(npoints+1), np.zeros(npoints+1), np.zeros(npoints+1)
    Ux[0], Uy[0], Uz[0], Sx[0], Sy[0], Sz[0] = Ux0, Uy0, Uz0, Sx0, Sy0, Sz0
    gaussian_noise = sigma*np.random.normal(loc = 0., scale = np.sqrt(dt), size=(npoints,6))
    for i in range(npoints):
        fX=pos_hill(Sz[i], S0, n)
        fY=pos_hill(Sx[i], S0, n)
        fZ=pos_hill(Sy[i], S0, n)

        Ux[i+1] = Ux[i] + dt*( k*fX - Ux[i] )+ gaussian_noise[i][0]
        Uy[i+1] = Uy[i] + dt*( k*fY - Uy[i] )+ gaussian_noise[i][1]
        Uz[i+1] = Uz[i] + dt*( k*fZ - Uz[i] )+ gaussian_noise[i][2]
        Sx[i+1] = Sx[i] + dt*( Ux[i] - g*Sx[i] ) + gaussian_noise[i][3]
        Sy[i+1] = Sy[i] + dt*( Uy[i] - g*Sy[i] ) + gaussian_noise[i][4]
        Sz[i+1] = Sz[i] + dt*( Uz[i] - g*Sz[i] ) + gaussian_noise[i][5]
    return Ux, Uy, Uz, Sx, Sy, Sz


def time_course_tristable(T, dt, Ux0, Uy0, Sx0, Sy0, k=4, g=0.1, nxy=1, nyx=1, nxx=3, nyy=3, lxy=0.1, lyx=0.1, lxx=10, lyy=10, Scross=150, Sself=100, sigma=0.002):
    '''
    time course starting from given initial conditions
    T, dt: time interval and timestep
    Ux0, Sx0, Uy0, Sy0: initial conditions
    S0X, S0Y, n, gm, kx, ky: controllable parameters of hill function
    sigma: the white noise amplitude
    returns full vectorts of time course
    '''
    npoints = int(T/dt)
    Ux, Uy, Sx, Sy = np.zeros(npoints+1), np.zeros(npoints+1), np.zeros(npoints+1), np.zeros(npoints+1)
    Ux[0], Uy[0], Sx[0], Sy[0] = Ux0, Uy0, Sx0, Sy0
    gaussian_noise = sigma*np.random.normal(loc = 0., scale = np.sqrt(dt), size=(npoints,4))
    for i in range(npoints):
        fX=shifted_hill(Sx[i], Sself, lxx, nxx)*shifted_hill(Sy[i], Scross, lyx, nyx)
        fY=shifted_hill(Sy[i], Sself, lyy, nyy)*shifted_hill(Sx[i], Scross, lxy, nxy)

        Ux[i+1] = Ux[i] + dt*( k*fX - Ux[i] )+ gaussian_noise[i][0]
        Uy[i+1] = Uy[i] + dt*( k*fY - Uy[i] )+ gaussian_noise[i][1]
        Sx[i+1] = Sx[i] + dt*( Ux[i] - g*Sx[i] ) + gaussian_noise[i][2]
        Sy[i+1] = Sy[i] + dt*( Uy[i] - g*Sy[i] ) + gaussian_noise[i][3]
    return Ux, Uy, Sx, Sy

jit_module()


def generate_data(x0, pars, circ='double_pos', n_traj=2000, noise_scale_initial = 0.1,noise_scale_traj = 1e-3, T_pert = 10, dt_pert = 0.1):
  N = int(T_pert / dt_pert) + 1

  if circ=='double_pos' or circ=='double_neg':
      n, g, kx, ky, k0, S0X, S0Y, hill_X, hill_Y = pars
      U_data, S_data, nvar = np.zeros((n_traj * N, 2)), np.zeros((n_traj * N, 2)), 4
  elif circ=='repressilator':
      S0, n, g, k = pars
      U_data, S_data, nvar = np.zeros((n_traj * N, 3)), np.zeros((n_traj * N, 3)), 6
  elif circ=='tristable':
      k, g, nxy, nyx, nxx, nyy, lxy, lyx, lxx, lyy, Scross, Sself = pars
      U_data, S_data, nvar = np.zeros((n_traj * N, 2)), np.zeros((n_traj * N, 2)), 4

  for i in range(n_traj):
    # pick a small, Gaussian displacement around the fixed point
    if circ=='repressilator':
        [UX0, UY0, UZ0, SX0, SY0, SZ0] = x0 + noise_scale_initial*np.random.normal(loc = 0., scale = 1, size=6)*x0
    else:
        [UX0, UY0, SX0, SY0] = x0 + noise_scale_initial*np.random.normal(loc=0., scale=1, size=nvar)*x0

    if circ=='double_pos' or circ=='double_neg':
        Ux_fit, Uy_fit, Sx_fit, Sy_fit = time_course_twogenes(T_pert, dt_pert, UX0, UY0, SX0, SY0, n=n, g=g, kx=kx,
                                                              ky=ky, k0=k0, S0X=S0X, S0Y=S0Y,
                                                              sigma=noise_scale_traj, hill_X=hill_X, hill_Y=hill_Y)
    elif circ=='repressilator':
        Ux_fit, Uy_fit, Uz_fit, Sx_fit, Sy_fit, Sz_fit = time_course_repr(T_pert, dt_pert, UX0, UY0, UZ0, SX0, SY0, SZ0,
                                                                          S0=S0, n=n, g=g, k=k, sigma=noise_scale_traj)
    elif circ=='tristable':
        Ux_fit, Uy_fit, Sx_fit, Sy_fit = time_course_tristable(T_pert, dt_pert, UX0, UY0, SX0, SY0, k=k, g=g, nxy=nxy,
                                                               nyx=nyx, nxx=nxx, nyy=nyy, lxy=lxy, lyx=lyx, lxx=lxx, lyy=lyy,
                                                               Scross=Scross, Sself=Sself, sigma=noise_scale_traj)

    ind_sample = list(range(i*N,(i+1)*N))
    U_data[ind_sample,0] = Ux_fit
    S_data[ind_sample,0] = Sx_fit
    U_data[ind_sample,1] = Uy_fit
    S_data[ind_sample,1] = Sy_fit
    if nvar>4:
        U_data[ind_sample, 2] = Uz_fit
        S_data[ind_sample, 2] = Sz_fit
  return U_data,S_data



def run_regression(x0, pars, circ='double_pos', n_traj=2000, n_sim=10, noise_scale_init = 1e-1, noise_scale_traj = 1e-2,
                   T_pert = 10, dt_pert = 0.1, return_traj=True, method='Ridge', alpha=1):

    if circ=='repressilator':
        B_est, G_est, K_est = np.zeros((3, 3)), np.zeros(3), np.zeros(3)
    else:
        B_est, G_est, K_est = np.zeros((2, 2)), np.zeros(2), np.zeros(2)

    for n_iter in range(n_sim):
        U_data, S_data = generate_data(x0, pars, circ=circ, n_traj=n_traj, noise_scale_initial=noise_scale_init,
                                       noise_scale_traj=noise_scale_traj, T_pert=T_pert, dt_pert=dt_pert)
        B, K, G = regr_twogenes(U_data, S_data, method=method, alpha=alpha)

        B_est = B_est + B
        G_est = G_est + G
        K_est = K_est + K

    B_est = B_est / n_iter
    G_est = G_est / n_iter
    K_est = K_est / n_iter

    if return_traj:
        return B_est, G_est, K_est, U_data, S_data
    else:
        return B_est, G_est, K_est


def regr_twogenes(U_data, S_data, method='Ridge', alpha=1):
    # simplified version with two genes - need to use .reshape(-1, 1) to transform into 2D array
    '''
    ridge regression to infer interaction matrix from U=BS+C equation
    '''

    assert method=='Ridge' or method=='Linear' or method=='Lasso', 'Please choose between Linear, Ridge or Lasso regression'

    if method=='Linear':
        reg = LinearRegression()
    elif method == 'Ridge':
        reg = Ridge(alpha=alpha)
    elif method == 'Lasso':
        reg = Lasso(alpha=alpha)

    ncell, ngene = U_data.shape
    B = np.zeros((ngene, ngene))
    C = np.zeros(ngene)
    G = np.zeros(ngene)

    # gene i=0
    reg.fit(S_data[:, 1].reshape(-1, 1), U_data[:, 0])
    B[0][1] = reg.coef_
    C[0] = reg.intercept_
    reg_g = LinearRegression(fit_intercept=False)
    reg_g.fit(S_data[:, [0]], U_data[:, 0])
    G[0] = reg_g.coef_

    # gene i=1
    reg.fit(S_data[:, 0].reshape(-1, 1), U_data[:, 1])
    B[1][0] = reg.coef_
    C[1] = reg.intercept_
    reg_g = LinearRegression(fit_intercept=False)
    reg_g.fit(S_data[:, [1]], U_data[:, 1])
    G[1] = reg_g.coef_

    return B, C, G


def regr_general(U_data, S_data, method='Ridge', alpha=1):

    if method=='Linear':
        reg = LinearRegression()
    elif method == 'Ridge':
        reg = Ridge(alpha=alpha)
    elif method == 'Lasso':
        reg = Lasso(alpha=alpha)

    ncell, ngene = U_data.shape
    B = np.zeros((ngene, ngene))
    C = np.zeros(ngene)
    G = np.zeros(ngene)

    for i in range(ngene):
        # delete i-th component of S_data on axis=1 (i.e. spliced counts of same gene)
        S_use = np.delete(S_data, i, 1)
        reg.fit(S_use, U_data[:, i])
        coeffs = reg.coef_

        B[i][0:i] = coeffs[0:i]
        B[i][i + 1:] = coeffs[i:]
        C[i] = reg.intercept_

        # fit spliced degradation rate
        reg_g = LinearRegression(fit_intercept=False)
        reg_g.fit(S_data[:, [i]], U_data[:, i])
        G[i] = reg_g.coef_

    return B, C, G
