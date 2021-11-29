import numpy as np
import matplotlib.pyplot as plt

import small_circ_func as scf


def plot_jac(ax, mat, a, title, tick=[0.5, 1.5], lab=['X', 'Y']):
    pt = plt.pcolor(mat, cmap='RdBu', vmin=-a, vmax=+a)
    plt.colorbar(pt)
    plt.xticks(tick, lab)
    plt.yticks(tick, lab)
    plt.title(title)


def plot_pert(ax, x, y, x0, y0, xlim, ylim):
    plt.scatter(x, y, label='FP perturbation')
    plt.scatter([x0], [y0], label='Stable fixed point')
    plt.xlabel('$U_x$')
    plt.ylabel('$U_y$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='upper right')


def plot_double_positive(B, true_jac, x0, U_data, x = np.arange(0., 1.5, 0.01), S0X=1, S0Y = 1., n = 2., g = 0.1, kx = 1, ky = 1, savefig=True, showfig=False):
    UXss, UYss, SXss, SYss = x0

    plt.figure(figsize=(14, 4))

    ax1 = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=4)
    plt.plot(x, kx * scf.pos_hill(x, S0Y * g, n), label='$\\frac{dU_x}{dt}=0$')
    plt.plot(ky * scf.pos_hill(x, S0X * g, n), x, label='$\\frac{dU_y}{dt}=0$')
    plt.xlabel('$U_x$')
    plt.ylabel('$U_y$')
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.5])
    plt.legend(loc='upper left')

    # visualize perturbation trajectories
    pert_X, pert_Y = U_data[:, 0], U_data[:, 1]
    ax2 = plt.subplot2grid((2, 14), (0, 4), rowspan=2, colspan=4)
    plot_pert(ax2, pert_X[0:2000], pert_Y[0:2000], UXss, UYss, [0, 2], [0, 2])

    # jacobians
    a = max(np.amax(np.abs(B)), np.amax(np.abs(true_jac)))
    ax3 = plt.subplot2grid((2, 14), (0, 8), rowspan=1, colspan=3)
    plot_jac(ax3, true_jac, a, 'True Jacobian')
    ax4 = plt.subplot2grid((2, 14), (0, 11), rowspan=1, colspan=3)
    plot_jac(ax4, B, a, 'Estimated Jacobian')

    plt.tight_layout()
    if savefig:
        plt.savefig('results/double_positive.pdf', format='pdf')
    if showfig:
        plt.show()



def plot_double_negative(x01, x02, U1_data, U2_data, B1, B2, mat1, mat2, S0X=100, S0Y = 100., n = 4., g = 0.1, k0 = 5, kx = 20, ky = 20, savefig=True, showfig=False):
    UX1, UY1, SX1, SY1 = x01
    UX2, UY2, SX2, SY2 = x02

    pert1_X, pert1_Y = U1_data[:, 0], U1_data[:, 1]
    pert2_X, pert2_Y = U2_data[:, 0], U2_data[:, 1]

    plt.figure(figsize=(14, 4))

    x = np.arange(0.1, 30, 0.1)
    ax1 = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=4)
    plt.plot(x, k0 + kx * scf.neg_hill(x, S0Y * g, n), label='$\\frac{dU_x}{dt}=0$')
    plt.plot(k0 + ky * scf.neg_hill(x, S0X * g, n), x, label='$\\frac{dU_y}{dt}=0$')
    plt.xlabel('$U_x$');
    plt.ylabel('$U_y$');
    plt.xlim([0, 30])
    plt.ylim([0, 30])
    plt.legend(loc='upper right');

    ax2 = plt.subplot2grid((2, 14), (0, 4), rowspan=2, colspan=4)
    plt.scatter(pert1_X[0:2000], pert1_Y[0:2000], label='FP1 perturbation')
    plt.scatter(pert2_X[0:2000], pert2_Y[0:2000], label='FP2 perturbation')
    plt.scatter([UX1, UY1], [UX2, UY2], label='Stable fixed points')
    plt.xlabel('$U_x$');
    plt.ylabel('$U_y$');
    plt.xlim([0, 30])
    plt.ylim([0, 30])
    plt.legend(loc='upper right');

    a = max(np.amax(np.abs(B1)), np.amax(np.abs(B2)), np.amax(np.abs(mat1)), np.amax(np.abs(mat2)))

    ax3 = plt.subplot2grid((2, 14), (0, 8), rowspan=1, colspan=3)
    plot_jac(ax3, mat1, a, 'True Jacobian-FP1')

    ax3 = plt.subplot2grid((2, 14), (0, 11), rowspan=1, colspan=3)
    plot_jac(ax3, B1, a, 'Estimated Jacobian-FP1')

    ax3 = plt.subplot2grid((2, 14), (1, 8), rowspan=1, colspan=3)
    plot_jac(ax3, mat2, a, 'True Jacobian-FP2')

    ax4 = plt.subplot2grid((2, 14), (1, 11), rowspan=1, colspan=3)
    plot_jac(ax4, B2, a, 'Estimated Jacobian-FP2')

    plt.tight_layout()
    if savefig:
        plt.savefig('results/double_negative.pdf', format='pdf')
    if showfig:
        plt.show()



def plot_repr(B, true_jac, x0, U_data, x = np.arange(0., 20.1, 0.01), S0=10., n=2, g=0.1, k=10., savefig=True, showfig=False):

    UX1, UY1, UZ1, SX1, SY1, SZ1 = x0
    pert_X, pert_Y = U_data[:, 0], U_data[:, 1]

    plt.figure(figsize=(14, 4))

    ax1 = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=4)
    z = k * scf.pos_hill(x, S0 * g, n)
    plt.plot(x, k * scf.pos_hill(x, S0 * g, n), label='$\\frac{dU_x}{dt}=0$, $\\frac{dU_z}{dt}=0$')
    plt.plot(k * scf.pos_hill(z, S0 * g, n), x, label='$\\frac{dU_y}{dt}=0$, $\\frac{dU_z}{dt}=0$')
    plt.xlabel('$U_x$');
    plt.ylabel('$U_y$');
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.legend(loc='upper right');

    ax2 = plt.subplot2grid((2, 14), (0, 4), rowspan=2, colspan=4)
    plt.scatter(pert_X[0:2000], pert_Y[0:2000], label='FP perturbation')
    plt.scatter([UX1], [UY1], label='Stable fixed point')
    plt.xlabel('$U_x$');
    plt.ylabel('$U_y$');
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.legend(loc='upper right');

    a = max(np.amax(np.abs(B)), np.amax(np.abs(true_jac)))

    ax3 = plt.subplot2grid((2, 14), (0, 8), rowspan=1, colspan=3)
    pt = plt.pcolor(true_jac, cmap='RdBu', vmin=-a, vmax=+a)
    plt.colorbar(pt)
    plt.xticks([0.5, 1.5, 2.5], ['X', 'Y', 'Z']);
    plt.yticks([0.5, 1.5, 2.5], ['X', 'Y', 'Z']);
    plt.title('True Jacobian');

    ax3 = plt.subplot2grid((2, 14), (0, 11), rowspan=1, colspan=3)
    pt = plt.pcolor(np.transpose(B), cmap='RdBu', vmin=-a, vmax=+a)
    plt.colorbar(pt)
    plt.xticks([0.5, 1.5, 2.5], ['X', 'Y', 'Z']);
    plt.yticks([0.5, 1.5, 2.5], ['X', 'Y', 'Z']);
    plt.title('Estimated Jacobian');

    plt.tight_layout()
    if savefig:
        plt.savefig('results/repressilator.pdf', format='pdf')
    if showfig:
        plt.show()


def plot_tristable(x1, x2, x3, B1, B2, B3, J1, J2, J3, U1, U2, U3, savefig=True, showfig=False):
    plt.figure(figsize=(14, 6))

    Ux1, Uy1, Sx1, Sy1 = x1
    Ux2, Uy2, Sx2, Sy2 = x2
    Ux3, Uy3, Sx3, Sy3 = x3

    pert_x1, pert_y1 = U1[:, 0], U1[:, 1]
    pert_x2, pert_y2 = U2[:, 0], U2[:, 1]
    pert_x3, pert_y3 = U3[:, 0], U3[:, 1]

    ax1 = plt.subplot2grid((3, 14), (0, 0), rowspan=2, colspan=4)
    plt.scatter([Ux1, Ux2, Ux3], [Uy1, Uy2, Uy3])
    plt.xlabel('$U_x$');
    plt.ylabel('$U_y$');
    plt.xlim([0, 40])
    plt.ylim([0, 40])

    ax2 = plt.subplot2grid((3, 14), (0, 4), rowspan=2, colspan=4)
    plt.scatter(pert_x1[0:2000], pert_y1[0:2000], label='FP1 perturbation')
    plt.scatter(pert_x2[0:2000], pert_y2[0:2000], label='FP2 perturbation')
    plt.scatter(pert_x3[0:2000], pert_y3[0:2000], label='FP3 perturbation')
    plt.scatter([Ux1, Ux2, Ux3], [Uy1, Uy2, Uy3], label='Stable fixed points')
    plt.xlabel('$U_x$');
    plt.ylabel('$U_y$');
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.legend(loc='upper right');

    a = max(np.amax(np.abs(B1)), np.amax(np.abs(J1)))
    ax3 = plt.subplot2grid((3, 14), (0, 8), rowspan=1, colspan=3)
    plot_jac(ax3, J1, a, 'True Jacobian - FP1')
    ax4 = plt.subplot2grid((3, 14), (0, 11), rowspan=1, colspan=3)
    plot_jac(ax4, B1, a, 'Estimated Jacobian - FP1')

    a = max(np.amax(np.abs(B2)), np.amax(np.abs(J2)))
    ax5 = plt.subplot2grid((3, 14), (1, 8), rowspan=1, colspan=3)
    plot_jac(ax5, J2, a, 'True Jacobian - FP2')
    ax6 = plt.subplot2grid((3, 14), (1, 11), rowspan=1, colspan=3)
    plot_jac(ax6, B2, a, 'Estimated Jacobian - FP2')

    a = max(np.amax(np.abs(B3)), np.amax(np.abs(J3)))
    ax7 = plt.subplot2grid((3, 14), (2, 8), rowspan=1, colspan=3)
    plot_jac(ax7, J3, a, 'True Jacobian - FP3')
    ax8 = plt.subplot2grid((3, 14), (2, 11), rowspan=1, colspan=3)
    plot_jac(ax8, B3, a, 'Estimated Jacobian - FP3')

    plt.tight_layout()
    if savefig:
        plt.savefig('results/tristable.pdf', format='pdf')
    if showfig:
        plt.show()


def plot_sens(Jxy, n_traj_vec, n_iter_vec, est_ntraj, est_niter, est, savefig=True, figname='results/sens.pdf', format='pdf', showfig=False):

    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot2grid((2, 10), (0, 0), rowspan=1, colspan=4)
    plt.plot(n_traj_vec, est_ntraj, 'bo-', label='Estimated')
    plt.plot([0.9 * np.amin(n_traj_vec), 1.1 * np.amax(n_traj_vec)], [Jxy, Jxy], 'r--', label='True')
    plt.xlim([0.9 * np.amin(n_traj_vec), 1.1 * np.amax(n_traj_vec)])
    plt.xlabel('$N_{traj}$')
    plt.ylabel('$J_{XY}$')
    plt.legend(loc='best')

    ax2 = plt.subplot2grid((2, 10), (1, 0), rowspan=1, colspan=4)
    plt.plot(n_iter_vec, est_niter, 'bo-', label='Estimated')
    plt.plot([0.9 * np.amin(n_iter_vec), 1.1 * np.amax(n_iter_vec)], [Jxy, Jxy], 'r--', label='True')
    plt.xlim([0.9 * np.amin(n_iter_vec), 1.1 * np.amax(n_iter_vec)])
    plt.xlabel('$N_{iter}$')
    plt.ylabel('$J_{XY}$')
    plt.legend(loc='best')

    lim = np.amax(np.abs(est - Jxy))
    ax3 = plt.subplot2grid((2, 10), (0, 4), rowspan=2, colspan=6)
    pt = plt.pcolor(est - Jxy, cmap='RdBu', vmax=lim, vmin=-lim)
    plt.xticks(np.arange(0.5, n_iter_vec.size, 1), n_iter_vec)
    plt.yticks(np.arange(0.5, n_traj_vec.size, 1), n_traj_vec)
    plt.xlabel('$N_{iter}$')
    plt.ylabel('$N_{traj}$')
    cbar = plt.colorbar(pt, label='Error')

    plt.tight_layout()
    if savefig:
        plt.savefig(figname, format=format)
    if showfig:
        plt.show()