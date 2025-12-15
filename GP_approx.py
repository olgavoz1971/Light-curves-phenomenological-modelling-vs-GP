# https://www.researchgate.net/publication/371588335_Sparse_Logistic_Regression_for_RR_Lyrae_versus_Binaries_Classification#fullTextFileContent

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ExpSineSquared, WhiteKernel, ConstantKernel

plt.rcParams.update({
    'font.size': 20,  # default font size for all labels
    'axes.labelsize': 16,  # axis labels
    'axes.titlesize': 24,  # axes title
    'xtick.labelsize': 16,  # x-axis tick labels
    'ytick.labelsize': 16,  # y-axis tick labels
    'legend.fontsize': 20  # legend
})


def normalize_flux(flux, flux_err):
    f_min = np.min(flux)
    f_max = np.max(flux)
    scale = f_max - f_min if f_max > f_min else 1.0
    return (flux - f_min) / scale, flux_err / scale, f_min, f_max


def build_fixed_kernel():
    """Matern + Periodic + White + Constant (fixed parameters)"""
    matern = Matern(length_scale=5.0, nu=1.5)
    periodic = ExpSineSquared(length_scale=2.0, periodicity=3.0)
    white = WhiteKernel(noise_level=5e3)
    constant = ConstantKernel(1.0)
    return matern + periodic + white + constant


def fit_gp_phase_folded(phase, flux, flux_err, n_repeat=3):
    """Fit GP, repeat light curve to enforce periodicity"""
    phase_tiled = np.concatenate([phase + i for i in range(-n_repeat // 2 + 1, n_repeat // 2 + 1)])
    flux_tiled = np.tile(flux, n_repeat)
    err_tiled = np.tile(flux_err, n_repeat)

    X = phase_tiled.reshape(-1, 1)
    y = flux_tiled
    alpha = err_tiled ** 2

    kernel = build_fixed_kernel()
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
    gp.fit(X, y)
    return gp


def gp_interpolate(gp, n_points=300):
    x_plot = np.linspace(0, 1, n_points)
    mu, std = gp.predict(x_plot.reshape(-1, 1), return_std=True)
    # Shift so maximum is at phase = 0
    shift_idx = np.argmax(mu)
    mu = np.roll(mu, -shift_idx)
    std = np.roll(std, -shift_idx)
    x_plot = np.roll(x_plot, -shift_idx)

    # Sort x_plot and apply the same order to mu and std
    sort_idx = np.argsort(x_plot)
    x_plot = x_plot[sort_idx]
    mu = mu[sort_idx]
    std = std[sort_idx]
    return x_plot, mu, std


def main(path_to_esvs):
    tab = ascii.read(path_to_esvs)
    flux = np.array(tab["flux"])
    flux_err = np.array(tab["flux_err"])
    phase = np.array(tab["phase"])

    # Replace bad errors
    bad_err = (~np.isfinite(flux_err)) | (flux_err <= 0)
    if np.any(bad_err):
        fallback = max(1e-3, 0.01 * np.nanstd(flux))
        flux_err[bad_err] = fallback
        print(f"Warning: replaced bad flux_err with {fallback}")

    # Normalize
    flux, flux_err, f_min, f_max = normalize_flux(flux, flux_err)

    # Plot original data
    plt.figure(figsize=(14, 6))
    plt.errorbar(phase, flux, yerr=flux_err, fmt='.', alpha=0.6, label='Observed')
    plt.xlabel("Phase")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Fit GP
    print("Fitting Gaussian Process...")
    gp = fit_gp_phase_folded(phase, flux, flux_err, n_repeat=3)

    # Interpolate
    x_gp, mu, std = gp_interpolate(gp, n_points=300)

    # Plot GP

    plt.figure(figsize=(16, 10))
    plt.errorbar(phase, flux, yerr=flux_err, fmt='.', alpha=0.6, label='Observed')
    plt.plot(x_gp, mu, 'r-', lw=1, label='GP fit')
    plt.fill_between(x_gp, mu - std, mu + std, alpha=0.2, label='±1σ')
    plt.xlabel("Phase")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title('Gaussian Process')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # path = '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/lc_tess_FFI__AB_And_sparce.ecvs'
    # path = '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/lc_gaia_1936512041221649536_G.ecsv'
    # path = '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/lc_gaia_1936512041221649536_Bp.ecsv'
    # filename = 'lc_gaia_4585381817643702528_G.ecsv'
    # filename = 'lc_gaia_1936512041221649536_Bp.ecsv'    # AA And, for illustration
    filename = 'lc_tess_FFI__AB_And_sparce.ecvs'         # for illustration

    # filename = 'lc_tess_TPF_V0477_Lyr_TIC_423311936_sector_26_SPOC.ecsv'
    path = '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/' + filename
    main(path)
