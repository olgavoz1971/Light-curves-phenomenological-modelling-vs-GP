import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
plt.rcParams.update({
    'font.size': 20,  # default font size for all labels
    'axes.labelsize': 16,  # axis labels
    'axes.titlesize': 24,  # axes title
    'xtick.labelsize': 16,  # x-axis tick labels
    'ytick.labelsize': 16,  # y-axis tick labels
    'legend.fontsize': 20  # legend
})


# ---------- MODEL COMPONENTS ----------
def eclipses(theta, B, A1, A2, D, Gamma):
    """Two eclipse dips."""
    total = np.full_like(theta, B, dtype=float)
    for k, Ak in enumerate((A1, A2), start=1):
        phi_k = theta - 0.5 * (k - 1) - np.round(theta - 0.5 * (k - 1))
        x = np.clip(phi_k / D, -50, 50)
        inner = 1 - np.exp(1 - np.cosh(x))
        total += Ak * (1 - inner ** Gamma)
    return total


def harmonics(theta, np_terms, nc, A_cos, A_sin=None):
    """Cosine and optional sine terms."""
    theta = np.asarray(theta)
    fp = np.zeros_like(theta)
    for j in range(np_terms):
        fp += A_cos[j] * np.cos(2 * np.pi * (j + 1) * theta)
    fc = np.zeros_like(theta)
    if nc == 1 and A_sin is not None:
        fc = A_sin * np.sin(2 * np.pi * theta)
    return fp + fc


def full_model(theta, np_terms, nc, params):
    """
    params: [phase_shift, B, A1, A2, D, Gamma, A_cos..., (A_sin)]
    """
    phase_shift = params[0]
    theta_shifted = np.mod(theta - phase_shift, 1.0)
    B, A1, A2, D, Gamma = params[1:6]
    if nc == 1:
        A_cos = params[6:6 + np_terms]
        A_sin = params[6 + np_terms]
    else:
        A_cos = params[6:6 + np_terms]
        A_sin = None

    base = eclipses(theta_shifted, B, A1, A2, D, Gamma)
    harm = harmonics(theta_shifted, np_terms, nc, A_cos, A_sin)
    return base + harm


# ---------- FITTING HELPERS ----------
def guess_initial_params(phase, flux, np_terms, nc):
    B_guess = np.percentile(flux, 95)
    low5 = np.percentile(flux, 5)
    depth_est = B_guess - low5
    return (
            [0.0, B_guess, -0.6 * depth_est, -0.3 * depth_est, 0.08, 2.0]
            + [0.01] * np_terms
            + ([0.0] if nc == 1 else [])
    )


def robust_fit(phase, flux, flux_err, np_terms, nc, p0, lower, upper):
    """Robust least-squares fit with soft_l1 loss."""

    def res_fun(params, x, y, yerr):
        return (full_model(x, np_terms, nc, params) - y) / yerr

    res = least_squares(
        res_fun, x0=p0, args=(phase, flux, flux_err),
        bounds=(lower, upper),
        loss='soft_l1', f_scale=1.0, max_nfev=200000
    )
    residuals = flux - full_model(phase, np_terms, nc, res.x)
    chi2 = np.sum((residuals / flux_err) ** 2)
    ndof = len(flux) - len(res.x)
    return res.x, chi2, ndof, residuals


def normalize_flux(flux, flux_err):
    """
    Normalize flux and errors to the [0, 1] range.

    Parameters
    ----------
    flux : array-like
        Observed flux values.
    flux_err : array-like
        Corresponding flux uncertainties.

    Returns
    -------
    norm_flux, norm_flux_err, f_min, f_max : np.ndarray, np.ndarray, float, float
        Normalized flux, normalized errors, and the original min/max values.
    """
    f_min = np.min(flux)
    f_max = np.max(flux)
    scale = f_max - f_min if f_max > f_min else 1.0

    norm_flux = (flux - f_min) / scale
    norm_flux_err = flux_err / scale
    return norm_flux, norm_flux_err, f_min, f_max


# ---------- MAIN ----------


def main():
    print("Loading observations...")
    # filename = 'lc_gaia_4585381817643702528_G.ecsv'
    # filename = 'AB_AND_56.ecsv'
    filename = 'lc_tess_FFI__AB_And_sparce.ecvs'    # this was taken for the illustration
    # filename = 'lc_gaia_1936512041221649536_Bp.ecsv'    # AA And; and this was taken for the illustration
    # filename = 'lc_tess_TPF_V0477_Lyr_TIC_423311936_sector_26_SPOC.ecsv'
    # path = '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/AB_AND_56.ecsv'
    # path = '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/lc_tess_FFI__AB_And_sparce.ecvs'
    # path = '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/lc_gaia_1936512041221649536_Bp.ecsv'
    path = '/home/voz/projects/UPJS/VO/skvo_veb_project/skvo_veb/auxiliary/data/' + filename

    tab = ascii.read(path)

    flux = np.array(tab["flux"])
    flux_err = np.array(tab["flux_err"])

    normalized_flux, normalized_flux_err, f_min, f_max = normalize_flux(flux, flux_err)
    tab['normalized_flux'] = normalized_flux
    tab["normalized_flux_err"] = normalized_flux_err

    # tab["normalized_flux_err"] = [0.001] * len(tab)

    phase = np.array(tab["phase"])
    flux = np.array(tab["normalized_flux"])
    flux_err = np.array(tab["normalized_flux_err"])

    np_terms = 1  # number of cosine harmonics
    nc = 1  # include sine term for asymmetry
    p0 = guess_initial_params(phase, flux, np_terms, nc)

    #  Define lower and upper bounds for all model parameters in the same order as in p0:
    #  phase_shift, B, A1,A2, D, Gamma, A_cos, A_sin
    """
        | Parameter        | Lower | Upper | Physical meaning                                              |
        | ---------------- | ----- | ----- | ------------------------------------------------------------- |
        | phase_shift      | −0.5  | +0.5  | limits phase offset to within half a cycle                    |
        | B                | 0.5   | 1.5   | keeps baseline near normalised level                          |
        | A1, A2           | −2    | 0     | central depths                 |
        | D                | 0.001 | 0.5   | eclipses half-width: can’t be zero or more than half the cycle |
        | Gamma            | 0.1   | 10    | kurtosis coefficients (equal for both               |
        | A_cos, A_sin     | −1    | +1    | small modulation terms                                        |
    """
    lower = [-0.5, 0.5, -2, -2, 1e-3, 0.1] + [-1] * np_terms + ([-1] if nc == 1 else [])
    upper = [0.5, 1.5, 0, 0, 0.5, 10.0] + [1] * np_terms + ([1] if nc == 1 else [])

    print("Fitting light curve with phenomenological model...")
    popt, chi2, ndof, resid = robust_fit(phase, flux, flux_err, np_terms, nc, p0, lower, upper)
    # p0 = popt
    # popt, chi2, ndof, resid = robust_fit(phase, flux, flux_err, np_terms, nc, p0, lower, upper)

    # new_phase = np.linspace(0, 1, 100, endpoint=True)
    # new_flux = full_model(new_phase, np_terms, nc, popt)

    chi2_red = chi2 / ndof

    print("\nBest-fit parameters:")
    param_names = ["phase_shift", "B", "A1", "A2", "D", "Gamma"] \
                  + [f"A_cos{i + 1}" for i in range(np_terms)] \
                  + (["A_sin"] if nc == 1 else [])
    for name, val in zip(param_names, popt):
        print(f" {name:>10s} = {val:.6f}")
    print(f"\nχ² = {chi2:.3f},  reduced χ² = {chi2_red:.3f}")

    # Plot
    new_phase = np.linspace(0, 1, 400)
    new_flux = full_model(new_phase, np_terms, nc, popt)

    plt.figure(figsize=(16, 10))
    plt.errorbar(phase, flux, yerr=flux_err, fmt=".", alpha=0.6, label="Observed")
    plt.plot(new_phase, new_flux, "r-", lw=1, label="Model fit")
    plt.xlabel("Phase")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title('Phenomenological Modeling')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
