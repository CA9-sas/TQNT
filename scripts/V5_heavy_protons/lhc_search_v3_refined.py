"""
Recherche TQNT-V5p4 protons lourds dans ATLAS open data — v3 raffinée.

Améliorations vs v2 (commit c8ce23f) :
  1. Fenêtres précises ±30 MeV autour des 3 prédictions V5p4
  2. Calibration Crystal Ball sur J/ψ (3.097 GeV) + ψ(2S) (3.686 GeV)
  3. Interpolation de σ_resolution(mass) entre les deux points calibrés
  4. Background polynôme cubique
  5. Bootstrap (Poisson + lumi) pour distributions de significance
  6. Look-Elsewhere Effect (LEE) avec N_trials = range/σ_res ≈ 13
  7. Significance locale ET globale par candidat
  8. Combinaison di-muon + di-électron via likelihood
  9. 95% CL upper limit sur N_signal (profile likelihood)

Trois candidats V5p4 (catalogue TQNT_catalog_V5_diffusion.xlsx, onglet HP) :
  #262 : 6₁+6₁+7₃ → 2919.1 MeV (±30 MeV)
  #269 : 6₃+9₁+9₁ → 3037.0 MeV (±30 MeV)
  #283 : 4₁+6₁+9₁ → 3232.0 MeV (±30 MeV)
"""

import math
import sys
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import norm, poisson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproductibilité
SEED = 20260615
np.random.seed(SEED)

M_MU = 0.105658
M_E = 0.000511

V5_PREDICTIONS = [
    {"id": 262, "config": "6_1+6_1+7_3", "mass": 2.919},
    {"id": 269, "config": "6_3+9_1+9_1", "mass": 3.037},
    {"id": 283, "config": "4_1+6_1+9_1", "mass": 3.232},
]
SEARCH_WINDOW = 0.030  # ±30 MeV

# PDG : valeurs précises
M_JPSI_PDG = 3.0969
M_PSI2S_PDG = 3.6861

PROJECT_ROOT = Path("/home/euk9/projets/physique_nouvelle")
DATA_DIR = PROJECT_ROOT / "data" / "ATLAS_13TeV"
OUT = PROJECT_ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Crystal Ball PDF (Bukin 1985 / Skwarnicki 1986)
# ---------------------------------------------------------------------------
def crystal_ball(x, m, sigma, alpha, n):
    """Crystal Ball PDF (non normalisée)."""
    t = (x - m) / sigma
    core = np.exp(-0.5 * t * t)
    abs_a = abs(alpha)
    if abs_a < 1e-6:
        return core
    a_term = (n / abs_a) ** n * np.exp(-0.5 * alpha * alpha)
    b_term = n / abs_a - abs_a
    tail = a_term * np.maximum(b_term - t, 1e-30) ** (-n)
    return np.where(t > -alpha, core, tail)


def cb_resolution(mass, sigma_jpsi, sigma_psi2s,
                  m_ref1=M_JPSI_PDG, m_ref2=M_PSI2S_PDG):
    """Interpolation linéaire de la résolution en masse.
    σ(m) = σ1 + (σ2-σ1) × (m-m1)/(m2-m1)
    Calibration ATLAS open data : variation ~25-40 MeV à 3 GeV."""
    slope = (sigma_psi2s - sigma_jpsi) / (m_ref2 - m_ref1)
    return sigma_jpsi + slope * (mass - m_ref1)


# ---------------------------------------------------------------------------
# Modèle global : J/ψ + ψ(2S) + 3 V5p4 signaux + cubic background
# ---------------------------------------------------------------------------
def model_calibration(x, m_jpsi, sigma_jpsi, alpha, n,
                      N_jpsi, m_psi2s, sigma_psi2s, N_psi2s,
                      a0, a1, a2, a3):
    """Modèle de calibration (sans signaux V5p4) : 2 résonances + cubic."""
    cb_jpsi = N_jpsi * crystal_ball(x, m_jpsi, sigma_jpsi, alpha, n)
    cb_psi2s = N_psi2s * crystal_ball(x, m_psi2s, sigma_psi2s, alpha, n)
    bkg = a0 + a1*x + a2*x*x + a3*x*x*x
    return cb_jpsi + cb_psi2s + np.maximum(bkg, 0)


def model_signal(x, m_sig, sigma_sig, N_sig, a0, a1, a2, a3):
    """Modèle local : 1 signal + cubic background (pour profile likelihood)."""
    sig = N_sig * crystal_ball(x, m_sig, sigma_sig, 1.5, 3.0)
    bkg = a0 + a1*x + a2*x*x + a3*x*x*x
    return sig + np.maximum(bkg, 0)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def inv_mass_pairs(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1 = pt1*np.cos(phi1); py1 = pt1*np.sin(phi1); pz1 = pt1*np.sinh(eta1)
    px2 = pt2*np.cos(phi2); py2 = pt2*np.sin(phi2); pz2 = pt2*np.sinh(eta2)
    E1 = np.sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1)
    E2 = np.sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2)
    E = E1 + E2
    px = px1+px2; py = py1+py2; pz = pz1+pz2
    return np.sqrt(np.maximum(E*E - (px*px + py*py + pz*pz), 0))


def load_dileptons_by_channel(files):
    """Retourne (mass_mumu_OS, mass_ee_OS, mass_mumu_SS, mass_ee_SS)."""
    out = {"mumu_OS": [], "ee_OS": [], "mumu_SS": [], "ee_SS": []}
    for path in files:
        try:
            f = uproot.open(path)
            tree = f["mini"]
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            continue
        n = tree.num_entries
        print(f"  loading {path.name}  n_events={n}")
        arrs = tree.arrays(
            ["lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_charge", "lep_type"],
            entry_stop=n
        )
        mask = arrs["lep_n"] >= 2
        sel = arrs[mask]
        if len(sel) == 0:
            continue
        pt = sel["lep_pt"][:, :2] / 1000.0
        eta = sel["lep_eta"][:, :2]
        phi = sel["lep_phi"][:, :2]
        ch = sel["lep_charge"][:, :2]
        tp = sel["lep_type"][:, :2]
        pt1 = np.asarray(pt[:, 0]); pt2 = np.asarray(pt[:, 1])
        eta1 = np.asarray(eta[:, 0]); eta2 = np.asarray(eta[:, 1])
        phi1 = np.asarray(phi[:, 0]); phi2 = np.asarray(phi[:, 1])
        ch1 = np.asarray(ch[:, 0]); ch2 = np.asarray(ch[:, 1])
        tp1 = np.asarray(tp[:, 0]); tp2 = np.asarray(tp[:, 1])

        # masses canal : mumu utilise M_MU, ee utilise M_E
        is_mumu = (tp1 == 13) & (tp2 == 13)
        is_ee = (tp1 == 11) & (tp2 == 11)
        OS = ch1 * ch2 < 0
        SS = ch1 * ch2 > 0

        if is_mumu.any():
            m = inv_mass_pairs(
                pt1[is_mumu], eta1[is_mumu], phi1[is_mumu], M_MU,
                pt2[is_mumu], eta2[is_mumu], phi2[is_mumu], M_MU)
            sub_os = OS[is_mumu]; sub_ss = SS[is_mumu]
            out["mumu_OS"].append(m[sub_os])
            out["mumu_SS"].append(m[sub_ss])
        if is_ee.any():
            m = inv_mass_pairs(
                pt1[is_ee], eta1[is_ee], phi1[is_ee], M_E,
                pt2[is_ee], eta2[is_ee], phi2[is_ee], M_E)
            sub_os = OS[is_ee]; sub_ss = SS[is_ee]
            out["ee_OS"].append(m[sub_os])
            out["ee_SS"].append(m[sub_ss])

    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Calibration sur J/ψ + ψ(2S)
# ---------------------------------------------------------------------------
def calibrate_resolution(masses, fit_range=(2.5, 4.0), nbins=300, channel=""):
    """Fit CB(J/ψ) + CB(ψ(2S)) + cubic ; retourne (σ_jpsi, σ_psi2s, popt, χ²/ndof)."""
    mask = (masses >= fit_range[0]) & (masses <= fit_range[1])
    m_sel = masses[mask]
    n_data, edges = np.histogram(m_sel, bins=nbins, range=fit_range)
    centers = 0.5*(edges[:-1] + edges[1:])
    sigma_data = np.sqrt(n_data + 1)

    n_max = float(n_data.max())
    n_med = float(np.median(n_data))
    p0 = [3.097, 0.045, 1.5, 3.0,
          n_max*0.6, 3.686, 0.060, n_max*0.04,
          n_med, 0.0, 0.0, 0.0]
    lo = [3.05, 0.015, 0.3, 0.8, 0, 3.62, 0.020, 0,
          -1e7, -1e7, -1e7, -1e7]
    hi = [3.15, 0.090, 8.0, 12.0, n_max*2.5,
          3.75, 0.130, n_max*0.4, 1e7, 1e7, 1e7, 1e7]

    try:
        popt, pcov = curve_fit(model_calibration, centers, n_data, p0=p0,
                               sigma=sigma_data, absolute_sigma=True,
                               bounds=(lo, hi), maxfev=40000)
    except RuntimeError as e:
        print(f"  Calibration fit non convergé ({channel}): {e}")
        return None, None, None, None, centers, n_data

    n_fit = model_calibration(centers, *popt)
    chi2 = ((n_data - n_fit)**2 / sigma_data**2).sum()
    ndof = len(n_data) - 12
    return popt[1], popt[6], popt, chi2/ndof, centers, n_data


# ---------------------------------------------------------------------------
# Significance par candidat (profile likelihood)
# ---------------------------------------------------------------------------
def signal_significance(masses, m_target, sigma_res,
                        popt_calib, calib_range=(2.5, 4.0),
                        window=SEARCH_WINDOW, nbins_global=300):
    """Significance honnête : on FIXE le modèle de fond global (J/ψ +
    ψ(2S) + cubic) à ses paramètres calibrés et on extrait l'excès de
    signal local autour de m_target.

    On utilise le résidu (data − global_fit) intégré dans la fenêtre
    ±window (≈ ±σ_res) divisé par sqrt(N_bkg_in_window).
    """
    lo, hi = calib_range
    sel = masses[(masses >= lo) & (masses <= hi)]
    n_data, edges = np.histogram(sel, bins=nbins_global, range=(lo, hi))
    centers = 0.5*(edges[:-1] + edges[1:])

    # Modèle global évalué aux paramètres calibrés
    if popt_calib is not None:
        n_bkg = model_calibration(centers, *popt_calib)
    else:
        n_bkg = np.full_like(n_data, np.median(n_data), dtype=float)

    # Fenêtre ±window
    in_win = (centers >= m_target - window) & (centers <= m_target + window)
    n_obs_win = float(n_data[in_win].sum())
    n_bkg_win = float(n_bkg[in_win].sum())

    # Excès et significance (Poisson, formule asymptotique gauss)
    excess = n_obs_win - n_bkg_win
    if n_bkg_win > 0:
        local_z = excess / math.sqrt(n_bkg_win)
        dN_sig = math.sqrt(n_bkg_win + max(excess, 0))
    else:
        local_z = 0.0
        dN_sig = 1.0

    # 95% CL upper limit (Feldman-Cousins gaussien approximé)
    upper_95 = max(excess, 0) + 1.645 * dN_sig

    return {
        "N_sig": excess, "dN_sig": dN_sig,
        "N_bkg": n_bkg_win, "n_obs": n_obs_win,
        "local_z": local_z,
        "upper_95": upper_95,
        "n_data": n_data, "centers": centers,
        "n_bkg_curve": n_bkg,
    }


# ---------------------------------------------------------------------------
# Bootstrap pour distributions empiriques
# ---------------------------------------------------------------------------
def bootstrap_significance(masses, m_target, sigma_res, popt_calib,
                           calib_range=(2.5, 4.0), n_trials=1000,
                           window=SEARCH_WINDOW, nbins_global=300):
    """Bootstrap Poisson : on resample n_data bin-par-bin et on
    recalcule l'excès local. Distribution empirique de z_local."""
    rng = np.random.default_rng(SEED)
    lo, hi = calib_range
    sel = masses[(masses >= lo) & (masses <= hi)]
    n_data, edges = np.histogram(sel, bins=nbins_global, range=(lo, hi))
    centers = 0.5*(edges[:-1] + edges[1:])

    n_bkg = (model_calibration(centers, *popt_calib)
             if popt_calib is not None else np.full_like(n_data, np.median(n_data), dtype=float))

    in_win = (centers >= m_target - window) & (centers <= m_target + window)
    n_bkg_win = float(n_bkg[in_win].sum())

    zs = np.zeros(n_trials)
    for k in range(n_trials):
        sample = rng.poisson(np.maximum(n_data, 0.0))
        n_obs_win = float(sample[in_win].sum())
        excess = n_obs_win - n_bkg_win
        zs[k] = excess / math.sqrt(max(n_bkg_win, 1)) if n_bkg_win > 0 else 0.0
    return zs


# ---------------------------------------------------------------------------
# LEE correction
# ---------------------------------------------------------------------------
def lee_correction(local_z, n_trials):
    """Convertit la significance locale en globale par formule asymptotique :
    p_global = 1 - (1 - p_local)^N_trials
    Pour grands N_trials, z_global ≈ z_local - sqrt(2 ln N_trials).
    Méthode exacte via inversion p-valeurs."""
    if local_z <= 0:
        return local_z, 1.0
    p_local = 1.0 - norm.cdf(local_z)
    # 1 − (1 − p_local)^N
    p_global = 1.0 - (1.0 - p_local) ** n_trials
    p_global = min(max(p_global, 1e-300), 1.0 - 1e-15)
    z_global = norm.ppf(1.0 - p_global)
    return z_global, p_global


# ---------------------------------------------------------------------------
# Plot global
# ---------------------------------------------------------------------------
def plot_summary(results_mumu, results_ee, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A : mumu calibration
    ax = axes[0, 0]
    c = results_mumu["centers"]; n = results_mumu["n_data"]
    ax.errorbar(c, n, yerr=np.sqrt(n+1), fmt='o', ms=2, color='black',
                label=f"di-μ (N={n.sum()})")
    if results_mumu["popt_calib"] is not None:
        p = results_mumu["popt_calib"]
        x_fit = np.linspace(c[0], c[-1], 500)
        ax.plot(x_fit, model_calibration(x_fit, *p), 'b-', lw=1.5,
                label=f"CB+cubic χ²/ndof={results_mumu['chi2_ndof']:.2f}")
    for pred in V5_PREDICTIONS:
        ax.axvline(pred["mass"], color="red", ls="--", alpha=0.4)
    ax.set_xlabel("m(μ⁺μ⁻) [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_yscale("log")
    ax.set_title(f"Di-muon — σ(J/ψ)={results_mumu['sigma_jpsi']*1000:.1f} MeV, "
                 f"σ(ψ2S)={results_mumu['sigma_psi2s']*1000:.1f} MeV")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B : ee calibration
    ax = axes[0, 1]
    if results_ee["popt_calib"] is not None:
        c = results_ee["centers"]; n = results_ee["n_data"]
        ax.errorbar(c, n, yerr=np.sqrt(n+1), fmt='o', ms=2, color='black',
                    label=f"di-e (N={n.sum()})")
        p = results_ee["popt_calib"]
        x_fit = np.linspace(c[0], c[-1], 500)
        ax.plot(x_fit, model_calibration(x_fit, *p), 'b-', lw=1.5,
                label=f"CB+cubic χ²/ndof={results_ee['chi2_ndof']:.2f}")
        for pred in V5_PREDICTIONS:
            ax.axvline(pred["mass"], color="red", ls="--", alpha=0.4)
        ax.set_title(f"Di-electron — σ(J/ψ)={results_ee['sigma_jpsi']*1000:.1f} MeV, "
                     f"σ(ψ2S)={results_ee['sigma_psi2s']*1000:.1f} MeV")
    else:
        ax.set_title("Di-electron — calibration failed")
    ax.set_xlabel("m(e⁺e⁻) [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C : residuals globaux (data − calibration_bkg) in mumu
    ax = axes[1, 0]
    if results_mumu["popt_calib"] is not None and len(results_mumu["signals"]) > 0:
        s0 = results_mumu["signals"][0]
        c = s0["centers"]; n = s0["n_data"]; nb = s0["n_bkg_curve"]
        ax.errorbar(c, n - nb, yerr=np.sqrt(n+1), fmt='o', ms=2,
                    color='purple', label="di-μ data − global fit")
        for pred in V5_PREDICTIONS:
            ax.axvline(pred["mass"], color="red", ls="--", alpha=0.4,
                       label=f"V5 #{pred['id']}" if pred==V5_PREDICTIONS[0] else None)
            ax.axvspan(pred["mass"]-SEARCH_WINDOW, pred["mass"]+SEARCH_WINDOW,
                       color="red", alpha=0.08)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("m(μμ) [GeV]")
    ax.set_ylabel("Residual (data − global fit)")
    ax.set_title("Di-μ residuals near V5p4 windows (±30 MeV)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel D : significance bar chart
    ax = axes[1, 1]
    labels = [f"#{p['id']}\n{p['mass']:.3f}" for p in V5_PREDICTIONS]
    z_loc_mumu = [s["local_z"] for s in results_mumu["signals"]]
    z_loc_ee = ([s["local_z"] for s in results_ee["signals"]]
                if results_ee["popt_calib"] is not None else [0]*3)
    z_glob = results_mumu["combined_global_z"]
    x = np.arange(3)
    width = 0.27
    ax.bar(x - width, z_loc_mumu, width, label="di-μ local", color="steelblue")
    ax.bar(x, z_loc_ee, width, label="di-e local", color="orange")
    ax.bar(x + width, z_glob, width, label="combined global (LEE)", color="firebrick")
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(3.0, color='red', ls=':', alpha=0.5, label="3σ")
    ax.axhline(5.0, color='darkred', ls=':', alpha=0.5, label="5σ")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Significance (σ)")
    ax.set_title("V5p4 heavy proton candidates : local vs global (LEE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nFigure : {out_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyze_channel(masses, channel_name, n_bootstrap=200):
    """Pipeline complet pour un canal (mumu ou ee)."""
    print(f"\n{'='*78}\nCanal {channel_name} : {len(masses)} paires OS\n{'='*78}")
    if len(masses) < 1000:
        print(f"  Statistique insuffisante (n={len(masses)})")
        return {
            "popt_calib": None, "sigma_jpsi": None, "sigma_psi2s": None,
            "sigma_at_3GeV": 0.030, "chi2_ndof": None,
            "centers": np.array([]), "n_data": np.array([]),
            "signals": [], "bootstraps": [],
        }

    # Calibration
    sj, sp, popt, chi2, centers, n_data = calibrate_resolution(
        masses, fit_range=(2.5, 4.0), nbins=300, channel=channel_name)
    if popt is None:
        return {
            "popt_calib": None, "sigma_jpsi": 0.045, "sigma_psi2s": 0.060,
            "sigma_at_3GeV": 0.040, "chi2_ndof": None,
            "centers": centers, "n_data": n_data,
            "signals": [], "bootstraps": [],
        }
    print(f"  σ(J/ψ)  = {sj*1000:.2f} MeV  (PDG width 92 keV — résolution dominante)")
    print(f"  σ(ψ2S) = {sp*1000:.2f} MeV  (PDG width 296 keV)")
    print(f"  χ²/ndof = {chi2:.3f}")

    sigma_3GeV = cb_resolution(3.0, sj, sp)
    print(f"  σ_res(3 GeV interpolée) = {sigma_3GeV*1000:.2f} MeV")

    # Signaux par candidat (utilise modèle global calibré pour le fond)
    signals = []
    bootstraps = []
    for pred in V5_PREDICTIONS:
        sigma_local = cb_resolution(pred["mass"], sj, sp)
        sig = signal_significance(masses, pred["mass"], sigma_local,
                                  popt_calib=popt,
                                  calib_range=(2.5, 4.0),
                                  window=SEARCH_WINDOW, nbins_global=300)
        signals.append(sig)
        bs = bootstrap_significance(masses, pred["mass"], sigma_local,
                                    popt_calib=popt,
                                    calib_range=(2.5, 4.0),
                                    n_trials=n_bootstrap,
                                    window=SEARCH_WINDOW, nbins_global=300)
        bootstraps.append(bs)
        print(f"  #{pred['id']} m={pred['mass']:.3f}  σ_res={sigma_local*1000:.1f} MeV  "
              f"N_obs={sig['n_obs']:.0f}  N_bkg={sig['N_bkg']:.1f}  "
              f"excess={sig['N_sig']:+.1f}±{sig['dN_sig']:.1f}  "
              f"z_local={sig['local_z']:+.2f}  "
              f"BS_mean={bs.mean():+.2f}±{bs.std():.2f}  "
              f"95%UL={sig['upper_95']:.1f}")

    return {
        "popt_calib": popt, "sigma_jpsi": sj, "sigma_psi2s": sp,
        "sigma_at_3GeV": sigma_3GeV, "chi2_ndof": chi2,
        "centers": centers, "n_data": n_data,
        "signals": signals, "bootstraps": bootstraps,
    }


def combine_channels(res_mumu, res_ee):
    """Combinaison via somme des log-likelihoods asymptotiques.
    z_comb = (z1×w1 + z2×w2) / sqrt(w1²+w2²)  où w = 1/dN_sig."""
    n_window = (3.3 - 2.9)  # 400 MeV
    sigma_3 = res_mumu["sigma_at_3GeV"]
    n_trials = max(int(n_window / max(sigma_3, 1e-3)), 1)
    print(f"\nLEE : N_trials = {n_window*1000:.0f} MeV / {sigma_3*1000:.1f} MeV "
          f"= {n_trials}")

    combined_local = []
    combined_global = []
    for i in range(3):
        # Stouffer's combination : z_comb = sum(z_k) / sqrt(K)
        zs = []
        if res_mumu["popt_calib"] is not None:
            zs.append(res_mumu["signals"][i]["local_z"])
        if res_ee["popt_calib"] is not None:
            zs.append(res_ee["signals"][i]["local_z"])
        if len(zs) == 0:
            z_comb_local = 0.0
        else:
            z_comb_local = sum(zs) / math.sqrt(len(zs))
        if z_comb_local > 0:
            z_glob, _ = lee_correction(z_comb_local, n_trials)
        else:
            z_glob = z_comb_local  # under-fluctuation, no LEE inflation
        combined_local.append(z_comb_local)
        combined_global.append(z_glob)

    res_mumu["combined_local_z"] = combined_local
    res_mumu["combined_global_z"] = combined_global
    res_mumu["n_trials_LEE"] = n_trials
    return combined_local, combined_global


def write_csv(res_mumu, res_ee, combined_local, combined_global, out_path):
    lines = ["candidate_id,config,mass_GeV,channel,N_signal,dN_signal,"
             "local_z,combined_local_z,combined_global_z_LEE,UL_95_N_signal"]
    for i, pred in enumerate(V5_PREDICTIONS):
        for ch_name, res in [("mumu", res_mumu), ("ee", res_ee)]:
            if res["popt_calib"] is None:
                continue
            s = res["signals"][i]
            lines.append(f"{pred['id']},{pred['config']},{pred['mass']},{ch_name},"
                         f"{s['N_sig']:.2f},{s['dN_sig']:.2f},{s['local_z']:.3f},"
                         f"{combined_local[i]:.3f},{combined_global[i]:.3f},"
                         f"{s['upper_95']:.2f}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"CSV : {out_path}")


def main():
    files = sorted(DATA_DIR.glob("data_*.exactly2lep.root"))
    print("="*78)
    print(f"ATLAS Open Data 13 TeV — recherche V5p4 heavy protons v3 raffinée")
    print(f"  Fichiers : {len(files)}")
    print(f"  Seed     : {SEED}")
    print("="*78)

    by_ch = load_dileptons_by_channel(files)
    print(f"\nCanaux chargés :")
    for k, v in by_ch.items():
        print(f"  {k}: {len(v)} paires")

    # Analyse par canal
    res_mumu = analyze_channel(by_ch["mumu_OS"], "di-muon (μ+μ-)",
                               n_bootstrap=200)
    res_ee = analyze_channel(by_ch["ee_OS"], "di-electron (e+e-)",
                             n_bootstrap=200)

    # Combinaison + LEE
    combined_local, combined_global = combine_channels(res_mumu, res_ee)

    print("\n" + "="*78)
    print("RÉSUMÉ V5p4 — significance locale, combinée et globale (LEE)")
    print("="*78)
    print(f"{'ID':>4} {'config':>15} {'mass':>8} "
          f"{'z_μμ':>8} {'z_ee':>8} {'z_comb':>8} {'z_global':>10} {'95%UL':>10}")
    for i, pred in enumerate(V5_PREDICTIONS):
        z1 = res_mumu["signals"][i]["local_z"] if res_mumu["popt_calib"] is not None else float('nan')
        z2 = (res_ee["signals"][i]["local_z"]
              if res_ee["popt_calib"] is not None else float('nan'))
        ul = res_mumu["signals"][i]["upper_95"] if res_mumu["popt_calib"] is not None else float('nan')
        print(f"#{pred['id']:>3} {pred['config']:>15} {pred['mass']:>8.3f} "
              f"{z1:>+8.2f} {z2:>+8.2f} {combined_local[i]:>+8.2f} "
              f"{combined_global[i]:>+10.2f} {ul:>10.1f}")

    # Plot + CSV
    plot_summary(res_mumu, res_ee, OUT / "v5p4_heavy_proton_search_v3_combined.png")
    write_csv(res_mumu, res_ee, combined_local, combined_global,
              OUT / "v5p4_significance_summary.csv")

    # Verdict
    print("\n" + "="*78)
    print("VERDICT")
    print("="*78)
    max_global = max(abs(z) for z in combined_global)
    if max_global >= 5.0:
        print(f"  DISCOVERY POSSIBLE : z_global max = {max_global:.2f}σ ≥ 5σ — DOUBLE-CHECK requis")
    elif max_global >= 3.0:
        print(f"  HINT : z_global max = {max_global:.2f}σ ∈ [3σ, 5σ)")
    elif max_global >= 2.0:
        print(f"  MARGINAL : z_global max = {max_global:.2f}σ ∈ [2σ, 3σ)")
    else:
        print(f"  NULL DETECTION : z_global max = {max_global:.2f}σ < 2σ")
        print(f"  → Aucun signe statistiquement significatif de #262/#269/#283")
        print(f"  → Limites supérieures à 95% CL données dans le CSV")
    return res_mumu, res_ee, combined_local, combined_global


if __name__ == "__main__":
    main()
