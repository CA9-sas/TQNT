"""
V5.6 RE-VALIDATION --- SPARC c(M_h) slope under the V5.3 master pipeline.

OBJECTIF
========
Re-implementer le pipeline V5.3 master (kappa_CJ universality + c(M_h))
AUSSI EXACTEMENT QUE POSSIBLE et verifier la pente

    d log c / d log M_h

sur les 175 galaxies SPARC.  La V5.3 master (docs/V5p3_synthesis_FR.tex,
commit 6d06ac6) revendique +0,575 +/- 0,038, en tension a 18 sigma avec
Wechsler-Bullock observe (-0,10).  L'agent 2 (commit c8ce23f) a mesure
+0,276 / +0,286.  On reproduit ici la specification de V5.3 master.

PIPELINE V5.3 MASTER
====================
Pour chaque galaxie SPARC :

1. Vflat lu depuis SPARC_Lelli2016c.mrt (catalogue Lelli+2016)
2. Mh = (Vflat/200 km/s)^3 * 1e12 Msun   (abundance matching simple)
3. r_vir = (3 Mh / 4 pi 200 rho_crit)^(1/3)
4. Densite baryonique LOCALE au demi-rayon baryonique :
       rho_b = (1/(4 pi r^2)) dM_bar/dr   evaluee a r = r_half_bar
   ou M_bar(r) provient des courbes Rotmod (V_gas, V_disk, V_bul) :
       M_bar(r) = r * (Upsilon_d V_disk^2 + Upsilon_b V_bul^2
                       + sign(V_gas) V_gas^2) / G
   avec Upsilon_d = 0.5, Upsilon_b = 0.7 (V5.3 master, sec. 5).
5. L_coh^J = c_s / sqrt(4 pi G rho_b),   c_s = 10 km/s
6. r_s = kappa_CJ * L_coh^J,  kappa_CJ = 62.7 universel
7. c = r_vir / r_s
8. Fit log c vs log Mh -> pente, erreur bootstrap
9. Comparaison a WB (slope=-0,10) et au claim V5.3 (+0,575)

LIVRABLE
========
- scripts/V5_6_re_validation/sparc_refit_V5p3_master_standardized.py
- output/v5_6_v5p3_master_refit.png
- tests/test_sparc_v5p3_master_refit.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constantes (V5.3 master)
# ---------------------------------------------------------------------------

G_KPC_KMS2_PER_MSUN = 4.30091e-6     # G en kpc (km/s)^2 / Msun
RHO_CRIT_MSUN_KPC3 = 1.40e2          # densite critique Planck 2018 ~ 1.4e2 Msun/kpc^3
KAPPA_CJ_V5p3 = 62.7
C_SOUND_KMS = 10.0                   # vitesse du son baryonique froid (V5.3)
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7
WB_AMPL = 9.6                        # Wechsler-Bullock c(M=1e12)
WB_SLOPE = -0.10

# Claim V5.3 master
V5p3_CLAIMED_SLOPE = 0.575
V5p3_CLAIMED_SLOPE_ERR = 0.038

SPARC_ROOT = Path("/mnt/data/data_recherche/sparc")
SPARC_MAIN = SPARC_ROOT / "SPARC_Lelli2016c.mrt"
ROTMOD_DIR = SPARC_ROOT / "Rotmod_LTG"

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Catalog reader (Lelli+2016 MRT)
# ---------------------------------------------------------------------------


@dataclass
class SPARCGalaxy:
    name: str
    T: int
    D_Mpc: float
    Vflat_kms: float
    e_Vflat_kms: float
    Q: int


def parse_sparc_main(path: Path = SPARC_MAIN) -> list[SPARCGalaxy]:
    """Parse le fichier MRT principal (175 galaxies). Robuste a la dispersion
    de colonnes : on parse le nom en fixed-width [0:11] puis le reste en
    whitespace."""
    galaxies: list[SPARCGalaxy] = []
    with open(path, "r") as f:
        lines = f.readlines()

    # On localise la derniere ligne uniquement faite de tirets : les donnees
    # commencent juste apres.
    dash_indices = [
        i for i, line in enumerate(lines)
        if set(line.strip()) == {"-"} and len(line.strip()) > 10
    ]
    if not dash_indices:
        return galaxies
    data_start = dash_indices[-1] + 1

    for line in lines[data_start:]:
        if len(line.rstrip()) < 50:
            continue
        if line.startswith(("Note", "Acknow")):
            continue
        name = line[0:11].strip()
        if not name:
            continue
        rest = line[11:].split()
        if len(rest) < 17:
            continue
        try:
            T = int(rest[0])
            D = float(rest[1])
            Vflat = float(rest[14])
            e_Vflat = float(rest[15])
            Q = int(rest[16])
        except (ValueError, IndexError):
            continue
        if Vflat <= 0:
            continue
        galaxies.append(SPARCGalaxy(name, T, D, Vflat, e_Vflat, Q))
    return galaxies


def parse_rotmod(name: str, rotmod_dir: Path = ROTMOD_DIR) -> np.ndarray | None:
    """Charge le fichier Rotmod (R, Vobs, errV, Vgas, Vdisk, Vbul,
    SBdisk, SBbul)."""
    fname = rotmod_dir / f"{name}_rotmod.dat"
    if not fname.exists():
        candidates = list(rotmod_dir.glob(f"{name.strip()}_rotmod.dat"))
        if not candidates:
            return None
        fname = candidates[0]
    try:
        data = np.loadtxt(fname, comments="#")
    except Exception:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


# ---------------------------------------------------------------------------
# Halo (abundance matching simple) + virial radius
# ---------------------------------------------------------------------------


def M_halo_from_Vflat(Vflat_kms: float) -> float:
    """Mh = (Vflat / 200 km/s)^3 * 1e12 Msun (V5.3 master, sec. 5)."""
    return 1.0e12 * (Vflat_kms / 200.0) ** 3


def r_vir_kpc(M_h_Msun: float) -> float:
    """r_vir tel que la masse moyenne enclose = 200 rho_crit."""
    return (3.0 * M_h_Msun / (4.0 * math.pi * 200.0 * RHO_CRIT_MSUN_KPC3)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Profil baryonique : M_bar(r), r_half, rho_b(r_half)
# ---------------------------------------------------------------------------


def M_bar_profile(rotmod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (R, M_bar(R)) a partir des composantes SPARC.

    V5.3 master, sec. 5 :
        V_bar^2 = Upsilon_d V_disk^2 + Upsilon_b V_bul^2 + sign(V_gas) V_gas^2
        M_bar(r) = r * V_bar^2(r) / G
    """
    R = rotmod[:, 0]
    V_gas = rotmod[:, 3]
    V_disk = rotmod[:, 4]
    V_bul = rotmod[:, 5]

    V_gas2 = np.sign(V_gas) * V_gas ** 2
    V_disk2 = UPSILON_DISK * V_disk ** 2
    V_bul2 = UPSILON_BULGE * V_bul ** 2
    V_bar2 = V_gas2 + V_disk2 + V_bul2
    V_bar2 = np.clip(V_bar2, 1e-6, None)

    M_bar = R * V_bar2 / G_KPC_KMS2_PER_MSUN
    return R, M_bar


def r_half_baryonic(R: np.ndarray, M_bar: np.ndarray) -> float:
    """Demi-rayon baryonique : r tel que M_bar(r) = M_bar_tot / 2.
    On considere M_bar_tot = M_bar(R_max) (mesure SPARC tronquee)."""
    M_tot = M_bar[-1]
    target = 0.5 * M_tot
    # Interpolation lineaire monotone
    if M_tot <= 0 or not np.all(np.diff(M_bar) >= -1e-3):
        # Forcer monotonie via cummax
        M_mono = np.maximum.accumulate(M_bar)
        return float(np.interp(0.5 * M_mono[-1], M_mono, R))
    return float(np.interp(target, M_bar, R))


def rho_bar_at(R: np.ndarray, M_bar: np.ndarray, r_target: float) -> float:
    """rho_b(r_target) = (1/(4 pi r^2)) dM_bar/dr a r_target.

    On utilise la derivee centree de M_bar(R).  Densite en Msun / kpc^3.
    """
    if len(R) < 3:
        return float("nan")
    # Trier au cas ou
    order = np.argsort(R)
    Rs = R[order]
    Ms = M_bar[order]
    # Derivee numerique
    dM_dr = np.gradient(Ms, Rs)
    rho = dM_dr / (4.0 * math.pi * Rs ** 2)
    # Interpolation a r_target
    r_clamped = float(np.clip(r_target, Rs[0], Rs[-1]))
    rho_target = float(np.interp(r_clamped, Rs, rho))
    return max(rho_target, 1e-3)  # plancher pour eviter div par 0


def L_coh_J_kpc(rho_bar_Msun_per_kpc3: float) -> float:
    """L_coh^J = c_s / sqrt(4 pi G rho_b)."""
    denom = math.sqrt(4.0 * math.pi * G_KPC_KMS2_PER_MSUN * rho_bar_Msun_per_kpc3)
    return C_SOUND_KMS / denom


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def compute_concentration_for_galaxy(gal: SPARCGalaxy) -> dict | None:
    rotmod = parse_rotmod(gal.name)
    if rotmod is None or rotmod.shape[0] < 5:
        return None

    R, M_bar = M_bar_profile(rotmod)
    if not np.isfinite(M_bar).all() or M_bar[-1] <= 0:
        return None

    r_half = r_half_baryonic(R, M_bar)
    if not math.isfinite(r_half) or r_half <= 0:
        return None

    rho_b = rho_bar_at(R, M_bar, r_half)
    if not math.isfinite(rho_b) or rho_b <= 0:
        return None

    L_coh = L_coh_J_kpc(rho_b)
    r_s = KAPPA_CJ_V5p3 * L_coh
    M_h = M_halo_from_Vflat(gal.Vflat_kms)
    r_vir = r_vir_kpc(M_h)
    c = r_vir / r_s

    return {
        "name": gal.name,
        "Vflat": gal.Vflat_kms,
        "Q": gal.Q,
        "M_h": M_h,
        "r_vir": r_vir,
        "r_half": r_half,
        "rho_b": rho_b,
        "L_coh": L_coh,
        "r_s": r_s,
        "c": c,
    }


def fit_log_log_slope(log_M: np.ndarray, log_c: np.ndarray) -> tuple[float, float]:
    """Regression lineaire log c vs log M.  Retourne (slope, intercept)."""
    A = np.vstack([log_M, np.ones_like(log_M)]).T
    slope, intercept = np.linalg.lstsq(A, log_c, rcond=None)[0]
    return float(slope), float(intercept)


def bootstrap_slope(log_M: np.ndarray, log_c: np.ndarray,
                    n_boot: int = 2000, seed: int = 42) -> tuple[float, float, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    N = len(log_M)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        s, _ = fit_log_log_slope(log_M[idx], log_c[idx])
        slopes[i] = s
    mean = float(np.mean(slopes))
    std = float(np.std(slopes, ddof=1))
    lo, hi = np.percentile(slopes, [16, 84])
    return mean, std, (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_pipeline(verbose: bool = True) -> dict:
    galaxies = parse_sparc_main()
    if verbose:
        print(f"[V5.6 re-fit] SPARC catalogue : {len(galaxies)} galaxies avec Vflat")

    results = []
    for gal in galaxies:
        r = compute_concentration_for_galaxy(gal)
        if r is not None:
            results.append(r)

    if verbose:
        print(f"[V5.6 re-fit] Pipeline reussi sur {len(results)} galaxies")

    if not results:
        raise RuntimeError("Aucune galaxie utilisable")

    M_h = np.array([r["M_h"] for r in results])
    c = np.array([r["c"] for r in results])

    # Filtrage minimal : c > 0, M_h > 0
    mask = (c > 0) & (M_h > 0) & np.isfinite(c) & np.isfinite(M_h)
    M_h = M_h[mask]
    c = c[mask]

    log_M = np.log10(M_h / 1e12)
    log_c = np.log10(c)

    slope, intercept = fit_log_log_slope(log_M, log_c)
    slope_boot, slope_err, slope_ci = bootstrap_slope(log_M, log_c)

    # Tension vs claim V5.3
    tension_v5p3 = abs(slope - V5p3_CLAIMED_SLOPE) / V5p3_CLAIMED_SLOPE_ERR
    # Tension vs WB
    tension_wb = abs(slope - WB_SLOPE) / max(slope_err, 1e-6)

    report = {
        "n_galaxies_catalog": len(galaxies),
        "n_galaxies_used": int(mask.sum()),
        "slope_measured": slope,
        "intercept_measured": intercept,
        "slope_bootstrap_mean": slope_boot,
        "slope_bootstrap_std": slope_err,
        "slope_bootstrap_ci68": slope_ci,
        "V5p3_claimed_slope": V5p3_CLAIMED_SLOPE,
        "V5p3_claimed_err": V5p3_CLAIMED_SLOPE_ERR,
        "WB_slope": WB_SLOPE,
        "tension_vs_V5p3_master_sigma": tension_v5p3,
        "tension_vs_WB_sigma": tension_wb,
        "M_h": M_h,
        "c": c,
    }

    if verbose:
        print()
        print("=" * 64)
        print("RESULTAT REGRESSION log10(c) vs log10(M_h / 1e12 Msun)")
        print("=" * 64)
        print(f"  N galaxies utilisees     : {report['n_galaxies_used']}")
        print(f"  Slope mesure             : {slope:+.4f}")
        print(f"  Bootstrap mean +/- std   : {slope_boot:+.4f} +/- {slope_err:.4f}")
        print(f"  Bootstrap 68% CI         : [{slope_ci[0]:+.4f}, {slope_ci[1]:+.4f}]")
        print(f"  Intercept (log c at 1e12): {intercept:+.4f}")
        print()
        print(f"  Claim V5.3 master        : {V5p3_CLAIMED_SLOPE:+.3f} +/- {V5p3_CLAIMED_SLOPE_ERR:.3f}")
        print(f"  Wechsler-Bullock observe : {WB_SLOPE:+.3f}")
        print()
        print(f"  Tension vs V5.3 master   : {tension_v5p3:.1f} sigma (claim)")
        print(f"  Tension vs WB observed   : {tension_wb:.1f} sigma (measured)")
        print("=" * 64)

    return report


def plot_results(report: dict, out_path: Path) -> None:
    M_h = report["M_h"]
    c = report["c"]
    slope = report["slope_measured"]
    intercept = report["intercept_measured"]
    slope_err = report["slope_bootstrap_std"]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    ax.scatter(M_h, c, s=18, alpha=0.7, color="tab:purple",
               label=f"V5.3 pipeline re-fit (N = {len(M_h)})", zorder=3)

    # Notre fit
    M_grid = np.logspace(np.log10(M_h.min()), np.log10(M_h.max()), 200)
    log_M_grid = np.log10(M_grid / 1e12)
    log_c_grid = slope * log_M_grid + intercept
    ax.plot(M_grid, 10 ** log_c_grid, "b-", lw=2.0,
            label=f"Fit re-validation: c $\\propto M_h^{{{slope:+.3f} \\pm {slope_err:.3f}}}$")

    # Claim V5.3 master
    log_c_v5p3 = V5p3_CLAIMED_SLOPE * log_M_grid + intercept
    ax.plot(M_grid, 10 ** log_c_v5p3, "m--", lw=1.8,
            label=f"V5.3 master claim: slope $= {V5p3_CLAIMED_SLOPE:+.3f} \\pm {V5p3_CLAIMED_SLOPE_ERR:.3f}$")

    # Wechsler-Bullock
    c_WB = WB_AMPL * (M_grid / 1e12) ** WB_SLOPE
    ax.plot(M_grid, c_WB, "k:", lw=2.0,
            label=f"Wechsler-Bullock obs: slope $= {WB_SLOPE:+.3f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_h$ [$M_\odot$]")
    ax.set_ylabel(r"NFW concentration $c = r_{\rm vir} / r_s$")
    ax.set_title("V5.6 re-validation: c(M_h) sur SPARC, pipeline V5.3 master\n"
                 f"slope re-mesure = {slope:+.3f} (claim master {V5p3_CLAIMED_SLOPE:+.3f}, WB obs {WB_SLOPE:+.3f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    report = run_pipeline(verbose=True)
    out_png = OUTPUT_DIR / "v5_6_v5p3_master_refit.png"
    plot_results(report, out_png)
    print(f"\n[V5.6 re-fit] Figure ecrite : {out_png}")

    # Verdict explicite
    slope = report["slope_measured"]
    err = report["slope_bootstrap_std"]
    print()
    print("VERDICT")
    print("-------")
    if abs(slope - V5p3_CLAIMED_SLOPE) > 3 * V5p3_CLAIMED_SLOPE_ERR:
        print(f"  Le claim V5.3 master (+0.575) est INCOMPATIBLE avec la")
        print(f"  re-implementation standardisee : slope mesure {slope:+.3f}")
        print(f"  +/- {err:.3f}, ecart de {abs(slope - V5p3_CLAIMED_SLOPE)/V5p3_CLAIMED_SLOPE_ERR:.1f} sigma vs claim.")
    else:
        print(f"  Le claim V5.3 master (+0.575) est compatible avec le")
        print(f"  pipeline standardise : slope = {slope:+.3f} +/- {err:.3f}.")
    if abs(slope - WB_SLOPE) > 5 * err:
        print(f"  La 'tension 18 sigma vs WB' est REELLE : slope")
        print(f"  re-mesure incompatible avec WB (-0.10) a {abs(slope - WB_SLOPE)/err:.1f} sigma.")
    else:
        print(f"  La pente mesuree {slope:+.3f} +/- {err:.3f} n'est PAS")
        print(f"  incompatible avec WB (-0.10) a un niveau de tension extreme.")
