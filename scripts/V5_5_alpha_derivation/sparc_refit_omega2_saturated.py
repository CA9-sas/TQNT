"""
V5.5 ÉTAPE 9 — SPARC re-fit avec correction Ω²(M_h) saturée.

Objectif :
    Valider le modèle Ω²(M_h) saturé sigmoïdal de l'étape 8 sur les 175
    galaxies SPARC (Lelli, McGaugh, Schombert 2016). Comparaison directe
    au baseline V5.3 master :
        r_s = κ_CJ · L_coh^J(r_{1/2,b})         (κ_CJ = 62.7 universel)
        χ²/N = 3.05
    contre V5.5 :
        r_s = κ_CJ · L_coh^J(r_{1/2,b}) / sqrt(Ω²(M_h))

Pipeline par galaxie :
    1. Lire le rotmod_LTG/<gal>_rotmod.dat (R, Vobs, errV, Vgas, Vdisk, Vbul)
    2. V_flat (de SPARC_Lelli2016c.mrt) → M_h via TF baryonique-halo
       (relation simple M_h = (V_flat/V_norm)^3 × M_norm, calibrée sur MW)
    3. Construire V_bar(R)² = Vgas² + Vdisk² + Vbul²
    4. Calculer Ω²(M_h) via theory.cgq_envelope_saturated
    5. r_s = κ_CJ · L_coh^J / sqrt(Ω²(M_h)) (V5.5)
       r_s = κ_CJ · L_coh^J                  (V5.3 baseline pour comparaison)
    6. Halo NFW de masse virielle M_h donnée → V_NFW(R)
    7. V_model² = V_bar² + V_NFW²
    8. χ² = Σ (V_obs - V_model)² / errV²

Référence V5.3 : κ_CJ = 62.7, χ²/N = 3.05.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from theory.cgq_envelope_saturated import (  # noqa: E402
    omega_squared, h_saturated, M_STAR_MSUN, H_MAX,
)

# Paths
SPARC_ROOT = Path("/mnt/data/data_recherche/sparc")
SPARC_MAIN = SPARC_ROOT / "SPARC_Lelli2016c.mrt"
ROTMOD_DIR = SPARC_ROOT / "Rotmod_LTG"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Constants
G_KPC_KMS2_PER_MSUN = 4.30091e-6  # G in (kpc · (km/s)² / M_sun)
RHO_CRIT_MSUN_KPC3 = 1.40e2       # critical density Planck 2018 (M_sun / kpc^3)
                                  # = 9.47e-30 g/cm^3 converted
KAPPA_CJ_V5p3 = 62.7
C_SOUND_KMS = 10.0                # baryonic sound speed
WB_AMPL = 9.6                     # Wechsler-Bullock amplitude at 1e12 M_sun
WB_SLOPE = -0.10                  # WB slope d log c / d log M_h


# ---------------------------------------------------------------------------
# SPARC catalog reader
# ---------------------------------------------------------------------------

def parse_sparc_main(path: Path = SPARC_MAIN) -> list[dict]:
    """Parse the fixed-width main SPARC catalog (175 galaxies).

    The MRT format has a byte-by-byte description ending in a line of dashes
    around line 98. Galaxy data lines start right after (line 99) and run
    until end-of-file.
    """
    galaxies = []
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the LAST dash-only line — galaxy data starts right after.
    dash_indices = [
        i for i, line in enumerate(lines)
        if set(line.strip()) == {"-"} and len(line.strip()) > 10
    ]
    if not dash_indices:
        return galaxies
    data_start = dash_indices[-1] + 1

    # Per the MRT byte spec the columns are :
    #   1-11 name, 12-13 T, 14-19 D, 20-24 e_D, 25-26 f_D, 27-30 Inc,
    #   31-34 e_Inc, 35-41 L36, 42-48 e_L36, 49-53 Reff, 54-61 SBeff,
    #   62-66 Rdisk, 67-74 SBdisk, 75-81 MHI, 82-86 RHI, 87-91 Vflat,
    #   92-96 e_Vflat, 97-99 Q, 100-113 Ref.
    # In practice the actual file uses whitespace alignment that drifts
    # for entries with wider columns. The safest approach is to parse on
    # whitespace: the name occupies cols 1-11 (fixed-width), then the
    # remaining fields are space-separated numeric tokens in known order.
    for line in lines[data_start:]:
        if len(line.rstrip()) < 50:
            continue
        if line.startswith("Note") or line.startswith("Acknow"):
            continue
        # Fixed-width name (cols 1-11)
        name = line[0:11].strip()
        if not name:
            continue
        rest = line[11:].split()
        # Expected order: T D e_D f_D Inc e_Inc L36 e_L36 Reff SBeff Rdisk
        #                 SBdisk MHI RHI Vflat e_Vflat Q [Ref…]
        if len(rest) < 17:
            continue
        try:
            T = int(rest[0])
            D = float(rest[1])
            Vflat = float(rest[14])
            e_Vflat = float(rest[15])
            Q = int(rest[16])
            galaxies.append({
                "name": name,
                "T": T,
                "D_Mpc": D,
                "Vflat_kms": Vflat,
                "e_Vflat_kms": e_Vflat,
                "Q": Q,
            })
        except (ValueError, IndexError):
            continue
    return galaxies


def parse_rotmod(name: str, rotmod_dir: Path = ROTMOD_DIR) -> np.ndarray | None:
    """Read rotmod file for a galaxy. Returns (N, 8) array or None."""
    fname = rotmod_dir / f"{name}_rotmod.dat"
    if not fname.exists():
        # Try variants (some SPARC names have leading spaces)
        candidates = list(rotmod_dir.glob(f"{name.strip()}_rotmod.dat"))
        if not candidates:
            return None
        fname = candidates[0]
    try:
        data = np.loadtxt(fname, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Halo mass and NFW
# ---------------------------------------------------------------------------

def M_halo_from_Vflat(Vflat_kms: float) -> float:
    """Halo mass from Vflat (kms), calibrated on Milky Way analog.

    Use the abundance-matching relation Vflat ~ V_vir ~ 200 km/s ↔ M_h ~ 1e12.
    Pure r_vir ∝ M_h^(1/3) and ρ_vir = 200 ρ_crit fixes :
        V_vir² = G M_h / r_vir,   r_vir = (3 M / (4π·200·ρ_crit))^(1/3)
    Solving : M_h ∝ V_vir³ / sqrt(ρ_crit).

    Calibration: V_vir = 200 km/s → M_h = 1.5e12 (MW-like).
    """
    # M_h = (V/V0)^3 * M0 with V0=200, M0=1.5e12
    return 1.5e12 * (Vflat_kms / 200.0) ** 3


def r_vir_kpc(M_h_Msun: float) -> float:
    """Virial radius (200 ρ_crit definition)."""
    return (3.0 * M_h_Msun / (4.0 * math.pi * 200.0 * RHO_CRIT_MSUN_KPC3)) ** (1.0 / 3.0)


def V_NFW(R_kpc: np.ndarray, M_h_Msun: float, r_s_kpc: float) -> np.ndarray:
    """NFW circular velocity at radius R.

    V_NFW²(R) = G M(R) / R, with
        M(R) = 4π ρ_s r_s³ [ln(1+x) - x/(1+x)],  x = R/r_s
    Normalized such that the total enclosed mass at r_vir equals M_h.
    """
    R = np.asarray(R_kpc, dtype=float)
    r_vir = r_vir_kpc(M_h_Msun)
    c = r_vir / r_s_kpc
    # NFW mass profile factor
    mu = math.log(1.0 + c) - c / (1.0 + c)
    if mu <= 0:
        return np.zeros_like(R)
    rho_s_factor = M_h_Msun / mu  # = 4π ρ_s r_s³
    x = np.maximum(R / r_s_kpc, 1e-9)
    M_R = rho_s_factor * (np.log(1.0 + x) - x / (1.0 + x))
    V2 = G_KPC_KMS2_PER_MSUN * M_R / R
    V2 = np.where(V2 > 0, V2, 0.0)
    return np.sqrt(V2)


# ---------------------------------------------------------------------------
# Coherence length (V5.3 baseline — Jeans length at r_{1/2,baryonic})
# ---------------------------------------------------------------------------

def r_half_baryon_kpc(M_h_Msun: float) -> float:
    """Mo-Mao-White (1998) baryonic half-mass radius (kpc).

    r_d ≈ 0.0185 · r_vir · λ · 10    (calibration factor 10 cf. press_schechter_modified)
    r_{1/2,b} ≈ 1.68 · r_d
    """
    rv = r_vir_kpc(M_h_Msun)
    lam = 0.04  # Bullock+2001 spin parameter
    r_d = 0.0185 * rv * lam * 10.0
    return 1.68 * r_d


def rho_NFW_at_r_half(M_h_Msun: float) -> float:
    """ρ_NFW at the baryonic half-mass radius (M_sun/kpc³).

    Uses Wechsler-Bullock concentration for the canonical NFW reference profile.
    This is the LOCAL density that sets the Jeans length, NOT the mean
    baryonic density of the halo.
    """
    c = WB_AMPL * (M_h_Msun / 1e12) ** WB_SLOPE
    rv = r_vir_kpc(M_h_Msun)
    r_s = rv / c
    r_half = r_half_baryon_kpc(M_h_Msun)
    alpha = r_half / r_s
    delta_NFW = (200.0 / 3.0) * c ** 3 / (math.log(1.0 + c) - c / (1.0 + c))
    rho_s = RHO_CRIT_MSUN_KPC3 * delta_NFW
    return rho_s / (alpha * (1.0 + alpha) ** 2)


def L_coh_J_kpc(M_h_Msun: float) -> float:
    """Jeans coherence length L_coh^J = c_s / sqrt(4 π G ρ_b) (kpc).

    V5.3 master uses ρ_b = ρ_NFW(r_{1/2,b}) (LOCAL density at baryonic
    half-radius), not the halo-mean density. This scales with M_h and
    produces κ_CJ ≈ 62.7 calibration with χ²/N = 3.05.

    Note: in V5.3 a global baryonic density per halo is taken proportional
    to ρ_NFW at r_{1/2,b}; the c_s sound speed is fixed at 10 km/s.
    """
    rho = rho_NFW_at_r_half(M_h_Msun)  # M_sun / kpc^3
    denom = math.sqrt(4.0 * math.pi * G_KPC_KMS2_PER_MSUN * rho)  # km/s/kpc
    return C_SOUND_KMS / denom  # kpc


# ---------------------------------------------------------------------------
# Per-galaxy chi^2
# ---------------------------------------------------------------------------

def chi2_galaxy(rotmod: np.ndarray, M_h_Msun: float, r_s_kpc: float,
                rmin_kpc: float = 0.1) -> tuple[float, int]:
    """Compute chi^2 and N points for one galaxy under given r_s."""
    R = rotmod[:, 0]
    V_obs = rotmod[:, 1]
    e_V = rotmod[:, 2]
    V_gas = rotmod[:, 3]
    V_disk = rotmod[:, 4]
    V_bul = rotmod[:, 5]

    # Baryon V² (Vgas can be negative for centrifugal-deficient inner regions,
    # SPARC convention is V_gas² with sign — square then sum).
    V_bar2 = np.sign(V_gas) * V_gas ** 2 + V_disk ** 2 + V_bul ** 2
    V_bar2 = np.clip(V_bar2, 0.0, None)

    # NFW (capped at very small R)
    R_use = np.maximum(R, rmin_kpc)
    V_dm = V_NFW(R_use, M_h_Msun, r_s_kpc)

    V_model = np.sqrt(V_bar2 + V_dm ** 2)

    # Sanitize errors
    e_use = np.where(e_V > 0.5, e_V, 0.5)
    sel = (R > rmin_kpc) & np.isfinite(V_obs) & np.isfinite(V_model) & (V_obs > 0)
    if sel.sum() < 3:
        return float("nan"), 0
    resid = (V_obs[sel] - V_model[sel]) / e_use[sel]
    return float(np.sum(resid ** 2)), int(sel.sum())


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_refit(verbose: bool = True) -> dict:
    galaxies = parse_sparc_main()
    if verbose:
        print(f"[parse] {len(galaxies)} galaxies from main catalog")

    results = []
    skipped = 0
    for gal in galaxies:
        if gal["Vflat_kms"] <= 5.0:
            # No reliable Vflat (typically gas-dominated dwarves with V_flat unset)
            skipped += 1
            continue
        rot = parse_rotmod(gal["name"])
        if rot is None or rot.shape[0] < 4:
            skipped += 1
            continue

        Vflat = gal["Vflat_kms"]
        M_h = M_halo_from_Vflat(Vflat)
        Omega2 = omega_squared(M_h)
        h_val = h_saturated(M_h)

        # Baseline coherence length (V5.3) and V5.5 correction
        L_coh = L_coh_J_kpc(M_h)
        r_s_V53 = KAPPA_CJ_V5p3 * L_coh
        # V5.5 étape 8b : test sign convention via env var SIGN_CONVENTION
        # "multiply" : r_s × sqrt(Ω²) → c plus petit aux hautes M (sens WB)
        # "divide"   : r_s / sqrt(Ω²) → c plus grand aux hautes M
        import os
        _sign = os.environ.get("SIGN_CONVENTION", "divide")
        if _sign == "multiply":
            r_s_V55 = r_s_V53 * math.sqrt(Omega2)
        else:
            r_s_V55 = r_s_V53 / math.sqrt(Omega2)

        # Constrain r_s to be physical (< r_vir)
        rv = r_vir_kpc(M_h)
        r_s_V53 = min(r_s_V53, 0.9 * rv)
        r_s_V55 = min(r_s_V55, 0.9 * rv)
        c_V53 = rv / r_s_V53
        c_V55 = rv / r_s_V55

        chi2_53, N_53 = chi2_galaxy(rot, M_h, r_s_V53)
        chi2_55, N_55 = chi2_galaxy(rot, M_h, r_s_V55)

        results.append({
            "name": gal["name"],
            "Vflat": Vflat,
            "Q": gal["Q"],
            "M_h": M_h,
            "log_M_h": math.log10(M_h),
            "Omega2": Omega2,
            "h": h_val,
            "L_coh_kpc": L_coh,
            "r_s_V53_kpc": r_s_V53,
            "r_s_V55_kpc": r_s_V55,
            "c_V53": c_V53,
            "c_V55": c_V55,
            "r_vir_kpc": rv,
            "chi2_V53": chi2_53,
            "chi2_V55": chi2_55,
            "N": N_53,
        })

    if verbose:
        print(f"[refit] {len(results)} galaxies fitted, {skipped} skipped")
        print(f"[fraction] used = {len(results)/len(galaxies)*100:.1f} %")

    # Aggregate chi^2/N
    arr_chi53 = np.array([r["chi2_V53"] for r in results])
    arr_chi55 = np.array([r["chi2_V55"] for r in results])
    arr_N = np.array([r["N"] for r in results])
    valid = np.isfinite(arr_chi53) & np.isfinite(arr_chi55) & (arr_N > 0)

    total_chi53 = float(np.sum(arr_chi53[valid]))
    total_chi55 = float(np.sum(arr_chi55[valid]))
    total_N = int(np.sum(arr_N[valid]))
    chi2_per_N_V53 = total_chi53 / total_N if total_N > 0 else float("nan")
    chi2_per_N_V55 = total_chi55 / total_N if total_N > 0 else float("nan")

    summary = {
        "n_galaxies_used": int(valid.sum()),
        "n_galaxies_total": len(galaxies),
        "n_data_points": total_N,
        "chi2_per_N_V5p3": chi2_per_N_V53,
        "chi2_per_N_V5p5": chi2_per_N_V55,
        "chi2_per_N_V5p3_ref": 3.05,
        "improvement_factor": chi2_per_N_V53 / chi2_per_N_V55 if chi2_per_N_V55 > 0 else float("nan"),
        "results": results,
    }
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(summary: dict, outfile: Path = OUT / "v5p5_sparc_refit_omega2.png") -> Path:
    results = summary["results"]
    M_h = np.array([r["M_h"] for r in results])
    c_V53 = np.array([r["c_V53"] for r in results])
    c_V55 = np.array([r["c_V55"] for r in results])
    chi53 = np.array([r["chi2_V53"] / max(r["N"], 1) for r in results])
    chi55 = np.array([r["chi2_V55"] / max(r["N"], 1) for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (1) c(M_h) prediction vs WB
    ax = axes[0, 0]
    ax.scatter(M_h, c_V53, alpha=0.5, s=18, color="blue",
               label=f"V5.3 (κ_CJ universel) median c={np.nanmedian(c_V53):.1f}")
    ax.scatter(M_h, c_V55, alpha=0.5, s=18, color="red",
               label=f"V5.5 (Ω² saturé) median c={np.nanmedian(c_V55):.1f}")
    M_smooth = np.logspace(np.log10(M_h.min()), np.log10(M_h.max()), 100)
    c_WB = WB_AMPL * (M_smooth / 1e12) ** WB_SLOPE
    ax.plot(M_smooth, c_WB, "k--", lw=2,
            label=f"WB c={WB_AMPL}(M/10¹²)^{WB_SLOPE}")
    ax.axvline(M_STAR_MSUN, color="green", ls=":", alpha=0.7,
               label=f"M_* = 10¹²")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M_h (M_sun)")
    ax.set_ylabel("concentration c = r_vir / r_s")
    ax.set_title("Concentration vs halo mass (knee at 10¹²?)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, which="both")

    # (2) Ω²(M_h) profile
    ax = axes[0, 1]
    M_smooth = np.logspace(9, 15, 200)
    Omega2_smooth = np.array([omega_squared(m) for m in M_smooth])
    ax.semilogx(M_smooth, Omega2_smooth, "k-", lw=2, label="Ω²(M_h)")
    Omega2_data = np.array([r["Omega2"] for r in results])
    ax.scatter(M_h, Omega2_data, alpha=0.5, s=20, color="purple",
               label=f"SPARC galaxies (n={len(results)})")
    ax.axvline(M_STAR_MSUN, color="green", ls=":", label="M_*")
    ax.axhline(1.0 + H_MAX, color="red", ls="--", alpha=0.6,
               label=f"1+h_max = {1+H_MAX}")
    ax.set_xlabel("M_h (M_sun)")
    ax.set_ylabel("Ω²(M_h) = 1 + h(M_h)")
    ax.set_title("Profile saturé Ω²(M_h) — V5.5 étape 8")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # (3) chi^2 per galaxy
    ax = axes[1, 0]
    ax.hist(chi53, bins=30, alpha=0.5, color="blue",
            label=f"V5.3 mean χ²/N = {np.nanmean(chi53):.2f}")
    ax.hist(chi55, bins=30, alpha=0.5, color="red",
            label=f"V5.5 mean χ²/N = {np.nanmean(chi55):.2f}")
    ax.axvline(3.05, color="black", ls=":", label="V5.3 ref. 3.05")
    ax.set_xlabel("χ²/N per galaxy")
    ax.set_ylabel("# galaxies")
    ax.set_title("Distribution χ²/N par galaxie")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (4) chi^2_V55 / chi^2_V53 ratio vs M_h
    ax = axes[1, 1]
    ratio = chi55 / np.where(chi53 > 0, chi53, np.nan)
    ax.scatter(M_h, ratio, alpha=0.5, s=20, color="darkgreen")
    ax.axhline(1.0, color="black", ls=":", alpha=0.6, label="no change")
    ax.axvline(M_STAR_MSUN, color="green", ls=":", label="M_*")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M_h (M_sun)")
    ax.set_ylabel("χ²_V5.5 / χ²_V5.3")
    ax.set_title("Per-galaxy improvement V5.5/V5.3")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(outfile, dpi=130)
    plt.close()
    return outfile


def check_knee(results: list[dict]) -> dict:
    """Test if there's a slope change in c(M_h) at log M_h ~ 12."""
    M_h = np.array([r["M_h"] for r in results])
    c = np.array([r["c_V55"] for r in results])
    log_M = np.log10(M_h)
    log_c = np.log10(np.clip(c, 1e-3, None))

    # Fit two-segment power law: low (<M_*) and high (>M_*)
    low = log_M < 12.0
    high = log_M >= 12.0
    knee_info = {}
    if low.sum() >= 5:
        slope_low, _ = np.polyfit(log_M[low], log_c[low], 1)
        knee_info["slope_low"] = float(slope_low)
        knee_info["n_low"] = int(low.sum())
    if high.sum() >= 5:
        slope_high, _ = np.polyfit(log_M[high], log_c[high], 1)
        knee_info["slope_high"] = float(slope_high)
        knee_info["n_high"] = int(high.sum())
    # Overall slope
    slope_all, _ = np.polyfit(log_M, log_c, 1)
    knee_info["slope_all"] = float(slope_all)
    knee_info["slope_WB_target"] = WB_SLOPE
    knee_info["delta_slope"] = (knee_info.get("slope_high", float("nan"))
                                - knee_info.get("slope_low", float("nan")))
    return knee_info


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("V5.5 ÉTAPE 9 — SPARC re-fit avec Ω²(M_h) saturé")
    print("=" * 78)

    summary = run_refit(verbose=True)

    print()
    print("=" * 78)
    print("RÉSUMÉ")
    print("=" * 78)
    print(f"  Galaxies utilisées  : {summary['n_galaxies_used']} / {summary['n_galaxies_total']}")
    print(f"  Points cinématiques : {summary['n_data_points']}")
    print(f"  χ²/N V5.3 (calculé) : {summary['chi2_per_N_V5p3']:.3f}")
    print(f"  χ²/N V5.3 (ref.)    : {summary['chi2_per_N_V5p3_ref']}")
    print(f"  χ²/N V5.5 saturé    : {summary['chi2_per_N_V5p5']:.3f}")
    print(f"  Ratio V5.3 / V5.5   : {summary['improvement_factor']:.3f}")

    # Per-galaxy table top 10
    print()
    print("  Top 10 galaxies par |Δχ²/N| :")
    sorted_res = sorted(
        summary["results"],
        key=lambda r: abs((r["chi2_V53"] - r["chi2_V55"]) / max(r["N"], 1)),
        reverse=True,
    )
    print(f"    {'name':<14} {'V_flat':>7} {'log M_h':>8} {'Ω²':>6} "
          f"{'χ²/N V53':>9} {'χ²/N V55':>9}")
    for r in sorted_res[:10]:
        print(f"    {r['name']:<14} {r['Vflat']:>7.1f} {r['log_M_h']:>8.2f} "
              f"{r['Omega2']:>6.3f} {r['chi2_V53']/max(r['N'],1):>9.2f} "
              f"{r['chi2_V55']/max(r['N'],1):>9.2f}")

    # Knee check
    print()
    print("=" * 78)
    print("VÉRIFICATION DU KNEE c(M_h) à M_h ~ 10¹²")
    print("=" * 78)
    knee = check_knee(summary["results"])
    print(f"  Pente globale c(M_h)        : {knee.get('slope_all', float('nan')):+.4f}")
    print(f"  Pente low  (log M_h < 12)   : {knee.get('slope_low', float('nan')):+.4f}  "
          f"(N = {knee.get('n_low', 0)})")
    print(f"  Pente high (log M_h ≥ 12)   : {knee.get('slope_high', float('nan')):+.4f}  "
          f"(N = {knee.get('n_high', 0)})")
    print(f"  Δ slope (high − low)        : {knee.get('delta_slope', float('nan')):+.4f}")
    print(f"  Cible WB                    : {knee['slope_WB_target']:+.4f}")

    # Figure
    outfile = plot_results(summary)
    print()
    print(f"Figure : {outfile}")

    return summary, knee


if __name__ == "__main__":
    main()
