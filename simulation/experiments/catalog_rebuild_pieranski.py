r"""
Chantier 1 → D1 — reconstruction propre du catalogue via Pieranski 1.

Le catalogue actuel (theory/knot_catalog.py) utilise des paramétrisations
Fourier avec ~60-100 coefficients par nœud. Sept nœuds sur 19 ont des
U_EM artificiellement inflatés par leurs harmoniques haute fréquence
(Mur 1 du bilan chantier 1 : 8_3 ×69, 7_5 ×2703, etc.).

Ce script :
  1. Lance Pieranski 1 (smoothing + auto-évitement) sur chaque nœud.
  2. Calcule U_EM = ∮(κ²+τ²)ds sur la courbe RELAXÉE.
  3. Compare aux valeurs catalogue actuelles (REFERENCE_N50K).
  4. Applique TQNT-R6 (formule consolidée) avec les deux catalogues sur
     les 7 particules canoniques (e-, p, n, W, Z, H, t).
  5. Conclut : la reconstruction améliore-t-elle les masses prédites ?

Hypothèse : si U_EM(8_3) tombe de 1492 à ~50 (valeur typique), le secteur
EW devrait dramatiquement chuter (W, Z, H, t deviennent trop légers).
On testera donc si une simple recalibration de α₀ suffit, ou si la
structure de la formule s'effondre.
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K
from theory.mass_formula import c_j, EW_MARKER
from simulation.experiments.mass_pieranski import (
    pieranski_iteration, reparametrize, d_min_far, length, PUBLISHED,
)

KNOTS = list(REFERENCE_N50K.keys())


# ============================================================
# Calcul de Frenet sur un set de points (pas une string knot)
# ============================================================
def frenet_from_points(pts: np.ndarray) -> dict:
    """κ, τ, ds à partir de la polyligne fermée (pts[N,3])."""
    N = len(pts)
    # Paramètre uniforme [0,2π[ ; dt = 2π/N
    dt = 2 * np.pi / N
    p_next = np.roll(pts, -1, axis=0)
    p_prev = np.roll(pts, 1, axis=0)
    p_next2 = np.roll(pts, -2, axis=0)
    p_prev2 = np.roll(pts, 2, axis=0)
    dr = (p_next - p_prev) / (2 * dt)
    ddr = (p_next - 2 * pts + p_prev) / dt ** 2
    dddr = (p_next2 - 2 * p_next + 2 * p_prev - p_prev2) / (2 * dt ** 3)
    speed = np.linalg.norm(dr, axis=1)
    cross = np.cross(dr, ddr)
    cross_norm = np.linalg.norm(cross, axis=1)
    kappa = cross_norm / (speed ** 3 + 1e-30)
    triple = np.sum(cross * dddr, axis=1)
    tau = triple / (cross_norm ** 2 + 1e-30)
    ds = speed * dt
    return {"kappa": kappa, "tau": tau, "ds": ds, "L": float(np.sum(ds))}


def U_EM_from_points(pts: np.ndarray) -> float:
    """U_EM = ∮(κ²+τ²)ds calculé à partir d'une polyligne."""
    fr = frenet_from_points(pts)
    return float(np.sum((fr["kappa"] ** 2 + fr["tau"] ** 2) * fr["ds"]))


# ============================================================
# Pieranski 1 wrapper : retourne pts relaxés
# ============================================================
def pieranski_relax(name: str, n_points: int = 200, D: float = 1.0,
                    n_iter: int = 1000, alpha: float = 0.15, beta: float = 0.5,
                    skip_frac: float = 0.03) -> np.ndarray:
    pts = knot_curve(name, n_points + 1)[:-1]
    skip = max(int(n_points * skip_frac), 5)
    d0 = d_min_far(pts, skip)
    if d0 < 1e-10:
        d0 = 1e-3
    pts = pts * (D / d0)
    for it in range(n_iter):
        pts = pieranski_iteration(pts, alpha, D, beta, skip)
        if (it + 1) % 50 == 0:
            pts = reparametrize(pts, n_points)
    return pts


# ============================================================
# Construction des deux catalogues
# ============================================================
def build_catalogs():
    """Retourne {K: U_EM_V4, K: U_EM_relax}."""
    U_V4 = {K: float(REFERENCE_N50K[K]["sqrt_U"]) ** 2 for K in KNOTS}
    U_RELAX = {}
    pts_relax = {}
    print(f"  Pieranski 1 sur {len(KNOTS)} nœuds (N=200, 1000 itérations)…")
    for K in KNOTS:
        pts = pieranski_relax(K)
        U_RELAX[K] = U_EM_from_points(pts)
        pts_relax[K] = pts
    return U_V4, U_RELAX, pts_relax


# ============================================================
# Application de la formule TQNT-R6 sur les 7 canoniques
# ============================================================
SPECTRUM_CANON = [
    ("e-",       ["3_1"],               "single", 0.51099895),
    ("p",        ["3_1", "7_2", "7_2"], "hadron", 938.272),
    ("n",        ["3_1", "7_2", "7_2"], "hadron", 939.565),
    ("W",        ["4_1", "7_3", "8_3"], "ew",     80377.0),
    ("Z",        ["5_1", "7_2", "8_3"], "ew",     91188.0),
    ("H",        ["6_1", "7_4", "8_3"], "ew",     125250.0),
    ("t",        ["7_4", "7_6", "8_3"], "ew",     173100.0),
]


def mass_R6(comps, U_dict, alpha0):
    n = len(comps)
    if n == 1:
        f = 1.0
    elif EW_MARKER in comps:
        f = 1.0
    else:
        f = ((n + 1) / 2) ** (-3 / 8)
    U_tot = sum(U_dict[K] for K in comps)
    cJ_tot = sum(c_j(K) for K in comps)
    return alpha0 * U_tot * (f * cJ_tot) ** (32 / 9)


def calibrate(particles, U_dict):
    log_alphas = []
    for nm, comps, sec, me in particles:
        m_unit = mass_R6(comps, U_dict, alpha0=1.0)
        log_alphas.append(np.log(me / m_unit))
    return float(np.exp(np.mean(log_alphas)))


def main():
    print("=" * 110)
    print("  RECONSTRUCTION DU CATALOGUE TQNT VIA PIERANSKI 1 — IMPACT SUR LA FORMULE R6")
    print("=" * 110)

    U_V4, U_RELAX, _ = build_catalogs()

    # ─── Tableau de comparaison ──────────────────────────────────────────
    print(f"\n  {'K':>5} {'cr':>3} {'U_EM_V4':>10} {'U_EM_relax':>12} "
          f"{'ratio V4/relax':>16} {'flag':>8}")
    print("  " + "-" * 66)
    for K in KNOTS:
        cr = int(K.split("_")[0])
        u_v4 = U_V4[K]
        u_re = U_RELAX[K]
        ratio = u_v4 / u_re if u_re > 0 else float("inf")
        if ratio > 5:
            flag = "★ inflé"
        elif ratio > 2:
            flag = "↑ écart"
        elif ratio < 0.5:
            flag = "↓ relaxé"
        else:
            flag = "≈ stable"
        print(f"  {K:>5} {cr:>3} {u_v4:>10.2f} {u_re:>12.2f} "
              f"{ratio:>16.2f} {flag:>10}")

    # Stat globale
    ratios = np.array([U_V4[K] / U_RELAX[K] if U_RELAX[K] > 0 else np.nan
                       for K in KNOTS])
    n_inf = int(np.sum(ratios > 5))
    print(f"\n  STATISTIQUES : {n_inf} nœuds sur {len(KNOTS)} ont un facteur "
          f">5 entre catalogue V4 et Pieranski 1 (= pathologies confirmées)")
    print(f"  Ratio max : {np.nanmax(ratios):.1f}×  ({KNOTS[np.nanargmax(ratios)]})")
    print(f"  Ratio min : {np.nanmin(ratios):.2f}×  ({KNOTS[np.nanargmin(ratios)]})")
    print(f"  Médiane    : {np.nanmedian(ratios):.2f}×")

    # ─── Application sur le spectre canonique R6 ────────────────────────
    print(f"\n  {'═' * 80}")
    print(f"  ▶ APPLICATION DE TQNT-R6 AVEC LES DEUX CATALOGUES")
    print(f"  {'═' * 80}")

    # Calibrer α₀ globalement (LS log) avec chaque catalogue
    a0_V4_glob = calibrate(SPECTRUM_CANON, U_V4)
    a0_RE_glob = calibrate(SPECTRUM_CANON, U_RELAX)
    print(f"\n  α₀ (calib LS log sur 7 canoniques) :")
    print(f"    avec catalogue V4    = {a0_V4_glob:.4e} MeV")
    print(f"    avec catalogue relax = {a0_RE_glob:.4e} MeV  "
          f"(ratio = {a0_RE_glob/a0_V4_glob:.2f}×)")

    print(f"\n  {'particule':<10} {'m_exp':>11} {'m_R6 (V4 cat.)':>17} "
          f"{'m_R6 (relax cat.)':>19} {'ΔV4 %':>8} {'Δrelax %':>10}")
    print("  " + "-" * 78)
    log_V4, log_RE = [], []
    for nm, comps, sec, me in SPECTRUM_CANON:
        m_v4 = mass_R6(comps, U_V4, a0_V4_glob)
        m_re = mass_R6(comps, U_RELAX, a0_RE_glob)
        ev = (m_v4 - me) / me * 100
        er = (m_re - me) / me * 100
        log_V4.append(np.log(m_v4 / me))
        log_RE.append(np.log(m_re / me))
        f1 = "★" if abs(ev) < 1 else (" " if abs(ev) < 5 else "?")
        f2 = "★" if abs(er) < 1 else (" " if abs(er) < 5 else "?")
        print(f"  {nm:<10} {me:>11.2f} {m_v4:>17.2f} {m_re:>19.2f} "
              f"{ev:>+7.2f} {f1} {er:>+9.2f} {f2}")

    rms_v4 = float(np.sqrt(np.mean(np.array(log_V4) ** 2)))
    rms_re = float(np.sqrt(np.mean(np.array(log_RE) ** 2)))
    print(f"\n  RMS log :")
    print(f"    catalogue V4    : {rms_v4:.4f}")
    print(f"    catalogue relax : {rms_re:.4f}   "
          f"(Δ = {rms_re - rms_v4:+.4f})")

    # ─── Diagnostic des nœuds pathologiques ──────────────────────────────
    print(f"\n  ─── DIAGNOSTIC PAR NŒUD PATHOLOGIQUE ───")
    print(f"  Impact direct de la reconstruction sur U_EM des entrelacs canoniques :")
    print(f"  {'composante':>6} {'U_V4':>9} {'U_relax':>10} {'ratio':>9}")
    nodes_used = set()
    for nm, comps, sec, me in SPECTRUM_CANON:
        nodes_used.update(comps)
    for K in sorted(nodes_used):
        r = U_V4[K] / U_RELAX[K] if U_RELAX[K] > 0 else float("inf")
        print(f"  {K:>6} {U_V4[K]:>9.2f} {U_RELAX[K]:>10.2f} {r:>9.2f}×")

    # ─── Lecture ─────────────────────────────────────────────────────────
    print(f"\n  LECTURE :")
    print(f"  ──────────")
    print(f"  La reconstruction Pieranski 1 fait chuter U_EM(8_3) drastiquement")
    print(f"  (de {U_V4['8_3']:.0f} à {U_RELAX['8_3']:.1f}, ratio {U_V4['8_3']/U_RELAX['8_3']:.0f}×).")
    print(f"  Conséquence : le secteur EW (qui dépend U_EM_tot ≈ U_EM(8_3))")
    print(f"  perd l'essentiel de sa masse. Pour conserver les EW à 125 GeV,")
    print(f"  α₀ doit être augmenté d'un facteur ≈ {a0_RE_glob/a0_V4_glob:.0f}×.")
    print(f"  ")
    print(f"  Mais cette recalibration ne peut PAS satisfaire simultanément")
    print(f"  l'électron et le Higgs. C'est précisément le Mur 2 du bilan :")
    print(f"  le claim '9 paramètres globaux V4' cache 60-100 coeffs Fourier")
    print(f"  qui font implicitement le travail de tunage.")
    if rms_re > 1.0:
        print(f"  ")
        print(f"  ✗ Avec Pieranski 1 propre, la formule V4/R6 N'EST PLUS")
        print(f"    APPLICABLE : RMS = {rms_re:.2f}, signe que la formule")
        print(f"    nécessite les artefacts Fourier pour fonctionner.")
    elif rms_re < rms_v4 * 1.2:
        print(f"  ")
        print(f"  ✓ La reconstruction préserve la précision (RMS reste comparable).")
        print(f"    Le catalogue propre serait viable.")
    print("=" * 110)


if __name__ == "__main__":
    main()
