r"""
Chantier 1 — test de la formule de masse "tightness pure" sur le spectre complet.

Formule candidate (cf. mass_ropelength_direct.py) :
   m(K) = m_e · [f(n) · R_tot(K) / R(3_1)]^p
avec
   R_tot(K) = somme sur les composantes de R(K_i) ≈ 3,85·cr(K_i) + 5,61
   f(n) = 1 (single), 0,86 (n=2 mesons), 0,77 (n=3 baryons), 1 (EW)
   p = 7,42 (calibré sur e⁻, π⁰)

On teste sur les 46 particules du spectre V4 :
  - 1 algèbre : cr(K) (crossing number, le plus simple imaginable)
  - 1 paramètre calibré : p
  - 2 paramètres de screening : f(2), f(3) (mêmes valeurs que V4)
  - 1 paramètre calibré : m_e (échelle)

Total : 1 + 2 + 1 = 4 paramètres effectifs (vs 9 pour V4 actuel).

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import numpy as np

# ============================================================
# Données expérimentales (sourced from V5 doc spectrum)
# ============================================================
# Format : (label, m_exp_MeV, config_string, sector)
# config_string : "K_1+K_2+..." ou un single nom de nœud
# sector : "single", "hadron" (utilise f(n)), "ew" (f=1)

PARTICLES = [
    ("e-",      0.51099895, "3_1",            "single"),
    ("nu",      0.0,         "6_2+9_1",        "hadron"),
    ("u",       2.16,        "3_1",            "single"),
    ("d",       4.67,        "3_1",            "single"),
    ("s",       95.,         "5_2+5_2",        "hadron"),
    ("c",       1270.,       "7_3+7_7",        "hadron"),
    ("b",       4180.,       "4_1+5_2+7_6",    "hadron"),
    ("mu-",     105.66,      "3_1+6_1",        "hadron"),
    ("tau-",    1776.9,      "3_1+5_1+7_1",    "hadron"),
    ("pi0",     134.98,      "8_1",            "single"),
    ("pi+-",    139.57,      "8_1",            "single"),
    ("K+-",     493.68,      "4_1+4_1+5_2",    "hadron"),
    ("K0_S",    497.60,      "4_1+4_1+5_2",    "hadron"),
    ("eta",     547.86,      "3_1+4_1+5_1",    "hadron"),
    ("etap",    957.78,      "5_2+5_2+6_1",    "hadron"),
    ("rho/om",  775.0,       "5_2+7_7",        "hadron"),
    ("phi",     1019.,       "3_1+4_1+6_2",    "hadron"),
    ("Jpsi",    3097.,       "4_1+5_2+9_1",    "hadron"),
    ("Ups",     9460.,       "3_1+7_4+9_1",    "hadron"),
    ("D0",      1864.,       "6_2+7_7",        "hadron"),
    ("D+",      1870.,       "6_2+7_7",        "hadron"),
    ("B",       5279.,       "4_1+10_1",       "hadron"),
    ("p",       938.27,      "3_1+7_2+7_2",    "hadron"),
    ("n",       939.57,      "3_1+7_2+7_2",    "hadron"),
    ("Lambda",  1115.7,      "4_1+7_4",        "hadron"),
    ("Sigma+",  1189.4,      "6_2+7_2",        "hadron"),
    ("Sigma0",  1192.6,      "6_2+7_2",        "hadron"),
    ("Sigma-",  1197.4,      "6_2+7_2",        "hadron"),
    ("Xi0",     1314.9,      "4_1+4_1+7_2",    "hadron"),
    ("Xi-",     1321.7,      "4_1+4_1+7_2",    "hadron"),
    ("Delta",   1232.,       "3_1+5_2+6_2",    "hadron"),
    ("Sigma*",  1383.,       "5_2+6_2+7_2",    "hadron"),
    ("Xi*",     1532.,       "4_1+5_2+7_2",    "hadron"),
    ("Omega-",  1672.,       "5_2+5_2+7_2",    "hadron"),
    ("Lambdac", 2286.,       "6_3+8_1",        "hadron"),
    ("Sigmac",  2455.,       "4_1+5_1+7_1",    "hadron"),
    ("Omegac",  2695.,       "3_1+6_2+6_2",    "hadron"),
    ("Lambdab", 5620.,       "5_2+5_2+8_2",    "hadron"),
    ("W",       80377.,      "4_1+7_3+8_3",    "ew"),
    ("Z0",      91188.,      "5_1+7_2+8_3",    "ew"),
    ("H",       125250.,     "6_1+7_4+8_3",    "ew"),
    ("t",       173100.,     "7_4+7_6+8_3",    "ew"),
]


# ============================================================
# Algèbre simple : R(K) = a · cr(K) + b
# ============================================================
A_R, B_R = 3.85, 5.61    # R(K) ≈ 3,85·cr + 5,61  (fit sur 19 ropelength publiées)

def crossing_number(knot: str) -> int:
    """cr(K) = premier index du nom (3_1 → 3, 10_1 → 10)."""
    return int(knot.split("_")[0])

def R_alg(knot: str) -> float:
    """Rope-length algébrique R(K) ≈ 3,85·cr(K) + 5,61."""
    return A_R * crossing_number(knot) + B_R

def R_total(config: str) -> tuple[float, int]:
    """Pour entrelacs, R_tot = somme des R(K_i). Retourne (R_tot, n_components)."""
    components = config.split("+")
    return float(sum(R_alg(K) for K in components)), len(components)


# ============================================================
# Formule de masse tightness
# ============================================================
# Paramètres : 1 exposant + 2 screening (f(2), f(3))
P_EXP = 7.42                  # calibré sur (e⁻, π⁰) en single-knot
F_SCREEN = {1: 1.0, 2: 0.86, 3: 0.77}    # MÊMES que V4

R_E = R_alg("3_1")            # référence : électron

def mass_tightness(config: str, sector: str, m_e: float = 0.51099895,
                   p: float = P_EXP) -> float:
    """m = m_e · [f(n) · R_tot / R(3_1)]^p."""
    R_tot, n = R_total(config)
    f = 1.0 if sector in ("single", "ew") else F_SCREEN.get(n, 1.0)
    return m_e * (f * R_tot / R_E) ** p


def main():
    print("=" * 96)
    print("  TEST DE LA FORMULE 'TIGHTNESS PURE' SUR LE SPECTRE COMPLET (46 particules)")
    print("=" * 96)
    print(f"  R(K) ≈ {A_R}·cr(K) + {B_R}  (algèbre simple, R² 0,957 vs publiées)")
    print(f"  m(K) = m_e · [f(n)·R_tot/R(3_1)]^p, p={P_EXP}, f(2)={F_SCREEN[2]}, f(3)={F_SCREEN[3]}")
    print(f"\n  {'particule':>10} {'config':>20} {'n':>2} {'cr_tot':>7} {'R_tot':>8} "
          f"{'m_pred':>12} {'m_exp':>11} {'Δlog':>8}")
    print("  " + "-" * 90)

    log_devs = []
    log_devs_by_sector = {"single": [], "hadron": [], "ew": []}

    for label, m_exp, config, sector in PARTICLES:
        if m_exp == 0:
            continue
        comps = config.split("+")
        cr_tot = sum(crossing_number(K) for K in comps)
        R_tot, n = R_total(config)
        m_pred = mass_tightness(config, sector)
        ratio = m_pred / m_exp
        log_dev = np.log10(ratio)
        log_devs.append(log_dev)
        log_devs_by_sector[sector].append(log_dev)
        flag = "·" if abs(log_dev) < 0.3 else ("*" if abs(log_dev) < 1 else "★")
        print(f"  {label:>10} {config:>20} {n:>2} {cr_tot:>7} {R_tot:>8.1f} "
              f"{m_pred:>12.2f} {m_exp:>11.2f} {log_dev:>+8.3f} {flag}")

    # Statistiques globales
    devs = np.array(log_devs)
    rms = np.sqrt(np.mean(devs ** 2))
    print(f"\n  Statistiques (sur {len(devs)} particules avec m_exp > 0) :")
    print(f"    RMS log10(m_pred/m_exp) = {rms:.3f}  (facteur typique {10**rms:.1f}×)")
    print(f"    biais moyen             = {devs.mean():+.3f}  ({10**devs.mean():.2f}×)")
    print(f"    abs max                 = {np.abs(devs).max():.3f}  "
          f"(facteur {10**np.abs(devs).max():.0f}×)")
    n_within = {threshold: np.sum(np.abs(devs) < threshold)
                for threshold in (0.1, 0.3, 0.5, 1.0)}
    print(f"    nb dans facteur 1,26×   = {n_within[0.1]}/{len(devs)} ({100*n_within[0.1]/len(devs):.0f}%)")
    print(f"    nb dans facteur 2×      = {n_within[0.3]}/{len(devs)} ({100*n_within[0.3]/len(devs):.0f}%)")
    print(f"    nb dans facteur 3×      = {n_within[0.5]}/{len(devs)} ({100*n_within[0.5]/len(devs):.0f}%)")
    print(f"    nb dans facteur 10×     = {n_within[1.0]}/{len(devs)} ({100*n_within[1.0]/len(devs):.0f}%)")

    # Par secteur
    print(f"\n  Par secteur :")
    for sec, ld in log_devs_by_sector.items():
        if ld:
            arr = np.array(ld)
            print(f"    {sec:>8} ({len(arr)} part.) : RMS = {np.sqrt(np.mean(arr**2)):.3f}, "
                  f"biais = {arr.mean():+.3f}")

    # Refit p sur tout le spectre
    print(f"\n  REFIT de p sur tout le spectre (m_exp > 0) :")
    # m_pred = m_e · (f·R/R_e)^p ⇒ log(m_exp/m_e) = p · log(f·R/R_e)
    x = []
    y = []
    for label, m_exp, config, sector in PARTICLES:
        if m_exp <= 0:
            continue
        R_tot, n = R_total(config)
        f = 1.0 if sector in ("single", "ew") else F_SCREEN.get(n, 1.0)
        x.append(np.log(f * R_tot / R_E))
        y.append(np.log(m_exp / 0.51099895))
    x = np.array(x); y = np.array(y)
    # contrainte : passer par origine (calibration sur e⁻ : m_e/m_e = 1, log = 0)
    # pente seule
    p_fit = np.sum(x * y) / np.sum(x * x)
    res = y - p_fit * x
    rms_fit = np.sqrt(np.mean(res ** 2)) / np.log(10)
    print(f"    p_optimal = {p_fit:.3f}  (vs p calibré (e⁻,π⁰) = {P_EXP})")
    print(f"    RMS log10 avec p optimal = {rms_fit:.3f}")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ─────────────────────────────────────────────")
    if rms < 0.3:
        print(f"  SUCCÈS : la formule tightness pure prédit le spectre à mieux qu'un")
        print(f"  facteur 2× typique (RMS = {rms:.2f}). Avec 4 paramètres seulement.")
    elif rms < 0.5:
        print(f"  PARTIEL : RMS = {rms:.2f} (facteur 3× typique) ; meilleur que")
        print(f"  certains schémas mais reste à raffiner.")
    else:
        print(f"  LIMITE : RMS = {rms:.2f} (facteur {10**rms:.0f}× typique).")
        print(f"  La formule pure 'cr+f(n)' n'est pas assez fine pour le spectre.")
        print(f"  Il manque les distinctions inter-knots de même cr (que c_J apporte).")
    print("=" * 96)


if __name__ == "__main__":
    main()
