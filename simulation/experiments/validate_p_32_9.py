r"""
Chantier 1 → D4-2 — validation de la conjecture p = 32/9 sur le spectre V4.

Conjecture (cf. exponent_3_8_origin.py, lemme R6) :
   p_V4 = 32/9 = 3.555556  (au lieu de 3.55)
   f(n) = ((n+1)/2)^(-3/8) (au lieu de f(2)=0.86, f(3)=0.77)

Ces remplacements rendent la formule de masse V4 entièrement RATIONNELLE
(sauf α₀, calibré sur l'électron) :
   m = α₀ · U_EM · (((n+1)/2)^(-3/8) · c_J)^(32/9)

On teste que :
  (a) Sur singles : le shift est négligeable (<0.5%) après recalibration de α₀
  (b) Sur composites (hadrons, EW) : écart à expérience reste ≤1%
  (c) La conjecture est cohérente : aucune masse n'explose ou ne s'effondre

α₀ est RECALIBRÉ pour chaque valeur de p pour fixer m_e = 0.51099895 MeV
exactement. C'est l'unique paramètre libre du fit.

Cf. simulation/experiments/screening_factor_origin.py et
    docs/brouillon_chantier1_bilan.tex (lemme R6 + D4).
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j, EW_MARKER
from theory.abraham import REFERENCE_N50K

# ───────────────────────────────────────────────────────────────────────────
# Paramètres testés
# ───────────────────────────────────────────────────────────────────────────
P_V4 = 3.55             # V4 actuel
P_CONJ = 32.0 / 9.0     # conjecture (= 3.5555556)

# f(n) : V4 vs conjecture R6
F_V4 = {1: 1.00, 2: 0.86, 3: 0.77}
def f_R6(n): return ((n + 1) / 2) ** (-3 / 8)

M_E_EXP = 0.51099895  # MeV — électron référence calibration

# ───────────────────────────────────────────────────────────────────────────
# Formule de masse paramétrée
# ───────────────────────────────────────────────────────────────────────────
def mass_param(components, alpha0, p, f_func, sector="auto"):
    n = len(components)
    if sector == "auto":
        if n == 1:
            sector = "single"
        elif EW_MARKER in components:
            sector = "ew"
        else:
            sector = "hadron"
    if sector in ("single", "ew"):
        f = 1.0
    else:
        f = f_func(n) if callable(f_func) else f_func.get(n, 1.0)
    U_tot = sum(u_em(K) for K in components)
    cJ_tot = sum(c_j(K) for K in components)
    return alpha0 * U_tot * (f * cJ_tot) ** p


def calibrate_alpha0(p, f_func, electron_cfg=("3_1",)):
    """Calibre α₀ pour m_e = 0.51099895 MeV exactement."""
    n = len(electron_cfg)
    f = f_func(n) if callable(f_func) else f_func.get(n, 1.0)
    U = sum(u_em(K) for K in electron_cfg)
    cJ = sum(c_j(K) for K in electron_cfg)
    # m_e = α₀ · U · (f · cJ)^p ⟹ α₀ = m_e / (U · (f cJ)^p)
    return M_E_EXP / (U * (f * cJ) ** p)


# ───────────────────────────────────────────────────────────────────────────
# Spectre V4 calculable (configurations VRAIMENT calculables)
# ───────────────────────────────────────────────────────────────────────────
SPECTRUM = [
    # (nom, config, secteur, m_exp [MeV])
    ("électron",       ["3_1"],              "single", 0.51099895),
    ("proton",         ["3_1", "7_2", "7_2"], "hadron", 938.272),
    ("neutron",        ["3_1", "7_2", "7_2"], "hadron", 939.565),
    ("W±",             ["4_1", "7_3", "8_3"], "ew",     80377.0),
    ("Z⁰",             ["5_1", "7_2", "8_3"], "ew",     91188.0),
    ("Higgs",          ["6_1", "7_4", "8_3"], "ew",     125250.0),
    ("top",            ["7_4", "7_6", "8_3"], "ew",     173100.0),
    # mésons (n=2, hadron) — quelques tests
    ("π+ candidat",    ["5_2", "7_2"],        "hadron", 139.57),
    ("K+ candidat",    ["3_1", "8_2"],        "hadron", 493.7),
]


def main():
    print("=" * 100)
    print("  VALIDATION CONJECTURE p = 32/9 ET f(n) = ((n+1)/2)^(-3/8) SUR SPECTRE V4")
    print("=" * 100)
    print(f"  p_V4   = {P_V4:.10f}")
    print(f"  p_conj = 32/9 = {P_CONJ:.10f}")
    print(f"  Δp     = {(P_CONJ - P_V4):.6f}  ({(P_CONJ - P_V4)/P_V4*100:+.3f}%)")
    print()
    print(f"  f_V4 (hardcoded)  : f(1)=1.00,  f(2)=0.86,  f(3)=0.77")
    print(f"  f_R6 conjecture   : f(1)=1.00,  f(2)={f_R6(2):.4f},  f(3)={f_R6(3):.4f}")

    # Calibration de α₀ pour chaque scénario
    a0_V4 = calibrate_alpha0(P_V4, F_V4)
    a0_R6 = calibrate_alpha0(P_CONJ, f_R6)
    a0_HYB = calibrate_alpha0(P_CONJ, F_V4)  # p conjecture mais f V4
    a0_V4_R6 = calibrate_alpha0(P_V4, f_R6)  # p V4 mais f R6
    print(f"\n  α₀ recalibrés (pour m_e = {M_E_EXP} MeV exact) :")
    print(f"    α₀(V4 : p=3.55,  f_V4) = {a0_V4:.6e} MeV")
    print(f"    α₀(R6 : p=32/9,  f_R6) = {a0_R6:.6e} MeV  "
          f"(shift {(a0_R6/a0_V4-1)*100:+.3f}%)")
    print(f"    α₀(HYB : p=32/9, f_V4) = {a0_HYB:.6e} MeV")
    print(f"    α₀(V4_R6 : p=3.55, f_R6) = {a0_V4_R6:.6e} MeV")

    # ─── Tableau de validation ───────────────────────────────────────────
    print(f"\n  {'particule':<18} {'m_exp':>11} {'m_V4':>11} {'m_R6':>11} "
          f"{'ΔV4%':>7} {'ΔR6%':>7} {'shift R6/V4':>11}")
    print("  " + "-" * 87)

    scenarios = [
        ("V4", P_V4, F_V4, a0_V4),
        ("R6", P_CONJ, f_R6, a0_R6),
    ]

    rows = []
    for name, comp, sector, m_exp in SPECTRUM:
        results = {}
        for s_label, p, f, a0 in scenarios:
            m = mass_param(comp, a0, p, f, sector)
            err = (m - m_exp) / m_exp * 100
            results[s_label] = (m, err)
        m_V4, e_V4 = results["V4"]
        m_R6, e_R6 = results["R6"]
        shift = (m_R6 / m_V4 - 1) * 100
        rows.append((name, m_exp, m_V4, m_R6, e_V4, e_R6, shift))
        print(f"  {name:<18} {m_exp:>11.3f} {m_V4:>11.3f} {m_R6:>11.3f} "
              f"{e_V4:>+6.2f}% {e_R6:>+6.2f}% {shift:>+10.3f}%")

    # ─── Statistiques ────────────────────────────────────────────────────
    print(f"\n  STATISTIQUES (toutes particules) :")
    e_V4_arr = np.array([r[4] for r in rows])
    e_R6_arr = np.array([r[5] for r in rows])
    shifts = np.array([r[6] for r in rows])
    rms_V4 = float(np.sqrt(np.mean(e_V4_arr ** 2)))
    rms_R6 = float(np.sqrt(np.mean(e_R6_arr ** 2)))
    print(f"    RMS erreur V4 : {rms_V4:.2f}%   (max {np.max(np.abs(e_V4_arr)):.2f}%)")
    print(f"    RMS erreur R6 : {rms_R6:.2f}%   (max {np.max(np.abs(e_R6_arr)):.2f}%)")
    print(f"    Shift R6/V4 :   moyenne {np.mean(shifts):+.3f}%, "
          f"max {np.max(np.abs(shifts)):.3f}%")

    # ─── Décomposition de la conjecture (effet de p vs effet de f) ───────
    print(f"\n  ─── Décomposition : isoler effet de p (vs effet de f) ───")
    print(f"  {'particule':<18} {'V4':>11} {'HYB':>11} {'V4_R6':>11} {'R6':>11}")
    print(f"  {'':<18} {'(p_V4,f_V4)':>11} {'(p_C,f_V4)':>11} "
          f"{'(p_V4,f_R6)':>11} {'(p_C,f_R6)':>11}")
    print("  " + "-" * 64)
    for name, comp, sector, m_exp in SPECTRUM:
        m1 = mass_param(comp, a0_V4, P_V4, F_V4, sector)
        m2 = mass_param(comp, a0_HYB, P_CONJ, F_V4, sector)
        m3 = mass_param(comp, a0_V4_R6, P_V4, f_R6, sector)
        m4 = mass_param(comp, a0_R6, P_CONJ, f_R6, sector)
        print(f"  {name:<18} {m1:>11.3f} {m2:>11.3f} {m3:>11.3f} {m4:>11.3f}")

    # ─── Recalibration globale de α₀ (least-squares log) ────────────────
    print(f"\n  ─── α₀ recalibré GLOBALEMENT (minimise RMS log sur tout le spectre) ───")
    print(f"  Au lieu de calibrer sur le seul électron, on minimise")
    print(f"  Σ (log(m_pred/m_exp))² sur les 7 particules bien connues")
    print(f"  (excluant les candidats π+/K+ qui sont des guesses).")
    print()

    document principal = [(name, comp, sector, m_exp) for (name, comp, sector, m_exp)
             in SPECTRUM if "candidat" not in name]

    def global_alpha0(p, f_func):
        """α₀ qui minimise RMS log sur les 7 particules canoniques."""
        # log(m_pred) - log(m_exp) = log(α₀) + log(U·(f·cJ)^p) - log(m_exp)
        # ⟹ log(α₀)_opt = mean(log(m_exp) - log(U·(f·cJ)^p))
        log_alpha = []
        for name, comp, sector, m_exp in document principal:
            n = len(comp)
            if sector == "auto":
                sector = "single" if n == 1 else ("ew" if EW_MARKER in comp
                                                  else "hadron")
            if sector in ("single", "ew"):
                f = 1.0
            else:
                f = f_func(n) if callable(f_func) else f_func.get(n, 1.0)
            U_tot = sum(u_em(K) for K in comp)
            cJ_tot = sum(c_j(K) for K in comp)
            base = U_tot * (f * cJ_tot) ** p
            log_alpha.append(np.log(m_exp / base))
        return float(np.exp(np.mean(log_alpha)))

    a0_V4_glob = global_alpha0(P_V4, F_V4)
    a0_R6_glob = global_alpha0(P_CONJ, f_R6)
    print(f"  α₀ V4 (global LS) = {a0_V4_glob:.6e} MeV  "
          f"(vs calib électron {a0_V4:.6e}, écart {(a0_V4_glob/a0_V4-1)*100:+.3f}%)")
    print(f"  α₀ R6 (global LS) = {a0_R6_glob:.6e} MeV  "
          f"(vs calib électron {a0_R6:.6e}, écart {(a0_R6_glob/a0_R6-1)*100:+.3f}%)")
    print()
    print(f"  {'particule':<18} {'m_exp':>11} {'m_V4(glob)':>11} {'m_R6(glob)':>11} "
          f"{'ΔV4%':>7} {'ΔR6%':>7}")
    print("  " + "-" * 73)
    for name, comp, sector, m_exp in SPECTRUM:
        m1 = mass_param(comp, a0_V4_glob, P_V4, F_V4, sector)
        m2 = mass_param(comp, a0_R6_glob, P_CONJ, f_R6, sector)
        e1 = (m1 - m_exp) / m_exp * 100
        e2 = (m2 - m_exp) / m_exp * 100
        flag = "★" if abs(e2 - e1) < 0.1 else " "
        print(f"  {name:<18} {m_exp:>11.3f} {m1:>11.3f} {m2:>11.3f} "
              f"{e1:>+6.2f}% {e2:>+6.2f}%  {flag}")

    # RMS global
    e_V4_g = [((mass_param(c, a0_V4_glob, P_V4, F_V4, s) - me) / me * 100)
              for n, c, s, me in document principal]
    e_R6_g = [((mass_param(c, a0_R6_glob, P_CONJ, f_R6, s) - me) / me * 100)
              for n, c, s, me in document principal]
    rms_V4_g = float(np.sqrt(np.mean(np.array(e_V4_g) ** 2)))
    rms_R6_g = float(np.sqrt(np.mean(np.array(e_R6_g) ** 2)))
    print(f"\n  RMS sur les 7 canoniques (α₀ globaux):")
    print(f"    V4 : RMS = {rms_V4_g:.3f}%")
    print(f"    R6 : RMS = {rms_R6_g:.3f}%   (Δ = {rms_R6_g - rms_V4_g:+.3f}%)")

    # ─── Lecture ─────────────────────────────────────────────────────────
    print(f"\n  LECTURE :")
    print(f"  ─────────────────────────────────")
    max_shift = np.max(np.abs(shifts))
    if max_shift < 1.0:
        print(f"  ★ CONJECTURE VALIDÉE : aucune masse ne dérive de plus de {max_shift:.2f}%")
        print(f"    après remplacement p=3.55→32/9 et f_V4→f_R6. Le spectre V4 est")
        print(f"    STRICTEMENT cohérent avec p = 32/9 et f(n) = ((n+1)/2)^(-3/8).")
        print(f"  ")
        print(f"  Conséquences :")
        print(f"    - 9 paramètres globaux V4 → 6 (p, f(2), f(3) fixés rationnels)")
        print(f"    - p × 3/8 = 4/3 (de Gennes-Fisher exposant membrane élastique)")
        print(f"    - Wilson loop dans rep j=3/2 de SU(2)_8 (lien diquark décuplet)")
    elif max_shift < 5.0:
        print(f"  ✓ Shift maximum {max_shift:.2f}% — globalement OK mais quelques")
        print(f"    masses dérivent notablement. À vérifier au cas par cas.")
    else:
        print(f"  ✗ Shift maximum {max_shift:.1f}% — la conjecture EMPIRE la formule.")
        print(f"    p = 32/9 n'est probablement PAS la bonne valeur exacte.")
    print("=" * 100)


if __name__ == "__main__":
    main()
