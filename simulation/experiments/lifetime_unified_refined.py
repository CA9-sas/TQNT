r"""
Formule factorisée RAFFINÉE — durée de vie faible par classe topologique.

Trois classes, chacune avec sa physique propre :
  (A) ondes libres 3-corps / semi-lept. (n, μ, τ, D, Λc, Ωc, B, Λb) :
        Γ = C_A · θ_CKM · Q^ν      (Sargent ; ν ajusté ~ 5)
  (B) 2-corps leptonique (π±, K± → μν) :
        Γ = C_B · |V|² · f_P² m_ℓ² m_P (1 − m_ℓ²/m_P²)²   (hélicité)
  (C) non-leptonique noué (Λ, Σ, Ξ, Ω, K0_S) :
        Γ = C_C · θ_CKM            (Q-indépendant ; reconnexion)

θ_CKM = |V|² (valeurs mesurées, NON ajustées). Paramètres ajustés : C_A, ν,
C_B, C_C (4). On compare au précédent (un C par classe, ν=5, 2-corps plat :
RMS 0,95) pour voir si le raffinement 2-corps + ν libre fait tomber le RMS.

Usage :  python3 -m simulation.experiments.lifetime_unified_refined
"""
from __future__ import annotations

import numpy as np

Vud2, Vus2, Vcs2, Vcb2 = 0.949, 0.0497, 0.949, 0.00169
M_MU, M_PI, M_K = 105.66, 139.57, 493.68
F_PI, F_K = 130.2, 155.7   # constantes de désintégration (MeV)

# classe A : (nom, Q, θ, facteur_effectif, τ_exp)
#   facteur = N_canaux × amplification de matrice :
#     μ : 1 (pur leptonique, point)
#     n : (1+3g_A²) ≈ 5,86 (amplification axiale Gamow-Teller du nucléon, g_A=1,272)
#     τ, quarks lourds : 5 (2 leptoniques + 3 couleurs hadroniques, inclusif)
GA = 1.2723
W_N = 1.0 + 3.0 * GA**2   # ≈ 5.86 (facteur axial nucléon)
A = [
    ("n",       0.78,  Vud2, W_N, 878.4),
    ("mu-",     105.7, 1.0,  1.0, 2.197e-6),
    ("tau-",    1777., 1.0,  5.0, 2.903e-13),
    ("D0",      1270., Vcs2, 5.0, 4.10e-13),
    ("D+",      1270., Vcs2, 5.0, 1.033e-12),
    ("Lambdac", 1270., Vcs2, 5.0, 2.03e-13),
    ("Omegac",  1270., Vcs2, 5.0, 2.68e-13),
    ("B",       4180., Vcb2, 5.0, 1.519e-12),
    ("Lambdab", 4180., Vcb2, 5.0, 1.471e-12),
]
# classe B : (nom, f, m_P, |V|², τ_exp)
B = [
    ("pi+-", F_PI, M_PI, Vud2, 2.603e-8),
    ("K+-",  F_K,  M_K,  Vus2, 1.238e-8),
]
# classe C : (nom, θ, τ_exp)  — Q-indépendant
C = [
    ("Lambda", Vus2, 2.617e-10), ("Sigma+", Vus2, 8.018e-11),
    ("Sigma-", Vus2, 1.479e-10), ("Xi0", Vus2, 2.90e-10),
    ("Xi-", Vus2, 1.639e-10), ("Omega-", Vus2, 8.21e-11),
    ("K0_S", Vus2, 8.954e-11),
]


def main():
    print("=" * 86)
    print("  FORMULE FACTORISÉE RAFFINÉE — 2-corps hélicité + ν Sargent ajusté")
    print("=" * 86)
    resid_all, names_all = [], []

    # --- classe A : log τ = -logC_A - logθ - log N_ch - ν logQ  (fit logC_A, ν) ---
    lt = np.array([np.log10(r[4]) for r in A])
    lth = np.array([np.log10(r[2]) for r in A])
    lnch = np.array([np.log10(r[3]) for r in A])
    lQ = np.array([np.log10(r[1]) for r in A])
    X = np.column_stack([np.ones(len(A)), -lQ])      # y' = lt + logθ + logN_ch = c - ν logQ
    y = lt + lth + lnch
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    nu = c[1]
    resid = y - X @ c
    rmsA = np.sqrt(np.mean(resid ** 2))
    print(f"\n  (A) ondes libres 3-corps/semi-lept. (+N_ch) : ν = {nu:.2f}, RMS = {rmsA:.3f} déc ({len(A)} part.)")
    for r, dd in sorted(zip(A, resid), key=lambda z: z[0][4]):
        print(f"      {r[0]:>8} Q={r[1]:>7.1f} θ={r[2]:.3f} N_ch={r[3]}  résidu={dd:>+6.2f}")
    resid_all += list(resid); names_all += [r[0] for r in A]

    # --- classe B : 2-corps hélicité (fit logC_B) ---
    def rateB(f, mP, V2):
        return V2 * f**2 * M_MU**2 * mP * (1 - M_MU**2/mP**2)**2
    ltB = np.array([np.log10(r[4]) for r in B])
    base = np.array([-np.log10(rateB(r[1], r[2], r[3])) for r in B])
    cB = np.mean(ltB - base)
    residB = ltB - (cB + base)
    rmsB = np.sqrt(np.mean(residB ** 2))
    print(f"\n  (B) 2-corps leptonique (hélicité) : RMS = {rmsB:.3f} déc ({len(B)} part.)")
    for r, dd in zip(B, residB):
        print(f"      {r[0]:>8} f={r[1]:.0f} m_P={r[2]:.0f}  résidu={dd:>+6.2f}")
    resid_all += list(residB); names_all += [r[0] for r in B]

    # --- classe C : Q-indépendant (fit logC_C) ---
    ltC = np.array([np.log10(r[2]) for r in C])
    baseC = np.array([-np.log10(r[1]) for r in C])
    cC = np.mean(ltC - baseC)
    residC = ltC - (cC + baseC)
    rmsC = np.sqrt(np.mean(residC ** 2))
    print(f"\n  (C) non-leptonique noué (Q-indép.) : RMS = {rmsC:.3f} déc ({len(C)} part.)")
    for r, dd in sorted(zip(C, residC), key=lambda z: z[0][2]):
        print(f"      {r[0]:>8}  résidu={dd:>+6.2f}")
    resid_all += list(residC); names_all += [r[0] for r in C]

    glob = np.sqrt(np.mean(np.square(resid_all)))
    print(f"\n  RMS GLOBAL ({len(resid_all)} particules) = {glob:.3f} déc")
    print(f"  Paramètres : C_A, ν, C_B, C_C (4) ; θ_CKM pris des valeurs mesurées.")
    print(f"  Repères : un-seul-C 7,0 ; C/classe ν=5 2-corps plat 0,95 ; ici {glob:.2f}.")
    print("=" * 86)


if __name__ == "__main__":
    main()
