r"""
Test de la FORMULE UNIFIÉE proposée pour la durée de vie :

    Γ ∝ D · θ_gen · Q^p ,   p = 5 (émission d'ondes libres 3-corps / semi-lept.)
                                  p = 0 (2-corps lept. ou non-leptonique noué)

  - D = (Lk − Lk_opt)²  (boucles ; verrou de stabilité)
  - θ_gen = |V_CKM|²    (changement de génération, valeurs connues)
  - Q = énergie libérée (légers) ou masse du quark (charm/bottom)

On ajuste UN seul paramètre (C, normalisation) en log, θ pris des valeurs CKM
mesurées (pas ajusté). On compare deux variantes : avec D et sans D (D seulement
comme verrou de stabilité). On signale les exceptions D=0-mais-instable.

Usage :  python3 -m simulation.experiments.lifetime_unified_test
"""
from __future__ import annotations

import numpy as np

HBAR = 6.582e-22  # MeV·s
# CKM |V|²
Vud2, Vus2, Vcs2, Vcb2 = 0.949, 0.0497, 0.949, 0.00169

# nom, n, Lk, Q(MeV), p, θ, τ_exp(s), [single-knot ?]
W = [
    ("n",      3,  2, 0.78,   5, Vud2, 878.4,     False),
    ("mu-",    2,  0, 105.7,  5, 1.0,  2.197e-6,  False),
    ("tau-",   3,  0, 1777.,  5, 1.0,  2.903e-13, False),
    ("pi+-",   1,  0, 33.9,   0, Vud2, 2.603e-8,  True),   # 8_1 simple -> annihilation
    ("K+-",    3,  1, 388.,   0, Vus2, 1.238e-8,  False),
    ("K0_S",   3,  0, 218.5,  0, Vus2, 8.954e-11, False),
    ("Lambda", 2,  0, 37.8,   0, Vus2, 2.617e-10, False),
    ("Sigma+", 2,  1, 113.,   0, Vus2, 8.018e-11, False),
    ("Sigma-", 2, -1, 118.,   0, Vus2, 1.479e-10, False),
    ("Xi0",    3,  0, 64.2,   0, Vus2, 2.90e-10,  False),
    ("Xi-",    3, -1, 66.5,   0, Vus2, 1.639e-10, False),
    ("Omega-", 3,  0, 63.0,   0, Vus2, 8.21e-11,  False),
    ("D0",     2,  0, 1270.,  5, Vcs2, 4.10e-13,  False),
    ("D+",     2,  1, 1270.,  5, Vcs2, 1.033e-12, False),
    ("Lambdac",2,  1, 1270.,  5, Vcs2, 2.03e-13,  False),
    ("Omegac", 3,  0, 1270.,  5, Vcs2, 2.68e-13,  False),
    ("B",      2,  0, 4180.,  5, Vcb2, 1.519e-12, False),
    ("Lambdab",3,  0, 4180.,  5, Vcb2, 1.471e-12, False),
]


def lk_opt(n):
    return n * (n - 1) // 2


def _base(rows, use_D):
    lt = np.array([np.log10(r[6]) for r in rows])
    base = []
    for r in rows:
        n, lk, Q, p, th = r[1], r[2], r[3], r[4], r[5]
        D = (lk - lk_opt(n)) ** 2
        Dp = max(D, 1) if use_D else 1.0
        base.append(-np.log10(Dp) - np.log10(th) - p * np.log10(Q))
    return lt, np.array(base)


def fit_C(rows, use_D):
    """UN seul C : const = log10(ħ/C)."""
    lt, base = _base(rows, use_D)
    const = np.mean(lt - base)
    resid = lt - (const + base)
    return float(np.sqrt(np.mean(resid ** 2))), resid


def fit_C_per_class(rows, use_D):
    """UN C par classe d'espace des phases p (préfacteur de couplage distinct)."""
    lt, base = _base(rows, use_D)
    ps = np.array([r[4] for r in rows])
    resid = np.zeros_like(lt)
    for p in set(ps):
        m = ps == p
        const = np.mean(lt[m] - base[m])
        resid[m] = lt[m] - (const + base[m])
    return float(np.sqrt(np.mean(resid ** 2))), resid


def main():
    print("=" * 92)
    print("  FORMULE UNIFIÉE  Γ ∝ D·θ_CKM·Q^p  —  test sur 18 particules faibles")
    print("=" * 92)

    # verrou de stabilité D : exceptions D=0 mais instables
    print("\n  Verrou de stabilité D = (Lk − Lk_opt)² :")
    exc = []
    for r in W:
        n, lk = r[1], r[2]
        D = (lk - lk_opt(n)) ** 2
        if D == 0:
            exc.append(r[0])
    print(f"    Particules INSTABLES avec D=0 (le verrou échoue) : "
          f"{', '.join(exc) if exc else 'aucune'}")
    print("    -> le pion (8_1 simple, annihilation), Σ⁺/D⁺/Λc (Lk à l'optimum")
    print("       mais changent de SAVEUR vers plus léger) : D n'est PAS un verrou universel.")

    # fits
    print("\n  --- Ajustement à UN SEUL C (formule vraiment unique) ---")
    for use_D, lab in [(False, "Γ ∝ θ_CKM·Q^p"),
                       (True,  "Γ ∝ max(D,1)·θ_CKM·Q^p")]:
        rms, _ = fit_C(W, use_D)
        print(f"    {lab:>28} : RMS = {rms:.3f} déc  (1 param)")
    print("    -> ÉCHEC : p=5 (Sargent) et p=0 (noué) ont des PRÉFACTEURS de")
    print("       couplage différents ; un seul C ne peut pas relier les classes.")

    print("\n  --- Ajustement à UN C PAR CLASSE d'espace des phases (p) ---")
    for use_D, lab in [(False, "Γ ∝ C_p·θ_CKM·Q^p"),
                       (True,  "Γ ∝ C_p·max(D,1)·θ_CKM·Q^p")]:
        rms, resid = fit_C_per_class(W, use_D)
        print(f"    {lab:>30} : RMS = {rms:.3f} déc  (1 C par classe)")
        if use_D:
            for r, dl in sorted(zip(W, resid), key=lambda z: z[0][6]):
                print(f"        {r[0]:>8} p={r[4]} Q={r[3]:>7.1f} θ={r[5]:.3f}"
                      f"  résidu={dl:>+6.2f}")

    print("\n  CONCLUSION : pas de formule UNIQUE à une constante. La forme")
    print("  défendable est FACTORISÉE par classe : Γ = C_classe · θ_CKM · Q^p,")
    print("  la classe (ondes libres 3-corps / 2-corps / noué) fixant p ET C.")
    print("=" * 92)


if __name__ == "__main__":
    main()
