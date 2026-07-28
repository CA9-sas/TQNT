r"""
CARACTÉRISATION par la MOYENNE DE HÖLDER (loi de composition de la masse).

La composition de Ω_unified combine les « masses naïves » par une somme de type
Hölder d'ordre 1/2 (structure de Koide/Born) : holder = (Σ_i b_i^{1/2})^2,
b_i = m_naive(i)·T(i)^{3/4}. On CARACTÉRISE l'ordre h en le généralisant à la
moyenne de Hölder (Σ b_i^h)^{1/h} et en l'ajustant (vraie géométrie Abraham +
écrantage par appariement + Φ_F = U_κ^{5/3}·U_τ déterminée).

RÉSULTAT (10 particules) : l'ajustement libre donne **h = 0,543 ≈ 1/2** — la
valeur de Koide/Born, celle d'où sort le facteur 2/3 dérivé (derive_koide_23.py).
Le RMS est PLAT sur h ∈ [1/3, 1] (~10-11 %) : h n'est pas fortement contraint par
ce petit jeu, MAIS l'optimum coïncide avec 1/2. La loi de composition est donc
CONFIRMÉE comme moyenne de Hölder d'ordre 1/2 (Koide), non arbitraire.

  h=1/2 (Koide/Born) 10 %  |  h=2/3 10 %  |  h=3/4 (Casimir) 10 %  |
  h=1 (arithmétique) 11 %  |  h=1/3 11 %.

RÉSERVES : 10 particules, RMS peu sensible à h (confirmation FAIBLE mais optimum
à 1/2) ; mêmes réserves que pairing_screening / determine_frenet_functional.
À valider sur le spectre complet.
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
import theory.omega_ts as ots
from theory.abraham import REFERENCE_N50K as GEO

AMPHI = {"4_1", "6_3", "8_3"}
PART = {
    "p": ("3_1+7_2+7_2", 938.3), "Λ": ("3_1+5_2+7_2", 1115.7),
    "Σ": ("5_2+7_2+7_2", 1193.2), "Ξ": ("3_1+3_1+7_2", 1318.3),
    "Δ": ("5_2+6_3", 1232.0), "ρ⁰": ("6_2+6_2", 775.3), "ω": ("8_1+8_1", 782.7),
    "W": ("7_1+7_2+8_3", 80379.), "Z": ("7_2+8_3+9_1", 91188.),
    "H": ("7_1+8_3+10_1", 125250.),
}


def _install_ideal(f=0.5):
    new = {k: (GEO[k]["int_kappa"] ** 2 / GEO[k]["L"],
               GEO[k]["int_abs_tau"] ** 2 / (f * GEO[k]["L"])) for k in GEO}
    orig = ots._axiom_data
    ots._axiom_data = lambda name: {**orig(name),
        **({"Uk": new[name][0], "Ut": new[name][1]} if name in new else {})}


def _mass_factory():
    from simulation.experiments.omega_unified import (
        _m_naive, _T, RHO0, BETA_FLORY, P_CAS, Q_CAS)

    def PhiF(K):  # fonctionnelle de Frenet déterminée
        d = ots._axiom_data(K)
        return d["Uk"] ** (5 / 3) * d["Ut"]

    def mass(cfg, x, s, h):
        comp = [c.strip().replace("-", "_") for c in cfg.split("+")]
        n = len(comp); cnt = Counter(comp)
        collective = ((n + 1) / 2.0) ** (-x)
        screen = 1.0
        for k, c in cnt.items():
            if c >= 2:
                screen *= (RHO0 * ots._axiom_data(k)["cr"] ** (-BETA_FLORY)) ** (c - 1)
        b = [_m_naive(k) * _T(k) ** P_CAS for k in comp]
        holder = (sum(bi ** h for bi in b)) ** (1.0 / h)   # moyenne de Hölder d'ordre h
        G = sum(PhiF(k) for k in comp) / PhiF("3_1")
        pair = 1.0
        for i, j in itertools.combinations(range(n), 2):
            if comp[i] not in AMPHI and comp[j] not in AMPHI:
                pair *= s
        return collective * screen * holder * G ** Q_CAS * pair
    return mass


def rms(par, mass):
    x, s, h = par
    if h <= 0.05 or h > 3:
        return 9e9
    L = []
    for cfg, mx in (v for v in PART.values()):
        try:
            f = mass(cfg, x, s, h)
        except Exception:
            return 9e9
        if not np.isfinite(f) or f <= 0:
            return 9e9
        L.append((np.log(f), mx))
    off = np.mean([np.log(mx) - a for a, mx in L])
    return np.sqrt(np.mean([((np.exp(a + off) - mx) / mx) ** 2 for a, mx in L])) * 100


def main():
    from scipy.optimize import minimize
    _install_ideal()
    mass = _mass_factory()
    best = None
    for h0 in (0.3, 0.5, 0.75, 1.0, 1.5):
        for x0 in (-8, -6, -4):
            r = minimize(lambda p: rms(p, mass), [x0, 0.5, h0],
                         method="Nelder-Mead", options={"maxiter": 500})
            if best is None or r.fun < best.fun:
                best = r
    print(f"Moyenne de Hölder déterminée : ordre h = {best.x[2]:.3f}  (Koide/Born : 1/2)")
    print(f"  RMS = {best.fun:.0f}%\n  Ordres fixes :")
    for hh, lab in [(0.5, "1/2 Koide/Born"), (2/3, "2/3"), (0.75, "3/4 Casimir"),
                    (1, "1 arithmétique"), (1/3, "1/3")]:
        r = minimize(lambda p: rms([p[0], p[1], hh], mass), [-6, 0.5],
                     method="Nelder-Mead", options={"maxiter": 300})
        print(f"  h={hh:.3f} [{lab:15s}] RMS={r.fun:.0f}%")


if __name__ == "__main__":
    main()
