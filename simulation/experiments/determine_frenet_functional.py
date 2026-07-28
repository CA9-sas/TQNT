r"""
DÉTERMINATION de la fonctionnelle de Frenet Φ_F (couplage géométrique de la masse).

Le couplage géométrique de Ω_unified était G = Σ U_κ·U_τ (produit AD HOC). On
DÉTERMINE la fonctionnelle Φ_F = U_κ^a · U_τ^b par ajustement, sur la VRAIE
géométrie (Abraham idéal) + écrantage par appariement chiral (pairing_screening.py).

RÉSULTAT (10 particules) : l'ajustement libre donne (a,b)=(1,70 ; 0,95), et la
forme PROPRE
        Φ_F = U_κ^{5/3} · U_τ        (RMS 10 %, vs 14 % pour U_κ·U_τ)
donne le MÊME optimum. Le **5/3 est l'exposant de FLORY** déjà dérivé et présent
dans le cadre (écrantage de volume exclu, β_SAW=5/3) : la même physique 3D
d'auto-évitement gouverne la courbure. La torsion entre linéairement (b=1).

Comparaison des formes (x, s ré-optimisés) :
  U_κ^{5/3}·U_τ  10 %   |  U_κ·U_τ (ancien)  14 %  |  √U_κ·U_τ  19 %
  U_κ·√U_τ      21 %   |  U_τ seul          24 %  |  √(U_κU_τ) 26 %  |  U_κ seul 42 %

PROGRESSION (géométrie correcte) : 82 % (nu) → 14 % (+ appariement) → 10 %
(+ Φ_F déterminée). RÉSERVES : 10 particules ; ρ⁰(-22 %) outlier ; x=-6,6 ;
U_τ Abraham f=0,5. À valider sur le spectre complet.
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

    def patched(name):
        d = dict(orig(name))
        if name in new:
            d["Uk"], d["Ut"] = new[name]
        return d
    ots._axiom_data = patched


def make_mass(a, b):
    from simulation.experiments.omega_unified import (
        _m_naive, _T, RHO0, BETA_FLORY, P_CAS, Q_CAS)

    def PhiF(K):
        d = ots._axiom_data(K)
        return d["Uk"] ** a * d["Ut"] ** b

    def mass(cfg, x, s):
        comp = [c.strip().replace("-", "_") for c in cfg.split("+")]
        n = len(comp); cnt = Counter(comp)
        collective = ((n + 1) / 2.0) ** (-x)
        screen = 1.0
        for k, c in cnt.items():
            if c >= 2:
                screen *= (RHO0 * ots._axiom_data(k)["cr"] ** (-BETA_FLORY)) ** (c - 1)
        holder = sum(np.sqrt(_m_naive(k) * _T(k) ** P_CAS) for k in comp) ** 2
        G = sum(PhiF(k) for k in comp) / PhiF("3_1")
        pair = 1.0
        for i, j in itertools.combinations(range(n), 2):
            if comp[i] not in AMPHI and comp[j] not in AMPHI:
                pair *= s
        return collective * screen * holder * G ** Q_CAS * pair
    return mass


def rms(a, b, mass):
    from scipy.optimize import minimize

    def f(p):
        L = []
        for cfg, mx in ((c, m) for c, m in PART.values()):
            try:
                v = mass(cfg, p[0], p[1])
            except Exception:
                return 9e9
            if not np.isfinite(v) or v <= 0:
                return 9e9
            L.append((np.log(v), mx))
        off = np.mean([np.log(mx) - a2 for a2, mx in L])
        return np.sqrt(np.mean([((np.exp(a2 + off) - mx) / mx) ** 2 for a2, mx in L])) * 100
    r = minimize(f, [-6, 0.5], method="Nelder-Mead", options={"maxiter": 400})
    return r.fun, r.x


def main():
    _install_ideal()
    print("DÉTERMINATION de Φ_F = U_κ^a · U_τ^b (géométrie correcte + appariement) :\n")
    forms = [(5/3, 1, "U_κ^(5/3)·U_τ  ← DÉTERMINÉE (5/3=Flory)"),
             (1, 1, "U_κ·U_τ (ancien, ad hoc)"), (0.5, 1, "√U_κ·U_τ"),
             (1, 0.5, "U_κ·√U_τ"), (0, 1, "U_τ seul"), (0.5, 0.5, "√(U_κ·U_τ)"),
             (1, 0, "U_κ seul")]
    for a, b, lab in forms:
        r, x = rms(a, b, make_mass(a, b))
        print(f"  Φ_F=U_κ^{a:.3f}·U_τ^{b:.2f}  RMS={r:2.0f}%   [{lab}]")


if __name__ == "__main__":
    main()
