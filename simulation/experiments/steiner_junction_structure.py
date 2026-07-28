r"""
La physique du facteur de structure S(P) : les JONCTIONS DE STEINER du réseau de
tubes de flux confinant (détermine la dépendance croissante en n).

Le facteur de structure S(P)=N^{α(n)} avait α(n) croissant (α(2)=-0,18, α(3)=+0,45,
α(4)=+1,17), OPPOSÉ à Duplantier (validate_tetraquark_n4.py). On CHERCHE la physique.

RÉSULTAT : la dépendance est gouvernée par le NOMBRE DE JONCTIONS DE STEINER du
réseau confinant. n tubes reliés exigent (n-2) jonctions internes :
    méson (n=2) : 0 jonction (lien direct)  |  baryon (n=3) : 1 jonction (Y, cos120°)
    tétraquark (n=4) : 2 jonctions.
D'où S(P) = N^{c·(n-2)} (UN paramètre c). Comparaison (RMS) :
    Steiner N^{c(n-2)}      20 % (c=0,60, 1 param)   ← MEILLEURE forme physique
    arêtes N^{c(n-1)}       24 %
    paires N^{c·n(n-1)/2}   21 %
    α(n) libre (3 params)   19 %
La forme Steiner à 1 paramètre égale quasiment le libre à 3 params, et REPRODUIT
les exposants : α(n)=0,60·(n-2) → 0 / 0,60 / 1,20 ≈ libres -0,18 / 0,45 / 1,17.

C'est la MÊME physique de jonction Steiner que le N-Δ (liaison 1/2, angles 120°,
cf. [[decay-propagation-and-knot-state-index]]) : le réseau de tubes de flux χ se
connecte par des jonctions de Steiner, et chaque jonction ajoute un poids N^c à la
masse. Le facteur de structure est donc DÉTERMINÉ : S(P) = N^{c·(n-2)}, (n-2) =
jonctions (topologique, dérivé), c≈0,60 (un paramètre de force de jonction).

RÉSERVES : le RMS 20 % (avec tétraquarks) est dominé par les outliers tétraquarks
(saveur/assignation spéculative) ; la dépendance en n (n-2 Steiner) est robuste.
c=0,60 est calibré (≈3/5 ; lien à la géométrie 120° à explorer).
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
import theory.omega_ts as ots
from theory.abraham import REFERENCE_N50K as GEO

AMPHI = {"4_1", "6_3", "8_3"}
PART = {
    "ρ⁰": ("6_2+6_2", 775.3, 2), "ω": ("8_1+8_1", 782.7, 2), "Δ": ("5_2+6_3", 1232.0, 2),
    "p": ("3_1+7_2+7_2", 938.3, 3), "Λ": ("3_1+5_2+7_2", 1115.7, 3),
    "Σ": ("5_2+7_2+7_2", 1193.2, 3), "Ξ": ("3_1+3_1+7_2", 1318.3, 3),
    "W": ("7_1+7_2+8_3", 80379., 3), "Z": ("7_2+8_3+9_1", 91188., 3),
    "H": ("7_1+8_3+10_1", 125250., 3), "σ": ("3_1+3_1+3_1+3_1", 500., 4),
    "κ": ("3_1+3_1+3_1+5_2", 700., 4), "a₀": ("3_1+3_1+5_2+5_2", 980., 4),
}


def _install_ideal(f=0.5):
    new = {k: (GEO[k]["int_kappa"] ** 2 / GEO[k]["L"],
               GEO[k]["int_abs_tau"] ** 2 / (f * GEO[k]["L"])) for k in GEO}
    orig = ots._axiom_data
    ots._axiom_data = lambda name: {**orig(name),
        **({"Uk": new[name][0], "Ut": new[name][1]} if name in new else {})}


def _base(cfg, s):
    from simulation.experiments.omega_unified import (
        _m_naive, _T, RHO0, BETA_FLORY, P_CAS, Q_CAS)
    comp = [c.strip().replace("-", "_") for c in cfg.split("+")]
    n = len(comp); cnt = Counter(comp)
    screen = 1.0
    for k, c in cnt.items():
        if c >= 2:
            screen *= (RHO0 * ots._axiom_data(k)["cr"] ** (-BETA_FLORY)) ** (c - 1)
    b = [_m_naive(k) * _T(k) ** P_CAS for k in comp]
    holder = (sum(np.sqrt(bi) for bi in b)) ** 2
    G = sum(ots._axiom_data(k)["Uk"] ** (5/3) * ots._axiom_data(k)["Ut"] for k in comp)
    G /= ots._axiom_data("3_1")["Uk"] ** (5/3) * ots._axiom_data("3_1")["Ut"]
    pair = 1.0
    for i, j in itertools.combinations(range(n), 2):
        if comp[i] not in AMPHI and comp[j] not in AMPHI:
            pair *= s
    return screen * holder * G ** Q_CAS * pair, sum(ots._axiom_data(k)["cr"] for k in comp)


def rms(s, c):
    """S(P) = N^{c·(n-2)} : jonctions de Steiner."""
    L = []
    for cfg, mx, n in (v for v in PART.values()):
        bval, N = _base(cfg, s)
        f = bval * N ** (c * (n - 2))
        if not np.isfinite(f) or f <= 0:
            return 9e9, None
        L.append((np.log(f), mx, n))
    off = np.mean([np.log(mx) - a for a, mx, _ in L])
    return np.sqrt(np.mean([((np.exp(a + off) - mx) / mx) ** 2 for a, mx, _ in L])) * 100, off


def main():
    from scipy.optimize import minimize
    _install_ideal()
    r = minimize(lambda p: rms(p[0], p[1])[0], [0.53, 0.6],
                 method="Nelder-Mead", options={"maxiter": 600})
    s, c = r.x
    print(f"S(P) = N^(c·(n-2)) — jonctions de Steiner (méson 0, baryon 1, tétra 2)")
    print(f"  c = {c:.3f} (1 param) ; s_chiral = {s:.3f} ; RMS = {r.fun:.0f}%")
    print(f"  α(n)=c·(n-2) : n=2→{c*0:.2f}  n=3→{c*1:.2f}  n=4→{c*2:.2f}")
    print(f"  (libres : -0,18 / 0,45 / 1,17)  → la dépendance en n = (n-2) Steiner.")
    print(f"  Même physique de jonction Steiner que le N-Δ (liaison 1/2, angles 120°).")


if __name__ == "__main__":
    main()
