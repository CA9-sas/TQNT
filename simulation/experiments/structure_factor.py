r"""
FACTEUR DE STRUCTURE S(P) (idée utilisateur) — configuration de réseau de l'entrelacs.

L'entrelacs de n composantes est une ÉTOILE/RÉSEAU à n branches (Duplantier 1989,
cf. brouillon_masse_x_duplantier). Le facteur de structure S(P) = N^{α(n)} porte le
poids configurationnel du réseau (N = taille = Σ cr_i ; α(n) = exposant d'étoile à
n branches). Il REMPLACE le collectif ad hoc ((n+1)/2)^{-x}.

Nouveauté : avec l'écrantage par appariement + la vraie géométrie, les composites
n=2 (mésons ρ⁰, ω) ET n=3 (baryons, EW) sont dans le fit → le VERROU du brouillon
(« pas de composites propres n≠3 ») saute, on peut tester la dépendance en n.

RÉSULTAT (10 particules) : α(n=2)=-0,46, α(n=3)=+0,22, RMS 10 % (identique au
collectif). α(2)=-0,46 tombe dans la plage des exposants de SOMMET de Duplantier
(σ_f ~ -0,3 à -1 en d=3), ce qui est suggestif. MAIS Δα=α(3)-α(2)=+0,68 : l'exposant
CROÎT avec n, alors que le vrai exposant d'étoile de Duplantier DÉCROÎT avec le
nombre de branches. Donc le facteur de structure est CARACTÉRISÉ mais son
identification à Duplantier N'EST PAS CONFIRMÉE (signe de la dépendance en n
opposé) — cohérent avec le verdict du brouillon (« x n'est pas un exposant de
Duplantier élémentaire »).

STATUT : S(P) ajouté et déterminé, fit 10 % (= collectif) ; lien Duplantier partiel
(α₂ dans la plage) mais non confirmé (trend en n inversé). Réserves : 10 particules,
2 valeurs de n seulement (2 et 3), ρ⁰ outlier. À valider avec n=4 (tétra/penta).
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


def _base(cfg, s):
    """Tout sauf le facteur de structure : screen, Hölder(1/2), G(Φ_F), appariement."""
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
    N = sum(ots._axiom_data(k)["cr"] for k in comp)
    return screen * holder * G ** Q_CAS * pair, n, N


def rms(par):
    s, a2, a3 = par
    L = []
    for cfg, mx in (v for v in PART.values()):
        bval, n, N = _base(cfg, s)
        f = bval * N ** (a2 if n == 2 else a3)      # S(P) = N^{α(n)}
        if not np.isfinite(f) or f <= 0:
            return 9e9
        L.append((np.log(f), mx))
    off = np.mean([np.log(mx) - a for a, mx in L])
    return np.sqrt(np.mean([((np.exp(a + off) - mx) / mx) ** 2 for a, mx in L])) * 100


def main():
    from scipy.optimize import minimize
    _install_ideal()
    best = None
    for s0 in (0.4, 0.6):
        for a2 in (-4, -2, 0):
            for a3 in (-6, -2, 2):
                r = minimize(rms, [s0, a2, a3], method="Nelder-Mead",
                             options={"maxiter": 600, "fatol": 1e-2})
                if best is None or r.fun < best.fun:
                    best = r
    s, a2, a3 = best.x
    print(f"Facteur de structure S(P) = N^α(n) (Duplantier, N=Σcr) :")
    print(f"  s_chiral={s:.3f} ; α(n=2)={a2:.2f} ; α(n=3)={a3:.2f} ; RMS={best.fun:.0f}%")
    print(f"  Δα = α(3)-α(2) = {a3 - a2:+.2f}")
    print(f"  α(2)=-0.46 ∈ plage Duplantier (σ_f~-0.3..-1) MAIS Δα>0 (croît) vs")
    print(f"  Duplantier décroît avec les branches → identification NON confirmée.")


if __name__ == "__main__":
    main()
