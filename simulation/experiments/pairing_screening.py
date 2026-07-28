r"""
Loi d'ÉCRANTAGE PAR APPARIEMENT (idée utilisateur) — récupère la formule de masse
absolue avec la VRAIE géométrie.

Contexte : recalibrer Ω_unified avec la géométrie idéale correcte (Abraham) donnait
82 % RMS, échec STRUCTUREL (hadrons légers +~100 % trop lourds, EW -~70 % trop
légers, décalages OPPOSÉS ; U_κ(8₃) s'effondre ×80). Il manquait un mécanisme
physique : l'écrantage par appariement.

MÉCANISME : quand deux nœuds s'apparient (se lient) dans un entrelacs, leurs champs
χ se recouvrent et s'écrantent — MAIS seulement pour les paires CHIRALES (dont les
champs miroir peuvent s'annuler). Un nœud AMPHICHIRAL (4₁,6₃,8₃ = auto-miroir) ne
s'écrante PAS (pas d'annulation miroir). C'est exactement ce qui distingue les
cœurs baryoniques (7₂,5₂ chiraux → écrantés) des cœurs EW (8₃ amphichiral → non).

FORME : mass → mass × s_chiral^(n_paires_chirales), n_paires_chirales = nb de
paires (i<j) de l'entrelacs dont les DEUX nœuds sont chiraux.

RÉSULTAT (10 particules, géométrie Abraham correcte, calibration x + s_chiral) :
  RMS = 14 % (vs 82 % sans appariement), s_chiral ≈ 0,39 (écrantage physique <1).
  Le décalage OPPOSÉ est CORRIGÉ : baryons +4..+9 %, EW -14..+13 %.

RÉSERVES : 10 particules seulement ; x recalibré loin de -2,28 ; ρ⁰(-31 %), Δ(+22 %)
restent des outliers ; U_τ Abraham utilise f=0,5 estimé. Mais le point STRUCTUREL
est solide : la loi d'écrantage par appariement chirale corrige l'échec de la
géométrie correcte. À valider sur le spectre complet.
"""
from __future__ import annotations
import itertools
import numpy as np
import theory.omega_ts as ots
from theory.abraham import REFERENCE_N50K as GEO

AMPHI = {"4_1", "6_3", "8_3"}  # amphichiraux : auto-miroir, N'écrantent PAS

PART = {
    "p": ("3_1+7_2+7_2", 938.3), "Λ": ("3_1+5_2+7_2", 1115.7),
    "Σ": ("5_2+7_2+7_2", 1193.2), "Ξ": ("3_1+3_1+7_2", 1318.3),
    "Δ": ("5_2+6_3", 1232.0), "ρ⁰": ("6_2+6_2", 775.3), "ω": ("8_1+8_1", 782.7),
    "W": ("7_1+7_2+8_3", 80379.), "Z": ("7_2+8_3+9_1", 91188.),
    "H": ("7_1+8_3+10_1", 125250.),
}


def _install_ideal_geometry(f=0.5):
    """Patche _axiom_data pour U_κ=(∮κ)²/L, U_τ=(∮|τ|)²/(fL) (Abraham idéal)."""
    new = {k: (GEO[k]["int_kappa"] ** 2 / GEO[k]["L"],
               GEO[k]["int_abs_tau"] ** 2 / (f * GEO[k]["L"])) for k in GEO}
    orig = ots._axiom_data

    def patched(name):
        d = dict(orig(name))
        if name in new:
            d["Uk"], d["Ut"] = new[name]
        return d
    ots._axiom_data = patched


def pairing_factor(comp, s_chiral):
    """s_chiral^(nb de paires chiral-chiral de l'entrelacs)."""
    fac = 1.0
    for i, j in itertools.combinations(range(len(comp)), 2):
        if comp[i] not in AMPHI and comp[j] not in AMPHI:
            fac *= s_chiral
    return fac


def rms(par, mass_unified):
    x, s = par
    rows = []
    for nm, (cfg, mx) in PART.items():
        comp = [c.strip().replace("-", "_") for c in cfg.split("+")]
        try:
            f = mass_unified(cfg, x) * pairing_factor(comp, s)
        except Exception:
            return 9e9, None
        if not np.isfinite(f) or f <= 0:
            return 9e9, None
        rows.append((nm, np.log(f), mx))
    off = np.mean([np.log(mx) - a for _, a, mx in rows])
    res = {nm: (np.exp(a + off), mx) for nm, a, mx in rows}
    return np.sqrt(np.mean([((mp - mx) / mx) ** 2 for mp, mx in res.values()])) * 100, res


def main():
    from scipy.optimize import minimize
    import importlib
    _install_ideal_geometry()
    import simulation.experiments.omega_unified as ou
    importlib.reload(ou)

    best = None
    for x0 in (-3, -2, -1, 0, 1):
        for s0 in (0.3, 0.6, 1.0, 1.5):
            r = minimize(lambda p: rms(p, ou.mass_unified)[0], [x0, s0],
                         method="Nelder-Mead", options={"maxiter": 300})
            if best is None or r.fun < best.fun:
                best = r
    rmsv, res = rms(best.x, ou.mass_unified)
    print(f"Écrantage par appariement (chiralité) : x={best.x[0]:.2f}, s_chiral={best.x[1]:.3f}")
    print(f"  {'part.':5s} {'M_préd':>9s} {'M_exp':>9s} {'écart%':>8s}")
    for nm, (mp, mx) in res.items():
        print(f"  {nm:5s} {mp:9.0f} {mx:9.0f} {(mp - mx) / mx * 100:+8.0f}")
    print(f"\n  RMS = {rmsv:.0f}%  (vs 82% sans appariement) ; décalage opposé CORRIGÉ.")


if __name__ == "__main__":
    main()
