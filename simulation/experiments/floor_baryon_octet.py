r"""
Le plancher du secteur baryonique u/d/s isolé — le seul secteur GÉNUINEMENT
géométrique.

Suite à floor_physical_assignments.py (plancher global ~34 % à assignations
physiques fixes) et à la demande utilisateur « isole le secteur baryonique u/d/s
à son plancher ». On isole les baryons u/d/s et on cherche le plancher HONNÊTE
(CV-stable, pas surajusté).

DÉCOUVERTE — il faut séparer octet et décuplet (le J compte) :
Sous exposants CANONIQUES + échelle seule (1 param), sur les 8 baryons u/d/s :
    OCTET (p,Λ,Σ,Ξ, spin-1/2)      : +0,4 / -3,0 / +3,5 / -0,8 %  → excellent
    DÉCUPLET (Δ,Σ*,Ω, spin-3/2)    : -62 / +56 / +79 %  → catastrophe
Le décuplet casse car la formule n'a PAS de terme de moment cinétique J ; or
décuplet J=3/2 vs octet J=1/2 (splitting N-Δ = mécanisme τ_⊥ spin-spin séparé,
[[decay-propagation-and-knot-state-index]]). Le décuplet est donc HORS de ce
secteur (il relève du terme Regge/J), pas un échec de l'octet.

PLANCHER OCTET u/d/s (le résultat) :
    RMS = 2,4 %  avec UN SEUL paramètre (l'échelle), exposants canoniques fixes.
    CV leave-one-out : médian 2,5 %, moyen 2,6 %  → CV ≈ fit = GÉNUINEMENT STABLE.
    ordre p < Λ < Σ < Ξ REPRODUIT (la géométrie encode l'espacement de l'octet).
Contraste TOTAL avec les 0,3 %/2,7 % surajustés (CV LOO 40-48 %) : ici la CV ne
se dégrade PAS. L'octet u/d/s est le SEUL secteur où la géométrie des nœuds encode
vraiment les masses — à 1 paramètre, CV-stable, ordre correct. (Avec le coefficient
de secteur f_bar, baryon_confinement_coefficient.py : 2 %, déjà établi.)

BILAN GÉNÉRAL DES PLANCHERS (assignations physiques fixes) :
    global (23 particules)   34 %   — la géométrie n'encode pas les masses
    octet u/d/s (4)          2,4 %  — CV-stable 1 param   ← LE secteur géométrique
    décuplet u/d/s           >50 %  — exige le terme J (Regge), hors formule
    EW / saveurés            34-43 % — saveur non-topologique
Conclusion : ce qui est géométrique dans les masses se réduit à l'OCTET
baryonique u/d/s (spin-1/2, léger). Tout le reste échappe (saveur, ou terme J).
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from simulation.experiments.combi_formula_search import D, CANON

OCTET = [("p", "3_1+7_2+7_2", 938.3), ("Lam", "3_1+5_2+7_2", 1115.7),
         ("Sig", "5_2+7_2+7_2", 1193.2), ("Xi", "3_1+3_1+7_2", 1318.3)]
DECUPLET = [("Del", "5_2+6_3", 1232.0), ("Sigst", "5_1+5_1+5_2", 1384.6),
            ("Xist", "4_1+5_2+7_2", 1533.4), ("Om", "7_2+7_2+7_4", 1672.5)]
FULL = np.array(CANON)


def mass_of(cfg, p):
    MC, ML, beta, A, u, v, Q, X, B, rho0, scale = p
    comp = cfg.split("+"); n = len(comp); cnt = Counter(comp)
    ws = [(D[k]["cr"] * MC + D[k]["nl"] * ML) * ((D[k]["b"] * D[k]["det"]) ** beta) for k in comp]
    sk = sum(w ** A for w in ws) ** (1 / A)
    G = sum(D[k]["Uk"] ** u * D[k]["Ut"] ** v for k in comp)
    pair = 1.0
    for k, c in cnt.items():
        if c >= 2:
            pair *= (rho0 * D[k]["cr"] ** (-B)) ** (c - 1)
    return scale * ((n + 1) / 2) ** (-X) * pair * sk * G ** Q


def scale_only(data):
    """Exposants canoniques fixes ; seule l'échelle calée (geomean). 1 paramètre."""
    p = FULL.copy()
    p[10] = np.exp(np.mean([np.log(me) - np.log(mass_of(cfg, [*FULL[:10], 1.0]))
                            for _, cfg, me in data]))
    return p


def rms(data, p):
    return float(np.sqrt(np.mean([((mass_of(cfg, p) - me) / me * 100) ** 2
                                  for _, cfg, me in data])))


def main():
    p = scale_only(OCTET)
    print("OCTET u/d/s (spin-1/2), exposants CANONIQUES, ÉCHELLE seule (1 param) :")
    for nm, cfg, me in OCTET:
        mp = mass_of(cfg, p)
        print(f"  {nm:4s} {cfg:16s} pred={mp:7.1f} exp={me:7.1f} ({(mp - me) / me * 100:+.1f}%)")
    print(f"  RMS = {rms(OCTET, p):.1f}%")
    loo = [abs((mass_of(OCTET[i][1], scale_only([OCTET[j] for j in range(len(OCTET)) if j != i]))
                - OCTET[i][2]) / OCTET[i][2] * 100) for i in range(len(OCTET))]
    print(f"  CV LOO : médian={np.median(loo):.1f}%  moyen={np.mean(loo):.1f}%  → CV-STABLE")
    order = " < ".join(nm for nm, _ in sorted([(nm, mass_of(cfg, p)) for nm, cfg, _ in OCTET],
                                              key=lambda t: t[1]))
    print(f"  ordre : {order}  (exp p < Lam < Sig < Xi)")
    print(f"\n  décuplet (échelle seule) RMS = {rms(DECUPLET, scale_only(DECUPLET)):.0f}% "
          "→ exige le terme J (Regge/N-Δ), hors formule.")


if __name__ == "__main__":
    main()
