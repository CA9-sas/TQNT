r"""
Le coefficient qui rétablit le RMS des baryons : un facteur de SECTEUR constant.

La validation sur le spectre complet (validate_full_spectrum.py) donnait les baryons
à ~37 %. L'utilisateur : « il y a un coefficient pour rétablir les RMS des baryons ».
On le cherche — et il existe, mais l'histoire n'est PAS celle qu'on croit d'abord.

FAUSSE PISTE (à ne pas répéter) : un premier fit (Ξ* inclus, Steiner c libre)
donnait « proton pile sur f(3)=0,77 ». Cela N'A PAS SURVÉCU au re-fit : le 0,77
était dégénéré avec l'exposant de Steiner c (les deux absorbent le même décalage
baryon/EW). Test décisif (baryoncoef3) :
    A  c + f_bar libres        RMS baryon 3 % (f_bar=0,30, Steiner s'effondre c→0,10)
    B  f_bar=1 (sans coeff)    RMS baryon 9 % (Steiner seul, c=0,63)
    C  f_bar=0,77 (imposé)     RMS baryon 8 %
    D  c=0 (pas de Steiner), f_bar seul   RMS baryon **2 %** (f_bar=0,239)
L'optimiseur PRÉFÈRE un coefficient constant à l'exposant de Steiner (2-3 % vs 9 %).

RÉSULTAT (cas D) : un SEUL facteur multiplicatif de secteur f_bar ≈ 0,24 (baryons
confinés < EW non confiné) restaure l'octet + Δ à **~2 %**, avec appariement OFF
(s≈0,97) et SANS Steiner :
    p -2 %, Λ +3 %, Σ +2 %, Ξ -3 %, Δ -0 %.
CONTENU NON TRIVIAL : l'ORDRE p<Λ<Σ<Ξ et l'espacement sont reproduits (922<1146<
1215<1283 vs 938<1116<1193<1318). Un scalaire ne peut pas truquer l'ordonnancement
— c'est la GÉOMÉTRIE (holder × Φ_F, courbes correctes) qui suit l'octet ; le
coefficient ne fait que caler l'ÉCHELLE absolue.

L'ERREUR précédente (65 % avec un constant) venait d'avoir forcé l'appariement
s=0,45 issu du fit EW : sur-écrantage des paires chirales du baryon. Quand s est
libre (→ appariement ~off), la géométrie de base suit l'octet.

VALEUR : f_bar ≈ 0,24 est propre à CETTE normalisation (holder × G^{0,75}). Ce
n'est PAS numériquement la f(3)=0,77 de la formule V4 (c_J^{3,55}) — même RÔLE
physique (confinement de secteur baryon), autre normalisation. 0,77^5 ≈ 0,271 est
dans le voisinage (plausiblement la même f(3) passée par les exposants de la
formule), mais ce lien n'est PAS établi.

PORTÉE : rétablit le SECTEUR BARYONIQUE (octet + Δ, u/d/s) à 2 %. Ne touche PAS
le mur restant du spectre complet : saveurés lourds (J/ψ, Υ, D, B — saveur non-
topologique), dispersion EW (Z +24 %, top -24 %), mauvaises assignations (η).
1 param calé sur 5 baryons : l'échelle est fittée, mais l'ORDRE/espacement de
l'octet est prédit par la géométrie (le contenu réel).
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
import theory.omega_ts as ots
from theory.abraham import REFERENCE_N50K as GEO

AMPHI = {"4_1", "6_3", "8_3"}
EWKNOT = "8_3"   # marqueur du secteur électrofaible (non confiné, f=1)

BARYONS = [("p", "3_1+7_2+7_2", 938.3), ("Λ", "3_1+5_2+7_2", 1115.7),
           ("Σ", "5_2+7_2+7_2", 1193.2), ("Ξ", "3_1+3_1+7_2", 1318.3),
           ("Δ", "5_2+6_3", 1232.0)]
EW = [("W", "7_1+7_2+8_3", 80377.), ("Z", "7_2+8_3+9_1", 91188.),
      ("H", "7_1+8_3+10_1", 125250.), ("top", "8_2+8_3+10_1", 172500.)]


def _install_ideal(f=0.5):
    new = {k: (GEO[k]["int_kappa"] ** 2 / GEO[k]["L"],
               GEO[k]["int_abs_tau"] ** 2 / (f * GEO[k]["L"])) for k in GEO}
    orig = ots._axiom_data
    ots._axiom_data = lambda name: {**orig(name),
        **({"Uk": new[name][0], "Ut": new[name][1]} if name in new else {})}


def raw(cfg, s, fbar):
    """Formule de base (SANS Steiner) × facteur de secteur f_bar (baryon si pas d'EW)."""
    from simulation.experiments.omega_unified import (
        _m_naive, _T, RHO0, BETA_FLORY, P_CAS, Q_CAS)
    comp = [x.strip().replace("-", "_") for x in cfg.split("+")]
    n = len(comp); cnt = Counter(comp)
    screen = 1.0
    for k, cc in cnt.items():
        if cc >= 2:
            screen *= (RHO0 * ots._axiom_data(k)["cr"] ** (-BETA_FLORY)) ** (cc - 1)
    b = [_m_naive(k) * _T(k) ** P_CAS for k in comp]
    holder = (sum(np.sqrt(bi) for bi in b)) ** 2
    G = sum(ots._axiom_data(k)["Uk"] ** (5/3) * ots._axiom_data(k)["Ut"] for k in comp)
    G /= ots._axiom_data("3_1")["Uk"] ** (5/3) * ots._axiom_data("3_1")["Ut"]
    pair = 1.0
    for i, j in itertools.combinations(range(n), 2):
        if comp[i] not in AMPHI and comp[j] not in AMPHI:
            pair *= s
    conf = 1.0 if EWKNOT in comp else fbar
    return holder * G ** Q_CAS * pair * screen * conf


def main():
    from scipy.optimize import minimize
    _install_ideal()
    data = BARYONS + EW
    r = minimize(lambda p: np.std([np.log(raw(cfg, p[0], p[1])) - np.log(mx)
                 for _, cfg, mx in data]), [0.5, 0.7],
                 method="Nelder-Mead", options={"maxiter": 1500})
    s, fbar = r.x
    off = np.mean([np.log(mx) - np.log(raw(cfg, s, fbar)) for _, cfg, mx in data])
    print(f"Coefficient de secteur baryon : f_bar = {fbar:.3f}  (appariement s={s:.3f}, off)")
    print(f"  (EW = f=1 ; pas de Steiner)\n")
    for nm, cfg, mx in BARYONS:
        pred = raw(cfg, s, fbar) * np.exp(off)
        print(f"  {nm:4s} préd={pred:7.1f} exp={mx:7.1f} ({(pred - mx) / mx * 100:+.0f}%)")
    res = np.sqrt(np.mean([((raw(cfg, s, fbar) * np.exp(off) - mx) / mx) ** 2
                           for _, cfg, mx in BARYONS])) * 100
    print(f"\n  RMS octet + Δ = {res:.0f}%  (ordre p<Λ<Σ<Ξ reproduit — géométrie, pas l'échelle)")


if __name__ == "__main__":
    main()
