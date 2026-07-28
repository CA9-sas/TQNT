r"""
Le plancher prédictif : exposants ajustés sur des assignations PHYSIQUES FIXES.

Suite à combi_formula_search.py (best-match adaptatif → 0,3 %, mais surajustement :
null 4,1 %, CV test 13 %) et à la demande utilisateur « contraindre le fit aux
assignations physiques et voir le plancher ». On FIGE l'assignation de chaque
particule à sa config physique (dérivée de charge/spin/Koide, validate_full_spectrum)
et on n'ajuste QUE les 11 exposants — AUCUN best-match. C'est le vrai plancher
PRÉDICTIF de la formule.

NB leptons : e = 3_1. μ, τ sont AUSSI 3_1 (modes Z₃/génération, [[decay-propagation-and-knot-state-index]])
→ la formule de masse topologique leur donnerait la MÊME masse que e. Ils sont donc
EXCLUS du fit de masse (la génération n'est pas topologique). À elle seule, cette
exclusion dit que 2/3 des leptons échappent à la formule.

RÉSULTAT — PLANCHER ≈ 34 % (23 particules, 11 exposants libres) :
    par secteur : baryons u/d/s 21 % (le plus géométrique), flav 34 %, mésons 37 %, EW 43 %.
    CV : train 24 % → test 60 % (le 34 % est même optimiste).
    pires résidus : top -52 %, D +74 %, φ +70 %, K* +69 %, Σ* +46 %.

COMPARAISON DÉCISIVE :
    best-match (assignation ADAPTATIVE)   : 0,3 %
    assignations PHYSIQUES fixes          : 34 %
Le gap 0,3 % → 34 % EST exactement la LIBERTÉ D'ASSIGNATION. Avec des assignations
honnêtes, la formule de masse topologique plafonne à ~34 % : la géométrie des nœuds
n'encode PAS les masses précises. Confirme définitivement [[epistemic-assessment-V6]]
et le §Validation spectre complet. Ce qui reste géométrique : le SECTEUR baryonique
u/d/s (21 %, cf. §Le secteur baryonique se rétablit à 2 % avec un coefficient de
secteur) ; le reste (mésons, EW, saveurés) est dominé par la saveur non-topologique.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from simulation.experiments.combi_formula_search import D, CANON

# assignations PHYSIQUES fixes (validate_full_spectrum + e ; μ,τ exclus = modes Z₃)
PHYS = [
    ("e", "3_1", 0.511, "lep"),
    ("rho", "6_2+6_2", 775.3, "mes"), ("omega", "8_1+8_1", 782.7, "mes"),
    ("phi", "7_5+7_5", 1019.5, "mes"), ("Kstar", "5_2+7_5", 891.7, "mes"),
    ("eta", "3_1+4_1+5_1", 547.9, "mes"), ("etap", "5_2+5_2+6_1", 957.8, "mes"),
    ("K", "4_1+4_1+5_2", 493.7, "mes"),
    ("p", "3_1+7_2+7_2", 938.3, "bar"), ("Lam", "3_1+5_2+7_2", 1115.7, "bar"),
    ("Sig", "5_2+7_2+7_2", 1193.2, "bar"), ("Xi", "3_1+3_1+7_2", 1318.3, "bar"),
    ("Del", "5_2+6_3", 1232.0, "bar"), ("Sigst", "5_1+5_1+5_2", 1384.6, "bar"),
    ("Xist", "4_1+5_2+7_2", 1533.4, "bar"),
    ("W", "7_1+7_2+8_3", 80377., "EW"), ("Z", "7_2+8_3+9_1", 91188., "EW"),
    ("H", "7_1+8_3+10_1", 125250., "EW"), ("top", "8_2+8_3+10_1", 172570., "EW"),
    ("Jpsi", "7_4+7_4", 3096.9, "flav"), ("Ups", "7_6+7_6+4_1", 9460.3, "flav"),
    ("D", "3_1+5_1+7_2", 1864.8, "flav"), ("B", "8_1+8_2", 5279.3, "flav"),
]


def mass_of(cfg, p):
    MC, ML, beta, A, u, v, Q, X, B, rho0, scale = p
    comp = cfg.split("+"); n = len(comp); cnt = Counter(comp)
    ws = [(D[k]["cr"] * MC + D[k]["nl"] * ML) * ((D[k]["b"] * D[k]["det"]) ** beta) for k in comp]
    if any(w <= 0 for w in ws):
        return np.nan
    sk = sum(w ** A for w in ws) ** (1 / A)
    G = sum(D[k]["Uk"] ** u * D[k]["Ut"] ** v for k in comp)
    pair = 1.0
    for k, c in cnt.items():
        if c >= 2:
            pair *= (rho0 * D[k]["cr"] ** (-B)) ** (c - 1)
    return scale * ((n + 1) / 2) ** (-X) * pair * sk * G ** Q


def rms(p, data):
    e = []
    for _, cfg, me, _ in data:
        mp = mass_of(cfg, p)
        if not np.isfinite(mp) or mp <= 0:
            return 9.9
        e.append(np.log(mp) - np.log(me))
    return float(np.sqrt(np.mean(np.array(e) ** 2)))


def fit(data, restarts=30, seed=3):
    x0 = np.array(CANON); rng = np.random.default_rng(seed); best = None
    for r in range(restarts):
        xx = x0 * (1 + 0.25 * rng.standard_normal(len(x0))) if r else x0
        res = minimize(lambda p: rms(p, data), xx, method="Nelder-Mead",
                       options={"maxiter": 3000, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    return best


def main():
    b = fit(PHYS)
    print(f"PLANCHER (assignations physiques FIXES, {len(PHYS)} particules) : "
          f"RMS = {b.fun * 100:.0f}%")
    for sec in ["lep", "mes", "bar", "EW", "flav"]:
        d = [x for x in PHYS if x[3] == sec]
        if d:
            print(f"  {sec:5s} ({len(d)}) RMS={rms(b.x, d) * 100:.0f}%")
    print("  → best-match adaptatif donnait 0,3 % ; le gap = liberté d'assignation.")


if __name__ == "__main__":
    main()
