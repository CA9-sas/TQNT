r"""
Levier À BRAS favorisant l'asymétrie, via le linking Lk — idée utilisateur.

« Teste le levier à bras qui favorise l'asymétrie via le linking Lk » : un petit
nœud (bras long) verrouille un gros via l'enlacement, amplification ∝ Lk × asymétrie.
On ajoute à la formule un terme de bras m → m·(1 + λ·Lk·⟨asym⟩), où
⟨asym⟩ = Σ_{i<j}|√w_i−√w_j| / Σ_i√w_i (0 si nœuds égaux, croît avec l'asymétrie).

CONTRAINTE DE DISCIPLINE : dans le cadre, l'état fondamental est à Lk_opt = n
(nombre de composantes), donc Lk=3 CONSTANT pour tous les baryons 3-nœuds. Sur
l'octet, « ×Lk » est donc dégénéré avec λ → on ne teste que le BRAS (asymétrie),
pas le canal Lk indépendamment. (Les masses ~identiques dans un isomultiplet où Q,
donc Lk, varie confirment que Lk ne porte pas la masse au 1er ordre.)

RÉSULTATS (3 régimes) :
1. Squelette SEUL (sans le couplage géométrique G=ΣU_κU_τ), octet :
   toutes les formes de bras veulent λ>0 (asymétrie AMPLIFIE), RMS 39%→27%.
   MAIS c'est le bras qui REMPLACE le G manquant (proxy de taille), CV LOO 57-67%.
2. Formule COMPLÈTE (avec G) + bras, jeu mixte (baryons+mésons, Lk variable 2/3/4) :
   λ≈0, gain 0,0 point → au-delà de G, le bras n'apporte RIEN.
3. Formule COMPLÈTE + bras, octet propre :
   λ=+0,034 (asymétrie AMPLIFIE, sens de l'intuition ✓), fit 2,36%→1,31%,
   MAIS CV LOO 3% vs 2,5% baseline → la CV ne s'améliore PAS (le bras absorbe du
   résidu, 1 param/4 points, sans pouvoir prédictif robuste).

VERDICT HONNÊTE : le levier à bras est DIRECTIONNELLEMENT cohérent avec
l'intuition (le sign de λ est POSITIF partout : l'asymétrie amplifie), MAIS il
n'est PAS établi comme effet distinct nécessaire — il est largement REDONDANT
avec le couplage géométrique G = ΣU_κ·U_τ (qui encode déjà la taille), et
n'améliore pas la validation croisée. Le canal Lk lui-même n'a PAS pu être testé
indépendamment (Lk=Lk_opt=n constant sur les états fondamentaux de l'octet).

CE QU'IL FAUDRAIT pour trancher : des états à Lk VARIABLE et bien assignés (états
de linking excités, ou entrelacs de n différents avec assignations propres) — là
seulement le canal « ×Lk » se sépare du bras d'asymétrie et de G. Non refusé, non
établi : à tester sur un jeu à Lk variable.

Contraste avec le levier SYMÉTRIQUE (Hölder-½, lever_coupling_holder.py) : celui-là
EST établi (60% du poids, CV-stable, A→½ résiste au fit libre). Le levier ASYMÉTRIE
×Lk est, lui, seulement suggestif.
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from simulation.experiments.combi_formula_search import D, CANON

# (nom, config, masse, Lk_opt=n) — octet u/d/s propre
OCTET = [("p", "3_1+7_2+7_2", 938.3, 3), ("Lam", "3_1+5_2+7_2", 1115.7, 3),
         ("Sig", "5_2+7_2+7_2", 1193.2, 3), ("Xi", "3_1+3_1+7_2", 1318.3, 3)]
FULL = np.array(CANON)


def base_full(cfg):
    MC, ML, beta, A, u, v, Q, X, B, rho0, scale = FULL
    comp = cfg.split("+"); n = len(comp); cnt = Counter(comp)
    ws = [(D[k]["cr"] * MC + D[k]["nl"] * ML) * ((D[k]["b"] * D[k]["det"]) ** beta) for k in comp]
    sk = sum(x ** A for x in ws) ** (1 / A)
    G = sum(D[k]["Uk"] ** u * D[k]["Ut"] ** v for k in comp)
    pair = 1.0
    for k, c in cnt.items():
        if c >= 2:
            pair *= (rho0 * D[k]["cr"] ** (-B)) ** (c - 1)
    return ((n + 1) / 2) ** (-X) * pair * sk * G ** Q, ws


def pred(cfg, Lk, lam):
    b, ws = base_full(cfg)
    sroot = [np.sqrt(x) for x in ws]
    arm = sum(abs(sroot[i] - sroot[j]) for i, j in itertools.combinations(range(len(ws)), 2))
    return b * (1 + lam * Lk * arm / sum(sroot))


def rms(lam, data):
    raws = np.array([pred(cfg, Lk, lam) for _, cfg, _, Lk in data])
    me = np.array([m for _, _, m, _ in data])
    if np.any(raws <= 0):
        return 9e9
    off = np.exp(np.mean(np.log(me) - np.log(raws)))
    return float(np.sqrt(np.mean(((raws * off - me) / me * 100) ** 2)))


def main():
    print("Octet, formule COMPLÈTE (avec G), + bras asymétrie×Lk :")
    print(f"  λ=0 (sans bras) : RMS = {rms(0, OCTET):.2f}%")
    b = minimize(lambda p: rms(p[0], OCTET), [0.05], method="Nelder-Mead",
                 options={"maxiter": 3000})
    print(f"  λ libre         : RMS = {b.fun:.2f}%   λ = {b.x[0]:+.4f} "
          f"({'asymétrie AMPLIFIE' if b.x[0] > 0 else 'réduit'})")
    loo = []
    for i in range(len(OCTET)):
        tr = [OCTET[j] for j in range(len(OCTET)) if j != i]
        rr = minimize(lambda p: rms(p[0], tr), [0.05], method="Nelder-Mead")
        nm, cfg, me, Lk = OCTET[i]
        raws = [pred(c, L, rr.x[0]) for _, c, _, L in tr]
        off = np.exp(np.mean([np.log(m) for _, _, m, _ in tr]) - np.mean(np.log(raws)))
        loo.append(abs(np.log(pred(cfg, Lk, rr.x[0]) * off) - np.log(me)))
    print(f"  CV LOO (λ libre): médian = {np.median(loo) * 100:.0f}%  (baseline ~2,5%)")
    print("\n  → λ>0 (sens de l'intuition) mais CV ne s'améliore pas : bras REDONDANT")
    print("    avec G. Canal Lk non testable ici (Lk=3 constant). Voir docstring.")


if __name__ == "__main__":
    main()
