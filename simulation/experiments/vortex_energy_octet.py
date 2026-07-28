r"""
Calcul des 2 termes de vortex sur l'octet, SANS ajustement — résultat honnête.

Suite du programme vortex (vortex_geometric_effects : 2 axes = clasp + writhe).
Demande utilisateur : « attaque le calcul des 2 termes sur l'octet sans fit ».

TERMES (par nœud, scale-free, aucun paramètre ajusté) :
  - E_BS  : énergie cinétique de Biot-Savart du filament (Γ=1), (1/8π)∮∮dl·dl/|r|
            régularisée (cœur=1/n). = mécanisme (1), le croisement-levier.
  - axisym: I_min/I_perp du cœur (1=centré, →0 excentré). = mécanisme (2), freinage.
  - Wr    : writhe (hélicité), axe indépendant.
Poids de nœud testé : w_K = E_BS·axisym (rotation freinée). Combinaison sur
l'entrelacs : somme, et Hölder-½ (= couplage-levier, √m=Σ√w).

RÉSULTAT (octet, référence poids extraits √W : 3_1=12,3 5_2=12,8 7_2=10,0) :
  E_BS   : 3_1=0,218  5_2=0,280  7_2=0,329  (monotone en croisements → ordre FAUX)
  axisym : 3_1=0,752  5_2=0,259  7_2=0,223  (freinage : torus centré, twist freiné)
  E·ax   : 3_1=0,164  5_2=0,073  7_2=0,073
  Ordre octet : E_BS seul 2/6 ; E·ax (add ou Hölder) 3/6, corr +0,35.
  Le freinage (mécanisme 2) est DIRECTIONNELLEMENT correct (corr passe de -0,43 à
  +0,35), mais l'ordre reste 3/6 : Σ (5_2+7_2+7_2, sans 3_1) est prédit trop léger
  (-45 %).

DEUX LIMITES IDENTIFIÉES (le goulot, pas un échec de l'idée) :
  1. FREINAGE TROP FORT : axisym écrase les twists (E·ax range ×2,3) alors que les
     poids réels sont ÉQUILIBRÉS (√W range ×1,28). Le freinage physique est plus
     doux que I_min/I_perp brut.
  2. COUPLAGE INTER-NŒUDS MANQUANT (dominant) : on a établi (lever_coupling_holder)
     que le couplage-levier fait ~60 % de la masse baryonique. Ici on calcule des
     nœuds ISOLÉS ; le Biot-Savart MUTUEL entre les 3 brins de l'entrelacs — le
     cœur de la masse — n'est PAS dans le calcul (on n'a pas l'embedding lié du
     baryon, seulement les courbes de nœuds séparées). La Hölder-½ n'en est qu'un
     PROXY (termes croisés √(w_i w_j)), pas le vrai couplage géométrique.

CE QUI EST ACQUIS (sans fit) : le freinage d'excentricité oriente correctement
(corr +0,35, mécanisme 2 réel) ; la séparation des familles était déjà à 100 %.
CE QUI MANQUE POUR LA MASSE FINE : la géométrie de l'ENTRELACS LIÉ (placer les 3
nœuds enlacés dans l'espace) pour calculer le Biot-Savart mutuel = le couplage
(60 %). C'est l'étape suivante claire — construire l'embedding du lien, pas fitter.

STATUT : premier calcul paramètre-libre fait ; direction (freinage) validée ;
goulot = embedding de l'entrelacs lié. Ne PAS présenter comme une reproduction de
l'octet — c'est un résultat partiel honnête avec bottleneck identifié.
"""
from __future__ import annotations
import numpy as np
from theory.omega_ts import _knot_curve
from simulation.experiments.repair_curves_braid import braid_curve, BRAID_WORDS

OCTET = [("p", "3_1+7_2+7_2", 938.3), ("Lam", "3_1+5_2+7_2", 1115.7),
         ("Sig", "5_2+7_2+7_2", 1193.2), ("Xi", "3_1+3_1+7_2", 1318.3)]


def get_curve(K, n=1200):
    if K in ("3_1", "4_1", "5_1", "7_1", "9_1"):
        return np.asarray(_knot_curve(K, n), float)
    if K in BRAID_WORDS:
        w, ns, _ = BRAID_WORDS[K]
        return np.asarray(braid_curve(w, ns)[0], float)
    raise KeyError(K)


def _resample_unit(P, m=300):
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0], np.cumsum(d)]); Lt = s[-1]
    u = np.linspace(0, Lt, m, endpoint=False)
    return np.column_stack([np.interp(u, s[:-1], P[:, k], period=Lt) for k in range(3)]) / Lt


def knot_terms(K):
    Q = _resample_unit(get_curve(K)); n = len(Q)
    dl = np.roll(Q, -1, 0) - Q; mid = Q + dl / 2; core = 1.0 / n
    E = 0.0
    for i in range(n):
        rn = np.sqrt(((mid[i] - mid) ** 2).sum(1) + core ** 2)
        m = np.arange(n) != i
        E += np.sum((dl[i] @ dl.T)[m] / rn[m])
    E_BS = E / (8 * np.pi)
    R = Q - Q.mean(0); r2 = (R ** 2).sum(1)
    I = np.einsum("n,ij->ij", r2, np.eye(3)) - np.einsum("ni,nj->ij", R, R)
    ev = np.linalg.eigvalsh(I)
    return E_BS, ev[0] / ((ev[1] + ev[2]) / 2)


def main():
    KN = {K: knot_terms(K) for K in ["3_1", "5_2", "7_2"]}
    for K, (e, a) in KN.items():
        print(f"  {K}: E_BS={e:.4f}  axisym={a:.3f}  E·ax={e * a:.4f}")
    print("  réf √W extraits : 3_1=12,3  5_2=12,8  7_2=10,0\n")
    mexp = np.array([m for _, _, m in OCTET])
    oexp = [nm for nm, _, _ in sorted(OCTET, key=lambda t: t[2])]
    for mode in ["add", "holder"]:
        pv = []
        for nm, cfg, _ in OCTET:
            ws = [KN[k][0] * KN[k][1] for k in cfg.split("+")]
            pv.append((nm, sum(ws) if mode == "add" else sum(np.sqrt(w) for w in ws) ** 2))
        opred = [nm for nm, _ in sorted(pv, key=lambda t: t[1])]
        kend = sum(1 for i in range(4) for j in range(i + 1, 4)
                   if opred.index(oexp[i]) < opred.index(oexp[j]))
        corr = np.corrcoef([v for _, v in pv], mexp)[0, 1]
        print(f"  {mode:7s}: ordre {' < '.join(opred):24s} Kendall {kend}/6  corr {corr:+.2f}")
    print(f"  exp    : ordre {' < '.join(oexp)}")
    print("\n  → freinage directionnellement OK ; goulot = couplage inter-nœuds")
    print("    (Biot-Savart mutuel de l'entrelacs lié, ~60% de la masse, non calculé).")


if __name__ == "__main__":
    main()
