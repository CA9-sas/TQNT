"""Tentative de derivation des angles theta_Sigma (~67 deg) et theta_Xi (~97 deg)
depuis les plongements du catalogue. VERDICT : NON derivable avec les outils actuels.

Pipeline construit :
  1. Porteurs etranges repares par fermeture de tresse (BRAID_WORDS a deja 6_1, 6_2 ;
     mot independant trouve pour 6_1 = [1,1,2,-1,-3,2,-3] (KnotInfo), Alexander (2,-5,2)
     valide en croise avec celui du module).
  2. PROBLEME 1 : les fermetures de tresse sont PLANAIRES par construction -> tous les
     lobes coalignes (angles ~0 partout) : l'embedding de reparation n'est pas physique.
  3. Relaxation de Mobius (descente repulsive 1/r^4, re-echantillonnage, longueur fixee,
     garde Alexander) -> formes 3D, type de noeud PRESERVE sur tous les runs.
  4. Decomposition en lobes (arcs contigus par orientation d'aire locale) ; angle de
     chaque lobe vs l'aire vectorielle globale.

VERDICT (test de controle + stabilite) :
  - CONTROLE TREFLE (C3, attendu : lobes tous alignes ~0) : lobe dominant 26-29 deg,
    angle max 101-115 deg -> la decomposition produit des fragments anti-alignes SPURIEUX
    (regions de croisement), meme pour le noeud le plus symetrique. ECHEC du controle.
  - Dispersion inter-runs (3 jeux de parametres) : +/-15-20 deg sur les angles -- superieure
    a la resolution requise (separer 2 / 67 / 97 deg).
  - La 'signature 6_2 seul a un lobe >90 deg' du premier run etait un ARTEFACT (run 2 : 86 ;
    et le trefle-controle montre aussi >100).
=> les angles theta restent des LECTURES. Requis pour une vraie derivation :
  (i) relaxation de noeud ideal convergee sous contrainte de tube (classe SONO/ridgerunner) ;
  (ii) identification PRINCIPIELLE du lobe enfile -- qui exige la geometrie du systeme LIE
       a 3 composantes (l'angle vit dans le compose, pas dans le noeud isole) ;
  (iii) l'axe de spin du compose. Aucun de ces outils n'existe dans le repo.
"""
import numpy as np

from simulation.experiments.repair_curves_braid import braid_curve, alexander, BRAID_WORDS
from theory.knot_catalog import knot_curve

TARGET = {'7_2': (3, -5, 3), '6_1': (2, -5, 2), '6_2': (1, -3, 3, -3, 1), '3_1': (1, -1, 1)}


def resample(P, N):
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0], np.cumsum(d)])[:-1]
    su = np.linspace(0, s[-1] + d[-1], N, endpoint=False)
    return np.stack([np.interp(su, np.concatenate([s, [s[-1] + d[-1]]]),
                               np.concatenate([P[:, k], [P[0, k]]])) for k in range(3)], 1)


def relax(P, steps=400, lr=2e-3, N=180):
    """Descente repulsive type Mobius, longueur fixee ; preserve le type (garde Alexander externe)."""
    P = resample(P, N); P = P - P.mean(0)
    P = P / np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()
    idx = np.arange(N)
    dmi = np.minimum(np.abs(idx[:, None] - idx[None, :]), N - np.abs(idx[:, None] - idx[None, :]))
    mask = dmi > 3
    for _ in range(steps):
        diff = P[:, None, :] - P[None, :, :]
        r2 = (diff ** 2).sum(-1) + 1e-12
        F = (diff / r2[..., None] ** 2 * mask[..., None]).sum(1)
        P = P + lr * F / np.abs(F).max()
        P = resample(P, N); P = P - P.mean(0)
        P = P / np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()
    return P


def lobes(P):
    P = P - P.mean(0)
    dl = np.roll(P, -1, 0) - P
    dS = 0.5 * np.cross(P + 0.5 * dl, dl)
    Sg = dS.sum(0); Sgh = Sg / np.linalg.norm(Sg)
    lob, cur = [], np.zeros(3)
    for v in dS:
        if np.dot(cur, v) < 0 and np.linalg.norm(cur) > 0.02 * np.linalg.norm(Sg):
            lob.append(cur); cur = v.copy()
        else:
            cur = cur + v
    lob.append(cur)
    if len(lob) > 1 and np.dot(lob[0], lob[-1]) > 0:
        lob[0] = lob[0] + lob.pop()
    out = [(np.linalg.norm(L) / np.linalg.norm(Sg),
            float(np.degrees(np.arccos(np.clip(np.dot(L, Sgh) / np.linalg.norm(L), -1, 1)))))
           for L in lob if np.linalg.norm(L) > 0.03 * np.linalg.norm(Sg)]
    return sorted(out, reverse=True)


def run():
    print("cibles (lectures) : 7_2 -> 2 deg ; 6_1 -> 67 deg ; 6_2 -> 97 deg")
    for K in ('3_1', '7_2', '6_1', '6_2'):
        if K == '3_1':
            P0 = np.asarray(knot_curve('3_1', 2000), float)
        else:
            w, ns, _ = BRAID_WORDS[K]
            P0, _, _ = braid_curve(w, ns, pts_per_seg=40)
        rows = []
        for steps, lr, N in ((400, 2e-3, 180), (700, 1.5e-3, 200), (500, 3e-3, 160)):
            P = relax(np.asarray(P0, float), steps, lr, N)
            ok = tuple(map(int, alexander(resample(P, 800))[0])) == TARGET[K]
            lb = lobes(P)
            rows.append((ok, lb[0][1], max(a for _, a in lb)))
        lab = 'CONTROLE' if K == '3_1' else 'porteur'
        print(f"  {K} ({lab}): (type_ok, lobe_dominant, angle_max) = " +
              " | ".join(f"({'OK' if o else 'KO'},{d:.0f},{m:.0f})" for o, d, m in rows))
    print("\nVERDICT : controle trefle NON passe (fragments >100 deg spurieux) ; dispersion")
    print("+/-20 deg >> resolution requise. Les theta restent des LECTURES (cf. docstring).")


if __name__ == "__main__":
    run()
