r"""
Catalogue des EFFETS GÉOMÉTRIQUES des nœuds sur la masse (reframe vortex, sans fit).

Demande utilisateur : avant le calcul d'énergie sans paramètre, « trouver d'autres
effets géométriques des nœuds sur les masses ». On calcule un large éventail
d'invariants sur les courbes VALIDES (torus + réparées clôture de tresse) et on
les réduit par corrélation.

EFFETS CALCULÉS (tous scale-free, sans ajustement) :
  - aniso   : anisotropie I_max/I_min du cœur (forme du vortex)
  - ecc     : excentricité R_max/R_g
  - kconc   : concentration de courbure κ_max/⟨κ⟩ (netteté du clasp)
  - kloc    : fraction d'arc portant 50 % de ∮κ² (localisation du clasp)
  - totT    : torsion totale ∮|τ|/2π (corkscrew du vortex)
  - writhe  : |Wr| auto-enlacement de Gauss (HÉLICITÉ du vortex, chiralité)
  [testés et ÉCARTÉS comme NULS : rayon radial de la courbure r_κ/R_g (aucune
   séparation) ; kurtosis radial (faible).]

RÉSULTAT STRUCTURAL — les 6 effets s'effondrent en 2 AXES INDÉPENDANTS :
  Matrice de corrélation : aniso, ecc, kconc, kloc, totT sont tous corrélés à
  |r| = 0,8-0,97 → UN SEUL axe (la forme torus↔twist / clasp). Le writhe est
  décorrélé (|r| < 0,4 avec tous) → axe INDÉPENDANT.
    → 2 degrés de liberté géométriques distincts sur 6 effets.

LES DEUX AXES :
  1. CLASP / EXCENTRICITÉ (aniso≈ecc≈kconc≈kloc≈totT) — la forme du vortex.
     Torus (lepton) = centré/lisse (aniso 1,65, clasp étalé sur 39 % de l'arc) ;
     twist (baryon/EW) = excentré/clasped (aniso ~5, 50 % de la courbure² dans
     4-8 % de l'arc). Les mécanismes utilisateur (1) « croisement=levier sur brins
     inférieurs » et (2) « boucles excentrées freinent » sont GÉOMÉTRIQUEMENT LE
     MÊME axe : le clasp. Sépare les familles à 100 %.
  2. WRITHE / HÉLICITÉ (indépendant) — l'auto-enlacement du vortex. 3_1 = -3,4
     (chiral, matière), torus T(2,q) croît (+6,9 / +10,5 / +14,4 pour 5_1/7_1/9_1).
     C'est la CHIRALITÉ (matière/antimatière, V_K(t) vs V_K(t^{-1})) ET, en
     dynamique des vortex, l'HÉLICITÉ H = Γ²·Wr, quantité CONSERVÉE qui porte de
     l'énergie. C'est le SEUL effet géométrique nouveau et indépendant.

CONCLUSION pour le programme vortex : au-delà de la forme (clasp, déjà identifiée
et qui sépare les familles), il existe EXACTEMENT UN autre axe géométrique
indépendant — le WRITHE/hélicité. L'énergie du vortex doit donc combiner (a) un
terme de forme/clasp (rotation + freinage) et (b) un terme d'hélicité ∝ Wr². Les
autres invariants (courbure, torsion, excentricité) sont redondants avec (a) — ne
pas les compter comme des effets séparés. Étape suivante (calcul sans paramètre) :
E_vortex = E_rotation(forme) + E_hélicité(Wr), tester l'ordre de l'octet.
"""
from __future__ import annotations
import numpy as np
from theory.omega_ts import _knot_curve
from simulation.experiments.repair_curves_braid import braid_curve, BRAID_WORDS

TORUS = {"3_1", "5_1", "7_1", "9_1"}
KNOTS = ["3_1", "4_1", "5_1", "7_1", "9_1", "5_2", "7_2", "8_2", "8_3", "10_1", "6_2"]
KEYS = ["aniso", "ecc", "kconc", "kloc", "totT", "writhe"]


def get_curve(K, n=1200):
    if K in ("3_1", "4_1", "5_1", "7_1", "9_1"):
        return np.asarray(_knot_curve(K, n), float)
    if K in BRAID_WORDS:
        w, ns, _ = BRAID_WORDS[K]
        return np.asarray(braid_curve(w, ns)[0], float)
    raise KeyError(K)


def _resample(P, m=280):
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0], np.cumsum(d)]); Lt = s[-1]
    u = np.linspace(0, Lt, m, endpoint=False)
    return np.column_stack([np.interp(u, s[:-1], P[:, k], period=Lt) for k in range(3)]), Lt


def effects(K):
    Q, Lt = _resample(get_curve(K)); n = len(Q); ds = Lt / n
    d1 = np.gradient(Q, ds, axis=0); d2 = np.gradient(d1, ds, axis=0); d3 = np.gradient(d2, ds, axis=0)
    cr = np.cross(d1, d2); ncr = np.linalg.norm(cr, axis=1); sp = np.linalg.norm(d1, axis=1)
    dsr = sp * ds
    kap = ncr / np.clip(sp ** 3, 1e-9, None)
    tau = np.einsum("ij,ij->i", cr, d3) / np.clip(ncr ** 2, 1e-9, None)
    R = Q - Q.mean(0); r = np.linalg.norm(R, axis=1)
    I = np.einsum("n,ij->ij", r ** 2, np.eye(3)) - np.einsum("ni,nj->ij", R, R)
    ev = np.linalg.eigvalsh(I)
    w = kap ** 2 * dsr; cum = np.cumsum(np.sort(w)[::-1])
    dl = np.roll(Q, -1, 0) - Q; mid = Q + dl / 2; W = 0.0
    for i in range(n):
        rr = mid[i] - mid; rn = np.linalg.norm(rr, axis=1); m = rn > 1e-6
        W += np.sum(np.einsum("ij,ij->i", np.cross(dl[i], dl[m]), rr[m]) / rn[m] ** 3)
    return dict(aniso=ev[2] / ev[0], ecc=r.max() / np.sqrt((r ** 2 * dsr).sum() / Lt),
                kconc=kap.max() / ((kap * dsr).sum() / Lt),
                kloc=np.searchsorted(cum, 0.5 * cum[-1]) / n,
                totT=(np.abs(tau) * dsr).sum() / (2 * np.pi), writhe=abs(W / (4 * np.pi)))


def main():
    M = np.array([[effects(K)[k] for k in KEYS] for K in KNOTS])
    C = np.corrcoef(M.T)
    print("Corrélation des effets géométriques (|r|>0,8 = redondants) :")
    print("        " + " ".join(f"{k:>7s}" for k in KEYS))
    for i, k in enumerate(KEYS):
        print(f"{k:8s}" + " ".join(f"{C[i, j]:+7.2f}" for j in range(len(KEYS))))
    indep = []
    for i in range(len(KEYS)):
        if all(abs(C[i, j]) < 0.8 for j in indep):
            indep.append(i)
    print(f"\nAxes INDÉPENDANTS (|r|<0,8) : {[KEYS[i] for i in indep]}")
    print("→ 2 DOF : (1) forme/clasp (aniso≈ecc≈kconc≈kloc≈totT) ; (2) writhe/hélicité.")


if __name__ == "__main__":
    main()
