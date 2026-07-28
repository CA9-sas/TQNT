r"""
Le puits CGQ amplifie-t-il le vortex ? — test de l'hypothèse utilisateur.

Idée utilisateur : « regarde si le point le plus profond du puits de la CGQ
amplifie le vortex » (comme l'eau qui spirale vers une bonde : plus le puits est
profond, plus le centre tourne vite). NB utilisateur : « même ce n'est pas du même
ordre » — on teste l'amplification GÉOMÉTRIQUE du vortex par la profondeur du
puits, pas la magnitude gravitationnelle du CGQ (~10⁻³⁹, négligeable).

Le CGQ = puits de densité χ que le nœud creuse (g_q=(c²/2)∇(Ω²−1), Ω²=1+κ_g|Ψ|²).
Profondeur du puits = potentiel Φ(p)=∮ ds/|p−r| maximal sur l'intérieur (point le
plus enfermé par le tube). Calculé sur courbes valides, scale-free (L=1).

RÉSULTAT — l'hypothèse d'amplification est INFIRMÉE (le puits FREINE) :
  Φ_max : 3_1=15,8  5_2=26,7  7_2=33,0  (croît avec les croisements)
  corr(Φ_max, poids extrait √W) = −0,66 : le puits le PLUS profond (7_2) donne la
  masse la plus FAIBLE. Test octet :
    E×Φ (amplifie)  Kendall 2/6  corr −0,41  ← EMPIRE
    E/Φ (freine)    Kendall 4/6  corr +0,37  ← améliore
    Φ seul          Kendall 2/6  corr −0,41
  Le puits n'amplifie PAS ; le traiter en FREIN (E/Φ) donne le meilleur résultat
  géométrique de l'octet (4/6).

REDONDANCE : corr(Φ_max, anisotropie du clasp) = +0,89 → le puits CGQ est
REDONDANT avec l'axe du clasp (pas un degré de liberté nouveau, cf.
vortex_geometric_effects : 2 axes = clasp + writhe).

RAISON PHYSIQUE (l'insight) : le point le plus profond du puits est AU CLASP (là
où les brins s'entassent, |Ψ|² pique), donc HORS-AXE (zone excentrée). Une bonde
amplifierait le vortex si elle était CENTRÉE (drain axisymétrique) ; mais pour un
nœud le puits le plus profond est excentré (au clasp), donc il FREINE au lieu
d'accélérer. Torus (lepton, centré) → puits ~centré ; twist (baryon, clasped) →
puits excentré = freinant. Cohérent avec clasp = braking déjà établi.

CONCLUSION : réponse NON à l'hypothèse — le puits CGQ n'amplifie pas le vortex
(le sens est opposé : il freine), et il n'est de toute façon pas indépendant
(redondant avec le clasp). Ça ne change pas le tableau : familles = géométrie
(clasp/writhe), masse fine octet = étrangeté (Z₃, [[flavor-is-SM-artifact]]).
Le CGQ reste le champ fondamental ([[champ-gravitationnel-quantique]]) mais son
puits n'est pas un amplificateur de masse.
"""
from __future__ import annotations
import numpy as np
from theory.omega_ts import _knot_curve
from simulation.experiments.repair_curves_braid import braid_curve, BRAID_WORDS

WEIGHT = {"3_1": 12.3, "5_2": 12.8, "7_2": 10.0}   # √W extraits (per_knot_amplification)
OCTET = [("p", "3_1 7_2 7_2", 938.3), ("Lam", "3_1 5_2 7_2", 1115.7),
         ("Sig", "5_2 7_2 7_2", 1193.2), ("Xi", "3_1 3_1 7_2", 1318.3)]


def get_curve(K):
    if K in ("3_1", "4_1", "5_1", "7_1", "9_1"):
        P = np.asarray(_knot_curve(K, 400), float)
    else:
        w, ns, _ = BRAID_WORDS[K]
        P = np.asarray(braid_curve(w, ns)[0], float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0], np.cumsum(d)]); Lt = s[-1]
    u = np.linspace(0, Lt, 240, endpoint=False)
    return np.column_stack([np.interp(u, s[:-1], P[:, k], period=Lt) for k in range(3)]) / Lt


def cgq_well(Q):
    """Profondeur max du puits CGQ Φ(p)=∮ ds/|p−r| (point le plus enfermé)."""
    ds = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
    c = Q.mean(0); ext = np.ptp(Q, 0).max(); best = -1.0; bp = c
    rng = np.random.RandomState(0)
    for scale in (0.4, 0.2, 0.1):
        for p in c + scale * ext * (rng.rand(300, 3) - 0.5):
            phi = np.sum(ds / np.clip(np.linalg.norm(Q - p, axis=1), 0.02, None))
            if phi > best:
                best = phi; bp = p
        c = bp
    return best


def anisotropy(Q):
    R = Q - Q.mean(0)
    ev = np.linalg.eigvalsh(np.einsum("ni,nj->ij", R, R))
    return ev[2] / ev[0]


def main():
    PHI = {k: cgq_well(get_curve(k)) for k in ["3_1", "5_2", "7_2"]}
    print("Profondeur du puits CGQ (Φ_max, L=1) vs poids extrait √W :")
    for k in ["3_1", "5_2", "7_2"]:
        print(f"  {k}: Φ_max={PHI[k]:.1f}  √W={WEIGHT[k]}")
    phi = [PHI[k] for k in ["3_1", "5_2", "7_2"]]; w = [WEIGHT[k] for k in ["3_1", "5_2", "7_2"]]
    print(f"  corr(Φ_max, √W) = {np.corrcoef(phi, w)[0, 1]:+.2f}\n")

    mexp = np.array([m for _, _, m in OCTET])
    oexp = [nm for nm, _, m in sorted(OCTET, key=lambda t: t[2])]
    for mode, lab in [(+1, "E×Φ (amplifie)"), (-1, "E/Φ (freine)")]:
        vals = [sum(PHI[k] ** mode for k in cfg.split()) for _, cfg, _ in OCTET]
        opred = [OCTET[i][0] for i in np.argsort(vals)]
        kend = sum(1 for i in range(4) for j in range(i + 1, 4)
                   if opred.index(oexp[i]) < opred.index(oexp[j]))
        print(f"  {lab:16s}: Kendall {kend}/6  corr {np.corrcoef(vals, mexp)[0, 1]:+.2f}")
    ks = ["3_1", "5_1", "7_1", "5_2", "7_2", "8_2", "8_3", "10_1"]
    p = [cgq_well(get_curve(k)) for k in ks]; a = [anisotropy(get_curve(k)) for k in ks]
    print(f"\n  corr(Φ_max, anisotropie clasp) = {np.corrcoef(p, a)[0, 1]:+.2f} "
          "→ redondant avec le clasp.")
    print("  VERDICT : le puits CGQ ne l'amplifie pas — il freine (au clasp, hors-axe).")


if __name__ == "__main__":
    main()
