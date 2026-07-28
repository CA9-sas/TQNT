r"""
M₀ via la valeur propre du puits χ de l'ENTRELACS ENTIER (pas nœud par nœud).

Correction (utilisateur) : une particule est un ENTRELACS (nucléon = 3₁+7₂+7₂,
méson = K+K̄), pas un nœud isolé. Sa masse est l'auto-énergie du puits χ de
l'entrelacs COMPLET, où l'enlacement/recouvrement entre composantes remodèle le
puits globalement. On calcule donc E₀ de l'entrelacs, pas la somme des nœuds.

Méthode : place les courbes des composantes (RÉPARÉES pour les non-toriques,
catalogue pour les toriques) espacées le long de x, maille le complément commun
(generate_mesh_multi), assemble le potentiel de Jones sommé (assemble_system_multi),
résout E₀. Injecte les vraies courbes (le solve_link du dépôt utilise les courbes
catalogue invalides).

Testable : secteur LÉGER (composantes réparées/valides disponibles) —
nucléon, Λ, Σ, Ξ, ρ⁰, ω, η′, Δ. Les quarkonia lourds (7₄/7₅/7₆) restent hors de
portée (courbes non réparées).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

from theory.knot_catalog import knot_curve
from theory.schrodinger import generate_mesh_multi, assemble_system_multi, solve_eigenproblem
from simulation.experiments.repair_curves_braid import braid_curve, BRAID_WORDS

# 6₃ (amphichiral, Δ) — trouvé par recherche det-13 : Alexander (1,-3,5,-3,1)
EXTRA = {"6_3": ([1, 1, 1, 2, -1, -1, -2, -2], 3)}


def get_curve(knot_name, npts=400):
    """Courbe RÉPARÉE (tresse) si non-torique validée, sinon catalogue (torique)."""
    base = knot_name.replace("bar", "")  # miroir = même géométrie/retraction
    if base in BRAID_WORDS:
        w, ns, _ = BRAID_WORDS[base]
        P, _, _ = braid_curve(w, ns, pts_per_seg=40)
    elif base in EXTRA:
        w, ns = EXTRA[base]
        P, _, _ = braid_curve(w, ns, pts_per_seg=40)
    else:
        P = np.asarray(knot_curve(base, npts), float)  # torique valide
    P = P - P.mean(0)
    m = np.max(np.abs(P))
    return P / m if m > 0 else P  # extent ~1


def solve_entrelacs(components, distance=3.0, n_coarse=22):
    """E₀ du puits χ de l'entrelacs (composantes espacées le long de x)."""
    curves = []
    n = len(components)
    for i, K in enumerate(components):
        c = get_curve(K) * (0.35 * distance)          # rayon ~0.35 d
        c = c + np.array([(i - (n - 1) / 2) * distance, 0.0, 0.0])
        curves.append(c)
    pts, tets, bnd = generate_mesh_multi(curves, n_coarse=n_coarse, epsilon=0.2, refine=True)
    names = [K.replace("bar", "") for K in components]
    H, M = assemble_system_multi(pts, tets, bnd, names, curves, energy=0.0, epsilon=0.2)
    ev, _ = solve_eigenproblem(H, M, n_eigenvalues=3)
    return float(np.sort(ev)[0])


# entrelacs (composantes) + masse PDG (MeV) du secteur léger
PARTICLES = {
    "π⁰":   (["8_1", "8_1"], 135.0),
    "ρ⁰":   (["6_2", "6_2"], 775.3),
    "ω":    (["8_1", "8_1"], 782.7),        # même config que π⁰ -> teste la limite
    "η′":   (["5_2", "5_2", "6_1"], 957.8),
    "nucléon": (["3_1", "7_2", "7_2"], 938.9),
    "Λ":    (["3_1", "5_2", "7_2"], 1115.7),
    "Σ":    (["5_2", "7_2", "7_2"], 1193.2),
    "Ξ":    (["3_1", "3_1", "7_2"], 1318.3),
    "Δ":    (["5_2", "6_3"], 1232.0),
}


def main():
    import time
    print("E₀ du puits χ de l'ENTRELACS ENTIER (courbes réparées injectées)\n")
    print(f"{'particule':9s} {'entrelacs':22s} {'E₀':>10s} {'masse(MeV)':>10s} {'t(s)':>5s}")
    rows = []
    for nm, (comps, m) in PARTICLES.items():
        t = time.time()
        try:
            e0 = solve_entrelacs(comps)
            rows.append((nm, e0, m))
            print(f"{nm:9s} {'+'.join(comps):22s} {e0:10.3f} {m:10.1f} {time.time()-t:5.1f}", flush=True)
        except Exception as ex:
            print(f"{nm:9s} {'+'.join(comps):22s} ERREUR {ex}", flush=True)
    if len(rows) >= 3:
        E = np.array([r[1] for r in rows]); Ms = np.array([r[2] for r in rows])
        print(f"\ncorr(E₀, masse)     = {np.corrcoef(E, Ms)[0, 1]:.3f}")
        print(f"corr(|E₀|, log masse) = {np.corrcoef(-E, np.log(Ms))[0, 1]:.3f}")


if __name__ == "__main__":
    main()
