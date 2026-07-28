"""Attaque fraiche sur A = sqrt(2) (Koide) : le melange a 45 degres des canaux du trefle.

REFORMULATION EXACTE (algebre) : dans sqrt(m_k) = c0 + c1*cos(delta + 2pi k/3),
  A = c1/c0 = sqrt(2)  <=>  ||singlet||^2 = ||doublet||^2  (equipartition PAR CANAL)
  (3 c0^2 = c1^2 * sum cos^2 = (3/2) c1^2). L'alternative 'equipartition par
  DIMENSION' (le doublet compte double) donnerait Q = 1 : REFUTEE par la mesure.
  Q = 2/3 <=> angle de 45 deg entre le vecteur sqrt(m) et la direction democratique.

MECANISME CANDIDAT (teste ici, zero bouton) : un melange a 45 deg est l'etat propre
d'un couplage PUREMENT HORS-DIAGONAL entre deux canaux degeneres. Les deux canaux
TQNT du trefle : le mode uniforme psi_0 (singlet C3) et le doublet Z3 (psi_a, psi_b).
Si l'energie d'habillage quartique du champ chi (terme de retroaction P5,
E ~ int |Psi|^4) est SYMETRIQUE sous l'echange singlet <-> doublet
(I_0000 = I_1111 et I_0001 = I_0111), son extremum de melange tombe a 45 deg
exactement -> A = sqrt(2) sans parametre.

PROTOCOLE : solveur FEM (solve_knot '3_1', courbe torique VALIDE, C3 autour de z) ;
identification du doublet par caractere de rotation C3 (overlap <psi(Rx)|psi(x)>,
attendu ~ -0.5 par etat du doublet E) ; integrales quartiques par masse condensee ;
E_mix(theta) = int (cos(theta) psi_0 + sin(theta) psi_Z3)^4 ; angle extremal.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from theory.schrodinger import generate_mesh, assemble_system, solve_eigenproblem  # noqa: E402
from theory.knot_catalog import knot_curve  # noqa: E402


def lumped_volumes(points, tets):
    """Volume condense par sommet (somme des quarts de volumes des tetras)."""
    V = np.zeros(len(points))
    for t in tets:
        p = points[t]
        vol = abs(np.linalg.det(p[1:] - p[0])) / 6.0
        V[t] += vol / 4.0
    return V


def c3_overlap(points, V, psi_i, psi_j):
    """<psi_i(R x)|psi_j(x)> pour R = rotation de 2pi/3 autour de z (plus proche voisin)."""
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    tree = cKDTree(points)
    _, idx = tree.query(points @ R.T)
    return float(np.sum(V * psi_i[idx] * psi_j))


def run():
    print("=== A = sqrt(2) : test du melange a 45 deg (modes du trefle, solveur FEM) ===")
    curve = knot_curve("3_1", 500)
    knot_tree = cKDTree(curve)
    pts, tets, boundary = generate_mesh(curve, n_coarse=26, epsilon=0.2, refine=True)
    print(f"maillage : {len(pts)} points, {len(tets)} tetras", flush=True)
    H, M = assemble_system(pts, tets, boundary, "3_1", knot_tree, energy=0.0, epsilon=0.2)
    evals, evecs_free = solve_eigenproblem(H, M, n_eigenvalues=6)
    free = ~boundary
    evecs = np.zeros((len(pts), evecs_free.shape[1]))
    evecs[free, :] = evecs_free
    V = lumped_volumes(pts, tets)

    # normalisation L2 (masse condensee)
    modes = []
    for k in range(evecs.shape[1]):
        v = evecs[:, k]
        v = v / np.sqrt(np.sum(V * v * v))
        modes.append(v)

    print("\nmode | E | caractere C3 <psi(Rx)|psi(x)>")
    chars = []
    for k, v in enumerate(modes):
        ch = c3_overlap(pts, V, v, v)
        chars.append(ch)
        print(f"  {k}  | {evals[k]:9.4f} | {ch:+.3f}")

    # doublet Z3 : première paire quasi degeneree a caractere ~ -0.5
    pair = None
    for k in range(1, len(modes) - 1):
        if (abs(evals[k + 1] - evals[k]) / abs(evals[k]) < 0.05
                and chars[k] < 0 and chars[k + 1] < 0):
            pair = (k, k + 1)
            break
    if pair is None:
        print("\nPAS de doublet Z3 identifiable (maillage trop anisotrope ?) — ECHEC du protocole")
        return
    a, b = pair
    print(f"\ndoublet Z3 identifie : modes {a},{b} (E = {evals[a]:.4f}, {evals[b]:.4f} ; "
          f"caracteres {chars[a]:+.2f}, {chars[b]:+.2f})")

    psi0 = modes[0]
    # combinaison Z3 reelle normalisee (l'angle de melange est insensible au choix
    # de la combinaison dans le doublet si la forme quartique est isotrope dedans)
    results = []
    for lab, z in (("psi_a", modes[a]), ("psi_b", modes[b]),
                   ("(a+b)/sqrt2", (modes[a] + modes[b]) / np.sqrt(2))):
        z = z / np.sqrt(np.sum(V * z * z))
        I0000 = np.sum(V * psi0**4)
        I0001 = np.sum(V * psi0**3 * z)
        I0011 = np.sum(V * psi0**2 * z**2)
        I0111 = np.sum(V * psi0 * z**3)
        I1111 = np.sum(V * z**4)
        th = np.linspace(0, np.pi / 2, 2001)
        c, s = np.cos(th), np.sin(th)
        E = (c**4 * I0000 + 4 * c**3 * s * I0001 + 6 * c**2 * s**2 * I0011
             + 4 * c * s**3 * I0111 + s**4 * I1111)
        i_ext = int(np.argmax(E))
        th_max = np.degrees(th[i_ext])
        i_min = int(np.argmin(E))
        th_min = np.degrees(th[i_min])
        sym = abs(I0000 - I1111) / (I0000 + I1111)
        results.append((lab, th_max, th_min, sym, I0000, I0011, I1111, I0001, I0111))
        print(f"\n[{lab}] I0000={I0000:.4g} I0011={I0011:.4g} I1111={I1111:.4g} "
              f"I0001={I0001:.3g} I0111={I0111:.3g}")
        print(f"  asymetrie |I0000-I1111|/(somme) = {sym:.2f} ; "
              f"extremum E_mix : max a {th_max:.1f} deg, min a {th_min:.1f} deg")

    # sonde complete : balayage de l'orientation phi DANS le doublet
    print("\n=== Balayage de l'orientation du doublet (theta_max en fonction de phi) ===")
    za, zb = modes[a], modes[b]
    th = np.linspace(0, np.pi / 2, 2001)
    c, s_ = np.cos(th), np.sin(th)
    thmaxs = []
    for phi in np.linspace(0, np.pi, 36, endpoint=False):
        z = np.cos(phi) * za + np.sin(phi) * zb
        z = z / np.sqrt(np.sum(V * z * z))
        I = [np.sum(V * psi0**(4 - m) * z**m) for m in range(5)]
        E = (c**4 * I[0] + 4 * c**3 * s_ * I[1] + 6 * c**2 * s_**2 * I[2]
             + 4 * c * s_**3 * I[3] + s_**4 * I[4])
        thmaxs.append(np.degrees(th[int(np.argmax(E))]))
    thmaxs = np.array(thmaxs)
    generic = thmaxs[(thmaxs > 5) & (thmaxs < 85)]
    print(f"  36 orientations : mediane = {np.median(thmaxs):.1f} deg ; "
          f"generiques (hors bords) : {len(generic)}/36, "
          f"mediane = {np.median(generic):.1f} deg, "
          f"intervalle [{generic.min():.1f}, {generic.max():.1f}]")
    np.savez('/tmp/koide45_modes.npz', V=V, psi0=psi0, za=za, zb=zb,
             evals=evals[:4], pts=pts)

    print("\n=== Lecture ===")
    print("A=sqrt(2) exigerait un extremum de melange a 45.0 deg (equipartition par canal),")
    print("c-a-d I0000 ~ I1111 et I0001 ~ I0111 (forme quartique symetrique singlet<->doublet).")


if __name__ == "__main__":
    run()
