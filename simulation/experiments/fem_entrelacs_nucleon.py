"""Chantier f1/f2 -- etape 1 : FEM chi sur l'entrelacs nucleon complet (20/07).

PREMIERE resolution du champ chi par elements finis sur l'entrelacs
3_1 + 7_2 + 7b_2 ENTIER (le solveur n'avait servi que sur des noeuds
simples ; generate_mesh_multi/assemble_system_multi existaient sans
avoir ete appliques au nucleon).

MOTIFS :
  - NEUTRON : l'entrelacs relaxe SONO (output/nucleon_ideal_*.npy),
    |Lk| = (1,1,0) ;
  - PROTON : le meme entrelacs avec UN passage local controle de S2 a
    travers S1 (bosse gaussienne au point d'approche minimale, gardes
    Lk de Gauss + types d'Alexander) -> |Lk| = (1,1,1).

OBSERVABLES (cibles f1/f2, au niveau de precision de la maille ~%) :
  - E1 et l'ecart E2-E1 (etat fondamental non degenere = indicateur
    d'unicite DYNAMIQUE, volet f1) ;
  - E1(p) vs E1(n) : egalite attendue au % (le volet << U_EM egal a
    E_tension pres >> exige 0,14 % -- hors de portee de cette maille,
    declare) ;
  - R_g(|Psi|^2) : rayon d'enveloppe R_n vs R_p (volet f2 : R_n ~ R_p) ;
  - longueurs de brin L (Delta L physique 0,02 % : hors de portee ici,
    le passage local ajoute un biais geometrique declare).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "simulation/experiments")
from theory.schrodinger import (  # noqa: E402
    generate_mesh_multi, assemble_system_multi, solve_eigenproblem)
from simulation.ideal_embedding import linking_number  # noqa: E402
from repair_curves_braid import alexander  # noqa: E402

NAMES = ["3_1", "7_2", "7_2"]


def load_neutron():
    return [np.load("output/nucleon_ideal_%d.npy" % i) for i in range(3)]


def lk_pattern(comps):
    return tuple(abs(linking_number(comps[i], comps[j]))
                 for i, j in ((0, 1), (0, 2), (1, 2)))


def alex_types(comps):
    return [tuple(int(x) for x in alexander(C)[0]) for C in comps]


def make_proton(comps):
    """Passage local de S2 a travers S1 (bosse gaussienne), parametres
    cherches sous gardes : |Lk| -> (1,1,1), types d'Alexander intacts."""
    K, S1, S2 = [C.copy() for C in comps]
    d = np.linalg.norm(S2[:, None, :] - S1[None, :, :], axis=2)
    ranked = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    n2 = len(S2)
    idx = np.arange(n2)
    types0 = alex_types(comps)
    def min_gap(comps):
        g = 1e9
        for a in range(3):
            for b in range(a + 1, 3):
                dd = np.linalg.norm(comps[a][:, None] - comps[b][None, :], axis=2)
                g = min(g, float(dd.min()))
        return g

    best, best_gap, best_lab = None, -1.0, None
    for pt in range(4):                      # 4 sites d'approche les plus proches
        j, i = ranked[pt * 40]
        u = S1[i] - S2[j]
        u /= np.linalg.norm(u)
        ang = np.minimum(np.abs(idx - j), n2 - np.abs(idx - j))
        for width in (6, 8, 10, 12):
            for push in (2.1, 2.6, 3.2, 3.8, 4.5):
                for sg in (+1, -1):
                    bump = np.exp(-(ang / width) ** 2)
                    S2b = S2 + sg * push * bump[:, None] * u[None, :]
                    cand = [K, S1, S2b]
                    if lk_pattern(cand) == (1, 1, 1) and alex_types(cand) == types0:
                        g = min_gap(cand)
                        if g > best_gap:
                            best, best_gap = cand, g
                            best_lab = (pt, width, sg * push)
    if best is None:
        raise RuntimeError("aucun passage (1,1,1) trouve sous gardes")
    print("passage retenu : site %d, largeur %d, pousse %+.1f ; gap = %.2f" % (
        best_lab[0], best_lab[1], best_lab[2], best_gap), flush=True)
    return best


def fem_solve(comps, tag):
    print("[%s] Lk = %s ; types Alexander = %s" % (
        tag, lk_pattern(comps), alex_types(comps)), flush=True)
    L = [float(np.linalg.norm(np.diff(np.vstack([C, C[:1]]), axis=0),
                              axis=1).sum()) for C in comps]
    points, tets, boundary = generate_mesh_multi(comps, n_coarse=26,
                                                 epsilon=0.35, refine=True)
    print("[%s] maillage : %d points, %d tetraedres" % (
        tag, len(points), len(tets)), flush=True)
    H, M = assemble_system_multi(points, tets, boundary, NAMES, comps,
                                 epsilon=0.35)
    vals, vecs = solve_eigenproblem(H, M, n_eigenvalues=4)
    free = np.where(~boundary)[0]
    psi2 = np.zeros(len(points))
    psi2[free] = vecs[:, 0] ** 2
    psi2 /= psi2.sum()
    cg = (points * psi2[:, None]).sum(0)
    rg = float(np.sqrt((psi2 * ((points - cg) ** 2).sum(1)).sum()))
    print("[%s] E = %s" % (tag, np.round(vals, 4)), flush=True)
    print("[%s] R_g(|Psi|^2) = %.3f ; L = %s (total %.1f)" % (
        tag, rg, np.round(L, 1), sum(L)), flush=True)
    return {"E": vals, "Rg": rg, "L": sum(L)}


def prep(iters=3200):
    """Construit le proton et POLIT les deux motifs par sono_link (gardes)."""
    from simulation.ideal_embedding import sono_link
    neutron = load_neutron()
    proton = make_proton(neutron)
    types0 = alex_types(neutron)
    for tag, comps, lk_want in (("neutron", neutron, (1, 1, 0)),
                                ("proton", proton, (1, 1, 1))):
        c2, info = sono_link([C.copy() for C in comps], iters=iters,
                             loosen_start=False)
        ok = lk_pattern(c2) == lk_want and alex_types(c2) == types0
        print("%s poli : rho=%.1f gap=%.3f Lk=%s gardes %s" % (
            tag, info["ropelength"], info["min_gap"], lk_pattern(c2),
            "OK" if ok else "KO"), flush=True)
        if not ok:
            raise RuntimeError("gardes violees pour " + tag)
        for k, C in enumerate(c2):
            np.save("output/fem_%s_%d.npy" % (tag, k), C)
    print("motifs polis sauves (output/fem_*_*.npy)", flush=True)


def run():
    print("=== FEM chi sur l'entrelacs nucleon (etape 1 du chantier f1/f2) ===\n")
    neutron = [np.load("output/fem_neutron_%d.npy" % k) for k in range(3)]
    proton = [np.load("output/fem_proton_%d.npy" % k) for k in range(3)]
    assert lk_pattern(neutron) == (1, 1, 0), lk_pattern(neutron)
    assert lk_pattern(proton) == (1, 1, 1), lk_pattern(proton)
    assert alex_types(proton) == alex_types(neutron)
    print("motifs polis charges (gardes OK)\n", flush=True)

    rn = fem_solve(neutron, "neutron")
    print(flush=True)
    rp = fem_solve(proton, "proton ")

    print("\n=== Bilan (niveau maille ~%) ===")
    gap_n = (rn["E"][1] - rn["E"][0]) / abs(rn["E"][0])
    gap_p = (rp["E"][1] - rp["E"][0]) / abs(rp["E"][0])
    print("f1 (unicite dynamique) : fondamental NON degenere --")
    print("  ecart relatif E2-E1 : neutron %.1f %%, proton %.1f %%" % (
        100 * gap_n, 100 * gap_p))
    print("f1 (egalite) : E1(p)/E1(n) - 1 = %+.2f %%" % (
        100 * (rp["E"][0] / rn["E"][0] - 1)))
    print("f2 (enveloppe) : R_p/R_n - 1 = %+.2f %%" % (100 * (rp["Rg"] / rn["Rg"] - 1)))
    print("f2 (brin) : Delta L (passage local, biais declare) = %+.2f %%" % (
        100 * (rp["L"] / rn["L"] - 1)))
    print("\nLIMITES DECLAREES : maille au niveau du %% -- les cibles fines")
    print("(egalite U_EM a 0,14 %% pres, Delta L 0,02 %%) exigent l'etude de")
    print("convergence (epsilon, n_coarse) et l'embedding proton RELAXE ;")
    print("cette etape etablit le PIPELINE et les verdicts au niveau %%.")


def tighten():
    """Etape 2a : resserrement complet (sono long + rechauffes graduees)."""
    from simulation.ideal_embedding import sono_link, squeeze
    rng = np.random.default_rng(11)
    for tag, lk_want, do_reheat in (("proton", (1, 1, 1), True),
                                    ("neutron", (1, 1, 0), False)):
        comps = [np.load("output/fem_%s_%d.npy" % (tag, k)) for k in range(3)]
        types0 = alex_types(comps)
        best, binfo = sono_link([C.copy() for C in comps], iters=10000,
                                loosen_start=False)
        ok = lk_pattern(best) == lk_want and alex_types(best) == types0
        print("%s sono 10k : rho=%.1f gap=%.3f %s" % (
            tag, binfo["ropelength"], binfo["min_gap"],
            "OK" if ok else "KO"), flush=True)
        if not ok:
            raise RuntimeError("gardes violees (sono long) " + tag)
        best_rho = binfo["ropelength"]
        if do_reheat:
            for s_f in (0.7, 0.8):
                for t in range(2):
                    u = rng.normal(size=3)
                    sq = [squeeze(C, u, s_f) for C in best]
                    c2, i2 = sono_link(sq, iters=6000, loosen_start=False)
                    ok2 = lk_pattern(c2) == lk_want and alex_types(c2) == types0
                    print("  rechauffe s=%.1f essai %d : rho=%.1f %s" % (
                        s_f, t, i2["ropelength"], "OK" if ok2 else "KO"),
                        flush=True)
                    if ok2 and i2["ropelength"] < best_rho - 1e-3:
                        best, best_rho = c2, i2["ropelength"]
        for k, C in enumerate(best):
            np.save("output/fem_%s_tight_%d.npy" % (tag, k), C)
        print("%s resserre : rho = %.1f (sauve)" % (tag, best_rho), flush=True)


def conv():
    """Etape 2b : convergence de maille sur le motif neutron."""
    neutron = [np.load("output/fem_neutron_%d.npy" % k) for k in range(3)]
    res = []
    for n_c, eps in ((22, 0.35), (26, 0.35), (30, 0.35), (26, 0.30), (26, 0.42)):
        points, tets, boundary = generate_mesh_multi(neutron, n_coarse=n_c,
                                                     epsilon=eps, refine=True)
        H, M = assemble_system_multi(points, tets, boundary, NAMES, neutron,
                                     epsilon=eps)
        vals, _ = solve_eigenproblem(H, M, n_eigenvalues=2)
        res.append((n_c, eps, vals[0], vals[1]))
        print("n_coarse=%d eps=%.2f : E1=%.4f E2=%.4f (%d pts)" % (
            n_c, eps, vals[0], vals[1], len(points)), flush=True)
    e1 = [r[2] for r in res[:3]]
    print("\nconvergence maille (eps fixe) : dispersion E1 = %.2f %%" % (
        100 * (max(e1) - min(e1)) / abs(np.mean(e1))))
    e1e = [res[1][2], res[3][2], res[4][2]]
    print("sensibilite a eps (maille fixe) : dispersion E1 = %.2f %%" % (
        100 * (max(e1e) - min(e1e)) / abs(np.mean(e1e))))


def run_tight():
    """Etape 2c : FEM sur les motifs resserres."""
    neutron = [np.load("output/fem_neutron_tight_%d.npy" % k) for k in range(3)]
    proton = [np.load("output/fem_proton_tight_%d.npy" % k) for k in range(3)]
    rn = fem_solve(neutron, "neutron")
    print(flush=True)
    rp = fem_solve(proton, "proton ")
    print("\nE1(p)/E1(n) - 1 = %+.2f %% ; R_p/R_n - 1 = %+.2f %% ;"
          " Delta L = %+.2f %%" % (100 * (rp["E"][0] / rn["E"][0] - 1),
                                   100 * (rp["Rg"] / rn["Rg"] - 1),
                                   100 * (rp["L"] / rn["L"] - 1)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--prep":
        prep()
    elif mode == "--tighten":
        tighten()
    elif mode == "--conv":
        conv()
    elif mode == "--fem-tight":
        run_tight()
    else:
        run()
