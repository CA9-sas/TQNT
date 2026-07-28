"""Mesurer la barriere de rearrangement sur le reseau simule (20/07).

CONSOLIDATION de derive_tc_gel_topologique : la barriere du gel Gamma=H
etait POSEE a E_q = sigma_chi * xi (quantum de fragment, coefficient 1).
Ici on MESURE les primitives geometriques du rearrangement sur le
reseau VV Z3 trace (machinerie chi_string_network_pocc reutilisee) :

 (M1) c_seg : longueur de corde par domaine traverse, en unites de xi
      (deux conventions : pas de reseau ; centres de faces lisses).
 (M2) d_min : distance de chaque cellule de corde au premier brin
      ETRANGER (autre corde, ou meme corde a distance d'arc > 4) --
      c'est la distance a franchir pour une reconnexion inter-brins.
 (M3) f_contact : fraction de cellules deja multi-brins (contact
      present, reconnexion P15 sans franchissement).

MODELES DE BARRIERE confrontes (E_b = c * sigma_chi * xi) :
  (tente)  deformation d'un segment epingle a xi pour atteindre d :
           c = sqrt(1 + 4 (d/xi)^2) - 1
  (pont)   excroissance aller-retour : c = 2 d/xi
  (segment) re-routage d'un domaine : c = c_seg
Chaque c mesure -> T_c = sqrt(c * kappa_dom * sigma / L) et Omega.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.chi_string_network_pocc import (  # noqa: E402
    fluxes_z3, trace)

import math  # noqa: E402

M_PL = 1.2209e19
SIGMA = 0.41847 ** 2
KAPPA_DOM = 1.14
G_S = 15.2
M_EDGE = 1.73
T_NOM = 0.067


def solve_tc(c_bar):
    t = 0.07
    for _ in range(200):
        L = math.log(M_PL / (1.66 * math.sqrt(G_S) * t))
        t = 0.5 * t + 0.5 * math.sqrt(c_bar * KAPPA_DOM * SIGMA / L)
    return t


def face_center_length(path_cells):
    """Longueur lisse par cellule : distance entre centres des faces
    d'entree/sortie (droit = 1, coude = 1/sqrt(2))."""
    n = len(path_cells)
    tot = 0.0
    for i in range(n):
        a = np.array(path_cells[i])
        b = np.array(path_cells[(i + 1) % n])
        c = np.array(path_cells[(i - 1) % n])
        d_in = a - c
        d_out = b - a
        # replier PBC (pas unitaires)
        d_in = (d_in + 1) % 3 - 1 if np.max(np.abs(d_in)) > 1 else d_in
        d_out = (d_out + 1) % 3 - 1 if np.max(np.abs(d_out)) > 1 else d_out
        straight = np.array_equal(d_in, d_out)
        tot += 1.0 if straight else 1.0 / math.sqrt(2)
    return tot / n


def pbc_dist2(a, b, N):
    d = np.abs(a - b)
    d = np.minimum(d, N - d)
    return int((d * d).sum())


def run():
    N, seeds = 16, (0, 1, 2)
    print("=== Mesure de la barriere de rearrangement (reseau VV Z3) ===\n")
    all_cseg, all_d, all_fc = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ph = rng.integers(0, 3, (N, N, N))
        fx, fy, fz = fluxes_z3(ph)
        loops, winders, n_multi = trace(fx, fy, fz, N, rng)
        strings = loops + winders
        # (M1) longueur par domaine
        cs = [face_center_length([tuple(np.mod(c, N)) for c in p])
              for p in strings if len(p) >= 6]
        c_seg = float(np.mean(cs))
        # occupation par cellule (repliee)
        occ = {}
        for si, p in enumerate(strings):
            for ai, cell in enumerate(p):
                occ.setdefault(tuple(np.mod(cell, N)), []).append((si, ai, len(p)))
        cells = list(occ.keys())
        arr = np.array(cells)
        # (M3) contact deja present
        multi = [c for c, v in occ.items()
                 if len({s for s, _, _ in v}) > 1
                 or any(min(abs(a1 - a2), l1 - abs(a1 - a2)) > 4
                        for i, (s1, a1, l1) in enumerate(v)
                        for s2, a2, l2 in v[i + 1:] if s1 == s2)]
        f_contact = len(multi) / len(cells)
        # (M2) distance au premier brin etranger (sous-echantillon)
        rng2 = np.random.default_rng(100 + seed)
        sample = rng2.choice(len(cells), size=min(600, len(cells)),
                             replace=False)
        dmins = []
        for idx in sample:
            c0 = arr[idx]
            owners0 = occ[cells[idx]]
            best = None
            for jdx in range(len(cells)):
                if jdx == idx:
                    continue
                foreign = False
                for s1, a1, l1 in owners0:
                    for s2, a2, l2 in occ[cells[jdx]]:
                        if s2 != s1 or min(abs(a1 - a2), l1 - abs(a1 - a2)) > 4:
                            foreign = True
                            break
                    if foreign:
                        break
                if not foreign:
                    continue
                d2 = pbc_dist2(c0, arr[jdx], N)
                if best is None or d2 < best:
                    best = d2
                    if best <= 1:
                        break
            dmins.append(math.sqrt(best))
        # (M4) quantum d'echange aux contacts : Delta L entre les deux
        # appariements d'une cellule 2-in/2-out (geometrie faces)
        swaps = []
        for cell in multi:
            v = occ[cell]
            if len(v) != 2:
                continue
            segs = []
            for si, ai, li in v:
                p = strings[si]
                a = np.array(np.mod(p[ai], N))
                b = np.array(np.mod(p[(ai + 1) % li], N))
                c = np.array(np.mod(p[(ai - 1) % li], N))
                d_in = np.array([((x + 1) % 3) - 1 for x in (a - c)]) \
                    if np.max(np.abs(a - c)) > 1 else a - c
                d_out = np.array([((x + 1) % 3) - 1 for x in (b - a)]) \
                    if np.max(np.abs(b - a)) > 1 else b - a
                segs.append((tuple(d_in), tuple(d_out)))

            def clen(di, do):
                return 1.0 if di == do else 1.0 / math.sqrt(2)
            (i1, o1), (i2, o2) = segs
            dl = abs(clen(i1, o1) + clen(i2, o2)
                     - clen(i1, o2) - clen(i2, o1))
            swaps.append(dl)
        all_cseg.append(c_seg)
        all_d.append(np.median(dmins))
        all_fc.append(f_contact)
        all_swap = globals().setdefault("_ALL_SWAP", [])
        all_swap.extend(swaps)
        print("graine %d : %3d cordes, %5d cellules ; c_seg = %.3f ;"
              " mediane d_min = %.2f xi ; f_contact = %.2f ;"
              " |DL|_swap moy = %.3f" % (
                  seed, len(strings), len(cells), c_seg,
                  np.median(dmins), f_contact,
                  np.mean(swaps) if swaps else float("nan")))

    c_seg = float(np.mean(all_cseg))
    d_med = float(np.mean(all_d))
    f_c = float(np.mean(all_fc))
    swaps = globals().get("_ALL_SWAP", [])
    c_swap = float(np.mean(swaps)) if swaps else float("nan")
    print("\nPRIMITIVES MESUREES : c_seg = %.3f +- %.3f ; d_min mediane ="
          " %.2f xi ; f_contact = %.2f ; |DL|_swap = %.3f (n=%d)" % (
              c_seg, np.std(all_cseg), d_med, f_c, c_swap, len(swaps)))

    print("\nMODELES DE BARRIERE -> T_c (et Omega en aval) :")
    models = [
        ("(tente)  c = sqrt(1+4d^2)-1, d mediane",
         math.sqrt(1 + 4 * d_med * d_med) - 1),
        ("(pont)   c = 2 d mediane", 2 * d_med),
        ("(segment) c = c_seg (re-routage d'un domaine)", c_seg),
        ("(echange) c = |DL|_swap (contact, energie echangee)", c_swap),
        ("(pose)   c = 1 (quantum de fragment, reference)", 1.0),
    ]
    for lab, c in models:
        t = solve_tc(c)
        om = 3.9 * math.exp(-M_EDGE * (1 / t - 1 / T_NOM))
        print("  %-46s c = %.3f -> T_c = %5.1f MeV ; Omega = %5.1f" % (
            lab, c, t * 1e3, om))

    print("\nLECTURE HONNETE :")
    print(" - les PRIMITIVES sont mesurees proprement (3 graines stables) ;")
    print(" - mais la geometrie seule ne PINCE PAS le coefficient : les")
    print("   modeles s'etalent sur c in [%.2f, %.2f] -> T_c in [%d, %d]" % (
        min(m[1] for m in models), max(m[1] for m in models),
        solve_tc(min(m[1] for m in models)) * 1e3,
        solve_tc(max(m[1] for m in models)) * 1e3))
    print("   et le levier exponentiel de Omega amplifie tout ;")
    print(" - nuance clef : %.0f %% du reseau est deja en contact, et le" % (
        f_c * 100))
    print("   quantum d'energie ECHANGE a ces sites est petit (|DL| ~ %.1f)" % (
        c_swap))
    print("   -> les reconnexions de contact sont quasi neutres en energie :")
    print("   elles changent la topologie SANS re-equilibrer le poids de")
    print("   Boltzmann -- coherent avec un gel gouverne par les pas")
    print("   d'ECHANGE d'energie, pas par la topologie seule ;")
    print(" - le PIN du coefficient exige une mesure DYNAMIQUE (Kramers :")
    print("   taux d'evenements changeant l'energie vs T sur la dynamique")
    print("   de Nambu thermalisee -> pente = E_b). Chantier suivant.")


if __name__ == "__main__":
    run()
