"""Pincer Q : statistique complete des temps de scission (20/07).

Q est le seuil de qualite du critere de fragmentation (gamma =
omega/Q) : combien de periodes un fragment doit-il osciller librement
avant sa premiere scission. L'ancre historique etait la MEDIANE de la
generation 0 (t* = 0,40 periode, consolidation kappa_s). Ici : la
distribution COMPLETE sur la cascade branchee (tous fragments, toutes
generations), t* = (t_scission - t_naissance)/(L/2) (periode de Nambu
d'une boucle de longueur L : L/2).

Q_mesure = 2 pi x mediane(t*) ; propagation a T_form via la boite
(epsilon, Q) pour epsilon = 1 et 2.
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.departage_f_frac import (  # noqa: E402
    first_crossing_from, P_TWIST, DSIG_MIN, MAX_EV)
from simulation.experiments.derive_lambda_kinks import (  # noqa: E402
    Poly, embedded_fragments)

SIGMA = 0.41847 ** 2
KAPPA = 1.14
D_TUBE = 0.1084
ZETA3 = 1.2020569
N_ALL_T3 = (ZETA3 / math.pi ** 2) * (2 + 0.75 * (6 + 4 + 1.2))


def cascade_times(path, rng):
    """Cascade branchee instrumentee : liste des t* de scission
    (temps depuis la naissance du fragment, en periodes L/2)."""
    V = np.asarray(path, float)
    V = np.vstack([V, V[:1]])
    tstars = []   # (t*, generation)
    stack = [(Poly(V, np.zeros(3)), Poly(V, np.zeros(3)), 0.0, 0.0, [], 0)]
    n_ev = 0
    while stack:
        pa, pb, t0, t_birth, zones, gen = stack.pop()
        Lf = pa.L
        if Lf < 2 * DSIG_MIN + 0.05 or n_ev >= MAX_EV:
            continue
        hit = first_crossing_from(pa, pb, Lf, t0, zones)
        if hit is None:
            continue
        n_ev += 1
        t, s1, s2 = hit
        if rng.random() < P_TWIST:
            stack.append((pa, pb, t, t_birth, zones + [(t, s1, s2)], gen))
        else:
            tstars.append(((t - t_birth) / (Lf / 2.0), gen))
            if s2 < s1:
                s1, s2 = s2, s1
            for (u1, u2) in ((s1, s2), (s2, s1 + Lf)):
                Va = pa.slice(u1 - t, u2 - t)
                Vb = pb.slice(u1 + t, u2 + t)
                da = Poly(Va, Va[-1] - Va[0])
                db = Poly(Vb, Vb[-1] - Vb[0])
                stack.append((da, db, 0.0, 0.0, [], gen + 1))
    return tstars


def t_form(eps, q):
    t = 0.07
    for _ in range(80):
        beta = eps * N_ALL_T3 * math.log((KAPPA / t) / D_TUBE)
        t = 0.5 * t + 0.5 * math.sqrt(SIGMA / (q * KAPPA * beta))
    return t


def run():
    print("=== Pin de Q : statistique complete des t* de scission ===\n")
    all_t = []
    for seeds, rs in (((42, 43), 2026), ((44, 45), 3026)):
        rng = np.random.default_rng(rs)
        parents = []
        for seed in seeds:
            r2 = np.random.default_rng(7 + seed)
            ps = embedded_fragments(N=16, seed=seed, n_sample=30, rng=r2)
            parents += [p for p in ps if 6 <= len(p) <= 80]
        parents = parents[:24]
        batch_t = []
        for p in parents:
            batch_t += cascade_times(p, rng)
        all_t += batch_t
        print("lot graines %s : %d scissions" % (seeds, len(batch_t)))

    gen0 = np.array([t for t, g in all_t if g == 0])
    t_all = np.array([t for t, g in all_t])
    # avalanche des filles : t* ~ 0 = amas simultanes du MEME contact
    # (identifie et exclu par la consolidation kappa_s) -- filtre
    clean = np.array([t for t, g in all_t if t > 0.05])
    f_aval = 1 - len(clean) / len(t_all)
    print("\nCOMBINE : %d scissions ; brut mediane = %.3f (domine par" % (
        len(t_all), np.median(t_all)))
    print("   l'avalanche des filles : %.0f %% a t* < 0,05 = amas du" % (
        f_aval * 100))
    print("   meme contact, exclus par la consolidation kappa_s)")
    print("   generation 0 (propre)   : n = %d, mediane t* = %.3f" % (
        len(gen0), np.median(gen0)))
    q1, med, q3 = np.percentile(clean, (25, 50, 75))
    print("   toutes gen., amas exclus : n = %d, quartiles"
          " [%.3f, %.3f, %.3f]" % (len(clean), q1, med, q3))
    print("   (ancre historique gen 0 = 0,40 : %s)" % (
        "OK" if abs(np.median(gen0) - 0.40) < 0.15 else "ECART"))
    q_med = 2 * math.pi * med
    print("   Q(amas exclus) = 2 pi x %.3f = %.2f ; Q(gen 0) = %.2f\n" % (
        med, q_med, 2 * math.pi * np.median(gen0)))

    print("PROPAGATION (boite epsilon x Q ; pincement Planck : eQ = 4,94) :")
    for eps in (1.0, 2.0):
        t = t_form(eps, q_med)
        print("   epsilon = %.0f : e*Q = %.2f -> T_form = %.0f MeV" % (
            eps, eps * q_med, t * 1e3))
    print("\nLECTURE : Q est desormais MESURE (distribution complete) ;")
    print("le seul O(1) restant est epsilon (cinetique quantique de")
    print("l'echange d'ondelette). epsilon requis par le pincement :")
    print("   epsilon = %.2f" % (4.94 / q_med))


if __name__ == "__main__":
    run()
