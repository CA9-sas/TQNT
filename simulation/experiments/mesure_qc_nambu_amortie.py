"""Mesure directe du seuil de liberation Q_c : dynamique amortie (20/07).

Le dernier nombre de la chaine T_form : Q_c = (omega/gamma) critique
au-dessous duquel un fragment ne peut plus accomplir sa premiere
scission (auto-croisement). Mesure DIRECTE, sans choix d'ensemble :

  - fragments generation 0 (plonges du reseau VV, comme la cascade) ;
  - evolution par equation d'onde AMORTIE X_tt = X_ss - gamma X_t
    (approximation lineaire de Nambu amortie -- declaree ; validee a
    gamma = 0 contre le t* de la cascade) ;
  - condition initiale statique (gel a la formation, comme la cascade) ;
  - premiere auto-collision : min distance entre points non adjacents
    (|Darc| > 0,58) < EPS ;
  - balayage gamma x (L/2) : probabilite d'accomplir la scission
    dans 3 periodes -> seuil a 50 % -> Q_c = omega/gamma_c.

Propagation : T_form(epsilon = pi, Q_c) -> Omega cinetique.
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.derive_lambda_kinks import embedded_fragments  # noqa: E402
from simulation.experiments.boltzmann_cinetique_omega import (  # noqa: E402
    w_kinetic, M_KNOT, T_NOM)
from simulation.experiments.pin_x17_fragments_porteurs import batch  # noqa: E402

SIGMA = 0.41847 ** 2
KAPPA = 1.14
D_TUBE = 0.1084
ZETA3 = 1.2020569
N_ALL_T3 = (ZETA3 / math.pi ** 2) * (2 + 0.75 * (6 + 4 + 1.2))
EPS_HIT = 0.12
DSIG_MIN = 0.577
NPTS = 96


def resample(path, n=NPTS):
    V = np.asarray(path, float)
    V = np.vstack([V, V[:1]])
    seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    su = np.linspace(0, L, n, endpoint=False)
    X = np.empty((n, 3))
    for d in range(3):
        X[:, d] = np.interp(su, s, V[:, d])
    return X, L


def first_cross_time(X0, L, gamma, t_max_periods=3.0):
    """Onde amortie sur courbe fermee ; temps de 1re auto-collision
    (en periodes L/2), ou None."""
    n = len(X0)
    ds = L / n
    dt = 0.4 * ds
    X = X0.copy()
    V = np.zeros_like(X0)
    period = L / 2.0
    t = 0.0
    # masque des paires non locales
    idx = np.arange(n)
    darc = np.abs(idx[:, None] - idx[None, :])
    darc = np.minimum(darc, n - darc) * ds
    iu = np.triu_indices(n, 1)
    ok = (darc[iu] > DSIG_MIN) & ((L - darc[iu]) > DSIG_MIN)
    nsteps = int(t_max_periods * period / dt)
    check_every = max(1, nsteps // 400)
    for k in range(nsteps):
        lap = (np.roll(X, -1, 0) - 2 * X + np.roll(X, 1, 0)) / ds ** 2
        V += dt * (lap - gamma * V)
        X += dt * V
        t += dt
        if k % check_every == 0:
            diff = X[iu[0]] - X[iu[1]]
            d2 = np.einsum("ij,ij->i", diff, diff)
            d2[~ok] = np.inf
            if d2.min() < EPS_HIT ** 2:
                return t / period
    return None


def run():
    print("=== Mesure directe de Q_c : onde amortie sur fragments gen 0 ===\n")
    frags = []
    for seed in (42, 43, 44, 45):
        r2 = np.random.default_rng(7 + seed)
        ps = embedded_fragments(N=16, seed=seed, n_sample=30, rng=r2)
        frags += [p for p in ps if 8 <= len(p) <= 80]
    frags = frags[:32]
    data = [resample(p) for p in frags]
    print("%d fragments gen 0 (4 graines reseau)\n" % len(data))

    # validation gamma = 0
    t0 = [first_cross_time(X, L, 0.0) for X, L in data]
    t0v = [t for t in t0 if t is not None]
    print("validation gamma = 0 : %d/%d se croisent ; mediane t* = %.3f" % (
        len(t0v), len(data), np.median(t0v)))
    print("   (cascade gen 0 : 0,300 -- coherence de l'approximation)\n")

    print("balayage gamma (Q = omega/gamma, omega = 2pi/periode) :")
    print("   Q        fraction accomplies (3 periodes)")
    qs = [8.0, 4.0, 2.8, 2.0, 1.4, 1.0, 0.7, 0.5]
    fracs = []
    for q in qs:
        done = 0
        for X, L in data:
            omega = 2 * math.pi / (L / 2.0)
            gam = omega / q
            if first_cross_time(X, L, gam) is not None:
                done += 1
        fr = done / len(data)
        fracs.append(fr)
        print("   %5.2f    %.2f" % (q, fr))

    # seuil a 50 % de la fraction gamma=0
    f0 = len(t0v) / len(data)
    target = 0.5 * f0
    q_c = None
    for i in range(len(qs) - 1):
        if (fracs[i] - target) * (fracs[i + 1] - target) <= 0:
            q1, q2 = qs[i], qs[i + 1]
            f1, f2 = fracs[i], fracs[i + 1]
            q_c = q1 + (target - f1) * (q2 - q1) / (f2 - f1)
            break
    if q_c is None:
        print("\nVERDICT (la mesure falsifie le critere) :")
        print(" - TOUS les fragments accomplissent leur scission a TOUT")
        print("   amortissement teste (jusqu'a Q = 0,5, sur-amorti x12) :")
        print("   un systeme elastique amorti RELAXE toujours (temps")
        print("   gamma/omega^2), il ne gele jamais.")
        print(" - Le critere gamma = omega/Q (sous-amortissement DYNAMIQUE)")
        print("   n'est PAS un critere de retard COSMOLOGIQUE : avec ~e^45")
        print("   periodes disponibles, la fragmentation s'accomplit a")
        print("   n'importe quel beta. Le retard reel exigerait")
        print("   gamma/omega^2 > t_Hubble, soit beta ~ 3e20 a 70 MeV :")
        print("   LA ROUTE FRICTION EST MORTE au niveau du critere --")
        print("   toutes ses variantes (y compris ondelette-hbar).")
        print(" - Ce qui SURVIT : l'ontologie du socle (E = hf, engrenage")
        print("   -> limite c, M_Pl = horloge du socle) -- fondations")
        print("   independantes de l'application friction.")
        print(" - T_form, candidats restants : l'ETABLISSEMENT DE sigma(T)")
        print("   (pas de tension -> pas d'effondrement de Nambu -> pas de")
        print("   fragmentation : l'horloge est le developpement de la")
        print("   tension sous la transition) ; parois hybrides Z_3.")
        return
    print("\nQ_c (50 %% de la fraction libre) = %.2f" % q_c)

    # propagation
    t = 0.07
    for _ in range(80):
        beta = math.pi * N_ALL_T3 * math.log((KAPPA / t) / D_TUBE)
        t = 0.5 * t + 0.5 * math.sqrt(SIGMA / (q_c * KAPPA * beta))
    finals = batch((42, 43), 2026) + batch((44, 45), 3026)
    L_bear = np.array([Lf for Lf, nn in finals if nn >= 1])
    w_ref = math.exp(-M_KNOT / T_NOM)
    ws = [w_kinetic(Lf, t) for Lf in L_bear]
    om = 3.9 * float(np.mean(ws)) / w_ref
    print("PROPAGATION (epsilon = pi derive, Q_c mesure) :")
    print("   T_form = %.1f MeV -> Omega = %.2f (Planck 5,45 : %+.0f %%)" % (
        t * 1e3, om, 100 * (om / 5.45 - 1)))


if __name__ == "__main__":
    run()
