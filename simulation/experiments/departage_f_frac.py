"""Departager f_frac : la cascade de Nambu COMPLETE avec branchement (17/07).

QUESTION : f_frac = 0.447 (arbre de Poisson) ou 0.28 (MC de lignee sur le
modele de froissement) ? Les deux etaient des modeles de l'arbre ; ici,
l'arbre de lignee REEL est simule : la dynamique de Nambu exacte (celle
de derive_lambda_kinks), avec a chaque contact le branchement DERIVE :

    scission (reconnexion)  : p = 1/(1 + 2e^-alpha) = 0.628
    passage (DeltaLk = +-1) : p = 2e^-alpha/(1 + 2e^-alpha) = 0.372

Mecanique des passages : la trajectoire de Nambu continue inchangee (les
brins se traversent) ; le n\oe ud forme est LOCALISE en sigma (position du
croisement) et suit son fragment aux scissions ; une zone d'exclusion
(memes arcs, ~2 xi de temps) empeche la re-detection du meme contact.

PLANCHER : DSIG_MIN = 0.577 xi (les deux filles peuvent payer m(23)) ;
les fragments < 2 x 0.577 sont terminaux (les contacts sous le plancher
font des n\oe uds legers, secteur visible -- declare).

SORTIE DIRECTE : n\oe uds (fragments finals a >= 1 passage) par xi de
corde ->  Omega = 14.3 x (8/9) x (n_knot / L_tot), sans passer par la
factorisation f_frac/lambda.
"""
import sys
import math
import numpy as np

sys.path.insert(0, ".")
import simulation.experiments.derive_lambda_kinks as dk  # noqa: E402
from simulation.experiments.derive_lambda_kinks import (  # noqa: E402
    Poly, embedded_fragments)

ALPHA_COND = 1.217
P_TWIST = 2 * math.exp(-ALPHA_COND) / (1 + 2 * math.exp(-ALPHA_COND))
EPS_HIT = 0.12
DT = 0.10
DSIG_MIN = 0.577
T_ZONE = 2.0
Z_R = 1.0
MAX_EV = 300


def circ(a, b, L):
    d = np.abs(a - b)
    return np.minimum(d, L - d)


def first_crossing_from(pa, pb, Lf, t0, zones):
    m = max(24, int(2.2 * Lf))
    sig = np.arange(m) * (Lf / m)
    dsig = circ(sig[:, None], sig[None, :], Lf)
    ok = (dsig > DSIG_MIN) & (Lf - dsig > DSIG_MIN)
    iu = np.triu_indices(m, 1)
    base_mask = ok[iu]
    if not base_mask.any():
        return None
    si, sj = sig[iu[0]], sig[iu[1]]
    for t in np.arange(t0, t0 + Lf + DT, DT):
        mask = base_mask.copy()
        for (t_ev, z1, z2) in zones:
            if t - t_ev < T_ZONE:
                near = ((circ(si, z1, Lf) < Z_R) & (circ(sj, z2, Lf) < Z_R)) | \
                       ((circ(si, z2, Lf) < Z_R) & (circ(sj, z1, Lf) < Z_R))
                mask &= ~near
        if not mask.any():
            continue
        X = 0.5 * (pa.eval(sig - t) + pb.eval(sig + t))
        d2 = np.sum((X[iu[0]] - X[iu[1]]) ** 2, axis=1)
        d2[~mask] = np.inf
        j = np.argmin(d2)
        if d2[j] < EPS_HIT ** 2:
            return t, si[j], sj[j]
    return None


def cascade_branchee(path, rng):
    """Cascade complete scission/passage ; retourne [(L, n_passages)]."""
    V = np.asarray(path, float)
    V = np.vstack([V, V[:1]])
    finals = []
    stack = [(Poly(V, np.zeros(3)), Poly(V, np.zeros(3)), 0.0, [], [])]
    n_ev = 0
    while stack:
        pa, pb, t0, tws, zones = stack.pop()
        Lf = pa.L
        if Lf < 2 * DSIG_MIN + 0.05 or n_ev >= MAX_EV:
            finals.append((Lf, len(tws)))
            continue
        hit = first_crossing_from(pa, pb, Lf, t0, zones)
        if hit is None:
            finals.append((Lf, len(tws)))
            continue
        n_ev += 1
        t, s1, s2 = hit
        if rng.random() < P_TWIST:
            pos = (s1 + s2) / 2.0 if abs(s2 - s1) < Lf / 2 else \
                ((s1 + s2 + Lf) / 2.0) % Lf
            stack.append((pa, pb, t, tws + [pos], zones + [(t, s1, s2)]))
        else:
            if s2 < s1:
                s1, s2 = s2, s1
            for (u1, u2) in ((s1, s2), (s2, s1 + Lf)):
                Va = pa.slice(u1 - t, u2 - t)
                Vb = pb.slice(u1 + t, u2 + t)
                # partition des passages par position d'arc
                sub = []
                for p_ in tws:
                    pp = p_ if p_ >= u1 else p_ + Lf
                    if u1 <= pp < u2:
                        sub.append(pp - u1)
                da = Poly(Va, Va[-1] - Va[0])
                db = Poly(Vb, Vb[-1] - Vb[0])
                stack.append((da, db, 0.0, sub, []))
                tws = [p_ for p_ in tws
                       if not (u1 <= (p_ if p_ >= u1 else p_ + Lf) < u2)]
    return finals


def run():
    print("=== Departage de f_frac : cascade de Nambu branchee ===")
    print("    (P_twist = %.3f derive ; plancher 0.577 xi ; zones" % P_TWIST)
    print("     d'exclusion apres passage)\n")
    rng = np.random.default_rng(2026)
    parents = []
    for seed in (42, 43):
        r2 = np.random.default_rng(7 + seed)
        ps = embedded_fragments(N=16, seed=seed, n_sample=30, rng=r2)
        parents += [p for p in ps if 6 <= len(p) <= 80]
    parents = parents[:24]
    L_tot = float(sum(len(p) for p in parents))
    print("parents : %d fragments plonges, L_tot = %.0f xi" % (
        len(parents), L_tot))

    n_fin = n_knot = 0
    len_fin = 0.0
    tw_counts = []
    for k, p in enumerate(parents):
        fins = cascade_branchee(p, rng)
        for (Lf, ntw) in fins:
            n_fin += 1
            len_fin += Lf
            tw_counts.append(ntw)
            if ntw >= 1:
                n_knot += 1
    tw = np.array(tw_counts)
    lam = len_fin / n_fin
    f_frac = n_knot / n_fin
    kpx = n_knot / L_tot
    om = 14.3 * (8 / 9) * kpx
    print("\nfragments finals : %d ; <L> = %.2f xi (lambda attendu 0.66-1.3)" % (
        n_fin, lam))
    print("passages/fragment : moyenne %.3f (arbre Poisson : 0.593)" % tw.mean())
    print("f_frac (>= 1 passage) : %.3f   [Poisson 0.447 ; MC lignee 0.28]" % f_frac)
    print("surdispersion : P(0) = %.3f vs Poisson e^-mu = %.3f" % (
        (tw == 0).mean(), math.exp(-tw.mean())))
    print("\nSORTIE DIRECTE : n_knot/L_tot = %.3f/xi" % kpx)
    print("  Omega = 14.3 x (8/9) x %.3f = %.2f  (Planck 5.45 : %+.0f %%)" % (
        kpx, om, 100 * (om / 5.45 - 1)))


if __name__ == "__main__":
    run()
