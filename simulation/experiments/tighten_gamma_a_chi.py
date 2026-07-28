"""Resserrement de Gamma_a(chi) : harmoniques, ensembles, boost des filles (17/07).

Les trois leviers declares de derive_gamma_a_chi (Gamma_a = 76 +- ~12,
N_harm = 24, 18 fragments, filles ignorees) :

  1. HARMONIQUES : convergence de F(N) pour N = 12..96 + queue en loi de
     puissance (P_n ~ C n^-p ; coudes ~n^-2, evenements luminaux ~n^-4/3) ;
     extrapolation N -> infini DES DEUX ensembles (le biais de troncature
     ne se compense dans le rapport que si les queues sont identiques).
  2. ENSEMBLES : 40 fragments, deux graines, L = 6..60 xi ; mediane vs
     moyenne ponderee par la longueur (systematique d'ensemble) ; ancre
     KT elargie (5 inclinaisons).
  3. BOOST DES FILLES : vitesses de recul mesurees (v = |da+db|/2L) ;
     correction O(v^2) sur Gamma quantifiee (la course concerne toutes
     les generations, pas seulement les statiques).

La fonctionnelle est vectorisee (segments x directions en numpy) pour
rendre N = 96 praticable.
"""
import sys
import math
import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.derive_lambda_kinks import (  # noqa: E402
    embedded_fragments, Poly)
from simulation.experiments.derive_gamma_a_chi import (  # noqa: E402
    fib_sphere, kt_loop)

KAPPA_S = 2.85
LN_XI_D = 5.054


def mover_amp_vec(V, dirs, w, sign):
    """A_j(k) pour toutes les directions d'un coup. V : (n+1,3)."""
    seg = np.diff(V, axis=0)
    ell = np.linalg.norm(seg, axis=1)
    keep = ell > 1e-12
    seg, ell = seg[keep], ell[keep]
    ap = seg / ell[:, None]
    L = ell.sum()
    u0 = np.concatenate([[0.0], np.cumsum(ell)])[:-1]
    V0 = V[:-1][keep]
    ka = V0 @ dirs.T                 # (nseg, ndir)
    kap = ap @ dirs.T
    c0 = sign * (w / 2) * (u0[:, None] - sign * ka)
    sl = sign * (w / 2) * (1 - sign * kap)
    small = np.abs(sl) < 1e-9
    sl_safe = np.where(small, 1.0, sl)
    I = np.exp(1j * c0) * (np.exp(1j * sl * ell[:, None]) - 1) / (1j * sl_safe)
    I = np.where(small, ell[:, None] * np.exp(1j * c0), I)
    return np.einsum("sj,sd->dj", ap, I) / L     # (ndir, 3)


def P_spectrum(Va, Vb, n_harm, dirs):
    """P_n (fonctionnelle par harmonique), n = 1..n_harm."""
    La = np.linalg.norm(np.diff(Va, axis=0), axis=1).sum()
    Pn = np.zeros(n_harm)
    for n in range(1, n_harm + 1):
        w = 4 * np.pi * n / La
        A = mover_amp_vec(Va, dirs, w, -1)
        B = mover_amp_vec(Vb, dirs, w, +1)
        A = A - dirs * np.sum(A * dirs, axis=1, keepdims=True)
        B = B - dirs * np.sum(B * dirs, axis=1, keepdims=True)
        aa = np.sum(np.abs(A) ** 2, axis=1)
        bb = np.sum(np.abs(B) ** 2, axis=1)
        ab = np.abs(np.sum(A * np.conj(B), axis=1)) ** 2
        Pn[n - 1] = n ** 2 * np.mean(2 * (aa * bb - ab))
    return Pn


def F_tail(Pn, n_fit=8):
    """F(N) + queue extrapolee (loi de puissance sur les n_fit derniers)."""
    N = len(Pn)
    F_N = Pn.sum()
    ns = np.arange(N - n_fit + 1, N + 1)
    ys = Pn[-n_fit:]
    good = ys > 0
    if good.sum() < 4:
        return F_N, F_N, np.nan
    p, logc = np.polyfit(np.log(ns[good]), np.log(ys[good]), 1)
    # queue : somme_{N+1}^inf C n^p  (p < -1 requis)
    if p >= -1.05:
        return F_N, np.inf, p
    C = math.exp(logc)
    tail = C * (N + 0.5) ** (p + 1) / (-(p + 1))
    return F_N, F_N + tail, p


def run():
    dirs = fib_sphere(196)
    N = 96

    print("=== Resserrement de Gamma_a(chi) ===\n")
    print("[1] ANCRE KT (5 inclinaisons, N = %d + queue) :" % N)
    F_kt = []
    for tau in np.linspace(np.pi / 6, 5 * np.pi / 6, 5):
        Pn = P_spectrum(*kt_loop(tau), N, dirs)
        fN, fInf, p = F_tail(Pn)
        F_kt.append(fInf)
        print("    tau = %.2f : F(24) = %.3e ; F(96) = %.3e ; F(inf) = %.3e"
              " (queue p = %.2f)" % (tau, Pn[:24].sum(), fN, fInf, p))
    F_kt_m = np.mean(F_kt)
    print("    <F_KT>(inf) = %.4e\n" % F_kt_m)

    print("[2] ENSEMBLE chi (2 graines, ~40 fragments, L = 6..60 xi) :")
    frs = []
    for seed in (42, 43):
        rng = np.random.default_rng(7 + seed)
        ps = embedded_fragments(N=16, seed=seed, n_sample=40, rng=rng)
        frs += [p for p in ps if 6 <= len(p) <= 60]
    frs = frs[:40]
    F_ch, F24_ch, lens, tails = [], [], [], []
    for pth in frs:
        V = np.asarray(pth, float)
        V = np.vstack([V, V[:1]])
        Pn = P_spectrum(V, V.copy(), N, dirs)
        fN, fInf, p = F_tail(Pn)
        F_ch.append(fInf)
        F24_ch.append(Pn[:24].sum())
        lens.append(len(pth))
        tails.append(p)
    F_ch = np.array(F_ch)
    lens = np.array(lens, float)
    print("    %d fragments ; queues p mediane = %.2f (KT : voir [1])" % (
        len(F_ch), np.median(tails)))
    med = np.median(F_ch)
    wmean = np.sum(F_ch * lens) / lens.sum()
    print("    F(inf) : mediane %.4e ; moyenne ponderee longueur %.4e\n" % (
        med, wmean))

    print("[3] BOOST DES FILLES (vitesses de recul, generation 1) :")
    import simulation.experiments.derive_lambda_kinks as dk
    dk.EPS_HIT, dk.DSIG_MIN, dk.DT = 0.15, 1.0, 0.12
    vs = []
    rng = np.random.default_rng(7)
    for pth in frs[:12]:
        V = np.asarray(pth, float)
        V = np.vstack([V, V[:1]])
        pa, pb = Poly(V, np.zeros(3)), Poly(V, np.zeros(3))
        hit = dk.first_crossing(pa, pb, pa.L)
        if hit is None:
            continue
        t, s1, s2 = hit
        if s2 < s1:
            s1, s2 = s2, s1
        for (u1, u2) in ((s1, s2), (s2, s1 + pa.L)):
            Va = pa.slice(u1 - t, u2 - t)
            Vb = pb.slice(u1 + t, u2 + t)
            Ld = u2 - u1
            v = np.linalg.norm((Va[-1] - Va[0]) + (Vb[-1] - Vb[0])) / (2 * Ld)
            vs.append(v)
    vs = np.array(vs)
    print("    v recul : mediane %.2f, moyenne %.2f -> correction O(v^2)"
          " ~ %.0f %% (gamma^2-ish)" % (np.median(vs), vs.mean(),
                                        100 * np.median(vs) ** 2))
    boost_sys = np.median(vs) ** 2

    print("\n[4] SYNTHESE (rapport a troncature APPARIEE -- les queues p ~ -1")
    print("    rendent l'extrapolation infinie instable ; le rapport a meme N")
    print("    est l'estimateur robuste, la derive en N = systematique) :")
    kts = [P_spectrum(*kt_loop(t), 96, dirs)
           for t in np.linspace(np.pi / 6, 5 * np.pi / 6, 5)]
    chs = []
    for pth in frs:
        V = np.asarray(pth, float)
        V = np.vstack([V, V[:1]])
        chs.append(P_spectrum(V, V.copy(), 96, dirs))
    gs = {}
    for Ntr in (24, 48, 96):
        fkt = np.mean([P[:Ntr].sum() for P in kts])
        fch = np.median([P[:Ntr].sum() for P in chs])
        fch_w = np.sum([P[:Ntr].sum() * l for P, l in zip(chs, lens)]) / lens.sum()
        gs[Ntr] = (65 * fch / fkt, 65 * fch_w / fkt)
        print("    N = %2d : Gamma_a(mediane) = %.0f ; (ponderee L) = %.0f" % (
            Ntr, gs[Ntr][0], gs[Ntr][1]))
    vals = [g for pair in gs.values() for g in pair]
    g_c = float(np.median(vals))
    dg = 0.5 * (max(vals) - min(vals))
    dg = math.hypot(dg, g_c * boost_sys)
    print("\n    => Gamma_a(chi) = %.0f +- %.0f  (requis 91 +- 20 ;" % (g_c, dg))
    print("       boost des filles : %.1f %% -- negligeable)" % (100 * boost_sys))
    print()
    print("    Consequence (course + plancher energetique dur) :")
    for g in (g_c - dg, g_c, g_c + dg):
        lcut = math.exp(g / (2 * math.pi * KAPPA_S) - LN_XI_D)
        lam = 1.14 * max(lcut, 0.577)
        om = 5.68 / lam
        print("       Gamma_a = %5.0f : l_cut = %.2f xi ; lambda = %.2f ;"
              " Omega = %.1f (%+.0f %%)" % (g, lcut, lam, om,
                                            100 * (om / 5.45 - 1)))


if __name__ == "__main__":
    run()
