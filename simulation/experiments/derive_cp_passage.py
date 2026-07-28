"""Deriver c_p : geometrie exacte du passage + mesure des contacts (20/07).

c_p = cout d'ouverture d'un passage de brin sur le tube serre, en
unites de sqrt(sigma(T) sigma_chi) D0. Cible (pincement Planck,
etape 9) : c_p = 3,5.

[1] FORMULE EXACTE (etat de transition, modele epingle) : partenaire
    rectiligne (exclusion de rayon D, centre a centre), brin epingle
    par ses contacts voisins a (+-s, -D). Le chemin minimal qui passe
    PAR-DESSUS est tangente-arc-tangente ; la longueur tangente vaut
    exactement s (theoreme de la tangente), donc le SURCOUT est un arc
    pur :
        extra(s) = D * c_p(s),
        c_p(s) = pi + 2 atan(D/s) - 2 atan(s/D).
    Verifications : c_p(D) = pi ; c_p(0+) = 2pi ; c_p(inf) = 0.
    Inversion de la cible : c_p = 3,5  <=>  s = 0,84 D.

[2] MESURE de s : SONO sur le trefle (proxy declare du T(2,23) --
    meme famille (2,q)), detection des zones de contact (approche
    non adjacente < 1,05 D), espacement des zones -> s_mesure.

[3] PROPAGATION : c_p(s_mesure) -> serrage E_pass = T -> T*, x* ->
    Omega (modele a interrupteur, etape 9).
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.ideal_embedding import sono  # noqa: E402
from theory.knot_catalog import knot_curve  # noqa: E402

SIGMA = 0.41847 ** 2
M_EDGE = 1.73
T_TRANS = 0.155
EXPO = 2 * 0.6301
D0_SIG = 0.1084 * SIGMA     # sigma_chi * D0 en GeV
T_REF = 0.067


def c_p_of_s(u):
    """u = s/D."""
    return math.pi + 2 * math.atan(1 / u) - 2 * math.atan(u)


def s_of_cp(cp):
    lo, hi = 1e-3, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if c_p_of_s(mid) > cp:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def serrage(cp):
    """Resout c_p sqrt(sigma(T) sigma_chi) D0 = T -> (T*, x*)."""
    t = 0.05
    for _ in range(200):
        x = max(1e-9, 1 - t / T_TRANS) ** EXPO
        t = 0.5 * t + 0.5 * cp * D0_SIG * math.sqrt(x)
    x = (1 - t / T_TRANS) ** EXPO
    return t, x


def run():
    print("=== Deriver c_p : formule exacte + contacts mesures ===\n")
    print("[1] FORMULE : c_p(s) = pi + 2 atan(D/s) - 2 atan(s/D)")
    for u in (0.25, 0.5, 0.84, 1.0, 1.5, 2.0):
        print("    s = %.2f D -> c_p = %.2f" % (u, c_p_of_s(u)))
    print("    cible c_p = 3,5 <=> s = %.2f D\n" % s_of_cp(3.5))

    print("[2] MESURE des contacts (SONO trefle, proxy du T(2,q)) :")
    P0 = knot_curve("3_1", 160)
    P, info = sono(P0, R=1.0, iters=9000)
    n = len(P)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])[:-1]
    L_tot = arc[-1] + seg[-1]
    D = 2.0
    print("    longueur/D = %.2f (ropelength ideale 16,37 : %+.1f %%)" % (
        L_tot / D, 100 * (L_tot / D / 16.372 - 1)))
    # approche non adjacente par point (separation d'arc > 2D)
    dmin = np.full(n, np.inf)
    for i in range(n):
        d_arc = np.abs(arc - arc[i])
        d_arc = np.minimum(d_arc, L_tot - d_arc)
        mask = d_arc > 2 * D
        if mask.any():
            d = np.linalg.norm(P[mask] - P[i], axis=1)
            dmin[i] = d.min()
    contact = dmin < 1.05 * D
    frac = contact.mean()
    # zones de contact -> ecarts (gaps) entre zones, en longueur d'arc
    idx = np.arange(n)
    runs = []
    in_run = False
    for i in list(idx) + [0]:
        if contact[i] and not in_run:
            start = i
            in_run = True
        elif not contact[i] and in_run:
            runs.append((start, i - 1))
            in_run = False
    gaps = []
    if len(runs) >= 2:
        for k in range(len(runs)):
            end_prev = runs[k][1]
            start_next = runs[(k + 1) % len(runs)][0]
            g = (arc[start_next] - arc[end_prev]) % L_tot
            if 0 < g < L_tot / 2:
                gaps.append(g)
    print("    fraction en contact = %.2f ; zones = %d" % (frac, len(runs)))
    if gaps:
        s_med = np.median(gaps) / 2.0 / D
        print("    ecarts entre zones (arc) : mediane = %.2f D -> s = %.2f D"
              % (np.median(gaps) / D, s_med))
    else:
        s_med = None
        print("    CONTACT QUASI CONTINU : pas d'ecart mesurable ->")
        print("    s -> 0, c_p -> 2 pi = 6,28 (borne du modele epingle)")

    print("\n[2b] CONTACT CONTINU -> modele DECOLLEMENT SOUS PRESSION :")
    print("    le brin decolle sur 2L (libre), enveloppe le partenaire ;")
    print("    cout = extra de chemin (formule [1] avec s = L) + travail")
    print("    contre la pression de contact f = sigma kappa_perp :")
    print("    c_eff = min_u [ c_p(u) + 2 q u ],  q = kappa_perp D,")
    print("    minimum analytique u* = sqrt(2/q - 1).")
    # mesure de q sur le trefle SONO : composante de courbure vers le contact
    T_hat = np.diff(np.vstack([P, P[:1]]), axis=0)
    T_hat /= np.linalg.norm(T_hat, axis=1)[:, None]
    kv = (T_hat - np.roll(T_hat, 1, axis=0))
    kv /= seg[:, None]
    qs = []
    for i in range(n):
        d_arc = np.abs(arc - arc[i])
        d_arc = np.minimum(d_arc, L_tot - d_arc)
        mask = d_arc > 2 * D
        if not mask.any():
            continue
        d = np.linalg.norm(P[mask] - P[i], axis=1)
        j = np.flatnonzero(mask)[np.argmin(d)]
        if d.min() < 1.05 * D:
            n_hat = (P[j] - P[i]) / d.min()
            q_i = max(0.0, float(kv[i] @ n_hat)) * D
            qs.append(q_i)
    qs = np.array(qs)
    q_med = float(np.median(qs))
    q_mean = float(np.mean(qs))
    print("    q mesure (trefle ideal) : mediane %.2f, moyenne %.2f" % (
        q_med, q_mean))
    print("    (controle Fary-Milnor : <kappa> D >= 4 pi D/L = %.2f ;" % (
        4 * math.pi * D / L_tot))
    print("     la composante VERS le contact en est une fraction)")

    def c_eff(q):
        u = math.sqrt(max(1e-9, 2 / q - 1))
        return c_p_of_s(u) + 2 * q * u, u
    ce_med, u_med = c_eff(q_med)
    ce_mean, u_mean = c_eff(q_mean)
    print("    c_eff(q mediane) = %.2f (u* = %.2f) ;"
          " c_eff(q moyenne) = %.2f" % (ce_med, u_med, ce_mean))
    print("    [cible 3,5 <=> q = 0,41]\n")

    print("[2c] LE VRAI NŒUD DU RESERVOIR : q sur T(2,23) ideal (SONO) :")
    from theory.knot_catalog import _torus_knot_curve
    P0b = _torus_knot_curve(2, 23, 420)
    Pb, infob = sono(P0b, R=1.0, iters=9000)
    nb = len(Pb)
    segb = np.linalg.norm(np.diff(np.vstack([Pb, Pb[:1]]), axis=0), axis=1)
    arcb = np.concatenate([[0], np.cumsum(segb)])[:-1]
    Lb = arcb[-1] + segb[-1]
    print("    longueur/D = %.1f (mesure historique 90,21 : %+.1f %%)" % (
        Lb / D, 100 * (Lb / D / 90.21 - 1)))
    Tb = np.diff(np.vstack([Pb, Pb[:1]]), axis=0)
    Tb /= np.linalg.norm(Tb, axis=1)[:, None]
    kvb = (Tb - np.roll(Tb, 1, axis=0)) / segb[:, None]
    qsb = []
    for i in range(nb):
        d_arc = np.abs(arcb - arcb[i])
        d_arc = np.minimum(d_arc, Lb - d_arc)
        mask = d_arc > 2 * D
        d = np.linalg.norm(Pb[mask] - Pb[i], axis=1)
        j = np.flatnonzero(mask)[np.argmin(d)]
        if d.min() < 1.05 * D:
            n_hat = (Pb[j] - Pb[i]) / d.min()
            qsb.append(max(0.0, float(kvb[i] @ n_hat)) * D)
    qsb = np.array(qsb)
    print("    fraction en contact = %.2f ; q : mediane %.2f,"
          " moyenne %.2f, Q1 %.2f" % (
              len(qsb) / nb, np.median(qsb), qsb.mean(),
              np.percentile(qsb, 25)))
    ce_23, u_23 = c_eff(float(np.median(qsb)))
    print("    c_eff(T(2,23), q mediane) = %.2f (u* = %.2f)"
          " [cible 3,5 <=> q = 0,41]\n" % (ce_23, u_23))

    print("[3] PROPAGATION (serrage E_pass = T, interrupteur) :")
    cands = [("cible (Planck)", 3.5),
             ("MESURE T(2,23) (reservoir)", ce_23),
             ("MESURE trefle (proxy)", ce_med)]
    if s_med:
        cands.append(("mesure s = %.2f D" % s_med, c_p_of_s(s_med)))
    cands.append(("borne contact continu (2 pi)", 2 * math.pi))
    cands.append(("borne s = D (pi)", math.pi))
    w_ref = math.exp(-M_EDGE / T_REF)
    for lab, cp in cands:
        t_s, x_s = serrage(cp)
        m_t = M_EDGE * math.sqrt(x_s)
        om = 3.9 * math.exp(-m_t / t_s) / w_ref
        print("    %-24s c_p = %.2f -> T* = %4.1f MeV, x* = %.2f,"
              " Omega = %8.2f" % (lab, cp, t_s * 1e3, x_s, om))


if __name__ == "__main__":
    run()
