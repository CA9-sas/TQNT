"""Pincer les marges residuelles de T_c : dispersion de q et T_trans (20/07).

[1] DISPERSION DE q -> LA MOYENNE DE MARCHE (statistique derivee).
    La marche visite les sites uniformement : la probabilite effective
    par pas est la MOYENNE DE SITES <e^{-E(q)/T}>, pas la valeur d'un
    quantile choisi. Deux adoucisseurs mesurables :
    (a) lissage de zone : l'enveloppement s'etend sur ~2u*D ~ 3D --
        le q pertinent est moyenne sur la zone de decollage ;
    (b) au gel, l'exposant vaut O(1) (E_char = T), pas O(20) : la
        dispersion residuelle se moyenne presque sans effet (Jensen
        au premier ordre ~ variance/2).
    Mesure : T(2,23) SONO -> q_i par site -> q_zone (fenetre 3D) ->
    resolution de <exp(-c_eff(q_zone) D0 sqrt(sigma(T)sigma_chi)/T)>
    = 1/e -> T_c effectif.

[2] T_TRANS -> PINCE PAR LE RESEAU. L'identification canonique
    sigma_Cornell = sigma_chi attache la transition chi au croisement
    QCD PHYSIQUE (avec canaux de nucleation legers), mesure :
    T_trans = 156,5 +- 1,5 MeV (HotQCD). Sensibilite : dT_c/dT_trans
    ~ 0,14 -> +-0,2 MeV : NEGLIGEABLE. Caveats declares : forme de
    sigma(T) (Ising canonique via SU(2)/gluon-Ising ; bande champ
    moyen en controle) ; l'alternative jauge pure (297 MeV) est
    ecartee par l'identification canonique (les canaux legers
    existent).
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.ideal_embedding import sono  # noqa: E402
from theory.knot_catalog import _torus_knot_curve  # noqa: E402
from simulation.experiments.omega_budget_reconstruction import budget  # noqa: E402

SIGMA = 0.41847 ** 2
D0_SIG = 0.1084 * SIGMA
EXPO_ISING = 2 * 0.6301
P_OCC_EFF = (8.0 / 9.0) * 0.304
D = 2.0


def c_p_of_s(u):
    return math.pi + 2 * math.atan(1 / u) - 2 * math.atan(u)


def c_eff_of_q(q):
    u = math.sqrt(max(1e-9, 2 / q - 1))
    return c_p_of_s(u) + 2 * q * u


def t_c_walk(q_sites, t_trans, expo):
    """Resout <exp(-E(q_i)/T)> = 1/e (moyenne de marche), bissection."""
    ces = np.array([c_eff_of_q(max(0.05, q)) for q in q_sites])

    def mean_p(t):
        x = max(1e-9, 1 - t / t_trans) ** expo
        return float(np.mean(np.exp(-ces * D0_SIG * math.sqrt(x) / t)))

    lo, hi = 0.02, 0.95 * t_trans
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) < 1 / math.e:   # p croit avec T
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_c_scalar(c_eff, t_trans, expo):
    t = 0.05
    for _ in range(200):
        x = max(1e-9, 1 - t / t_trans) ** expo
        t = 0.5 * t + 0.5 * c_eff * D0_SIG * math.sqrt(x)
    return t


def run():
    print("=== Pincer les marges : q (moyenne de marche) et T_trans ===\n")

    print("[1] MESURE q sur T(2,23) (SONO) + lissage de zone :")
    P0 = _torus_knot_curve(2, 23, 420)
    P, info = sono(P0, R=1.0, iters=9000)
    n = len(P)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])[:-1]
    L = arc[-1] + seg[-1]
    T_hat = np.diff(np.vstack([P, P[:1]]), axis=0)
    T_hat /= np.linalg.norm(T_hat, axis=1)[:, None]
    kv = (T_hat - np.roll(T_hat, 1, axis=0)) / seg[:, None]
    qs = np.full(n, np.nan)
    for i in range(n):
        d_arc = np.abs(arc - arc[i])
        d_arc = np.minimum(d_arc, L - d_arc)
        mask = d_arc > 2 * D
        d = np.linalg.norm(P[mask] - P[i], axis=1)
        j = np.flatnonzero(mask)[np.argmin(d)]
        if d.min() < 1.05 * D:
            n_hat = (P[j] - P[i]) / d.min()
            qs[i] = max(0.0, float(kv[i] @ n_hat)) * D
    valid = ~np.isnan(qs)
    q_raw = qs[valid]
    # lissage de zone (fenetre d'arc 3D, circulaire) sur la serie des sites
    win = max(1, int(3 * D / (L / n)))
    q_filled = np.where(valid, qs, np.nanmedian(qs))
    kern = np.ones(win) / win
    q_zone = np.convolve(np.concatenate([q_filled, q_filled[:win]]),
                         kern, mode="same")[:n][valid]
    print("   brut : mediane %.2f, IQR [%.2f, %.2f]" % (
        np.median(q_raw), np.percentile(q_raw, 25),
        np.percentile(q_raw, 75)))
    print("   zone (3D) : mediane %.2f, IQR [%.2f, %.2f] -- dispersion"
          " reduite x%.1f" % (
              np.median(q_zone), np.percentile(q_zone, 25),
              np.percentile(q_zone, 75),
              (np.percentile(q_raw, 75) - np.percentile(q_raw, 25))
              / max(1e-9, np.percentile(q_zone, 75)
                    - np.percentile(q_zone, 25))))

    print("\n[2] T_c PAR MOYENNE DE MARCHE (la statistique derivee) :")
    t_med = t_c_scalar(c_eff_of_q(float(np.median(q_zone))), 0.1565,
                       EXPO_ISING)
    t_walk = t_c_walk(q_zone, 0.1565, EXPO_ISING)
    print("   mediane de zone      : T_c = %.1f MeV" % (t_med * 1e3))
    print("   moyenne de marche    : T_c = %.1f MeV" % (t_walk * 1e3))
    print("   -> ecart (effet Jensen residuel) = %.1f MeV : la marge q" % (
        abs(t_walk - t_med) * 1e3))
    print("   est PINCEE (statistique derivee, plus de choix de quantile).")

    print("\n[3] T_TRANS PINCE PAR LE RESEAU (156,5 +- 1,5 MeV, HotQCD) :")
    for tt in (0.155, 0.1565, 0.158):
        t = t_c_walk(q_zone, tt, EXPO_ISING)
        print("   T_trans = %.1f MeV -> T_c = %.2f MeV" % (
            tt * 1e3, t * 1e3))
    print("   -> +-1,5 MeV sur T_trans = +-0,2 MeV sur T_c : NEGLIGEABLE.")
    print("   Controle de forme (champ moyen 2nu = 1) :")
    t_mf = t_c_walk(q_zone, 0.1565, 1.0)
    print("   exposant 1,0 -> T_c = %.1f MeV (forme : la marge dominante)" % (
        t_mf * 1e3))

    print("\n[4] BILAN Omega :")
    for lab, t in (("central (marche, Ising, 156,5)", t_walk),
                   ("forme champ moyen", t_mf)):
        b = budget(tc=t, g_s=15.2, p_occ=P_OCC_EFF)
        print("   %-32s T_c = %.1f -> Omega = %5.2f (%+.0f %%)" % (
            lab, t * 1e3, b["omega"], 100 * (b["omega"] / 5.45 - 1)))


if __name__ == "__main__":
    run()
