"""Torsion du tube de flux : equilibrage des masses et consequences (mu, tau).

Cadre : theoreme de Calugareanu-White-Fuller, Lk = Tw + Wr. La torsion (twist
Tw = (1/2pi) INT tau ds) et le vrillage (writhe Wr) s'echangent A TOPOLOGIE FIXEE :
'equilibrer les masses suite a la torsion' = redistribuer le reservoir d'energie de
torsion sans changer Lk. Deux canaux de relaxation :
  (a) twist -> writhe (White : dTw = -dWr)  -> change le VRILLAGE donc le dipole mu ;
  (b) dilution : E_tw = C(2pi Tw)^2/2L decroit par allongement L a Tw fixe -> change
      la longueur (masse M_topo) SANS toucher Wr ni mu.

RESULTATS (courbes VALIDES : toriques + 4_1) :
  1. White valide EXACTEMENT : Wr+Tw = -3.00, 10.00, 14.00, 18.00, -0.00 (entiers, SL Frenet).
  2. Reservoir de torsion f_tau = U_tau/(U_kappa+U_tau) : trefle 0.1318 = EXACTEMENT la
     partition Frenet V6 (briques m_c/m_l) -- coherence. Toriques : jusqu'a 0.49 (5_1) ;
     NB au-dela du trefle, f_tau est embedding-dependant (courbes non ideales).
  3. Levier Sargent (tau ~ Q^-5) : amplification 5M/Q. Neutron : dQ = 1.7 keV (delta=1.4e-5
     du reservoir) => d(tau)/tau = +1.1% -- l'ordre de l'ecart bouteille/faisceau (878.4 vs
     888 s). SPECULATIF, mais avec une SIGNATURE VERROUILLEE :
  4. Magnetisme : le dipole est porte par le WRITHE (vrillage = aire de boucle) ; le twist pur
     est dipole-silencieux. Canal (a) : dmu/mu = (Tw/Wr)*delta/2 = -0.104*delta/2 (trefle).
     Neutron : delta=1.4e-5 -> dmu/mu ~ 7e-7, au bord de la precision mu_n (2.4e-7) : TESTABLE.
  5. P23 (plasticite sous B_ext) : C_plast = -2.92 MeV = 21% du reservoir du muon
     (f_tau*m_mu = 13.9 MeV), signe coherent (liberation de twist). MAIS canal (a) donnerait
     dmu/mu ~ 1% pour le muon, EXCLU par la precession (~ppm, g-2). => si P23 = relaxation de
     torsion, elle passe par le canal (b) DILUTION (mu-silencieux). Contrainte identifiee.
"""
import numpy as np
from theory.abraham import frenet_geometry
from theory.knot_catalog import knot_curve

KNOTS = ['3_1', '5_1', '7_1', '9_1', '4_1']
FT31_V6 = 0.1318          # partition Frenet V6 (briques m_c/m_l)
C_PLAST, M_MU, M_N, Q_N = -2.92, 105.66, 939.57, 0.782
TAU_BOTTLE, TAU_BEAM = 878.4, 888.0


def writhe(P):
    P = np.asarray(P, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    n = len(P)
    dl = np.roll(P, -1, 0) - P
    mid = P + 0.5 * dl
    W = 0.0
    for i in range(n):
        r = mid[i] - mid
        r[i] = 1e9
        cr = np.cross(np.tile(dl[i], (n, 1)), dl)
        W += ((cr * r).sum(1) / np.linalg.norm(r, axis=1) ** 3).sum()
    return W / (4 * np.pi)


def frenet_data(K, npts=20000):
    g = frenet_geometry(K, npts)
    tau, kap, sp = (np.asarray(g[k]) for k in ('tau', 'kappa', 'speed'))
    ds = sp * (g['t'][1] - g['t'][0])
    Tw = float((tau * ds).sum() / (2 * np.pi))
    Uk = float((kap ** 2 * ds).sum())
    Ut = float((tau ** 2 * ds).sum())
    return Tw, Uk, Ut


def run():
    print("=== 1) White : SL = Wr + Tw (entier) ; reservoir f_tau ===")
    print(f"{'K':5}{'Wr':>8}{'Tw':>8}{'Wr+Tw':>8}{'f_tau':>8}")
    res = {}
    for K in KNOTS:
        Tw, Uk, Ut = frenet_data(K)
        Wr = writhe(np.asarray(knot_curve(K, 1600), float))
        res[K] = (Wr, Tw, Ut / (Uk + Ut))
        print(f"{K:5}{Wr:>8.2f}{Tw:>8.2f}{Wr+Tw:>8.2f}{Ut/(Uk+Ut):>8.3f}")
    ft = res['3_1'][2]
    print(f"  f_tau(3_1)={ft:.4f} = partition Frenet V6 ({FT31_V6}) : coherence exacte")

    print("\n=== 2) reservoirs et P23 ===")
    for nm, m in (('electron', 0.511), ('muon', M_MU), ('neutron', M_N)):
        print(f"  {nm:9}: f_tau*m = {ft*m:8.2f} MeV")
    delta_mu = abs(C_PLAST) / (ft * M_MU)
    print(f"  P23 : C_plast={C_PLAST} MeV = {delta_mu*100:.0f}% du reservoir muon (signe = liberation)")

    print("\n=== 3) levier Sargent + signature verrouillee ===")
    dtau = TAU_BEAM / TAU_BOTTLE - 1
    dQ = Q_N * dtau / 5
    delta_n = dQ / (ft * M_N)
    TwWr = res['3_1'][1] / res['3_1'][0]
    print(f"  neutron : d(tau)/tau={dtau*100:.1f}% (faisceau/bouteille) <- dQ={dQ*1e3:.1f} keV, delta={delta_n:.1e}")
    print(f"  canal (a) verrouille : dmu/mu = (Tw/Wr)*delta/2 = {abs(TwWr)*delta_n/2:.1e}"
          f"  (precision mu_n : 2.4e-7 -> TESTABLE au bord)")
    print(f"  muon P23 canal (a) : dmu/mu ~ {abs(TwWr)*delta_mu/2:.1e} ~ 1% -> EXCLU (precession ppm)")
    print(f"  => P23-torsion doit etre canal (b) DILUTION (allongement, mu-silencieux)")


if __name__ == "__main__":
    run()
