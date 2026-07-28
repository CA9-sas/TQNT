"""Verification INDEPENDANTE de la chaine Omega (pre-publication V6, 17/07).

Chaque maillon verifiable est reverifie par une METHODE INDEPENDANTE
(pas une re-execution) :
  [1] percage Z3 = 24/81 : re-denombrement par une arithmetique differente
      (phases cumulees mod 3) + Monte-Carlo 1e6 ;
  [2] g_s(T_c) = 15.2 : integrales d'entropie par scipy.integrate.quad
      (quadrature independante de la somme manuelle) ;
  [3] prefacteur Omega(P_occ=1) = 14.3 : re-implementation complete de la
      somme du cout de formation (extrapolation N(c) et m_geo propres) ;
  [4] dynamique de Nambu : etalon EXACT -- la boucle circulaire statique
      s'effondre en un point a t = L/4 (solution analytique connue) ;
      periodicite X(s, t+L) = X(s, t) verifiee ;
  [5] fonctionnelle de radiation : integrales de segment fermees vs
      quadrature numerique directe (scipy) sur une polyligne aleatoire ;
  [6] cascade branchee : reseau INDEPENDANT (N = 20, nouvelle graine
      reseau ET nouvelle graine de branchement) -> Omega ;
  [7] assemblage final : Omega = [3] x 8/9 x [6].
"""
import sys
import math
import numpy as np

sys.path.insert(0, ".")


def check(label, val, ref, tol):
    ok = abs(val / ref - 1) <= tol if ref != 0 else abs(val) <= tol
    print("  %-52s %10.4g  (ref %.4g)  %s" % (
        label, val, ref, "OK" if ok else "ECART !"))
    return ok


def run():
    oks = []
    print("=== Verification independante de la chaine Omega ===\n")

    # [1] percage Z3
    print("[1] Percage Z3 (methode : phases cumulees mod 3 + MC)")
    cnt = 0
    for cfg in range(81):
        a, b, c, d = (cfg % 3, (cfg // 3) % 3, (cfg // 9) % 3, (cfg // 27) % 3)
        w = 0
        for p, q in ((a, b), (b, c), (c, d), (d, a)):
            diff = (q - p) % 3
            w += diff if diff <= 1 else diff - 3
        if w != 0:
            cnt += 1
    oks.append(check("percage exact (denombrement)", cnt / 81, 24 / 81, 1e-12))
    rng = np.random.default_rng(99)
    ph = rng.integers(0, 3, (4, 1_000_000))
    w = np.zeros(1_000_000)
    for i in range(4):
        d = (ph[(i + 1) % 4] - ph[i]) % 3
        w += np.where(d <= 1, d, d - 3)
    oks.append(check("percage MC 1e6", float((w != 0).mean()), 24 / 81, 0.01))
    oks.append(check("offre de corde 3 x P", 3 * cnt / 81, 8 / 9, 1e-12))

    # [2] g_s par scipy
    print("\n[2] g_s(T_c = 67 MeV) par scipy.integrate.quad")
    from scipy.integrate import quad

    def s_dof(x, eta):   # eta = +1 fermion, -1 boson ; s = (rho+P)/T
        rho = quad(lambda u: u * u * math.sqrt(u * u + x * x)
                   / (math.exp(math.sqrt(u * u + x * x)) + eta), 0, 60)[0]
        P = quad(lambda u: u ** 4 / (3 * math.sqrt(u * u + x * x))
                 / (math.exp(math.sqrt(u * u + x * x)) + eta), 0, 60)[0]
        s0 = quad(lambda u: u ** 3 / (math.exp(u) - 1), 0, 60)[0] * 4 / 3
        return (rho + P) / s0

    T = 0.067
    g = (2 * s_dof(0, -1) + 6 * s_dof(0, 1) + 4 * s_dof(0.000511 / T, 1)
         + 4 * s_dof(0.10566 / T, 1) + 2 * s_dof(0.13957 / T, -1)
         + 1 * s_dof(0.13498 / T, -1))
    oks.append(check("g_s(67 MeV)", g, 15.25, 0.01))

    # [3] prefacteur Omega(P_occ = 1), re-implementation complete
    print("\n[3] Prefacteur Omega(P_occ = 1), somme re-implementee")
    HT = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165, 11: 552,
          12: 2176, 13: 9988, 14: 46972, 15: 253293, 16: 1388705,
          17: 8053393, 18: 48266466, 19: 294130458, 20: 1847319428}

    def Nc(c):
        if c in HT:
            return float(HT[c])
        n, r = float(HT[20]), 6.28
        for cc in range(21, c + 1):
            r = 6.5 + (r - 6.5) * math.exp(-1 / 8.0)
            n *= r
        return n

    def mg(c):
        return (310.0 / 16.372) * (16.372 + (38.62 - 16.372) / 6 * (c - 3)) / 1000

    a_, tc, kap, eta, s0ng, mp = 1.217, 0.067, 1.14, 6.12e-10, 7.04, 0.93827
    w_ = {c: Nc(c) * math.exp(-a_ * c - mg(c) / tc) for c in range(3, 150)}
    Z = sum(w_.values())
    Sd = sum(v for c, v in w_.items() if c >= 23)
    mbar = sum(w_[c] * mg(c) for c in range(23, 150)) / Sd
    om1 = (135 / (8 * math.pi ** 3 * g * kap ** 3)) * (Sd / Z) * s0ng * mbar / (eta * mp)
    oks.append(check("Omega(P_occ=1, g_s scipy)", om1, 14.3, 0.03))

    # [4] etalon exact de Nambu : boucle circulaire statique
    print("\n[4] Nambu : effondrement de la boucle circulaire (exact t* = L/4)")
    from simulation.experiments.derive_lambda_kinks import Poly
    n, R0 = 256, 5.0
    u = np.arange(n + 1) * 2 * np.pi / n
    V = np.stack([R0 * np.cos(u), R0 * np.sin(u), 0 * u], 1)
    pa = Poly(V, np.zeros(3))
    L = pa.L
    sig = np.arange(64) * L / 64
    r_at = lambda t: float(np.linalg.norm(
        0.5 * (pa.eval(sig - t) + pa.eval(sig + t)), axis=1).max())
    oks.append(check("rayon a t = L/4 (effondrement)", r_at(L / 4) / R0, 0.0, 0.01))
    oks.append(check("periodicite |X(t+L) - X(t)|/R0",
                     abs(r_at(0.31 * L) - r_at(0.31 * L + L)) / R0, 0.0, 1e-6))
    oks.append(check("R(t)/R0 = cos(2pi t/L) a t = 0.15 L",
                     r_at(0.15 * L) / R0, math.cos(2 * math.pi * 0.15), 0.01))

    # [5] fonctionnelle : segments fermes vs quadrature scipy
    print("\n[5] Radiation : integrales de segment vs quadrature directe")
    from simulation.experiments.tighten_gamma_a_chi import mover_amp_vec
    rng = np.random.default_rng(5)
    steps = rng.normal(size=(12, 3))
    steps -= steps.mean(0)
    steps /= np.linalg.norm(steps, axis=1)[:, None]
    Vp = np.vstack([np.zeros(3), np.cumsum(steps, 0)])
    Vp[-1] = Vp[0]        # fermee (approx : re-normalise le dernier pas)
    khat = np.array([[0.3, -0.5, 0.812403840463596]])
    khat /= np.linalg.norm(khat)
    Ltot = np.linalg.norm(np.diff(Vp, axis=0), axis=1).sum()
    w = 4 * np.pi * 3 / Ltot
    A_closed = mover_amp_vec(Vp, khat, w, -1)[0]

    def a_of(uu):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Vp, axis=0), axis=1))])
        uu = uu % s[-1]
        i = min(np.searchsorted(s, uu, side="right") - 1, len(s) - 2)
        f = (uu - s[i]) / (s[i + 1] - s[i])
        return Vp[i] + f * (Vp[i + 1] - Vp[i])

    from scipy.integrate import quad as q2
    A_num = np.zeros(3, complex)
    for j in range(3):
        for part, fn in ((1, np.cos), (1j, np.sin)):
            A_num[j] += part * q2(
                lambda uu, j=j, fn=fn: ((a_of(uu + 1e-6) - a_of(uu - 1e-6)) / 2e-6)[j]
                * fn(-(w / 2) * (uu + float(a_of(uu) @ khat[0]))),
                0, Ltot, limit=800)[0] / Ltot
    oks.append(check("|A_ferme - A_quad| / |A|",
                     float(np.linalg.norm(A_closed - A_num)
                           / np.linalg.norm(A_closed)), 0.0, 0.02))

    # [6] cascade branchee sur un reseau INDEPENDANT (N = 20, graines neuves)
    print("\n[6] Cascade branchee, reseau independant (N = 20, graines 777/778)")
    import simulation.experiments.departage_f_frac as df
    from simulation.experiments.derive_lambda_kinks import embedded_fragments
    rng = np.random.default_rng(777)
    ps = embedded_fragments(N=20, seed=778, n_sample=30, rng=rng)
    parents = [p for p in ps if 6 <= len(p) <= 80][:20]
    L_tot = float(sum(len(p) for p in parents))
    brng = np.random.default_rng(779)
    n_fin = n_knot = 0
    for p in parents:
        for (Lf, ntw) in df.cascade_branchee(p, brng):
            n_fin += 1
            if ntw >= 1:
                n_knot += 1
    kpx = n_knot / L_tot
    oks.append(check("n_knot / L (reseau N=20)", kpx, 0.335, 0.20))

    # [7] assemblage
    om = om1 * (8 / 9) * kpx / 1.0
    print("\n[7] ASSEMBLAGE : Omega = %.2f (chaine publiee : 4.3 +- 0.5)" % om)
    oks.append(check("Omega final", om, 4.3, 0.20))

    print("\nBILAN : %d/%d verifications passees." % (sum(oks), len(oks)))
    if all(oks):
        print("La chaine Omega est VERIFIEE par methodes independantes.")
    else:
        print("ECARTS a investiguer avant publication.")


if __name__ == "__main__":
    run()
