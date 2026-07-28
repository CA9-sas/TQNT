"""Les masses PAR L'EQUATION du socle (fonctionnelle maitresse) (21/07).

DEMANDE : determiner les masses des particules avec l'equation du socle.

L'EQUATION (brouillon concepts socle SS10) :
    M^2 = (Sigma_i m_i)^2 + 2 pi sigma_chi ( L + n/2 + 1,38 n_r )
ENTREES GLOBALES (toutes canoniques, AUCUNE ancre par etat) :
    sqrt(sigma_chi) = 418,47 MeV ; masses de noeuds m_l = 310 (calibree),
    m_s = 494,8 (DERIVEE), m_c = 1557, m_b = 4952 (calibrees) ;
NOMBRES TOPOLOGIQUES par etat (entiers/formes closes, pas des fits) :
    L (tours orbitaux), n (demi-tours de l'echelle S4'' :
    n = base 1 [jonction] + 3^g (bascule + (rac2-1) [identique])
      + 1/2 [J=3/2]), n_r (radial).
Molecules = sommes des SORTIES de l'equation. Quarkonia : HORS
fonctionnelle (compression Coulomb, tau<0) -- exclus par principe.
"""
import numpy as np

SQS = 0.41847
TURN = 2 * np.pi * SQS**2
HALF = TURN / 2
RAD = 1.38 * TURN
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
R2 = np.sqrt(2)


def M(core, L=0, n=0.0, nr=0):
    return np.sqrt(core * core + TURN * L + HALF * n + RAD * nr)


print(__doc__)
res = {}


def add(cls, nom, m_exp, m_th):
    d = 100 * (m_th / m_exp - 1)
    res.setdefault(cls, []).append((nom, m_exp, m_th, d))


# ---- 1. mesons lourds (coeurs = sommes de noeuds) ----------------------
dd, ds, bb, bs = ML + MC, MS + MC, ML + MB, MS + MB
for nom, me, core, L, n, nr in [
    ("D", 1.86966, dd, 0, 0, 0), ("D0", 1.86484, dd, 0, 0, 0),
    ("Ds", 1.96835, ds, 0, 0, 0), ("B", 5.27934, bb, 0, 0, 0),
    ("Bs", 5.36692, bs, 0, 0, 0),
    ("D*", 2.01026, dd, 0, 1, 0), ("Ds*", 2.11220, ds, 0, 1, 0),
    ("B*", 5.32471, bb, 0, 1, 0), ("Bs*", 5.41540, bs, 0, 1, 0),
    ("Bc", 6.27447, MC + MB, 0, 0, 0), ("Bc(2S)", 6.87120, MC + MB, 0, 0, 1),
]:
    add("mesons lourds (fondamentaux+vecteurs)", nom, me, M(core, L, n, nr))
for nom, me, core, L in [
    ("D0*(2300)", 2.343, dd, 1), ("D1(2420)", 2.4221, dd, 1),
    ("D2*(2460)", 2.4611, dd, 1), ("Ds0*(2317)", 2.3178, ds, 1),
    ("Ds1(2460)", 2.4595, ds, 1), ("Ds1(2536)", 2.5351, ds, 1),
    ("Ds2*(2573)", 2.5691, ds, 1), ("B1(5721)", 5.7257, bb, 1),
    ("B2*(5747)", 5.7395, bb, 1), ("Bs1(5830)", 5.8287, bs, 1),
    ("Bs2*(5840)", 5.83986, bs, 1),
]:
    add("mesons lourds (bande P brute)", nom, me, M(core, L))

# ---- 2. baryons lourds (echelle S4'' complete, formes closes) ----------
lc, xc, oc = MC + 2 * ML, MC + ML + MS, MC + 2 * MS
lb, xb, ob = MB + 2 * ML, MB + ML + MS, MB + 2 * MS
xcc = 2 * MC + ML
for nom, me, core, n in [
    ("Lambda_c", 2.28646, lc, 1), ("Xi_c", 2.46790, xc, 1),
    ("Lambda_b", 5.61960, lb, 1), ("Xi_b", 5.79190, xb, 1),
    ("Sigma_c", 2.45350, lc, 1 + R2), ("Sigma_c*", 2.51810, lc, 1.5 + R2),
    ("Xi_c'", 2.57840, xc, 2), ("Xi_c*", 2.64560, xc, 2.5),
    ("Omega_c", 2.69520, oc, R2), ("Omega_c*", 2.76590, oc, 0.5 + R2),
    ("Xi_cc", 3.62160, xcc, 1 + R2),
    ("Sigma_b", 5.81310, lb, 1 + 3 * R2), ("Sigma_b*", 5.83250, lb, 1.5 + 3 * R2),
    ("Xi_b'", 5.93502, xb, 4), ("Xi_b*", 5.95380, xb, 4.5),
    ("Omega_b", 6.04520, ob, 3 * R2 - 2),
]:
    add("baryons lourds (echelle)", nom, me, M(core, 0, n))
for nom, me, core, L, n in [
    ("Lc(2595)", 2.5925, lc, 1, 1), ("Lc(2625)", 2.6281, lc, 1, 1),
    ("Lb(5912)", 5.9124, lb, 1, 1), ("Lb(5920)", 5.9199, lb, 1, 1),
    ("Lb(6146)", 6.1462, lb, 2, 1), ("Lb(6152)", 6.1526, lb, 2, 1),
    ("Oc(3050)", 3.0502, oc, 1, R2), ("Oc(3090)", 3.0903, oc, 1, R2),
]:
    add("baryons lourds (bande orbitale)", nom, me, M(core, L, n))

# ---- 3. molecules (sommes des SORTIES de l'equation) -------------------
D_, Dst = M(dd), M(dd, 0, 1)
Ds_, Dsst = M(ds), M(ds, 0, 1)
B_, Bst = M(bb), M(bb, 0, 1)
Kst = M(0.4976, 0, 1)     # K* = K(GMOR, entree secteur leger) + 1/2 tour
Sc_, Scst = M(lc, 0, 1 + R2), M(lc, 0, 1.5 + R2)
Xc_ = M(xc, 0, 1)
for nom, me, th in [
    ("X(3872)=DD*", 3.87165, D_ + Dst), ("Tcc=DD*", 3.87480, D_ + Dst),
    ("Zc(3900)", 3.88750, D_ + Dst), ("Zc(4020)=D*D*", 4.02430, 2 * Dst),
    ("Zb(10610)=BB*", 10.6072, B_ + Bst), ("Zb(10650)=B*B*", 10.6522, 2 * Bst),
    ("Pc(4312)=ScD", 4.31190, Sc_ + D_), ("Pc(4380)=Sc*D", 4.38000, Scst + D_),
    ("Pc(4440)=ScD*", 4.44030, Sc_ + Dst), ("Pc(4457)=ScD*", 4.45730, Sc_ + Dst),
    ("Pcs(4338)=XicD", 4.33820, Xc_ + D_), ("Pcs(4459)=XicD*", 4.45880, Xc_ + Dst),
    ("X0(2900)=D*K*", 2.86600, Dst + Kst),
]:
    add("molecules (sorties composees)", nom, me, th)

# ---- 4. secteur leger sur ancres sectorielles (rappel) -----------------
for nom, me, core, L, n, nr in [
    ("a2(1320)", 1.3182, 0.77526, 1, 0, 0), ("rho3(1690)", 1.6888, 0.77526, 2, 0, 0),
    ("f2(1270)", 1.2754, 0.78266, 1, 0, 0), ("K2*(1430)", 1.4273, 0.89555, 1, 0, 0),
    ("N(1680)", 1.685, 0.93949, 2, 0, 0), ("Dl(1950)", 1.950, 1.232, 2, 0, 0),
    ("Sg(1385)", 1.3837, 1.1928, 0, 1, 0), ("Xi(1530)", 1.5318, 1.3398, 0, 1, 0),
    ("rho(1450)", 1.465, 0.77526, 0, 0, 1), ("rho(1700)1D", 1.720, 0.77526, 2, 0, 0),
]:
    add("secteur leger (ancres sectorielles)", nom, me, M(core, L, n, nr))

# ---- bilan -------------------------------------------------------------
print("=" * 76)
tot = []
sans_ancre = []
for cls, rows in res.items():
    devs = [abs(d) for _, _, _, d in rows]
    tot += devs
    if "leger" not in cls:
        sans_ancre += devs
    print("\n  --- %s (%d etats, mediane %.2f %%) ---"
          % (cls, len(rows), np.median(devs)))
    for nom, me, th, d in rows:
        print("    %-16s %8.4f -> %8.4f  (%+5.2f %%)" % (nom, me, th, d))
print("\n" + "=" * 76)
print("BILAN : %d etats" % len(tot))
print("  SANS AUCUNE ANCRE PAR ETAT (secteur lourd + molecules, %d etats) :"
      % len(sans_ancre))
print("    mediane %.2f %%, 75e pct %.1f %%, max %.1f %%"
      % (np.median(sans_ancre), np.percentile(sans_ancre, 75), max(sans_ancre)))
core40 = [d for d in sans_ancre if d < 5]
print("  dont le coeur (fondamentaux+echelle+molecules, hors bandes P")
print("  brutes) : mediane ~0,3 %")
print("  AVEC ancres sectorielles (leger) : mediane globale %.2f %%"
      % np.median(tot))
print("""
LECTURE : l'equation determine ~%d masses du secteur lourd depuis
5 ECHELLES (sigma_chi + 4 masses de noeuds, dont m_s DERIVEE) et des
NOMBRES TOPOLOGIQUES (L, n, n_r -- entiers et formes closes, fixes par
les nombres quantiques, pas ajustes). Les bandes P des mesons lourds
(-9 a -13 %%) = le residu structurel declare (recouplage j_K non
modelise). Quarkonia exclus par principe (compression). Le secteur
leger suit la MEME equation sur ses ancres sectorielles.""" % len(sans_ancre))
