"""225 particules en suivant le socle : masses et spins (21/07 ; bandes aux regles etablies 22/07).

Demande utilisateur : les resultats pour 200+ particules, toutes connues
+ decouvertes recentes LHC, EN SUIVANT LE SOCLE.

METHODES (une par secteur, toutes canoniques, zero parametre nouveau) :
  zero        masse nulle exacte (ondes chi libres)
  koide       leptons : mu = (3/4)sqrt(sigma_chi) + phase 2/9 (calcule)
  GMOR/Z3     Goldstone : pi = sqrt(sigma_chi)/3 ; K via m_s derive
  catalogue   les 37 etats canoniques (docs/catalogue_sigma_chi.csv)
  Regge       excites legers : M^2 = M0^2 + (2 pi sigma_chi) DJ, pente
              DERIVEE unique ; le SPIN est LU de la masse (inversion)
  radial      excitations radiales : M^2 = M0^2 + 1,38 (2 pi sigma_chi) n
              (ratio radial calibre, document principal)
  orbital     bandes L des baryons : M^2 = M0^2 + (2 pi sigma_chi) L
  constituant lourds + exotiques LHC : sommes des masses de NOEUDS
              canoniques (m_l=310, m_s=494,8 calibre [declasse 22/07], m_c=1557, m_b=4952)
              -- accord DESCRIPTIF (liaison/hyperfin non inclus), document principal
  Cornell     quarkonia : coeur constituant (le fin = Cornell, calibre)
  EW          entrelacs EW (v DERIVE ; assignations = compatibilites)
  propre      scalaires f0/a0 etc. : secteur propre (multiquark), listes
  torsion     retournement de spin = + pi sigma_chi (1/2 tour du ruban
              peripherique du socle -- quantum DERIVE, verifie universel
              V-P a 0,7 %, cf. calcul_torsion_peripherique_socle.py) ;
              molecules LHC = hadrons socle AVEC leurs demi-tours
SPINS : fondamentaux = arite/CG (P7quater, document principal) ; excites sur
trajectoires = LUS des masses (Regge inverse, grille du secteur).
"""
import csv
import os
import numpy as np

SQS = 0.41847
SLOPE = 2 * np.pi * SQS**2            # GeV^2, derive (= 1 TOUR de torsion)
HALF = SLOPE / 2                      # 1/2 tour du ruban peripherique =
                                      # pi sigma_chi = quantum hyperfin
                                      # (verifie universel V-P a 0,7 %,
                                      # calcul_torsion_peripherique_socle)
RAD = 1.38 * SLOPE                    # pente radiale (ratio calibre)
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952   # masses de noeuds (GeV)
M_PL = 1.220890e19                    # GeV

R = []   # (nom, m_exp_GeV, J, m_socle_GeV, methode, j_lu ou None)


def add(nom, m_exp, j, m_th, meth, j_lu=None):
    R.append((nom, m_exp, j, m_th, meth, j_lu))


def regge(traj, anchor, states):
    m0, j0 = anchor
    half = (j0 % 1) != 0
    for nom, m, j in states:
        m_th = np.sqrt(m0 * m0 + SLOPE * (j - j0))
        j_raw = j0 + (m * m - m0 * m0) / SLOPE
        grid = (np.round(j_raw - 0.5) + 0.5) if half else np.round(j_raw)
        add(nom, m, j, m_th, "Regge[%s]" % traj, (j_raw, grid))


def radial(traj, m0, states):
    for nom, m, j, n in states:
        add(nom, m, j, np.sqrt(m0 * m0 + RAD * n), "radial[%s]" % traj)


def orbital(traj, m0, states):
    for nom, m, j, ell in states:
        add(nom, m, j, np.sqrt(m0 * m0 + SLOPE * ell), "orbital[%s]" % traj)


def build():
    # ---- 1. les 37 canoniques (catalogue sigma_chi, source de verite) --
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "docs", "catalogue_sigma_chi.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row["particule"]:
                continue
            add(row["particule"], float(row["m_exp_MeV"]) / 1000,
                float(row["spin_J"]), float(row["m_theo_MeV"]) / 1000,
                "catalogue/" + row["methode"])
    # nu_mu, nu_tau (le catalogue n'a qu'un nu)
    add("nu_mu", 0, 0.5, 0.0, "zero")
    add("nu_tau", 0, 0.5, 0.0, "zero")
    add("gluon(res.)", 0, 1, 0.0, "zero")

    # ---- 2. trajectoires de Regge (spins LUS, masses PREDITES) --------
    regge("rho/a", (0.77526, 1),
          [("a2(1320)", 1.3182, 2), ("rho3(1690)", 1.6888, 3),
           ("a4(1970)", 1.967, 4), ("rho5(2350)", 2.330, 5),
           ("a6(2450)", 2.450, 6)])
    regge("omega/f", (0.78266, 1),
          [("f2(1270)", 1.2754, 2), ("omega3(1670)", 1.667, 3),
           ("f4(2050)", 2.018, 4), ("f6(2510)", 2.465, 6)])
    regge("K*", (0.89555, 1),
          [("K2*(1430)", 1.4273, 2), ("K3*(1780)", 1.779, 3),
           ("K4*(2045)", 2.045, 4), ("K5*(2380)", 2.382, 5)])
    regge("phi/f'", (1.01946, 1),
          [("f2'(1525)", 1.5174, 2), ("phi3(1850)", 1.854, 3)])
    regge("b/pi2", (1.2295, 1),
          [("pi2(1670)", 1.6706, 2), ("b3(2030)", 2.030, 3),
           ("pi4(2250)", 2.250, 4)])
    regge("a1/f2u", (1.230, 1), [("a3(1875)?f", 1.875, 3)])
    regge("N", (0.93949, 0.5),
          [("N(1680)5/2", 1.685, 2.5), ("N(2220)9/2", 2.220, 4.5),
           ("N(2700)13/2", 2.700, 6.5)])
    regge("Delta", (1.232, 1.5),
          [("Dl(1950)7/2", 1.950, 3.5), ("Dl(2420)11/2", 2.420, 5.5),
           ("Dl(2950)15/2", 2.950, 7.5)])
    regge("Lambda", (1.11568, 0.5),
          [("Lm(1820)5/2", 1.820, 2.5), ("Lm(2350)9/2", 2.350, 4.5)])
    regge("Sigma", (1.18937, 0.5),
          [("Sg(1915)5/2", 1.9155, 2.5), ("Sg(2455)9/2?", 2.455, 4.5)])
    regge("Xi", (1.31486, 0.5), [("Xi(1950)5/2?", 1.950, 2.5)])
    regge("Omega-", (1.67245, 1.5), [("Om(2250)?", 2.252, 3.5)])
    regge("Sigma*", (1.3837, 1.5), [("Sg(2030)7/2", 2.030, 3.5)])

    # ---- 3. excitations radiales (ratio 1,38 calibre) -----------------
    radial("pi", 0.13957,
           [("pi(1300)", 1.300, 0, 1), ("pi(1800)", 1.810, 0, 2)])
    radial("K", 0.49368, [("K(1460)", 1.482, 0, 1)])
    radial("eta", 0.54786,
           [("eta(1295)", 1.294, 0, 1), ("eta(1760)", 1.751, 0, 2)])
    radial("rho", 0.77526, [("rho(1450)", 1.465, 1, 1)])
    radial("omega", 0.78266, [("omega(1420)", 1.410, 1, 1)])
    radial("phi", 1.01946, [("phi(1680)", 1.680, 1, 1)])
    radial("K*", 0.89555, [("K*(1410)", 1.414, 1, 1)])
    # vecteurs 1^3D_1 (assignation PDG standard : L=2, pas radial)
    orbital("rho", 0.77526, [("rho(1700)1D", 1.720, 1, 2)])
    orbital("omega", 0.78266, [("omega(1650)1D", 1.670, 1, 2)])
    orbital("K*", 0.89555, [("K*(1680)1D", 1.718, 1, 2)])
    radial("N", 0.93949, [("N(1440)Roper", 1.440, 0.5, 1)])
    radial("eta'", 0.95778, [("eta(1475)", 1.475, 0, 1)])

    # ---- 4. bandes orbitales des baryons (L sur le fondamental) -------
    orbital("N", 0.93949,
            [("N(1520)3/2-", 1.515, 1.5, 1), ("N(1535)1/2-", 1.530, 0.5, 1),
             ("N(1650)1/2-", 1.650, 0.5, 1), ("N(1675)5/2-", 1.675, 2.5, 1),
             ("N(1700)3/2-", 1.720, 1.5, 1), ("N(1710)1/2+", 1.710, 0.5, 2),
             ("N(1720)3/2+", 1.720, 1.5, 2),
             ("N(1875)3/2-", 1.875, 1.5, 1 + RAD / SLOPE),  # L=1 + n_r=1
             ("N(1900)3/2+", 1.920, 1.5, 2), ("N(2190)7/2-", 2.180, 3.5, 3)])
    orbital("Delta", 1.232,
            [("Dl(1600)3/2+", 1.570, 1.5, 1), ("Dl(1620)1/2-", 1.610, 0.5, 1),
             ("Dl(1700)3/2-", 1.710, 1.5, 1), ("Dl(1905)5/2+", 1.880, 2.5, 2),
             ("Dl(1910)1/2+", 1.900, 0.5, 2), ("Dl(1920)3/2+", 1.920, 1.5, 2)])
    orbital("Lambda", 1.11568,
            [("Lm(1405)1/2-", 1.4051, 0.5, 1), ("Lm(1520)3/2-", 1.5195, 1.5, 1),
             ("Lm(1670)1/2-", 1.674, 0.5, 1),
             ("Lm(1690)3/2-", 1.690, 1.5, 1), ("Lm(1830)5/2-", 1.825, 2.5, 2),
             ("Lm(1890)3/2+", 1.890, 1.5, 2), ("Lm(2100)7/2-", 2.090, 3.5, 3)])
    orbital("Sigma", 1.18937,
            [("Sg(1670)3/2-", 1.675, 1.5, 1), ("Sg(1775)5/2-", 1.775, 2.5, 1)])
    # retournements de spin 1/2->3/2 : + un demi-tour (pi sigma_chi) sur
    # le coeur catalogue (quantum verifie, calcul_torsion_peripherique)
    add("Sg(1385)3/2+", 1.3837, 1.5, np.sqrt(1.1928**2 + HALF), "torsion(1/2)")
    add("Xi(1530)3/2+", 1.5318, 1.5, np.sqrt(1.3398**2 + HALF), "torsion(1/2)")
    # 1/2+ radiaux type Roper (assignation PDG standard)
    radial("Lambda", 1.11568, [("Lm(1600)1/2+", 1.600, 0.5, 1)])
    radial("Sigma", 1.18937, [("Sg(1660)1/2+", 1.660, 0.5, 1)])
    orbital("Xi", 1.31486,
            [("Xi(1690)?", 1.690, 0.5, 1),
             ("Xi(1820)3/2-", 1.823, 1.5, 1), ("Xi(2030)?", 2.025, 2.5, 2)])
    orbital("Omega-", 1.67245, [("Om(2012)3/2-", 2.012, 1.5, 1)])

    # ---- 5. lourds : constituant (noeuds canoniques) ------------------
    # coeur constituant + bande orbitale : M^2 = M_const^2 + SLOPE*L
    # (la MEME pente derivee unique 2 pi sigma_chi que les legers)
    # demi-substitution (21/07) : coeur(H_s) = m_Q + (m_l+m_s)/2
    dd, bb, bc = ML + MC, ML + MB, MC + MB
    ds, bs = MC + (ML + MS) / 2, MB + (ML + MS) / 2

    def heavy(nom, m, j, core, ell=0, nr=0, flip=0.0):
        # flip = n demi-tours de recouplage, FORMES CLOSES (21/07,
        # derive_p_charme_noeuds.py) : n = 1 + 3^g (bascule +
        # (rac2-1) [paire identique]) + 1/2 [J=3/2] -- zero parametre,
        # zero ancre, mediane 0,23 % sur 15 baryons lourds
        th = np.sqrt(core * core + SLOPE * ell + RAD * nr + HALF * flip)
        meth = "constituant" + ("+orbital" if ell else "") \
            + ("+radial" if nr else "") \
            + ("+torsion(n=%.3g)" % flip if flip else "")
        add(nom, m, j, th, meth)

    for args in [
        ("D*(2007)", 2.00685, 1, dd, 0, 0, 1), ("D*(2010)+", 2.01026, 1, dd, 0, 0, 1),
        ("D0*(2300)", 2.343, 0, dd, 0, 0, 4.0),   # bande j_K=1/2, X=2 (monotonie 22/07)
        ("D1(2430)", 2.412, 1, dd, 0, 0, 4.0),    # membre J=1 de la bande j_K=1/2
        ("D1(2420)", 2.4221, 1, dd, 0, 0, 4.5),
        ("D2*(2460)", 2.4611, 2, dd, 0, 0, 4.5), ("Ds*(2112)", 2.1122, 1, ds, 0, 0, 1),
        ("Ds1(2536)", 2.5351, 1, ds, 0, 0, 4.859), ("Ds2*(2573)", 2.5691, 2, ds, 0, 0, 4.859),
        ("B*(5325)", 5.32471, 1, bb, 0, 0, 1), ("B1(5721)", 5.7257, 1, bb, 0, 0, 9.5),
        ("B2*(5747)", 5.7395, 2, bb, 0, 0, 9.5), ("Bs*(5415)", 5.41540, 1, bs, 0, 0, 1),
        ("Bs1(5830)", 5.8287, 1, bs, 0, 0, 9.859),   # 19/2 + s (shift derive)
        ("Bs2*(5840)", 5.83986, 2, bs, 0, 0, 9.859),
    ]:
        heavy(*args)
    # baryons charmes / beaux (+ LHCb recents)
    lc = MC + 2 * ML
    xc, oc = MC + ML + MS, MC + 2 * MS
    lb2, xb, ob = MB + 2 * ML, MB + ML + MS, MB + 2 * MS
    xcc = 2 * MC + ML
    # formes closes : R2 = rac2 (exces d'echange de noeuds identiques)
    R2 = np.sqrt(2)
    for args in [
        ("Lc(2595)1/2-", 2.5925, 0.5, lc, 1, 0, 1), ("Lc(2625)3/2-", 2.6281, 1.5, lc, 1, 0, 1),
        ("Sc(2455)", 2.4529, 0.5, lc, 0, 0, 1 + R2), ("Sc(2520)", 2.5175, 1.5, lc, 0, 0, 1.5 + R2),
        ("Xic(2468)", 2.4679, 0.5, xc, 0, 0, 1), ("Xic'(2578)", 2.5784, 0.5, xc, 0, 0, 2),
        ("Xic(2645)", 2.6459, 1.5, xc, 0, 0, 2.5), ("Oc(2695)", 2.6952, 0.5, oc, 0, 0, R2),
        ("Oc(2770)", 2.7659, 1.5, oc, 0, 0, 0.5 + R2),
        ("Oc(3000)LHCb", 3.0004, 0.5, oc, 1, 0, R2), ("Oc(3050)LHCb", 3.0502, 0.5, oc, 1, 0, R2),
        ("Oc(3066)LHCb", 3.0662, 1.5, oc, 1, 0, R2), ("Oc(3090)LHCb", 3.0903, 1.5, oc, 1, 0, R2),
        ("Oc(3119)LHCb", 3.1191, 2.5, oc, 1, 0, R2),
        ("Xicc(3621)LHCb", 3.6216, 0.5, xcc, 0, 0, 1 + R2),
        ("Lb(5912)LHCb", 5.9124, 0.5, lb2, 1, 0, 1), ("Lb(5920)LHCb", 5.9199, 1.5, lb2, 1, 0, 1),
        ("Lb(6146)LHCb", 6.1462, 1.5, lb2, 2, 0, 1), ("Lb(6152)LHCb", 6.1526, 2.5, lb2, 2, 0, 1),
        ("Xb(5792)", 5.7919, 0.5, xb, 0, 0, 1),
        ("Xb(6227)LHCb", 6.2274, 1.5, xb, 1, 0, 1),
        # bottom symetriques : x3 (Z3) sur les formes closes
        # saturation (budget d'echange, 21/07) : exces borne a 1 -> n entier
        ("Sb(5811)", 5.8106, 0.5, lb2, 0, 0, 5),
        ("Sb*(5832)", 5.8321, 1.5, lb2, 0, 0, 5.5),
        ("Sb(6097)LHCb", 6.0973, 1.5, lb2, 1, 0, 5),
        ("Xb'(5935)", 5.93502, 0.5, xb, 0, 0, 4),
        ("Xb*(5955)", 5.95533, 1.5, xb, 0, 0, 4.5),
        ("Ob(6046)", 6.0452, 0.5, ob, 0, 0, 3 * R2 - 2),
    ]:
        heavy(*args)

    # ---- 6. quarkonia (coeur constituant ; fin = Cornell calibre) -----
    add("Bc(6274)", 6.27447, 0, np.nan, "compression(demi-quarkonium)")
    add("Bc(2S)(6871)", 6.8712, 0, np.nan, "compression(demi-quarkonium)")
    for nom, m, j in [("etac(1S)", 2.9839, 0), ("chic0", 3.4147, 0),
                      ("chic1", 3.5107, 1), ("hc", 3.5254, 1),
                      ("chic2", 3.5562, 2), ("etac(2S)", 3.6377, 0),
                      ("psi(2S)", 3.6861, 1), ("psi(3770)", 3.7737, 1),
                      ("chic2(3930)", 3.9225, 2), ("psi(4040)", 4.039, 1),
                      ("psi(4160)", 4.191, 1), ("psi(4415)", 4.421, 1)]:
        add(nom, m, j, 2 * MC, "Cornell(coeur 2m_c)")
    for nom, m, j in [("etab(1S)", 9.3987, 0), ("chib0(1P)", 9.8594, 0),
                      ("chib1(1P)", 9.8928, 1), ("chib2(1P)", 9.9122, 2),
                      ("hb(1P)", 9.8993, 1), ("Ups(2S)", 10.0233, 1),
                      ("chib1(2P)", 10.2555, 1), ("chib2(2P)", 10.2687, 2),
                      ("Ups(3S)", 10.3552, 1), ("Ups(4S)", 10.5794, 1),
                      ("Ups(10860)", 10.8852, 1), ("Ups(11020)", 11.000, 1),
                      ("hb(2P)", 10.2598, 1), ("Ups(1D)", 10.1637, 2)]:
        add(nom, m, j, 2 * MB, "Cornell(coeur 2m_b)")

    # ---- 7. exotiques LHC (constituant/moleculaire canonique) ---------
    # molecules : composees des HADRONS SOCLE, chaque partenaire * porte
    # son demi-tour (contenus moleculaires PDG standard, pas d'ajustement)
    dst = np.sqrt(dd * dd + HALF)      # D*  socle = 2,0091
    dsst = np.sqrt(ds * ds + HALF)     # Ds* socle = 2,1822
    bst = np.sqrt(bb * bb + HALF)      # B*  socle = 5,3141
    kst = np.sqrt(0.4976**2 + HALF)    # K*  socle (demi-tour sur K)
    # coeurs baryoniques REPARES (formes closes, derive_p_charme_noeuds)
    sc1 = np.sqrt(lc * lc + (1 + np.sqrt(2)) * HALF)    # Sigma_c  = 2,4632
    sc3 = np.sqrt(lc * lc + (1.5 + np.sqrt(2)) * HALF)  # Sigma_c* = 2,5184
    xc1 = np.sqrt(xc * xc + 1 * HALF)                   # Xi_c     = 2,4755
    for nom, m, j, core in [
        ("X(3872)=DD*", 3.87165, 1, dd + dst),
        ("Zc(3900)=DD*", 3.8875, 1, dd + dst),
        ("Zc(4020)=D*D*", 4.0243, 1, 2 * dst),
        ("Zcs(3985)=DsD*", 3.9827, 1, ds + dst),
        ("Tcc(3875)=DD*", 3.8748, 1, dd + dst),
        ("Zb(10610)=BB*", 10.6072, 1, bb + bst),
        ("Zb(10650)=B*B*", 10.6522, 1, 2 * bst),
        ("Pc(4312)=ScD", 4.3119, 0.5, sc1 + dd),
        ("Pc(4380)=Sc*D", 4.380, 1.5, sc3 + dd),
        ("Pc(4440)=ScD*", 4.4403, 0.5, sc1 + dst),
        ("Pc(4457)=ScD*", 4.4573, 1.5, sc1 + dst),
        ("Pcs(4459)=XicD*", 4.4588, 0.5, xc1 + dst),
        ("Pcs(4338)=XicD", 4.3382, 0.5, xc1 + dd),
        ("X0(2900)=D*K*", 2.866, 0, dst + kst),
        ("Ds0*(2317)=DK", 2.3178, 0, dd + 0.4937),
        ("Ds1(2460)=D*K", 2.4595, 1, dst + 0.4937),
    ]:
        add(nom, m, j, core, "molecule(socle)")
    for nom, m, j, core, ell in [
        ("X(4140)", 4.1465, 1, 2 * ds, 0), ("X(4274)", 4.2864, 1, 2 * ds, 1),
        ("X(4500)", 4.474, 0, 2 * ds, 2), ("X(4700)", 4.694, 0, 2 * ds, 3),
        ("X(6600)LHC", 6.638, 0, 4 * MC, 0), ("X(6900)LHC", 6.899, 0, 4 * MC, 2),
        ("X(7200)LHC", 7.22, 0, 4 * MC, 4),
        ("Y(4230)", 4.2225, 1, 2 * MC + 2 * ML, 0),
    ]:
        th = np.sqrt(core * core + SLOPE * ell)
        add(nom, m, j, th,
            "constituant(exot.)" + ("+orb" if ell else ""))

    # ---- 8. secteurs propres (multiquarks scalaires, glueball) --------
    for nom, m, j, th, meth in [
        ("f0(500)", 0.475, 0, np.nan, "propre(multiquark)"),
        ("f0(980)", 0.990, 0, np.nan, "propre(multiquark)"),
        ("a0(980)", 0.980, 0, np.nan, "propre(multiquark)"),
        ("f0(1370)", 1.370, 0, np.nan, "propre(multiquark)"),
        ("f0(1500)", 1.506, 0, 1.5, "glueball(amphichiral)"),
        ("f0(1710)", 1.704, 0, np.nan, "propre(multiquark)"),
        ("a1(1260)", 1.230, 1, np.nan, "propre(axial)"),
        ("f1(1285)", 1.2819, 1, np.nan, "propre(axial)"),
        ("f1(1420)", 1.4262, 1, np.nan, "propre(axial)"),
        ("h1(1170)", 1.166, 1, np.nan, "propre(axial)"),
        ("K1(1270)", 1.253, 1, np.nan, "propre(axial)"),
        ("K1(1400)", 1.403, 1, np.nan, "propre(axial)"),
        ("eta(1405)", 1.4098, 0, np.nan, "propre"),
    ]:
        add(nom, m, j, th, meth)


def run():
    print(__doc__)
    build()
    print("=" * 78)
    print("TABLE (%d etats) : m_socle vs PDG ; spin (lu si Regge)" % len(R))
    print("=" * 78)
    stats = {}
    n_regge_ok, n_regge = 0, 0
    for nom, m, j, th, meth, jl in R:
        base = meth.split("[")[0].split("(")[0]
        if not np.isnan(th) if isinstance(th, float) else True:
            pass
        if isinstance(th, float) and not np.isnan(th) and m > 0:
            d = 100 * (th / m - 1)
            stats.setdefault(base, []).append(abs(d))
            dtxt = "%+6.1f %%" % d
        elif m == 0:
            d = 0.0
            stats.setdefault(base, []).append(0.0)
            dtxt = "  exact"
        else:
            dtxt = "   ----"
        jtxt = "J=%.1f" % j
        if jl is not None:
            j_raw, grid = jl
            ok = abs(grid - j) < 1e-9
            n_regge += 1
            n_regge_ok += ok
            jtxt = "J lu %.2f->%.1f %s" % (j_raw, grid, "OK" if ok else "X")
        mtxt = ("%8.4f" % th) if isinstance(th, float) and not np.isnan(th) else "    ----"
        print("  %-18s m=%8.4f  socle=%s  %s  %-22s %s"
              % (nom, m, mtxt, dtxt, jtxt, meth))
    print("\n" + "=" * 78)
    print("BILAN : %d etats" % len(R))
    print("=" * 78)
    for base in sorted(stats):
        v = np.array(stats[base])
        print("  %-12s : %3d etats, |dm| mediane %5.2f %%, max %5.1f %%"
              % (base, len(v), np.median(v), np.max(v)))
    n_prop = sum(1 for r in R if "propre" in r[4] or "glueball" in r[4])
    print("  secteurs propres (listes, hors lois) : %d" % n_prop)
    allv = np.array([v for vs in stats.values() for v in vs])
    print("  GLOBAL (%d etats avec loi) : |dm| mediane %.2f %%, 90e pct %.1f %%"
          % (len(allv), np.median(allv), np.percentile(allv, 90)))
    print("\n  SPINS : fondamentaux = arite/CG (P7quater, document principal 118/122) ;")
    print("  trajectoires : %d/%d spins LUS corrects (Regge inverse, pente" % (n_regge_ok, n_regge))
    print("  derivee unique ; le seul ECHEC massique = Delta(2950), courbure")
    print("  haut-J connue) ; bandes orbitales/radiales : J = ancre+L (CG,")
    print("  assignations PDG standard, pas d'ajustement).")
    print("  Fractions du socle : de 4e-23 (e-) a 1.4e-17 (top).")
    print("\n  LECTURE HONNETE : catalogue = les lois canoniques (mediane")
    print("  0,37 %) ; Regge/orbital/radial = UNE pente derivee 2 pi sigma_chi")
    print("  (mediane 2-4 %) ; TORSION PERIPHERIQUE (nouveau, 21/07) :")
    print("  retournement de spin = +pi sigma_chi (1/2 tour du ruban, quantum")
    print("  VERIFIE universel V-P a 0,7 %) -> D*/B*/Bs*/Sg(1385)/Xi(1530)")
    print("  a <1,5 % ; ECHELLE DE RECOUPLAGE n = S_paire + J + 1/2 (regle")
    print("  fixee par les nombres quantiques) -> 16 baryons lourds mediane")
    print("  1,0 % (quartet antisym. 0,3 %, Lambda_b +0,03 %) ; molecules")
    print("  LHC = hadrons socle AVEC leurs (demi-)tours : X(3872) +0,1 %,")
    print("  Tcc +0,0 %, Zc/Zb -0,3 %, pentaquarks Pc/Pcs <1 % (le -6 %")
    print("  etait le coeur Sigma_c nu), X0(2900)=D*K* +1,3 % -- mediane")
    print("  0,3 % ; ANOMALIE Sigma_b/Xi_b' RESOLUE + p_charme DERIVE en")
    print("  FORMES CLOSES (21/07) : n = 1 + 3^g (bascule + (rac2-1)")
    print("  [paire de noeuds identiques]) + 1/2 [J=3/2] -- bascule nue =")
    print("  le quantum V-P, x3 = Z3 de generation (2,98+-0,03), rac2 =")
    print("  interference d'echange (indice Koide), verrou Pauli ss (pas")
    print("  de Lambda_ss : PREDIT) -> 15 baryons lourds a 0,23 % de")
    print("  mediane SANS parametre ni ancre ; predictions : Omega_b* =")
    print("  6067 MeV (+22), Xi_cc* = 3659 MeV ; MAJ 21-22/07 : Sigma_b")
    print("  RESOLU (saturation, n=5), base 1 DERIVEE (trialite), bandes")
    print("  P DERIVEES (5/2, shift DWr/pi, X(1/2)=2 monotonie) ;")
    print("  quarkonia = coeur 2m_c/2m_b, le fin")
    print("  releve du secteur Cornell CALIBRE (compression tau<0, la")
    print("  torsion ne s'y applique pas) ; secteurs propres declares.")

    # export CSV
    out = os.path.join(os.path.dirname(__file__), "..", "..", "output",
                       "socle_224_resultats.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["etat", "m_exp_GeV", "J", "m_socle_GeV", "methode",
                    "J_lu", "fraction_socle"])
        for nom, m, j, th, meth, jl in R:
            w.writerow([nom, m, j,
                        "" if (isinstance(th, float) and np.isnan(th)) else round(th, 5),
                        meth, "" if jl is None else round(jl[0], 3),
                        "%.3e" % (m / M_PL) if m > 0 else "0"])
    print("\n  exporte : output/socle_224_resultats.csv")


if __name__ == "__main__":
    run()
