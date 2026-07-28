"""Pincer m_l : fit du secteur constituant avec barres d'erreur (17/07).

OBJECTIF (chemin (i) du banc du mecanisme CS) : m_l a < 0.3 % pour donner
au test entier N = 2 pi lambda_sigma/D sa resolution (un pas = 0.72 %).

METHODE : le secteur lourd-ouvert (celui du test d'elimination de m_s)
est LINEAIRE en (m_l, m_c, m_b) avec m_s = r_geo m_l lie (r_geo =
l(5_2)/l(3_1) = 1.5960, la relation derivee) :
    m(etat) = a_l m_l + a_c m_c + a_b m_b (+ 140 pour les vecteurs).
Moindres carres relatifs ; sigma(m_l) depuis la covariance x la variance
residuelle (le bruit = l'erreur de modele ~2.7 %, pas le bruit PDG).
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import simulation.experiments.derive_ms_pincement as dm

R_GEO = dm.RATIO_GEO
HEAVY_CATS = ('charmé', 'charmé-V', 'beau', 'beau-V', 'charmé-B', 'beau-B',
              'double-c')


def model(nm, cat, ml, mc, mb):
    ms = R_GEO * ml
    m = dm.csv_constituent(nm, ms, mc, mb)
    if m is None:
        m = dm.heavy_open(nm, cat, ms, mc, mb)
        if m is not None and cat in ('charmé-V', 'beau-V'):
            m += 140
    return m


def run():
    PDG = dm.load_pdg()
    rows = []
    for nm, me, J, Q, c in PDG:
        if c not in HEAVY_CATS or nm in dm.SERVED or me <= 0:
            continue
        m0 = model(nm, c, 0.0, 0.0, 0.0)
        if m0 is None:
            continue
        al = model(nm, c, 1.0, 0.0, 0.0) - m0
        ac = model(nm, c, 0.0, 1.0, 0.0) - m0
        ab = model(nm, c, 0.0, 0.0, 1.0) - m0
        rows.append((nm, c, me, m0, al, ac, ab))
    print("etats du fit : %d" % len(rows))
    y = np.array([(me - m0) / me for _, _, me, m0, *_ in rows])
    A = np.array([[al / me, ac / me, ab / me]
                  for _, _, me, m0, al, ac, ab in rows])
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ theta
    dof = len(y) - 3
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    ml, mc, mb = theta
    sml, smc, smb = np.sqrt(np.diag(cov))
    print("\nFIT (m_s = %.4f m_l lie) :" % R_GEO)
    print("  m_l = %.1f +- %.1f MeV (%.2f %%)   [calibre : 310]" % (ml, sml, 100 * sml / ml))
    print("  m_c = %.0f +- %.0f ; m_b = %.0f +- %.0f" % (mc, smc, mb, smb))
    print("  RMS residuel = %.2f %% (dof = %d)" % (100 * np.sqrt(s2), dof))
    corr = cov[0, 1] / (sml * smc)
    print("  corr(m_l, m_c) = %.2f" % corr)

    # test entier
    SQS, HBARC, L31 = 417.65, 197.33, 16.372
    lam_s = HBARC / SQS
    for m in (ml - sml, ml, ml + sml):
        D = (m / (SQS ** 2 / HBARC)) / L31
        print("  m_l = %.1f -> N = %.1f" % (m, 2 * np.pi * lam_s / D))
    N_c = 2 * np.pi * lam_s / ((ml / (SQS ** 2 / HBARC)) / L31)
    sN = N_c * sml / ml
    print("\nTEST ENTIER : N = %.1f +- %.1f  (137 a %.1f sigma ; pas entier = %.2f sigma)" % (
        N_c, sN, abs(N_c - 137) / sN, 1.0 / sN))
    print("\nVERDICT :")
    if sml / ml < 0.003:
        print(" - m_l est PINCE a %.2f %% : resolution entiere atteinte." % (100 * sml / ml))
    else:
        print(" - Le secteur constituant pince m_l a %.1f %% seulement" % (100 * sml / ml))
        print("   (degenerescence m_l <-> m_c/m_b dans les sommes ; le bruit est")
        print("   l'erreur de MODELE ~%.1f %%, incompressible sans meilleure loi)." % (
            100 * np.sqrt(s2)))
        print(" - Le test entier reste NON RESOLU par cette voie : il faudrait")
        print("   %.1fx mieux. La voie (i) est epuisee honnetement ; restent" % (
            (100 * sml / ml) / 0.24))
        print("   (ii) deriver le rayon lambda_sigma, ou une loi de plateau plus")
        print("   fine (hyperfin constituant complet).")


if __name__ == "__main__":
    run()
