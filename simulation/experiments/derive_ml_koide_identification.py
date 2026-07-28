"""Reduire m_l/m_base a un rapport derive : l'identification Koide (19/07).

CANDIDAT STRUCTUREL (pas un scan de formes) : le trefle 3_1 porte DEUX
masses dans la theorie --
  - m_l = 310 MeV : masse constituante du trefle (plateau leger, CALIBREE) ;
  - mu_Koide = (3/4) sqrt(sigma_chi) = 313,85 MeV : l'echelle leptonique
    du MEME noeud (DERIVEE, recalibration 2026-07-18).
Si le 3_1 n'a qu'UNE echelle de masse (identification), alors :

    m_l = (3/4) sqrt(sigma_chi)   =>   m_l/m_base = 3/(4 S_octet)  DERIVE

et le comptage passe de 9 a 8 masses calibrees. Cout : +1,24 % sur m_l.

TEST DECISIF (le meme protocole que l'elimination de m_s) : le secteur
constituant lourd-ouvert, lineaire en (m_l, m_c, m_b) avec
m_s = r_geo m_l lie. Trois fits :
  A. libre (m_l, m_c, m_b)          -> optimum + RMS de reference ;
  B. m_l = 310 (calibre, document principal)     -> refit (m_c, m_b), RMS ;
  C. m_l = 313,85 (identification)  -> refit (m_c, m_b), RMS.
Verdict : si RMS_C - RMS_B est petit devant l'erreur de modele (~2,7 %),
la contrainte est GRATUITE et l'identification tient.

NB null : 3/4 n'est PAS scanne post-hoc -- c'est l'echelle Koide derivee
du meme noeud (pre-enregistree par la structure) ; test a UNE forme.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import simulation.experiments.derive_ms_pincement as dm

R_GEO = dm.RATIO_GEO
SQRT_SIGMA = 418.47
ML_KOIDE = 0.75 * SQRT_SIGMA          # 313.85
ML_CANON = 310.0
M_BASE = 938.240
S_OCTET = 2.242006
L_TREFLE = 16.372                     # ropelength 3_1
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


def build_rows():
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
    return rows


def fit(rows, ml_pin=None):
    """Moindres carres relatifs ; ml libre si ml_pin None, sinon pinne."""
    y = np.array([(me - m0) / me for _, _, me, m0, *_ in rows])
    if ml_pin is None:
        A = np.array([[al / me, ac / me, ab / me]
                      for _, _, me, m0, al, ac, ab in rows])
        theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ theta
        return theta, res
    y2 = y - np.array([ml_pin * al / me
                       for _, _, me, m0, al, ac, ab in rows])
    A = np.array([[ac / me, ab / me]
                  for _, _, me, m0, al, ac, ab in rows])
    theta, *_ = np.linalg.lstsq(A, y2, rcond=None)
    res = y2 - A @ theta
    return np.array([ml_pin, theta[0], theta[1]]), res


def rms(res):
    return 100.0 * float(np.sqrt(np.mean(res ** 2)))


def run():
    print("=== Identification m_l = (3/4) sqrt(sigma_chi) : test d'elimination ===\n")
    print("mu_Koide = (3/4) x %.2f = %.2f MeV ; m_l calibre = %.0f (%+.2f %%)" % (
        SQRT_SIGMA, ML_KOIDE, ML_CANON, 100 * (ML_KOIDE / ML_CANON - 1)))
    print("rapport vise : m_l/m_base = 3/(4 S_octet) = %.6f (mesure %.6f, %+.2f %%)\n" % (
        0.75 / S_OCTET, ML_CANON / M_BASE,
        100 * ((0.75 / S_OCTET) / (ML_CANON / M_BASE) - 1)))

    rows = build_rows()
    print("secteur lourd-ouvert : %d etats (m_s = %.4f m_l lie)\n" % (len(rows), R_GEO))

    (mlA, mcA, mbA), resA = fit(rows)
    (mlB, mcB, mbB), resB = fit(rows, ml_pin=ML_CANON)
    (mlC, mcC, mbC), resC = fit(rows, ml_pin=ML_KOIDE)
    print("A. libre         : m_l = %.1f ; m_c = %.0f ; m_b = %.0f ; RMS = %.2f %%" % (
        mlA, mcA, mbA, rms(resA)))
    print("B. m_l = 310     : m_c = %.0f ; m_b = %.0f ; RMS = %.2f %%" % (mcB, mbB, rms(resB)))
    print("C. m_l = 313,85  : m_c = %.0f ; m_b = %.0f ; RMS = %.2f %%" % (mcC, mbC, rms(resC)))
    dBC = rms(resC) - rms(resB)
    print("\nCOUT DE L'IDENTIFICATION : Delta RMS (C - B) = %+.2f point" % dBC)
    print("  refit : m_c %+.1f %% ; m_b %+.1f %% (vs B)" % (
        100 * (mcC / mcB - 1), 100 * (mbC / mbB - 1)))
    print("  m_s propage : %.1f -> %.1f MeV (%+.2f %%)" % (
        R_GEO * ML_CANON, R_GEO * ML_KOIDE, 100 * (ML_KOIDE / ML_CANON - 1)))

    # degenerescence avec le candidat CS (banc du 17/07)
    d_koide = 0.75 / L_TREFLE
    d_cs = 2 * np.pi / 137.0
    print("\nDEGENERESCENCE remarquable : sous l'identification, le tube")
    print("  d = D sqrt(sigma)/hbar c = 3/(4 l(3_1)) = %.5f ;" % d_koide)
    print("  le candidat CS (banc 17/07)  2 pi/137   = %.5f  (%+.2f %%)" % (
        d_cs, 100 * (d_cs / d_koide - 1)))
    print("  -> les deux voies (Koide et flux CS) coincident a 0,1 % :")
    print("  2 pi l(3_1) = 137,1 x (3/4)... la resolution entiere du banc")
    print("  (N = 137) devient COMPATIBLE si m_l = mu_Koide.")

    print("\nVERDICT (criteres du protocole m_s) :")
    if abs(dBC) < 0.15:
        print("  Delta RMS %.2f pt << erreur de modele 2,7 %% : la contrainte" % dBC)
        print("  est GRATUITE -- l'identification m_l = (3/4) sqrt(sigma_chi)")
        print("  TIENT le test d'elimination. m_l/m_base = 3/(4 S_octet)")
        print("  serait DERIVE (comptage 9 -> 8). Statut : candidat fort,")
        print("  a confronter au reste du plateau leger avant consolidation.")
    else:
        print("  Delta RMS %.2f pt : cout non negligeable -- l'identification" % dBC)
        print("  est DEFAVORISEE par le secteur constituant ; m_l reste calibre.")


if __name__ == "__main__":
    run()
