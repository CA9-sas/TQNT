"""Table canonique de ropelengths (ACPR) — remplace la table historique (2026-07-22).

Source : Ashton, Cantarella, Piatek, Rawdon, « Self-contact sets for 50
tightly knotted and linked tubes » (arXiv:math/0508248), Appendice A,
colonne Rop (borne serree par gradient contraint). CONVENTION :
longueur / RAYON de tube (diviser par 2 pour des diametres).

Verdict de l'audit 2026-07-22 (verify_ropelengths_acpr.py) sur la table
historique du depot (mass_ropelength_algebra.py) : 5 valeurs justes
(toriques + 4_1 + 6_1), 9 bornes laches (+2 a +8,8 %), 4 impossibles
(sous le minimum -- courbes de mauvaise topologie probables).
"""

ROPELENGTH_ACPR = {   # longueur / rayon
    "3_1": 32.7448, "4_1": 42.0928, "5_1": 47.2016, "5_2": 49.4704,
    "6_1": 56.7150, "6_2": 57.0271, "6_3": 57.8435,
    "7_1": 61.4109, "7_2": 63.8956, "7_3": 63.9327, "7_4": 64.2724,
    "7_5": 65.2609, "7_6": 65.7012, "7_7": 65.6108,
    "8_1": 71.0241, "8_2": 71.4107, "8_3": 71.1655, "9_1": 75.7252,
}
# entrelacs (Rop, longueur/rayon)
LINKS_ACPR = {
    "2^2_1": 25.1388, "4^2_1": 40.0169, "5^2_1": 49.7723,
    "6^2_1": 54.3893, "6^2_2": 56.7028, "6^2_3": 58.1044,
    "6^3_1": 57.8170, "6^3_2": 58.0145, "6^3_3": 50.5745,
    "7^2_1": 64.2353,
}
ML = 310.0   # masse du trefle (calibree, transmutation)


def m_topo(K):
    """Masse statique geometrique M_topo = m_l * l(K)/l(3_1) [MeV]."""
    return ML * ROPELENGTH_ACPR[K] / ROPELENGTH_ACPR["3_1"]
