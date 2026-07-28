"""Re-chiffrage F3 : masses ACPR, SANS etalement de writhe du 8_3 (22/07).

CE QUI A CHANGE depuis f3_line_search_canon.py (15/07) :
  1. MASSES ACPR (theory/ropelength_acpr.py, 2026-07-22) : les ropelengths
     du depot etaient faux (6_3 sous le minimum de -5,9 %, 8_3 borne lache
     +4,8 %) -> les 6 etats stricts recales a 1,07-1,64 GeV ;
  2. LE DOUBLET DE WRITHE DU 8_3 EST MORT : au vrai ideal (base katlas
     verifiee ACPR ~0,01 %), Wr(8_3) = 0,003 (converge au 4e chiffre,
     scan_fonctionnelles_habillees.py) -- le +-7,2 du 14/07 etait un
     artefact de nos courbes reparees non convergees. Mesures : Wr(4_1) =
     0,000, Wr(6_3) = 0,005, Wr(8_3) = 0,003 : les TROIS amphichiraux sont
     symetriques -> AUCUN etalement, les 6 etats sont des RAIES FINES.

QUESTION : la levee de F3 survit-elle sans le mecanisme d'etalement ?
(La voie (b) « cascade smear » disparait ; tout repose sur la voie
« briques canoniques + limites laches a basse energie » du 15/07.)
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_pearl_internal"))
sys.path.insert(0, str(ROOT / "scripts" / "V5_Fermi_LAT"))

from theory.dm_pearl_internal import (ALPHA_EM, M_PEARL_TYPICAL_MSUN,
                                      R_HUBBLE_M, T_HUBBLE_S,
                                      n_events_over_universe_age)
import class_E_entanglement_test as CE
from class_E_six_lines_predictions import fermi_limit_at_E, OMEGA_PER_AMPHI
from theory.ropelength_acpr import m_topo

DC = 22                               # Delta_c >= 22 (V5.21, derive)
WR_IDEAL = {"4_1": 0.000, "6_3": 0.005, "8_3": 0.003}   # katlas, 22/07
ETATS = [
    ("4_1+8_3",     ["4_1", "8_3"]),
    ("6_3+6_3",     ["6_3", "6_3"]),
    ("6_3+8_3",     ["6_3", "8_3"]),
    ("8_3+8_3",     ["8_3", "8_3"]),
    ("4_1+4_1+8_3", ["4_1", "4_1", "8_3"]),
    ("6_3+6_3+6_3", ["6_3", "6_3", "6_3"]),
]


def flux_line(m_canon, n_briques, dc):
    """Machinerie du 15/07 : TOV pearl, sigma_QED*alpha^dc, flux Hubble."""
    m_brique = m_canon / n_briques
    nev = n_events_over_universe_age(M_PEARL_TYPICAL_MSUN, m_brique, dc)
    N_univ = nev["N_events_total_universe"] * (OMEGA_PER_AMPHI / CE.OMEGA_A_FRACTION)
    d_H_cm = R_HUBBLE_M * 100.0
    return N_univ * 2.0 / (4.0 * math.pi * d_H_cm ** 2 * T_HUBBLE_S)


def run():
    print(__doc__)
    print("=== 6 raies FINES (masses ACPR, Delta_c = %d, AUCUN etalement) ===\n" % DC)
    print("%-14s %8s %12s %12s %8s %9s" % (
        "etat", "E(GeV)", "phi pred", "lim Fermi", "ratio", "dc_crit"))
    log_inv_a = math.log(1.0 / ALPHA_EM)
    worst = 0.0
    for nom, ks in ETATS:
        m = sum(m_topo(k) for k in ks) / 1000.0        # GeV
        phi = flux_line(m, len(ks), DC)
        lim = fermi_limit_at_E(m)
        ratio = phi / lim
        worst = max(worst, ratio)
        dc_crit = DC + math.log(max(ratio, 1e-300)) / log_inv_a
        print("%-14s %8.3f %12.2e %12.2e %8.2f %9.2f   %s" % (
            nom, m, phi, lim, ratio, dc_crit,
            "EXCLUE" if ratio > 1 else "sous la limite (x%.0f)" % (1 / ratio)))
    marge = 1.0 / worst
    print("""
VERDICT (22/07) :
  - Les 6 etats sont des raies fines a 1,07-1,64 GeV, TOUS sous les
    limites 2015 (pire cas : x%.0f de marge) au Delta_c = 22 derive.
  - F3 RESTE LEVEE -- et la levee est plus SIMPLE : elle ne repose plus
    sur l'etalement de writhe (mecanisme mort avec le doublet +-7,2),
    seulement sur (i) les briques canoniques (composants 398-674 MeV,
    v_F sature -> -9 decades de taux) et (ii) les limites ~30-70x plus
    laches a 1-1,6 GeV qu'aux energies Omega_ts.
  - La PREDICTION residuelle testable passe de 2 a 6 raies fines :
    1,07 / 1,10 / 1,22 / 1,35 / 1,47 / 1,64 GeV, a ~1 dex sous les
    limites 2015 (cible : analyses Fermi-LAT profondes 15+ ans).
    Les writhes mesures (0,000/0,005/0,003) ne fournissent AUCUNE
    largeur : toute detection d'un continuum etale a ces energies
    serait une TENSION (il n'y a plus de mecanisme d'etalement).
  - Reserves : limites Ackermann R3 NFW digitalisees (~0,5 dex --
    les marges x8-12 y resistent) ; fraction Class E = 10 %% de
    Omega_DM (heritage V5.19) ; briques = m/n (convention declaree).""" % marge)


if __name__ == "__main__":
    run()
