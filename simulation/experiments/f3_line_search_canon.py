"""Re-confrontation F3 (line search Fermi-LAT) aux ENERGIES CANONIQUES (15/07).

CONTEXTE : le test original (commit 06ad81f) confrontait les 6 raies Class E
aux energies Omega_ts (1.55-51.8 GeV) : 6/6 EXCLUES a Delta_c=22, ratios 1.57
a 7e12, Delta_c_crit 23-29 ; adouci 2026-07-14 (writhe du 8_3 etale les 4
grosses raies). Les masses canoniques (derive_strict_states_canon.py)
deplacent TOUT le spectre a 1.03-1.55 GeV : la confrontation est refaite.

CE QUI CHANGE :
  1. E_raie = m_canon (1.03-1.55 GeV) -> limites Fermi ~30-70x plus laches
     qu'a 16-52 GeV (la line search est moins sensible a basse energie) ;
  2. les BRIQUES du pearl sont les amphichiraux eux-memes (m_brique =
     m_canon/n : 397-706 MeV au lieu des 1.5-52 GeV passes comme m_knot
     dans l'ancien script) -> N_knots plus grand, taux PLUS ELEVES ;
  3. l'etalement de writhe E_tw/M est RESCALE par M_old/M_new (E_tw fixe,
     geometrie du 8_3) : x10-34 -> les 4 etats a 8_3 sont totalement
     etales (>>100%), la line search ne s'y applique plus du tout.

Scenario canonique : Delta_c >= 22 (V5.21, derive). Machinerie identique a
l'original (TOV pearl, sigma = sigma_QED alpha^dc, flux Hubble isotrope).
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
from class_E_six_lines_predictions import (fermi_limit_at_E, FERMI_DE_OVER_E,
                                           OMEGA_PER_AMPHI)

DC = 22            # Delta_c >= 22 (V5.21, canonique)
# (nom, m_canon_GeV, m_omega_ts_GeV, n_briques, contient_8_3)
CANON = [
    ("6_3+6_3",         1.030,  1.55, 2, False),
    ("6_3+6_3+6_3",     1.546,  4.70, 3, False),
    ("4_1+8_3",         1.104, 16.01, 2, True),
    ("8_3+8_3",         1.413, 17.24, 2, True),
    ("6_3+8_3",         1.221, 26.48, 2, True),
    ("4_1+4_1+8_3",     1.501, 51.78, 3, True),
]
# etalement de writhe aux masses Omega_ts (brouillon F3 writhe : 18-73 %
# pour les 4 etats a 8_3) -- on ne rescale que la BORNE INFERIEURE.
SMEAR_MIN_OLD = 0.18


def flux_line(m_canon, n_briques, dc):
    """Flux predit (ph/cm^2/s) : machinerie originale, briques canoniques."""
    m_brique = m_canon / n_briques
    nev = n_events_over_universe_age(M_PEARL_TYPICAL_MSUN, m_brique, dc)
    N_univ = nev["N_events_total_universe"] * (OMEGA_PER_AMPHI / CE.OMEGA_A_FRACTION)
    d_H_cm = R_HUBBLE_M * 100.0
    return N_univ * 2.0 / (4.0 * math.pi * d_H_cm ** 2 * T_HUBBLE_S)


def run():
    print("=== F3 re-confrontation aux energies canoniques (Delta_c = %d) ===\n" % DC)
    print("%-14s %8s %8s %12s %12s %9s  %s" % (
        "etat", "E(GeV)", "E_old", "phi pred", "lim Fermi", "ratio", "statut de raie"))
    log_inv_a = math.log(1.0 / ALPHA_EM)
    fins = []
    for nom, m, mold, nb, has83 in CANON:
        phi = flux_line(m, nb, DC)
        lim = fermi_limit_at_E(m)
        ratio = phi / lim
        if has83:
            smear = SMEAR_MIN_OLD * (mold / m)
            statut = "ETALE (writhe >= %.0f%% >> 12%% : line search N/A)" % (100 * smear)
        else:
            dc_crit = DC + math.log(max(ratio, 1e-300)) / log_inv_a
            statut = ("raie FINE : %s (dc_crit = %.1f)" % (
                "EXCLUE" if ratio > 1 else "sous la limite", dc_crit))
            fins.append((nom, m, ratio, dc_crit))
        print("%-14s %8.2f %8.2f %12.2e %12.2e %9.2f  %s" % (
            nom, m, mold, phi, lim, ratio, statut))

    print("\nVERDICT (15/07) :")
    print(" - Les 4 etats a 8_3 : etalement de writhe RESCALE par M_old/M_new")
    print("   (x10-34) : >> 100%% de largeur -- fondus dans le continu, la line")
    print("   search ne s'applique a AUCUN d'eux (la voie (b) devient totale")
    print("   pour ces etats).")
    for nom, m, ratio, dc_crit in fins:
        print(" - %s a %.2f GeV (raie fine) : ratio = %.2f -> %s" % (
            nom, m, ratio,
            "TENSION marginale" if 1 < ratio < 10 else
            ("EXCLUE" if ratio >= 10 else "COMPATIBLE")))
    print(" - Les Delta_c_crit calcules aux energies Omega_ts (23-29) sont")
    print("   remplaces par les valeurs ci-dessus.")
    print(" - Caveats : limites Ackermann R3 NFW digitalisees (~0.5 dex),")
    print("   extrapolation sous 1 GeV pour E < 1 GeV le cas echeant ;")
    print("   fraction Class E = 10%% de Omega_DM (heritage V5.19) ; briques")
    print("   canoniques m/n (l'ancien script passait la masse du produit).")


if __name__ == "__main__":
    run()
