"""Re-passe detection directe DM : les courbes de juin sous la
prediction DERIVEE de juillet (23/07).

CONTEXTE JUIN (data/analysis/DIRECT_DETECTION_6363_REPORT.md) : le
candidat strict 6_3+6_3 etait pris a m = 1,55 GeV (masse Omega_ts,
ARCHIVEE) et sigma_chiN restait OUVERT sur 44 decades (ansatze A/B/C :
1,5e-31 / 1,4e-23 / 5,8e-67 cm^2) -- « V4/V5 does not yet derive
alpha_top from first principles ».

JUILLET : la chaine est FERMEE --
  - masses ACPR : 6_3+6_3 = 1,10 GeV ; les 6 stricts 1,07-1,64 GeV
    (population SECONDAIRE) ; reservoir DOMINANT = noeuds isoles
    T(2,q>=23), bord c=23-25 a M = 1,7-1,9 GeV ;
  - canal SELECTIONNE (neutralisation par framing -> polarisabilite du
    tube, anneau conducteur ; origine microscopique = courant de rail
    verrouille, S50) : sigma_eq(c=23) = 1,1e-48 cm^2 [2,5e-49, 4,6e-48]
    (equivalence SI Ar/Xe calculee ; derive_alpha_pol_si_rk.py).
CONFRONTATIONS :
 [1] les 3 courbes de juin (LZ 2024, XENONnT 2023, PandaX-4T 2024) ;
 [2] DS-50 2022 Migdal/NQ basse masse (ancrage etabli, 6e-43 @ 3 GeV) ;
 [3] resolution de l'eventail de juin : les donnees FERMENT par le haut
     (A et B exclus), la derivation ferme par le bas -- la valeur
     derivee tombe dans la fenetre survivante.
"""
import csv
import math
import os

D = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                 "dm_direct_detection")

# prediction canonique (juillet)
SIGMA_EQ = 1.1e-48                    # cm^2, c=23, derive
SIGMA_BAND = (2.5e-49, 4.6e-48)
M_EDGE = (1.7, 1.9)                   # GeV, bord c=23-25 (34/22/15 %)
STRICTS = [("4_1+8_3", 1.07), ("6_3+6_3", 1.10), ("6_3+8_3", 1.22),
           ("8_3+8_3", 1.35), ("4_1+4_1+8_3", 1.47), ("6_3+6_3+6_3", 1.64)]

# ansatze du rapport de juin (a 1,55 GeV Omega_ts)
JUIN = [("A (U(1) topologique)", 1.5e-31), ("B (resonance chi)", 1.4e-23),
        ("C (P22 Planck)", 5.8e-67)]

# DS-50 2022 NQ (numerisation etablie, migdal_confrontation_digitized.py)
DS50 = [(1.2, 3e-40), (1.4, 4e-41), (1.6, 1.2e-41), (1.8, 5e-42),
        (2.0, 2.5e-42), (2.5, 1.1e-42), (3.0, 6e-43), (4.0, 4.5e-43)]


def interp_log(pts, m):
    xs = [math.log10(a) for a, _ in pts]
    ys = [math.log10(b) for _, b in pts]
    if m <= pts[0][0]:
        return None
    lx = math.log10(m)
    for i in range(len(xs) - 1):
        if xs[i] <= lx <= xs[i + 1]:
            t = (lx - xs[i]) / (xs[i + 1] - xs[i])
            return 10 ** (ys[i] + t * (ys[i + 1] - ys[i]))
    return 10 ** ys[-1]


def load_curve(path, mass_col=0, lim_col=1):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace("\t", ",").split(",") if p.strip()]
            try:
                m, s = float(parts[mass_col]), float(parts[lim_col])
                pts.append((m, s))
            except (ValueError, IndexError):
                continue
    return sorted(pts)


def main():
    print(__doc__)
    curves = {
        "LZ 2024": load_curve(os.path.join(D, "LZ_2024", "curve.csv")),
        "XENONnT 2023": load_curve(os.path.join(D, "XENONnT_2023", "curve.csv")),
        "PandaX-4T 2024": load_curve(
            os.path.join(D, "PandaX_4T_2024", "curve_pandax_digit.csv")),
    }
    print("=" * 78)
    print("[1] LES 3 COURBES DE JUIN vs LE RESERVOIR (M = 1,7-1,9 GeV)")
    print("=" * 78)
    for nom, pts in curves.items():
        m_min = pts[0][0]
        s_min = min(s for _, s in pts)
        print("  %-15s : seuil %.0f GeV, minimum sigma = %.2e cm^2" % (
            nom, m_min, s_min))
        print("     -> reservoir (%.1f-%.1f GeV) SOUS le seuil de masse ;"
              % M_EDGE)
        print("        et sigma_eq = %.1e est %.1f dex SOUS le minimum"
              " ABSOLU de la courbe" % (
                  SIGMA_EQ, math.log10(s_min / SIGMA_EQ)))
    print("""  -> ABSENCE PREDITE, DOUBLEMENT SATISFAITE : hors fenetre de
  masse ET hors portee en section efficace (la valeur derivee est sous
  le meilleur point de LZ, atteint a 40 GeV -- a 2 GeV la sensibilite
  s'effondre de plusieurs decades supplementaires).""")

    print("=" * 78)
    print("[2] DS-50 2022 BASSE MASSE (la SEULE experience dans la fenetre)")
    print("=" * 78)
    print("    M(GeV)  population        lim_DS50    sigma_eq    marge")
    for m in (1.7, 1.8, 1.9):
        lim = interp_log(DS50, m)
        print("    %.1f    bord c=23-25      %.1e    %.1e    %.1f dex" % (
            m, lim, SIGMA_EQ, math.log10(lim / SIGMA_EQ)))
    for nom, m in STRICTS:
        lim = interp_log(DS50, m)
        txt = "%.1e" % lim if lim else "sous seuil"
        marge = (" %.1f dex" % math.log10(lim / SIGMA_EQ)) if lim else "  --"
        print("    %.2f   strict %-10s %9s    %.1e   %s" % (
            m, nom, txt, SIGMA_EQ, marge))
    hi = SIGMA_BAND[1]
    lim18 = interp_log(DS50, 1.8)
    print("""  -> marge %.1f dex au bord (bande haute %.1e : %.1f dex) --
  l'absence est ROBUSTE aussi dans DS-20k/XLZD (gain attendu 2-3 dex,
  il en manque >= 4). Toute detection a 1,07-2 GeV REFUTERAIT.""" % (
        math.log10(lim18 / SIGMA_EQ), hi, math.log10(lim18 / hi)))

    print("=" * 78)
    print("[3] RESOLUTION DE L'EVENTAIL DE JUIN (44 decades -> ferme)")
    print("=" * 78)
    lim_155 = interp_log(DS50, 1.55)
    print("    limite DS-50 a 1,55 GeV (masse Omega_ts du rapport) :"
          " %.1e cm^2" % lim_155)
    for nom, s in JUIN:
        if s > lim_155:
            verdict = "EXCLU par les donnees (+%.0f dex)" % math.log10(
                s / lim_155)
        else:
            verdict = "permis (%.0f dex sous la limite)" % math.log10(
                lim_155 / s)
        print("    ansatz %-22s sigma = %.1e : %s" % (nom, s, verdict))
    print("""  -> les DONNEES ferment l'eventail par le HAUT (A exclu ~10 dex,
  B exclu ~18 dex) ; la DERIVATION (canal polarisabilite, R_K mesure
  SONO) le ferme par le BAS (C etait 19 dex trop faible : tue par le
  mecanisme, pas par les donnees). La valeur derivee sigma_eq = 1,1e-48
  tombe DANS la fenetre survivante -- le « 44-order spread / honest
  limitation » du rapport de juin est RESOLU des deux cotes.
  NB masses : le 6_3+6_3 de juin (1,55 GeV) etait Omega_ts ; masse
  canonique ACPR = 1,10 GeV (les 6 stricts : 1,07-1,64).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  (1) Les 3 courbes de juin (LZ/XENONnT/PandaX, seuils >= 6 GeV) ne
      touchent NI la fenetre de masse (1,07-2 GeV) NI la section
      efficace derivee (sous le minimum absolu de LZ) : l'absence
      standard predite est verifiee, trivialement mais proprement.
  (2) La seule experience dans la fenetre (DS-50 basse masse) laisse
      ~6,7 dex de marge au bord du reservoir : prediction d'ABSENCE
      robuste pour DS-20k/XLZD ; falsifiable (detection 1-2 GeV).
  (3) Le verdict OUVERT de juin (44 decades entre ansatze) est FERME :
      donnees par le haut, derivation par le bas, valeur derivee dans
      la fenetre. Sondes reelles du reservoir : gravitationnelles
      (perles V5.19) + cosmologiques (reconstruction Omega).""")


if __name__ == "__main__":
    main()
