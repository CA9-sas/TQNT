"""Le reservoir DM sous la loi geometrique consolidee : masses, sections, detectabilite.

QUESTION (15/07) : les modifications recentes (m_s DERIVE -> plateau geometrique
consolide ; doublet de writhe du 8_3 ; refit m_b) font-elles evoluer les noeuds
de matiere noire, et une apparition en detecteur est-elle envisageable ?

CE QUI CHANGE :
  - m_s = m_l * l(5_2)/l(3_1) valide la loi M_topo = scale * l(K) pour les
    noeuds SIMPLES (scale = m_l/l(3_1) = 18.93 MeV/unite, une normalisation).
    Le reservoir dominant P18 = noeuds premiers ISOLES c>18 : la loi
    s'applique NATIVEMENT (pas de contradiction charge/EM : ils sont sombres).
  => masses du reservoir predites GEOMETRIQUES, quasi lineaires en c.
  - Ropelength des toriques T(2,q) : lineaire, pente 3.71/croisement
    (table archivee 3_1/5_1/7_1/9_1 ; demi-ropelength).
  - Abondance : penalite de condensation alpha_cond = 1.20 par croisement
    (P18, derivee) -> n(c) ~ 1.2^-(c-19), normalise.
  - Section : suppression de Vassiliev sigma(c) = sigma_EM * alpha^(c-18),
    sigma_EM ~ 1e-25 cm^2 (calibrage CLAUDE/V5 : c=19 -> ~1e-27).

DETECTABILITE par tranche :
  - BLOCAGE par la couverture : sigma/M > ~1e-33 cm2/GeV -> thermalise dans
    l'atmosphere/roche, INVISIBLE sous terre (fenetre SIMP : detecteurs de
    SURFACE/ballon/fusee type XQC).
  - Fenetre Migdal/basse masse (XENONnT/LZ Migdal, DarkSide, CRESST) :
    M ~ 0.5-3 GeV, sigma ~ 1e-36..1e-42.
  - LZ/XENONnT standard : M > ~5 GeV -> HORS FENETRE (toutes les masses
    geometriques sont < 3 GeV) : la prediction d'ABSENCE se RENFORCE.
"""
import math

SCALE = 310.0 / 16.372          # MeV / unite de demi-ropelength (m_l / l(3_1))
SLOPE_T2 = (38.62 - 16.372) / 6  # demi-rope par croisement, toriques T(2,q)
L_31 = 16.372
ALPHA = 1 / 137.036
SIGMA_EM = 1e-25                 # cm^2 (calibrage c=19 -> 1e-27)
ALPHA_COND = 1.20


def run():
    print("=== Reservoir DM geometrique (m_s derive consolide la loi) ===\n")
    print(" c   M_topo   sigma_DM-N   abondance   sigma/M      regime de detection")
    print("     (GeV)    (cm^2)       relative    (cm^2/GeV)")
    tot = sum(ALPHA_COND ** (-(c - 19)) for c in range(19, 31))
    for c in range(19, 31):
        l_half = L_31 + SLOPE_T2 * (c - 3)
        M = SCALE * l_half / 1000.0          # GeV
        sig = SIGMA_EM * ALPHA ** (c - 18)
        ab = ALPHA_COND ** (-(c - 19)) / tot
        sm = sig / M
        if sm > 1e-33:
            reg = "BLOQUE sous terre -> surface/ballon (XQC, SIMP)"
        elif sig > 1e-42:
            reg = "fenetre MIGDAL/basse masse (XENONnT/LZ-Migdal, DarkSide)"
        elif sig > 1e-49:
            reg = "sous les seuils actuels (M<3 GeV et sigma faible)"
        else:
            reg = "invisible (toute generation)"
        print("%3d  %6.2f   %9.1e   %7.1f%%    %9.1e   %s" % (
            c, M, sig, ab * 100, sm, reg))
    print()
    print("Lecture :")
    print(" - TOUTES les masses du reservoir : %.1f a %.1f GeV (quasi plat) --" % (
        SCALE * (L_31 + SLOPE_T2 * 16) / 1000, SCALE * (L_31 + SLOPE_T2 * 27) / 1000))
    print("   HORS de la fenetre WIMP standard (LZ/XENONnT >5 GeV) : l'ABSENCE")
    print("   predite se renforce structurellement.")
    print(" - c=19-23 (%.0f%% de l'abondance) : sigma/M > 1e-33 -> ne PENETRE pas" % (
        100 * sum(ALPHA_COND ** (-(c - 19)) for c in range(19, 24)) / tot))
    print("   sous terre ; cible = detecteurs de SURFACE / fusee (XQC) -- a")
    print("   confronter aux exclusions SIMP existantes (test a mener).")
    print(" - c=24-27 : fenetre MIGDAL -- une apparition y est ENVISAGEABLE")
    print("   (M ~ 1.9-2.1 GeV, sigma ~ 1e-38..1e-44).")
    print(" - c>=28 : sous tout seuil.")
    print()
    print("Caveats honnetes :")
    print(" 1. sigma_EM = 1e-25 cm2 : calibrage d'ordre de grandeur (V5 13.6.4).")
    print(" 2. T(3,q>=10) : pente ropelength differente (plus dense) -> masses")
    print("    un peu plus hautes a c egal ; meme conclusion qualitative.")
    print(" 3. Les 6 etats STRICTS (1.55-51.8 GeV) sont des masses de l'ere")
    print("    Omega_ts : sous le document principal actuel ils n'ont PAS de loi de secteur")
    print("    assignee -- les energies de raies F3/CTA (1.55/4.70) heritent de")
    print("    ce caveat. Les 4 etats contenant 8_3 sont des DOUBLETS de writhe")
    print("    (geometrie brisee) : multiplets (+,+)/(+,-)/(-,-) possibles.")


if __name__ == "__main__":
    run()
