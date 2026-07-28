"""Sortie (b) : la distribution d'abondance du reservoir DEPUIS la cinetique canonique.

REGLE 6 : on part du modele CANONIQUE (alpha_refinement.py, commit bebfa51) :
    w(c) = m(c) * N(c) * exp(-alpha_cond * c),   alpha_cond = 1.20 (derive)
    N(c) = nombre de noeuds premiers (table Hoste-Thistlethwaite + 2.68^c/c)
et on le complete par la thermodynamique TQNT (Boltzmann EMERGENT, chapitre
thermo) a la temperature de condensation T_c = 67 MeV (celle du script
canonique), avec la loi de masse GEOMETRIQUE consolidee (m_s derive) :
    m_geo(c) = (m_l/l(3_1)) * l(T(2,c))  ~  0.066*c + 0.18 GeV.

TROIS VARIANTES de distribution dans la famille sombre (c = 19..30) :
  A. canonique-nombre : n(c) ~ N(c) e^{-alpha c}            (sans thermique)
  B. canonique-masse  : rho(c) ~ m_geo(c) N(c) e^{-alpha c} (poids du script)
  C. thermo complete  : n(c) ~ N(c) e^{-alpha c} e^{-m_geo(c)/T_c}
                        (Boltzmann emergent -- les regles TQNT completes)

QUESTION : la distribution evacue-t-elle le conflit SIMP/XQC (qui exige que
c=19-24 soit une fraction NEGLIGEABLE) ?

EFFETS COLLATERAUX a auditer (la refonte retroagit sur la cinetique) :
  - la derivation d'alpha_cond utilisait c_median du poids V5 (m ~ c^6.65,
    mediane haute ~30) ; sous la masse geometrique la mediane descend ->
    alpha(c_med) derive se deplace (sensibilite a chiffrer) ;
  - le ratio Omega_DM/Omega_b = 5.45 a ete calcule avec les masses V5.
"""
import math

RHO_ST = 2.68
HT = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165, 11: 552,
      12: 2176, 13: 9988, 14: 46972, 15: 253293, 16: 1388705}
ALPHA_COND = 1.20
T_C = 0.067            # GeV
SCALE = 310.0 / 16.372
SLOPE = (38.62 - 16.372) / 6
L31 = 16.372
ALPHA_EM = 1 / 137.036
SIGMA_EM = 1e-25


def N_knots(c):
    if c in HT:
        return float(HT[c])
    return RHO_ST ** c / c


def m_geo(c):
    return SCALE * (L31 + SLOPE * (c - 3)) / 1000.0   # GeV


def dist(kind, cs):
    w = []
    for c in cs:
        base = N_knots(c) * math.exp(-ALPHA_COND * c)
        if kind == 'A':
            w.append(base)
        elif kind == 'B':
            w.append(m_geo(c) * base)
        elif kind == 'C':
            w.append(base * math.exp(-m_geo(c) / T_C))
    s = sum(w)
    return [x / s for x in w]


def run():
    cs = list(range(19, 31))
    print("=== Distribution du reservoir depuis la cinetique canonique ===\n")
    print("pas de masse geometrique : %.1f MeV/croisement ; T_c = %.0f MeV" % (
        SCALE * SLOPE, T_C * 1000))
    print("(curiosite, QUARANTAINE regle du null : dm/pas = %.2f ~ 1 -- non exploite)\n" % (
        SCALE * SLOPE / 1000 / T_C))
    print("taux net par croisement (A) : rho_ST * e^-alpha = %.3f "
          "(le naif 1/alpha_cond = %.3f etait accidentellement proche)\n" % (
        RHO_ST * math.exp(-ALPHA_COND), 1 / ALPHA_COND))

    fA, fB, fC = dist('A', cs), dist('B', cs), dist('C', cs)
    print(" c    A nombre   B masse-geo   C thermo    (fractions)")
    for i, c in enumerate(cs):
        print("%3d   %7.2f%%   %8.2f%%   %8.3f%%" % (
            c, fA[i] * 100, fB[i] * 100, fC[i] * 100))
    for nom, f in (('A', fA), ('B', fB), ('C', fC)):
        print(" tranche c=19-24 (%s) : %.1f%%" % (nom, 100 * sum(f[:6])))

    # verdict SIMP : fraction c=19-24 vs les bornes (il faudrait <~1e-4 pour
    # echapper a XQC/CRESST-surface avec les sections nominales)
    print("\nVERDICT sortie (b) :")
    print(" - Les TROIS variantes canoniques donnent une distribution DOMINEE")
    print("   par les bas c (78-100%% dans c=19-24) : la cinetique ne fournit")
    print("   PAS de distribution croissante en c. La sortie (b) est REFUTEE ;")
    print("   la variante thermodynamique complete (C) DURCIT meme le conflit")
    print("   (c=19 seul porte ~%.0f%%)." % (fC[0] * 100))
    print(" - Le conflit SIMP/XQC (54-64%% du reservoir) TIENT sous la")
    print("   cinetique canonique. Restent : (a) c_min >= 26 a deriver, ou")
    print("   (c) sigma_EM surestime.")

    # collateral 1 : sensibilite de la derivation alpha_cond
    print("\nCOLLATERAL 1 -- retroaction sur la derivation d'alpha_cond :")
    d = 2 * math.cos(math.pi / 137)
    h = 0.75 / 137
    for cm, tag in ((30, "mediane V5 (m~c^6.65)"), (21, "mediane geo (variante B)"),
                    (19, "mediane thermo (variante C)")):
        nht = math.exp(math.log(RHO_ST) - 1.0 / cm)
        a = math.log(d ** 3 / nht) + math.sqrt(h)
        print("   c_med = %2d (%s) : alpha_derive = %.4f (%+.2f%% vs 1.20)" % (
            cm, tag, a, (a / 1.20 - 1) * 100))
    print("   -> le 0.076%% de la derivation DEPENDAIT de la loi de masse V5 ;")
    print("   sous la masse geometrique l'accord devient ~1-1.5%% (toujours un")
    print("   succes d'ordre, plus une coincidence fine).")

    print("\nCOLLATERAL 2 -- Omega_DM/Omega_b = 5.45 calcule avec les masses V5 :")
    print("   a re-auditer sous la masse geometrique (hors scope ici).")


if __name__ == "__main__":
    run()
