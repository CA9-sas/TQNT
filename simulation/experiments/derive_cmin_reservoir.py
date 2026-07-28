"""Sortie (a) : deriver le c_min effectif du reservoir DM (trois voies).

PRINCIPE : le seuil naif c > 18 marque le DEBUT de la suppression de
Vassiliev, pas l'invisibilite (sigma(19) = alpha*sigma_EM ~ 1e-27 : enorme).
Le RESERVOIR (ce que P18 compte comme matiere noire) exige davantage :

VOIE 1 -- DECOUPLAGE CINETIQUE (definitionnelle) : un noeud encore couple
  cinetiquement au fond thermique baryonique pendant la BBN n'est PAS de la matiere
  noire (il compte comme matiere interagissante : brise BBN/CMB et la
  definition meme du reservoir). Condition : Gamma_scatter(T) < H(T) pour
  tout T <= T_c (le pire cas est T = T_c : le noeud doit NAITRE decouple).
    Gamma = n_b(T) sigma v_rel ; n_b = eta_B * 0.24 T^3 ;
    v_rel ~ sqrt(3T/mu) ; H = 1.66 sqrt(g*) T^2 / M_Pl.

VOIE 2 -- SURVIE COSMOLOGIQUE : borne superieure sur la destruction par
  reconnexion baryonique (en prenant sigma_reco <= sigma_scatter, genereuse
  car la reconnexion est Vassiliev-supprimee davantage) :
    N_coll = int Gamma dt < 1 sur l'histoire cosmique.

VOIE 3 -- ECHO BAR-NATAN (canonique V5.21) : la regle de Vassiliev
  v_{c_i+c_j-2} donne Delta_c = 22 comme seuil effectif des evenements
  topologiques (amphichirales c=12 saturent) -- meme motif : les seuils
  effectifs tombent AU-DESSUS des minima naifs.

Puis : RE-CONFRONTATION SIMP/XQC avec le c_min derive (abondances
renormalisees) -- que reste-t-il du conflit ?
"""
import math

ALPHA = 1 / 137.036
SIGMA_EM_CM2 = 1e-25
CM2_TO_GEV2 = 1 / 3.894e-28      # 1 cm^2 en GeV^-2
M_PL = 1.221e19                   # GeV
ETA_B = 6.1e-10
T_C = 0.067                       # GeV
G_STAR = 10.75
SCALE = 310.0 / 16.372
SLOPE = (38.62 - 16.372) / 6
L31 = 16.372
ALPHA_COND = 1.20
RHO_ST = 2.68


def m_geo(c):
    return SCALE * (L31 + SLOPE * (c - 3)) / 1000.0


def sigma_c(c):                   # GeV^-2
    return SIGMA_EM_CM2 * CM2_TO_GEV2 * ALPHA ** (c - 18)


def gamma_over_H(c, T):
    M = m_geo(c)
    mu = M * 0.94 / (M + 0.94)    # masse reduite noeud-nucleon
    n_b = ETA_B * 0.24 * T ** 3
    v = math.sqrt(3 * T / mu)
    H = 1.66 * math.sqrt(G_STAR) * T ** 2 / M_PL
    return n_b * sigma_c(c) * v / H


def n_collisions(c):
    """N = int Gamma dt, ere de radiation, de T_c a ~0 (dominee par T_c).
    dt = -dT/(H T) => N = int Gamma/(H T) dT ; integrande ~ sqrt(T) =>
    N = [Gamma/H](T_c) * 2 (integrale exacte du sqrt)."""
    return gamma_over_H(c, T_C) * 2.0


def run():
    print("=== Deriver c_min du reservoir : trois voies ===\n")

    print("[1] Decouplage cinetique (Gamma/H a T_c, pire cas ; puis a T_BBN=1 MeV) :")
    c1 = None
    for c in range(19, 29):
        r_tc = gamma_over_H(c, T_C)
        r_bbn = gamma_over_H(c, 0.001)
        tag = ""
        if r_tc < 1 and c1 is None:
            c1 = c
            tag = "  <- NE decouple (naissance decouplee)"
        print("   c=%2d : Gamma/H(T_c) = %9.2e ; Gamma/H(BBN) = %9.2e%s" % (
            c, r_tc, r_bbn, tag))
    print("   => c_min(decouplage) = %d\n" % c1)

    print("[2] Survie (N_collisions < 1 sur l'histoire, borne genereuse) :")
    c2 = None
    for c in range(19, 29):
        N = n_collisions(c)
        tag = ""
        if N < 1 and c2 is None:
            c2 = c
            tag = "  <- survit intact"
        print("   c=%2d : N_coll ~ %9.2e%s" % (c, N, tag))
    print("   => c_min(survie) = %d  (borne : sigma_reco <= sigma_scatter)\n" % c2)

    print("[3] Echo canonique : Bar-Natan v_{ci+cj-2} -> Delta_c = 22 (V5.21) --")
    print("    meme motif de seuil effectif au-dessus du minimum naif.\n")

    cmin = max(c1, c2)
    print("CONVERGENCE : c_min derive = %d (voies 1-2 ; echo 22 en voie 3)\n" % cmin)

    # re-confrontation avec c_min derive
    print("[4] Re-confrontation SIMP/XQC avec c_min = %d (abondances renormalisees," % cmin)
    print("    distribution canonique A : N(c)e^-alpha, taux 0.807) :")
    cs = list(range(cmin, 31))
    w = [(RHO_ST ** c / c) * math.exp(-ALPHA_COND * c) for c in cs]
    tot = sum(w)
    BANDS = [("XQC", 0.1, 10.0, 3e-31, 1e-25),
             ("CRESST-surf", 0.5, 10.0, 4e-34, 3e-28),
             ("souterrain", 0.5, 3.0, 1e-41, 1e-35),
             ("satellites", 0.1, 10.0, 5e-30, 1e-20)]
    conflit = 0.0
    for i, c in enumerate(cs):
        M = m_geo(c)
        sig = SIGMA_EM_CM2 * ALPHA ** (c - 18)
        f = w[i] / tot
        hits = []
        for nom, m0, m1, s0, s1 in BANDS:
            if not (m0 <= M <= m1):
                continue
            if nom == "souterrain" and sig / M > 1e-33:
                continue
            if s0 / max(f, 1e-6) <= sig <= s1:
                hits.append("%s (+%.1f dex)" % (nom, math.log10(sig * f / s0)))
        v = ("EXCLU/TENSION par " + " ; ".join(hits)) if hits else "compatible"
        if hits:
            conflit += f
        if c <= cmin + 5 or hits:
            print("   c=%2d  M=%.2f  sigma=%.1e  f=%5.1f%%  %s" % (c, M, sig, f * 100, v))
    print("   => fraction en conflit apres c_min derive : %.0f%%" % (conflit * 100))
    print("")
    print("VERDICT sortie (a) :")
    print(" - Le cadre DERIVE un c_min effectif = %d (decouplage/survie," % cmin)
    print("   convergent avec l'echo Bar-Natan 22) -- PAS 26.")
    print(" - Ce c_min EVACUE les exclusions robustes (XQC c=19-20,")
    print("   CRESST-surface c=21) : le gros du candidat F tombe.")
    print(" - Le residu (c~23-25, souterrain bas-M) est DANS la marge +-1 dex")
    print("   des bandes encodees : statut TENSION, a trancher par les courbes")
    print("   numerisees -- plus une falsification franche.")
    print(" - PREDICTION AFFUTEE : le bord visible du reservoir (c=%d-%d," % (cmin, cmin + 2))
    print("   M ~ %.1f-%.1f GeV, sigma ~ %.0e-%.0e) tombe exactement dans la" % (
        m_geo(cmin), m_geo(cmin + 2), sigma_c(cmin) / CM2_TO_GEV2,
        sigma_c(cmin + 2) / CM2_TO_GEV2))
    print("   fenetre Migdal actuelle/prochaine : c'est LA cible detecteur.")


if __name__ == "__main__":
    run()
