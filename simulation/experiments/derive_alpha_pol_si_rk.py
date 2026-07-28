"""Resserrement de la bande +-1.5 dex de sigma_EM(neutre) : R_K, alpha_pol,
equivalence SI derives (17/07).

Les trois sources d'incertitude de derive_sigma_em_neutre.py, attaquees une
par une :

1. R_K -- DERIVE de la loi de masse geometrique. La longueur PHYSIQUE du
   tube est independante de la convention de ropelength :
       L_phys = M(c) / sigma_phys,   sigma_phys = sigma_chi / (hbar c).
   Le diametre de tube du plateau geometrique en decoule :
       D = L_phys / l(c) = 0.0214 fm  (uniforme -- c'est la normalisation
       m_l = 310 MeV, 'rayon de tube' du comptage V6).
   Le 'R_K = 3 fm' du script precedent utilisait le tube du PROTON
   (d_tube = lambdabar_C(p) = 0.21 fm, secteur Regge) : 10x trop grand
   pour le plateau geometrique. Rayon d'ensemble du T(2,q) : anneau a
   deux brins, R ~ L_phys/(4 pi) (encadre par SONO sur le T(2,23) ideal).

2. alpha_pol -- MECANISME SELECTIONNE. Deux lectures :
   (ISOLANT) deplacement transverse de la charge de boucle (Sigma_chi=+2)
   contre le courant de framing, amplitude ~ D/2, gap = premier mode de
   corde 2 pi hbar c / L : alpha_ins = 2 (Q e D/2)^2 / DeltaE ~ 2e-6 fm^3.
   (CONDUCTEUR) le courant de framing est une circulation PERSISTANTE
   (topologique -- le meme objet qui impose Q=0) ; la quantification de Lk
   verrouille la CIRCULATION, pas la densite de charge : la charge se
   redistribue librement LE LONG du tube (reponse adiabatique, v=1e-3).
   Le noeud repond comme un anneau conducteur : alpha_cond = k R^3,
   k dans [1/(ln(8R/a)-2), 1] (tore mince -> sphere).
   -> CONDUCTEUR selectionne. Le 'rescaling neutron' du script precedent
   est SUPPRIME : alpha_n est porte par le nuage de pions (isolant, gap
   m_pi) ; le noeud sombre n'a pas de nuage charge -- fausse analogie.

3. EQUIVALENCE SI -- CALCULEE. Le couplage polarisabilite voit le champ du
   NOYAU entier : V = -(Z^2 alpha alpha_pol/2)/r^4 -> amplitude ~ Z^2.
   Les limites publiees supposent un contact SI coherent A^2 :
       sigma_p^eq = sigma_p^naif x Z^4/A^2 x (mu_A/mu_p)^2
   (Ar : x479 ; Xe : x~4000). Le facteur est CONNU et pousse VERS LE HAUT.
   Born refait proprement : FT(1/r^4) = -pi^2 q  ->  sigma = 2 pi^3 mu^4
   v^2 A1^2, A1 = alpha alpha_pol / 2. (Le 2 pi alpha alpha_pol du script
   precedent surcomptait 4 pi.)
"""
import math

ALPHA = 1 / 137.036
HBARC = 0.19733            # GeV.fm
SQRT_SIGMA = 0.41765       # GeV
SIGMA_PHYS = SQRT_SIGMA ** 2 / HBARC   # GeV/fm
V = 1e-3
M_P = 0.93827
FM2_CM2 = 1e-26
LIM23 = 2.2e-41            # DS-50 (1.71 GeV) / f_bord -- migdal_confrontation
SCALE = 310.0 / 16.372     # MeV / unite de ropelength
L31, SLOPE = 16.372, (38.62 - 16.372) / 6


def m_geo(c):
    return SCALE * (L31 + SLOPE * (c - 3)) / 1000.0


def born_prefactor(alpha_pol, mu_GeV):
    """sigma_p (cm^2, eps=1) pour V = -(alpha alpha_pol/2)/r^4."""
    A1 = ALPHA * alpha_pol / 2.0       # fm^3
    mu = mu_GeV / HBARC                # fm^-1
    return 2 * math.pi ** 3 * mu ** 4 * V ** 2 * A1 ** 2 * FM2_CM2


def run(rg_sono=None):
    c = 23
    M = m_geo(c)
    lrope = L31 + SLOPE * (c - 3)
    Lphys = M / SIGMA_PHYS
    D = Lphys / lrope
    mu_p = M * M_P / (M + M_P)

    print("=== R_K, alpha_pol, equivalence SI : la bande resserree ===\n")
    print("[1] R_K derive de la loi de masse (c = %d, M = %.3f GeV) :" % (c, M))
    print("    L_phys = M/sigma_phys = %.3f fm ; D_tube = L/l = %.4f fm" % (Lphys, D))
    print("    (le 'R_K = 3 fm' precedent utilisait d_tube(proton) = 0.21 fm :")
    print("     tube Regge, PAS le tube du plateau geometrique)")
    R_lo = Lphys * math.cos(math.pi / 4) / (4 * math.pi)   # 2 brins, pas 45 deg
    R_hi = Lphys / (2 * math.pi)                           # 1 brin (borne haute)
    R_ring = Lphys / (4 * math.pi)                         # 2 brins, sans pas
    print("    anneau 2 brins : R = L/(4pi) = %.3f fm  [%.3f, %.3f]" % (
        R_ring, R_lo, R_hi))
    # MESURE SONO 2026-07-17 (T(2,23) ideal, n=560, 40k iters, min_gap 1.994) :
    # L/D = 90.21 (fit lineaire 90.53 : +0.4% -- valide la loi de masse) ;
    # R_g/D = 2.648, R_max/D = 3.811, R_out/D = 4.311 (enveloppe + tube).
    # Le T(2,23) ideal est un PELOTON compact, pas un anneau a 2 brins :
    # R_K mesure ~ 2x sous l'estimation analytique.
    SONO = {"L_D": 90.21, "Rg_D": 2.648, "Rout_D": 4.311}
    if rg_sono is None:
        D_m = Lphys / SONO["L_D"]
        Rg = SONO["Rg_D"] * D_m
        Rout = SONO["Rout_D"] * D_m
        # borne basse conducteur : volume du tube (alpha >= 3V/4pi, sphere)
        Vt = math.pi * (D_m / 2) ** 2 * Lphys
        R_vol = (3 * Vt / (4 * math.pi)) ** (1.0 / 3.0)
        print("    SONO (MESURE) : R_g = %.4f fm ; R_out = %.4f fm ;" % (Rg, Rout))
        print("      borne volume conducteur (3V_tube/4pi)^(1/3) = %.4f fm" % R_vol)
        R_lo, R_hi = max(Rg, R_vol), Rout
        R_ring = math.sqrt(R_lo * R_hi)
    print()

    # [2] alpha_pol
    DeltaE = 2 * math.pi * HBARC / Lphys
    a_ins = 2 * (2 ** 2) * (ALPHA * HBARC) * (D / 2) ** 2 / DeltaE
    print("[2] alpha_pol -- les deux lectures :")
    print("    (ISOLANT) gap corde DeltaE = 2 pi hbarc/L = %.3f GeV ;" % DeltaE)
    print("      alpha_ins = 2 Q^2 e^2 (D/2)^2/DeltaE = %.2e fm^3 (plancher)" % a_ins)
    # peloton compact (SONO) -> facteur de forme spherique, k ~ 1 :
    # bracket = [R_g (ou borne volume), R_out]^3
    a_lo = R_lo ** 3
    a_hi = R_hi ** 3
    a_c = math.sqrt(a_lo * a_hi)
    print("    (CONDUCTEUR, selectionne) peloton compact -> alpha = R^3,")
    print("      R dans [%.3f, %.3f] fm (R_g/volume -> enveloppe) :" % (R_lo, R_hi))
    print("      alpha_cond = %.2e fm^3  [%.1e, %.1e]" % (a_c, a_lo, a_hi))
    print("      ratio conducteur/isolant = %.0e : la circulation persistante" % (a_c / a_ins))
    print("      domine totalement la reponse EM du noeud neutre.\n")

    # [3] Born propre + equivalence SI
    print("[3] Born refait + equivalence SI :")
    for tgt, Zn, An, mA in (("Ar (DS-50/DS-20k)", 18, 40, 37.211),
                            ("Xe (XLZD)", 54, 131, 121.9)):
        mu_A = M * mA / (M + mA)
        f_si = (Zn ** 4 / An ** 2) * (mu_A / mu_p) ** 2
        print("    %s : Z^4/A^2 = %.1f ; (mu_A/mu_p)^2 = %.2f -> x%.0f (%+.1f dex)" % (
            tgt, Zn ** 4 / An ** 2, (mu_A / mu_p) ** 2, f_si, math.log10(f_si)))
    f_ar = (18 ** 4 / 40 ** 2) * ((M * 37.211 / (M + 37.211)) / mu_p) ** 2
    print()

    print("[4] sigma(c=23) equivalent-proton (Ar), Vassiliev alpha^5 :")
    for lab, aP in (("bas", a_lo), ("central", a_c), ("haut", a_hi)):
        sEM = born_prefactor(aP, mu_p)
        s23 = sEM * f_ar * ALPHA ** 5
        print("    alpha_pol %-8s %.1e fm^3 : sigma_EM = %.1e ; sigma_eq(23) = %.1e"
              "  (%+.1f dex vs DS-50)" % (lab, aP, sEM, s23, math.log10(s23 / LIM23)))
    print()

    # scenario 'a la frontiere' : quel alpha_pol / R_K faudrait-il ?
    s_c = born_prefactor(a_c, mu_p) * f_ar * ALPHA ** 5
    need = math.sqrt(LIM23 / s_c) * a_c
    R_need = (need / 1.0) ** (1.0 / 3.0)
    M_need = R_need / R_ring * M
    print("[5] Scenario 'bord a la frontiere' (sigma_eq = lim DS-50) :")
    print("    exigerait alpha_pol = %.1f fm^3, soit R_K = %.2f fm <-> M = %.0f GeV" % (
        need, R_need, M_need))
    print("    a M = 1.71 GeV fixe : EXCLU par la geometrie derivee (R suit M).\n")

    print("VERDICT :")
    print(" - R_K DERIVE puis MESURE (SONO, T(2,23) ideal) : L_phys = M/sigma")
    print("   independant des conventions ; L/D = 90.21 (+0.4% du fit lineaire :")
    print("   VALIDE la loi de masse geometrique) ; le peloton est COMPACT :")
    print("   R_K = 0.057-0.093 fm. Le 3 fm etait le tube du proton (x30-50).")
    print(" - alpha_pol : mecanisme SELECTIONNE (conducteur -- la circulation de")
    print("   framing est persistante ; Lk verrouille la circulation, pas la")
    print("   densite) : alpha_pol = R_K^3 = 3.8e-4 fm^3 [1.8e-4, 8.0e-4].")
    print("   Le rescaling neutron est SUPPRIME (nuage de pions = isolant).")
    print(" - Equivalence SI : facteur Z^4/A^2 (mu_A/mu_p)^2 CONNU (+2.7 dex Ar,")
    print("   +3.6 dex Xe) -- il REMONTE la section equivalente.")
    print(" - NET : sigma_eq(23) = 1.1e-48 cm^2 [2.5e-49, 4.6e-48] : le bord est")
    print("   ~7.3 dex SOUS la frontiere DS-50 (bande +-0.6, +-0.8 avec les O(1)")
    print("   Born/Migdal) -- bien plus profond que le '3 dex' du 16/07.")
    print("   La saturation 'a la frontiere' est EXCLUE par la geometrie.")
    print(" - CONSEQUENCE HONNETE : la detection directe EM du bord est HORS DE")
    print("   PORTEE (XLZD gagne ~2-3 dex, il en manque 4-5) : la prediction")
    print("   TQNT devient une ABSENCE ROBUSTE dans DS-20k/XLZD aussi ; toute")
    print("   detection a 1.7-2 GeV refuterait la lecture polarisabilite ou la")
    print("   loi de masse. Les sondes du reservoir restent gravitationnelles")
    print("   (perles/microlentilles, V5.19) et cosmologiques (budget Omega).")


if __name__ == "__main__":
    import sys
    rg = float(sys.argv[1]) if len(sys.argv) > 1 else None
    run(rg)
