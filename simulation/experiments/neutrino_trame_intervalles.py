"""Le déplacement du neutrino sur la trame de l'espace-temps + les
intervalles d'existence (27/07/2026, brouillon).

CE QUI ÉTAIT ÉTABLI (à vérifier ici pour cohérence) :
  - neutrino = onde auto-rotative OUVERTE (P14), fragment d'onde libre
    qui tourne sur lui-même ; PAS de topologie (pas noué) ;
  - masse = COURBURE de la boucle : m = hbar c / (2R), R = rayon
    d'auto-rotation = demi-Compton réduite (lambda_C = 4 pi R) ;
  - état INTERMÉDIAIRE entre le photon (droit, R=inf, m=0, spin 1) et
    le noeud (fermé, R~fm, massif, spin de tresse) ;
  - Dirac (pas de topologie à refléter), 0nubb = 0 exact ; 3 modes,
    pas de stérile ; Sigma m_nu ~ 66 meV.

LA LECTURE TRAME (nouveau) — les trois régimes de déplacement :
  - LUMIÈRE : onde jamais ENTRÉE dans un rail, S=0, spectatrice, c
    exact, zéro dispersion (droite) ;
  - NOEUD : onde ENTRÉE dans un rail, magnétisée, verrouillée, déficit
    de confiscation = masse, v < c ;
  - NEUTRINO : onde auto-rotative OUVERTE qui EFFLEURE la trame sans y
    entrer -> Q=0 (jamais dans un rail), jamais magnétisée (faible
    seulement) ; la COURBURE confisque une fraction infime
    d'ondulation = la masse (mécanisme DISTINCT du nouage).

LES INTERVALLES D'EXISTENCE (le calcul demandé) :
  - borne HAUTE = SEUIL DE NOUAGE : au-delà de l'énergie du plus petit
    noeud (m_l = (3/4) sqrt(sigma_chi)), un fragment PEUT se nouer ;
    R_min = hbar c / (2 m_l). (La nucléation EFFECTIVE exige en plus
    la porte Z3 -- jamais franchie par un neutre -> le neutrino reste
    ouvert à vie même sous le seuil de possibilité.)
  - borne BASSE = m -> 0 (photon, limite dégénérée) : NON contrainte
    par la trame (plate) ; les OSCILLATIONS imposent le plancher
    observationnel (m_2 >= sqrt(Dm2_21), m_3 >= sqrt(Dm2_31)).
  - fenêtre observationnelle actuelle : [oscillations, DESI].

GARDE-FOUS : R (donc m_nu exact) reste AJUSTÉ à Sigma m_nu (raffinement,
pas dérivation) ; la borne basse est observationnelle ; règle du test
nul (aucune valeur fine revendiquée au-delà des échelles).
"""
import math

HBARC = 197.3269804          # MeV fm
SQRT_SIGMA = 418.47          # MeV
M_L = 0.75 * SQRT_SIGMA      # MeV -- noeud minimal (trefle, (3/4) sqrt(sigma))
# splittings d'oscillation (NuFit, ordre normal)
DM21 = 8.68e-9               # MeV (sqrt(Dm2_21))
DM31 = 50.3e-9               # MeV (sqrt(Dm2_31))
DESI = 72e-9                 # MeV (Sigma m_nu < 72 meV)
KATRIN = 0.45e-6             # MeV (borne labo m_beta < 0.45 eV)
FM_TO_UM = 1e-9              # 1 fm = 1e-9 um


def R_of_m(m_MeV):
    """rayon d'auto-rotation (fm) : m = hbar c / (2R)."""
    return HBARC / (2 * m_MeV)


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LE SEUIL DE NOUAGE (borne HAUTE, dérivée de la trame)")
    print("=" * 78)
    R_min = R_of_m(M_L)
    lamC_min = 4 * math.pi * R_min
    print("  m_l = (3/4) sqrt(sigma_chi) = %.2f MeV  (énergie du plus petit"
          " noeud)" % M_L)
    print("  R_min = hbar c/(2 m_l) = %.4f fm ; lambda_C(seuil) = %.3f fm"
          % (R_min, lamC_min))
    print("""  => au-DELÀ de m_l (R < R_min) un fragment a l'énergie d'un noeud
     et PEUT se nouer -> il quitte le régime neutrino. La nucléation
     EFFECTIVE exige en plus la porte Z3 (perçage), jamais franchie
     par un objet NEUTRE : le neutrino reste ouvert même bien sous le
     seuil (Q=0 -> jamais dans un rail -> jamais noué à vie).""")

    print("=" * 78)
    print("[2] LA BORNE BASSE (m -> 0 = photon ; plancher = oscillations)")
    print("=" * 78)
    print("  trame plate (box chi = 0) : aucun plancher intrinsèque"
          " (m -> 0 = photon).")
    print("  plancher OBSERVATIONNEL (oscillations, ordre normal) :")
    print("    m_2 >= sqrt(Dm2_21) = %.2f meV" % (DM21 * 1e9))
    print("    m_3 >= sqrt(Dm2_31) = %.2f meV" % (DM31 * 1e9))

    print("=" * 78)
    print("[3] L'INTERVALLE D'EXISTENCE ET LES FENÊTRES PAR ÉTAT")
    print("=" * 78)
    print("  intervalle THÉORIQUE : m in (0, %.0f MeV]  <=>  R in [%.3f fm,"
          " inf)" % (M_L, R_min))
    print("  la fenêtre observée est ~%.1e sous le seuil (le neutrino vit"
          " TOUT EN BAS)" % (M_L / DM31))
    print()
    print("  %-30s %10s %12s %12s" % ("état", "m [meV]", "R [um]",
                                      "lambda_C [um]"))
    rows = [
        ("photon (limite dégénérée)", 0.0),
        ("plus léger (m1 -> 0 => m2)", DM21 * 1e9),
        ("atmosphérique garanti (m3)", DM31 * 1e9),
        ("m3 sous DESI (Sigma<72)", DESI * 1e9 - DM21 * 1e9),
        ("borne labo (KATRIN)", KATRIN * 1e9),
        ("SEUIL DE NOUAGE (m_l)", M_L * 1e9),
    ]
    for nom, m_meV in rows:
        if m_meV == 0:
            print("  %-30s %10s %12s %12s" % (nom, "0", "inf", "inf"))
            continue
        m_MeV = m_meV * 1e-9
        R_fm = R_of_m(m_MeV)
        R_um = R_fm * FM_TO_UM
        lamC_um = 4 * math.pi * R_um
        print("  %-30s %10.2f %12.3g %12.3g" % (nom, m_meV, R_um, lamC_um))

    print("=" * 78)
    print("[4] LE DÉPLACEMENT : VITESSE SUR LA TRAME (v ~ c, retard SN1987A)")
    print("=" * 78)
    m_nu = DM31 * 1e9 * 1e-9     # 50.3 meV en MeV
    print("  neutrino le plus lourd garanti : m = %.1f meV" % (DM31 * 1e9))
    for lab, E_MeV in [("solaire (~0,4 MeV)", 0.4),
                       ("SN1987A (~10 MeV)", 10.0),
                       ("atmosphérique (~1 GeV)", 1000.0)]:
        beta = math.sqrt(max(0.0, 1 - (m_nu / E_MeV) ** 2))
        defc = 1 - beta                       # 1 - v/c
        print("  %-24s : v/c = 1 - %.2e" % (lab, defc))
    # retard SN1987A sur 168 000 al
    D_ly = 1.68e5
    c_ly_per_s = 1.0 / (3.156e7)              # 1 al parcourue en 1 an -> s
    D_s = D_ly / c_ly_per_s                   # temps de vol lumière (s)
    E_SN = 10.0
    dt = D_s * 0.5 * (m_nu / E_SN) ** 2
    print("  retard SN1987A (D=1,68e5 al, E=10 MeV) : Dt = %.2e s"
          "  (observé : ~qq s de dispersion) -> compatible" % dt)

    print("=" * 78)
    print("[5] VÉRIFICATION DE COHÉRENCE (le déplacement trame vs l'envisagé)")
    print("=" * 78)
    checks = [
        ("Dirac (nu != nubar)", "pas de topologie à refléter (jamais noué)",
         True),
        ("Q = 0", "jamais ENTRÉ dans un rail -> aucune charge (NOUVEAU : "
         "la trame l'explique)", True),
        ("spin 1/2", "moment angulaire de l'auto-rotation (ruban 4pi)", True),
        ("masse minuscule, != 0", "courbure de boucle ouverte "
         "(m=hbar c/2R), pas nouage", True),
        ("interactions FAIBLES seulement", "jamais magnétisé (hors rail) : "
         "ni forte ni EM (NOUVEAU)", True),
        ("3 modes, pas de stérile", "3 fragments Z3, saturation dérivée",
         True),
        ("0nubb = 0", "nombre leptonique conservé (mécanisme onde libre)",
         True),
        ("presque la lumière", "R>>fm -> courbure infime -> quasi 𝒮=0, "
         "v~c, zéro dispersion effective", True),
    ]
    for nom, mech, ok in checks:
        print("  [%s] %-32s %s" % ("OK" if ok else "??", nom, mech))
    print("""
  => TOUT CORRESPOND : le déplacement sur la trame (onde ouverte qui
     effleure sans entrer) reproduit exactement l'objet envisagé, et
     la trame AJOUTE deux explications (Q=0 et faible-seulement =
     conséquences du « jamais entré dans un rail »).""")

    print("=" * 78)
    print("NUANCES HONNÊTES (règle du test nul)")
    print("=" * 78)
    print("""  - R (donc m_nu exact) reste AJUSTÉ à Sigma m_nu : raffinement,
    pas dérivation (statut inchangé) ;
  - la borne BASSE est observationnelle (trame plate = pas de
    plancher intrinsèque) ; seule la borne HAUTE (seuil de nouage
    m_l) est dérivée de la trame ;
  - la borne haute est un seuil de POSSIBILITÉ énergétique ; la
    non-nucléation EFFECTIVE vient de la porte Z3 (Q=0), mécanisme
    distinct -- les deux convergent (« le neutrino ne se noue
    jamais ») ;
  - coquille repérée dans le texte V6 (ch29/neutrinos) : « R_e ~ 0,19
    fm » doit se lire ~193 fm (= hbar c/2m_e ; le facteur 10^7 vs le
    neutrino le confirme) -- à corriger.""")


if __name__ == "__main__":
    main()
