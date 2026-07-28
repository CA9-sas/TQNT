"""LHCb : particules hors Modele Standard mais TQNT -- l'etat des lieux (22/07).

QUESTION : y a-t-il au LHCb des particules NON prevues par le MS mais
prevues par la TQNT ? Reponse structuree en quatre volets honnetes.

[1] LES EXOTIQUES LHCb (X, Z, T, P_c...) : hors du modele a quarks NAIF
    (pas de case qqbar/qqq), accommodees par QCD comme multiquarks/
    molecules SANS masses predites ; la TQNT les place sur la GRILLE DU
    SOCLE a zero parametre (hadrons socle + demi-tours).
[2] LES ETATS PROPRES A LA TQNT (sans case MS DU TOUT) : 6 stricts DM
    (1,07-1,64 GeV), EW-neutres (1,75/1,89), 111 realisables (~1-3 GeV)
    -- mais leur CREATION en collision est supprimee (V5.19 : alpha^c),
    => PREDICTION D'ABSENCE au LHCb ; la fenetre est ASTRO (Fermi-LAT).
[3] LES DISCRIMINANTS DATES (les masses tranchent TQNT vs QCD) :
    Xi_cc* TQNT 3659 vs reseau/modeles ~3690 ; B0*/B1' 5660-5666 vs
    modeles ~5700-5750 ; Omega_ccc 4782 strict vs reseau 4760-4800.
[4] LES ABSENCES PREVUES PAR THEOREME (le MS ne les interdit pas) :
    jamais de 4e generation (saturation Z3) ; aucun sterile ; Omega_bbb
    = secteur compression (M < 3 m_b, signe teste par le reseau).
"""
import numpy as np

ALPHA = 1 / 137.0


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] EXOTIQUES LHCb SUR LA GRILLE DU SOCLE (zero parametre, document principal)")
    print("=" * 78)
    # (etat, lecture socle, ecart document principal)
    table = [
        ("X(3872)", "D Dbar* (hadrons socle + 1/2-tours)", "+0,1 %"),
        ("T_cc+(3875)", "D D* ", "+0,0 %"),
        ("Z_c(3900)/Z_c(4020)", "D Dbar*/D* Dbar*", "-0,3/-0,2 %"),
        ("Z_b(10610/10650)", "B Bbar*/B* Bbar*", "-0,3/-0,2 %"),
        ("Z_cs(4003)", "D_s Dbar* (demi-substitution)", "-0,4 %"),
        ("P_c(4312/4440/4457)", "Sigma_c Dbar(*) (coeur torsionne)", "<1 %"),
        ("P_cs(4459)", "Xi_c Dbar*", "<1 %"),
        ("X0(2900)", "Dbar* K* ", "+1,3 %"),
        ("Ds0*(2317)/Ds1(2460)", "molecules DK/D*K (reclassees)", "+1,8 %"),
    ]
    for e, l, d in table:
        print("  %-22s %-38s %s" % (e, l, d))
    print("""  -> ~16 etats, mediane ~0,3 % : le modele a quarks NAIF ne les
  prevoyait pas ; QCD les ACCOMMODE (multiquarks) sans masses ; la TQNT
  les POSTDIT sur la grille a zero parametre. Edge TQNT = les nombres,
  pas l'existence.""")

    print("=" * 78)
    print("[2] LES ETATS SANS CASE MS : PREDICTION D'ABSENCE AU LHCb")
    print("=" * 78)
    for nom, c in [("4_1+8_3 (1,07 GeV)", 12), ("6_3+6_3 (1,10)", 12),
                   ("6_3+8_3 (1,22)", 14), ("8_3+8_3 (1,35)", 16),
                   ("6_3+6_3+6_3 (1,64)", 18)]:
        dc = c - 7           # depuis les briques hadroniques c<=7
        print("  %-22s c=%2d : creation ~ alpha^(dc=%d) ~ %.0e" % (
            nom, c, dc, ALPHA ** dc))
    print("""  -> la creation d'un entrelacs strict depuis des hadrons (c<=7)
  est supprimee en alpha^dc ~ 1e-11..1e-24 (motif V5.19 : le primordial
  direct est tue ; echo du null ATLAS heavy-proton, V5p4 propre).
  PREDICTION : LHCb ne verra JAMAIS les stricts/realisables -- leur
  fenetre est ASTRO (raies Fermi-LAT 1,10/1,64 GeV) et gravitationnelle.
  Toute detection collisionneur d'un etat neutre etroit a CES masses
  contredirait la suppression (falsifiable dans les deux sens).""")

    print("=" * 78)
    print("[3] DISCRIMINANTS DATES : LES MASSES TRANCHENT (TQNT vs QCD)")
    print("=" * 78)
    print("""  etat            TQNT (derive)      QCD (reseau/modeles)   discriminant
  Xi_cc*          3659 MeV           ~3690 (+-10)           ~30 MeV : NET
  B0*/B1'         5660/5666 (6,6)    ~5700-5750 (larges)    ~50 MeV : NET
  Omega_ccc       4782 strict        4760-4800              fin (~0,4 %)
  Omega_b(1P) j0  6297-6304          ~6300 (modeles)        faible
  Omega_b*        6067 (+22)         ~6066-6090             faible
  -> les DEUX discriminants nets : Xi_cc* (l'etoile du Xi_cc observe
  en mars 2026 -- LHCb l'aura) et B0*/B1'. Une mesure a ~3659 ou
  ~5665 selectionnerait la TQNT ; a ~3690/~5720, la refuterait
  (sur ces canaux).""")

    print("=" * 78)
    print("[4] LES ABSENCES PREVUES PAR THEOREME")
    print("=" * 78)
    print("""  - JAMAIS de hadrons de 4e generation (saturation Z3, derivee) --
    le MS ne l'interdit pas par principe ;
  - AUCUN neutrino sterile a aucune masse (le neutrino est une onde
    chi libre) -- toute confirmation refuterait ;
  - Omega_bbb : secteur COMPRESSION (M < 3 m_b = 14 856 ; le reseau
    donne ~14 371, liaison -485 MeV : le SIGNE de la categorisation
    est deja verifie par le reseau) -- pas une absence, un test de
    secteur ;
  - pas de « proton lourd » Gen 2/3 en collision (null ATLAS propre).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  Reponse honnete : il n'existe PAS (encore) au LHCb de particule
  observee qui soit hors-MS et TQNT-seule -- et la TQNT PREDIT cette
  absence (suppression alpha^dc de la creation de ses etats propres).
  Ce qui distingue la TQNT au LHCb : (1) la grille de masses a zero
  parametre sur les exotiques (~16 etats, 0,3 %) ; (2) deux
  discriminants dates NETS a venir : Xi_cc* = 3659 (vs ~3690) et
  B0*/B1' = 5660/5666 (vs ~5700-5750) ; (3) les absences par theoreme
  (4e generation, steriles). La fenetre des etats TQNT-seuls est
  ASTRO : raies 1,10/1,64 GeV (Fermi-LAT profond).""")


if __name__ == "__main__":
    main()
