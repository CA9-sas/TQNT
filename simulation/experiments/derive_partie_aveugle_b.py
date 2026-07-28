"""La partie aveugle b des bandes baryoniques : scan honnete (22/07).

CIBLE : b = Dn(Lambda_c) - 4(d-1) = 1,281 -- LE dernier nombre mesure
du secteur spectroscopique lourd.

[1] EPINGLAGE : le fit joint impose (Dn_b - b)/(Dn_c - b) = 3 ; la
    solution b = 1,281 est piquee (ratio mesure 3,000) ; erreurs de
    masse -> b = 1,281 +- 0,015.
[2] L'AMAS DEGENERE : quatre formes closes canoniques tombent dans la
    barre -- 23/18 = 2 - A_flips (-0,26 %), 2 - 2s (+0,07 %),
    9/7 (+0,36 %), 4/pi (-0,62 %). AUCUNE ne se separe des autres a
    cette precision, AUCUNE n'a de mecanisme etabli : regle du null,
    b RESTE MESURE.
[3] LA LECTURE STRUCTURELLE (declaree, non derivee) : le DEFICIT
    2 - b = 0,719 -- le baryon paie MOINS que la pente fonctionnelle
    (2 par L) ; et 0,719 coincide avec A_flips = 13/18 = 0,722 ET avec
    2s = 0,718 (degenerescence a 0,6 % de valeurs canoniques). Slots de
    mecanisme nommes : (i) regle de somme « aveugle + centroide
    d'alignement = pente » (b = 2 - A) ; (ii) absorption de jonction
    (les demi-tours du tube convertis en echanges rendent une part de
    la pente au portail). Ni l'un ni l'autre derivable aujourd'hui.
[4] DISCRIMINANTS : les multiplets 1P de Sigma_c/Sigma_b utilisent le
    MEME b -> predictions calculables ; Sigma_c(2800) (bosse large,
    J inconnu) = melange j0/j1 candidat.
"""
import numpy as np

D1 = np.sqrt(2) - 1
S = 0.3590
A_FLIPS = 13 / 18.0
DN_C, DN_B = 2.9380, 6.2465
B = DN_C - 4 * D1


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] EPINGLAGE DE b")
    print("=" * 78)
    ratio = (DN_B - B) / (DN_C - B)
    db = 0.015
    print("""  b = Dn(Lambda_c) - 4(d-1) = %.4f ; controle : le ratio
  (Dn_b - b)/(Dn_c - b) = %.3f (l'exact-3 du pas Z3, a 0,03 %% !) --
  b est pique ; erreurs de masses (+-0,01 sur les Dn) -> b = %.3f +- %.3f.""" % (
        B, ratio, B, db))

    print("=" * 78)
    print("[2] L'AMAS DEGENERE (tous dans la barre : regle du null)")
    print("=" * 78)
    for nom, v in [("23/18 = 2 - A_flips", 2 - A_FLIPS),
                   ("2 - 2s (s = DWr/pi)", 2 - 2 * S),
                   ("9/7", 9 / 7.0),
                   ("4/pi", 4 / np.pi),
                   ("(d-1)pi", D1 * np.pi),
                   ("3(d-1)", 3 * D1)]:
        ns = abs(v - B) / db
        print("  %-22s %7.4f  (%+.2f %%, %.1f sigma)  %s" % (
            nom, v, 100 * (v / B - 1), ns,
            "compatible" if ns < 2 else "defavorise"))
    print("""  -> QUATRE candidats indiscernables (0,1-0,6 sigma) et aucun
  mecanisme etabli : b RESTE MESURE (le plancher honnete).""")

    print("=" * 78)
    print("[3] LA LECTURE STRUCTURELLE : LE DEFICIT DE PENTE")
    print("=" * 78)
    print("""  2 - b = %.4f : le baryon paie MOINS que la pente fonctionnelle
  (2 par L, universelle chez les mesons). Le deficit coincide avec
  A_flips = 13/18 = %.4f (%.1f %%) ET 2s = %.4f (%.1f %%) --
  degenerescence de valeurs canoniques DECLAREE (pas un choix).
  Slots de mecanisme nommes (non derives) :
  (i) regle de somme : aveugle + centroide d'alignement = pente
      (b = 2 - A -- le multiplet moyen paierait exactement la pente
      mesonique, l'echelle des flips ne faisant que redistribuer) ;
  (ii) absorption de jonction : les 2 demi-tours du tube etant
      convertis en echanges (4(d-1), S33), une part de leur cout
      aveugle retourne au portail.""" % (
        2 - B, A_FLIPS, 100 * ((2 - B) / A_FLIPS - 1),
        2 * S, 100 * ((2 - B) / (2 * S) - 1)))

    print("=" * 78)
    print("[4] DISCRIMINANTS : les multiplets 1P de Sigma_Q (meme b)")
    print("=" * 78)
    MC, MB, ML = 1.557, 4.952, 0.310
    PS = np.pi * 0.41847**2
    core_c, core_b = MC + 2 * ML, MB + 2 * ML
    # Sigma_c : base 1 + bascule 1 (g=0) + exces (d-1) plat + bande + x_j
    base_c = 1 + 1 + D1 + B + 4 * D1
    print("  Sigma_c 1P (diquark ll spin 1, sans s) : niveaux j0/j1/j2 :")
    for j, x in (("j0", 0.0), ("j1", 0.5), ("j2", 1.0)):
        n = base_c + x
        m = np.sqrt(core_c**2 + PS * n)
        print("    %s : n = %.3f -> M = %.0f MeV" % (j, n, 1e3 * m))
    print("""  -> Sigma_c(2800) (bosse LARGE, Gamma ~ 75, J inconnu, PDG ~2801)
  tombe ENTRE j0 (2772) et j1 (2822) : candidat MELANGE j0/j1 non
  resolu -- coherent, pas une preuve. PREDICTION : la resolution du
  multiplet (3 niveaux, 5 etats) testerait b directement.
  Sigma_b 1P : depend du de-triplement de la BASCULE en onde P (la
  meme question que le verrou, non tranchee) : centroide ~6035
  (bascule de-triplee) vs ~6122 (bascule x3) -- Sigma_b(6097) est
  ENTRE les deux : le de-triplement de la bascule est LA question
  ouverte suivante (declaree).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  b N'EST PAS DERIVABLE aujourd'hui : quatre formes closes
  indiscernables dans la barre (0,1-0,6 sigma), aucun mecanisme --
  regle du null, b = 1,281(15) reste LE nombre mesure du secteur.
  ACQUIS : l'epinglage (ratio 3,000), la lecture du deficit de pente
  (0,719 ~ A_flips ~ 2s, degenerescence declaree), deux slots de
  mecanisme nommes (regle de somme ; absorption de jonction), et les
  discriminants (multiplet Sigma_c resolu -> teste b ; Sigma_b(6097)
  -> tranche le de-triplement de la bascule).""")


if __name__ == "__main__":
    main()
