"""Le de-triplement de la bascule en onde P : tranche par Sigma_b(6097) (22/07).

LA QUESTION (S34) : le verrou (exces d'echange) se DE-TRIPLE en onde P
(S31, verifie sur les centroides Omega). La bascule (le cout spin-1 de
la paire, x3^g au sol) fait-elle de meme ?

LA REGLE (canonique, zero hypothese nouvelle) : la distinction
STATIQUE / TRANSPORT -- deja TESTEE au chapitre saturation (verrou
statique non borne vs exces dynamique borne) :
 - l'exces d'echange est une quantite de TRANSPORT (genere par la
   rotation de l'anneau autour du cable, theoreme du maillon) : il
   reside sur l'anneau qui tourne -> en onde P il MIGRE sur l'anneau
   de la paire orbitante -> plat (d-1) [S31, verifie] ;
 - la bascule est un ETAT (la configuration spin-1 de la paire,
   stockee sur le chemin paire->lourd via la jonction : c'est le
   bilan de moment angulaire avec le lourd) : elle NE MIGRE PAS ->
   x3^g CONSERVE en onde P.
PREDICTION : bascule x3 (etat) + exces plat (transport) en onde P.

LE TEST : Sigma_b(6097) (LHCb 2019, Lambda_b pi, Gamma ~ 30, J inconnu)
= candidat 1P lambda-mode standard. Les QUATRE comptabilites possibles
sont enumerees ; la regle statique/transport en selectionne UNE.
"""
import numpy as np

SIGMA = 0.41847 ** 2
PS = np.pi * SIGMA
ML, MS_, MC, MB = 0.310, 0.4948, 1.557, 4.952
D1 = np.sqrt(2) - 1
B = 1.2811
X4 = 4 * D1
A = 13 / 18.0
XJ = {"j0": 0.0, "j1": 0.5, "j2": 1.0}
CORE_SB = MB + 2 * ML
M_OBS = 6.0969                          # Sigma_b(6097), moyenne +-


def levels(n_cen, core):
    return {j: np.sqrt(core**2 + PS * (n_cen - A + x)) for j, x in XJ.items()}


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LES QUATRE COMPTABILITES vs Sigma_b(6097)")
    print("=" * 78)
    opts = {
        "(a) bascule PLATE + exces plat":      1 + 1 + D1 + B + 3 * X4 + A,
        "(d) bascule PLATE + exces sature":    1 + 1 + 1.0 + B + 3 * X4 + A,
        "(b) bascule x3   + exces sature":     1 + 3 + 1.0 + B + 3 * X4 + A,
        "(c) bascule x3   + exces plat":       1 + 3 + D1 + B + 3 * X4 + A,
    }
    for nom, ncen in opts.items():
        lv = levels(ncen, CORE_SB)
        best = min(lv, key=lambda j: abs(lv[j] - M_OBS))
        d = 100 * (lv[best] / M_OBS - 1)
        ok = "EXCLU (niveau max %.0f < 6097)" % (1e3 * lv["j2"]) \
            if lv["j2"] < M_OBS - 0.012 else \
            "6097 = %s (%+.2f %%)" % (best, d)
        print("  %-36s j0/j1/j2 = %.0f/%.0f/%.0f  %s" % (
            nom, *(1e3 * lv[j] for j in ("j0", "j1", "j2")), ok))
    print("""
  -> les DEUX options a bascule plate sont EXCLUES (leur niveau le plus
  haut reste sous 6097 - 40 MeV) : LA BASCULE NE SE DE-TRIPLE PAS.
  Entre (b) et (c) : la regle statique/transport SELECTIONNE (c)
  (l'exces est du transport -> plat), et (c) est aussi le meilleur
  ajustement : Sigma_b(6097) = le niveau j1, a +0,02 %.""")

    print("=" * 78)
    print("[2] LA REGLE ETABLIE ET SES VERIFICATIONS CROISEES")
    print("=" * 78)
    print("""  ONDE P, paire au bout du cable :
    ETAT      (bascule, spin-1)    -> x3^g CONSERVE  [Sigma_b(6097)]
    TRANSPORT (exces d'echange)    -> PLAT (d-1)     [centroides Omega,
                                                      S31, -2,4 %]
  -- la meme dichotomie statique/dynamique que la saturation (document principal),
  appliquee a la migration : zero hypothese nouvelle. Coherences :
  au charme (3^0), la question est INVISIBLE (bascule x1 = plate) --
  Sigma_c(2800) entre j0 (2772) et j1 (2821), melange candidat,
  inchange ; les Omega (pas de bascule : verrou pur) restent au S31.""")

    print("=" * 78)
    print("[3] PREDICTIONS (quintet Sigma_b 1P complet, option (c))")
    print("=" * 78)
    lv = levels(opts["(c) bascule x3   + exces plat"], CORE_SB)
    unit = (3 / 16.0) * SIGMA / MB      # h_1/2, jambes ll : sans dilution
    s1, s2 = 1.5 * unit, 2.5 * unit
    print("""  j0 (J=1/2)         : %.0f MeV
  j1 (J=1/2, 3/2)    : %.0f / %.0f  (centroide %.0f = Sigma_b(6097))
  j2 (J=3/2, 5/2)    : %.0f / %.0f  (centroide %.0f)
  -> Sigma_b(6097) = le doublet j1 non resolu (Gamma ~ 30 le couvre) ;
  FALSIFIABLE : J(6097) = {1/2, 3/2} (pas 5/2) ; les niveaux j0 (~6076)
  et j2 (~6121) a chercher dans Lambda_b pi ; les deux J=3/2 (j1/j2, a
  ~19 MeV) se melangent a O(1/m_b) (meme motif que Omega_b).
  NB Xi_b' 1P (paire l-s, bascule sans exces — membres distincts) :
  j0 predit ~6243 ; Xi_b(6227) a -0,26 %% : candidat DECLARE avec
  reserve (rho-mode/molecule Lambda_b K debattus).""" % (
        1e3 * lv["j0"],
        1e3 * (lv["j1"] - (4 / 6) * s1), 1e3 * (lv["j1"] + (2 / 6) * s1),
        1e3 * lv["j1"],
        1e3 * (lv["j2"] - (6 / 10) * s2), 1e3 * (lv["j2"] + (4 / 10) * s2),
        1e3 * lv["j2"]))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  LA QUESTION EST TRANCHEE : la bascule NE se de-triple PAS (les
  options plates sont exclues par Sigma_b(6097), niveaux trop bas de
  40-70 MeV) ; l'exces d'echange, lui, reste plat (transport). La
  dichotomie ETAT/TRANSPORT du document principal (saturation) s'etend a la
  migration en onde P -- zero nombre nouveau, et Sigma_b(6097) = j1 a
  +0,02 %. Falsifiable : J(6097), les niveaux j0/j2 du quintet, le
  melange des 3/2.""")


if __name__ == "__main__":
    main()
