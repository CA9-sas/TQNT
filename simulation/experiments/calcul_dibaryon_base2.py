"""Le dibaryon base-2 : masse, canaux, largeur (22/07).

LA CATEGORIE (taxonomie [B2]) : le compose a DOUBLE trialite -- un
sommet 6-valent (la neutralite Z3 tient : 6 x 2pi/3 = 4pi = 0 mod 2pi,
enroulement w = 2) -> BASE 2 (la prediction structurelle du document principal,
derive_base_jonction_y). Machinerie : la fonctionnelle maitresse
etendue, M^2 = (somme m_i)^2 + pi sigma_chi n, avec
  n = 2 [base double] + verrous des paires identiques au sommet
(cc : verrou x3 de la generation de la paire, la lecture selectionnee
par Omega_cc ; les 4 trefles legers : distinguables par charge/Lk ->
pas d'exces force ; spin du fondamental : J = 1 si paire lourde
verrouillee spin 1, J = 0 sinon).

IDENTIFICATIONS DECLAREES : (i) extension de la fonctionnelle au
sommet 6-valent (meme rang que le reste du document principal) ; (ii) facteur de
liaison 6-valent inconnu (la geometrie 60 degres n'a pas son Steiner :
systematique O(1) dominante, +-40 MeV estimes) ; (iii) verrou de paire
lourde = 3^(g_paire) (d-1) (empirique, Xi_cc -0,6 %).
"""
import numpy as np

SIGMA = 0.41847 ** 2
PS = np.pi * SIGMA
ML, MS_, MC, MB = 0.310, 0.4948, 1.557, 4.952
D1 = np.sqrt(2) - 1
HBAR_EVS = 6.582e-16                    # eV.s


def mass(core, n):
    return np.sqrt(core**2 + PS * n)


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LES TROIS DIBARYONS COMPACTS (base 2)")
    print("=" * 78)
    cases = [
        ("H  (uuddss : 2s+4l)", 2 * MS_ + 4 * ML, 2 + D1,
         [("Lambda Lambda", 2 * 1.11568)]),
        ("H_cc (2c+4l)", 2 * MC + 4 * ML, 2 + 3 * D1,
         [("Lambda_c Lambda_c", 2 * 2.28646), ("Xi_cc + N", 3.6216 + 0.9383)]),
        ("H_bb (2b+4l)", 2 * MB + 4 * ML, 2 + 9 * D1,
         [("Lambda_b Lambda_b", 2 * 5.61960), ("Xi_bb + N", 10.3406 + 0.9383)]),
    ]
    for nom, core, n, seuils in cases:
        m = mass(core, n)
        print("  %-22s core %.3f, n = %.3f -> M = %.0f MeV" % (
            nom, core, n, 1e3 * m))
        for snom, sm in seuils:
            print("      vs %-20s %+.0f MeV %s" % (
                snom + " (%.0f)" % (1e3 * sm), 1e3 * (m - sm),
                "LIE" if m < sm else "au-dessus"))
    print("""
  -> H LEGER : +279 MeV AU-DESSUS de Lambda Lambda -- PAS d'etat lie :
  la machinerie base-2 POSTDIT l'absence du H-dibaryon (nulls
  experimentaux de 40 ans : coherence forte, gratuite) ;
  -> H_cc : -19 MeV sous Lambda_c Lambda_c, -6 sous Xi_cc N (seuils
  PDG) -- ETAT DE SEUIL (la systematique +-40 MeV couvre lie/cusp) ;
  -> H_bb : +45 au-dessus de Lambda_b Lambda_b, ~au seuil Xi_bb N
  (theorique) : non lie ou cusp.""")

    print("=" * 78)
    print("[2] H_cc : NOMBRES QUANTIQUES, CANAUX, LARGEUR")
    print("=" * 78)
    tau_lc = 2.02e-13
    gam = 2 * HBAR_EVS / tau_lc
    ctau = 3e8 * tau_lc / 2
    print("""  J^P = 1^+ (paire cc verrouillee spin 1, 4 trefles apparies a 0,
  S-wave) ; I = 0 (uudd symetrique).
  SI LIE (branche -19/-6 MeV) : desintegration FAIBLE seulement
  (c -> s) : Gamma ~ 2 hbar/tau(Lambda_c) ~ %.0e eV -- quasi stable a
  l'echelle du detecteur, c.tau ~ %.0f microns (VERTEX DEPLACE, la
  signature Xi_cc) ; canaux : Lambda_c Lambda pi+, Xi_c+ p K-,
  Lambda_c p K- pi+ ; production : evenements double-charme (LHCb en
  produit : Xi_cc, Omega_cc y ont ete vus).
  SI NON LIE (cusp) : resonance etroite au seuil Xi_cc N (structure
  type T_cc/X(3872) -- la meme physique de seuil que les molecules du
  document principal, mais a DOUBLE trialite).
  FALSIFIABLE : une structure a 4554 +- 40 MeV dans Lambda_c Lambda_c /
  Xi_cc N (ou son absence stricte au-dela de la fenetre).""" % (gam, 1e6 * ctau))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  La categorie double-jonction est PEUPLEE : base 2 + verrou de paire.
  - H leger : ABSENT (+279, postdiction des nulls -- le test de
    coherence passe) ;
  - H_cc = LA cible : 4554 +- 40 MeV, J^P = 1^+, etat de seuil
    Lambda_c Lambda_c / Xi_cc N -- lie -> vertex deplace ~30 um
    (signature Xi_cc-like), sinon cusp etroit ; production
    double-charme LHCb ;
  - H_bb : non lie / cusp Xi_bb N (~11 284).
  Identifications declarees : sommet 6-valent (fonctionnelle etendue),
  liaison 6-valente inconnue (systematique dominante), verrou de paire
  x3^g. Prochaine donnee qui tranche : le spectre Lambda_c Lambda_c
  de LHCb (Run 3).""")


if __name__ == "__main__":
    main()
