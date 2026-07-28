"""Confrontation aux donnees LHCb (juillet 2026) : verifications et statuts.

SOURCES : LHCb outreach 2026 (Xi_cc+ 17/03, Omega_cc+ ~3727 MeV 03/06,
Beauty 2026) ; LHCb 2017-2023 (quintet Omega_c, quartet Omega_b 2020,
Xi_b(6087/6095) 2023, Lb(5912/5920), Xi_cc++(3621)) ; PDG 2024.

QUATRE VOLETS :
 [A] Omega_cc OBSERVE ~3727 MeV vs prediction 3715-3788 (fourchette
     verrou/bascule du 21/07) : DANS la fourchette, pres du bord VERROU
     -> tranche l'ambiguite Xi_cc declaree, affute Omega_ccc.
     CHRONOLOGIE DECLAREE : annonce 03/06 < consolidation V6 21/07 --
     derivation aveugle en pratique (le document principal traitait Omega_cc comme
     non observe), mais PAS une pre-registration stricte.
 [B] Predictions OUVERTES (non observees, non contredites) :
     Xi_cc* 3659, Omega_b* 6067, B0*/B1' 5665, Omega_ccc -> 4782.
 [C] TESTS sur donnees LHCb existantes :
     C1. shift etrange des bandes BARYONIQUES (prediction S25 d'hier !) :
         Dn(Xi_c P)-Dn(Lambda_c P) et Dn(Xi_b P)-Dn(Lambda_b P) vs s ;
     C2. verrou ss des fonds Omega_c/Omega_b (re-verification donnees) ;
     C3. valeurs de bande P baryoniques MESUREES (le residu nomme).
"""
import numpy as np

SIGMA = 0.41847 ** 2
PS = np.pi * SIGMA                       # pi sigma_chi (demi-tour)
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
D = np.sqrt(2)


def nval(m, core):
    return (m * m - core * core) / PS


def cen(states):                          # (m, J) -> centroide (2J+1)
    w = np.array([2 * j + 1 for _, j in states])
    m = np.array([m for m, _ in states])
    return (w * m).sum() / w.sum()


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] OMEGA_cc : LA FOURCHETTE TIENT, LE VERROU EST SELECTIONNE")
    print("=" * 78)
    m_obs = 3.727                        # LHCb outreach 03/06/2026 (~3727)
    lo, hi = 3.715, 3.788                # prediction 21/07 (verrou / bascule)
    print("""  LHCb (03/06/2026, Beauty 2026) : Omega_cc+ ~ %.0f MeV (Omega_c0 pi+,
  donnees 2024 ; masse fine a paraitre). Prediction V6 (21/07) :
  %.0f-%.0f MeV. -> DANS LA FOURCHETTE ; ecart au bord VERROU (%.0f) :
  %+.1f MeV (%+.2f %%), au bord bascule : %+.1f MeV.
  CONSEQUENCES : (i) l'ambiguite Xi_cc verrou/bascule declaree est
  TRANCHEE par les donnees -> VERROU (la paire cc au verrou, comme ss) ;
  (ii) Omega_ccc s'affute : %.0f MeV (branche verrou ; la fourchette
  4782-4840 s'effondre) ; (iii) coherence reseau (~3,7 GeV) confirmee.
  CHRONOLOGIE : annonce 03/06 < consolidation 21/07 -- derivation
  aveugle en pratique (document principal : « non observe »), pas une
  pre-registration ; a declarer ainsi.""" % (
        1e3 * m_obs, 1e3 * lo, 1e3 * hi, 1e3 * lo,
        1e3 * (m_obs - lo), 100 * (m_obs - lo) / lo, 1e3 * (m_obs - hi),
        4782))

    print("=" * 78)
    print("[B] PREDICTIONS OUVERTES (aucune contredite au 22/07/2026)")
    print("=" * 78)
    print("""  - Xi_cc* (J=3/2) = 3659 MeV : le Xi_cc+ (isopartenaire, 17/03/2026)
    est le FOND ; l'etoile reste a observer ;
  - Omega_b* (S, J=3/2) = 6067 MeV (+22) : toujours non observe (le
    quartet 6316-6350 de 2020 = bandes P, pas l'etoile) ;
  - B0*/B1' : centroide 5665 MeV, membres ~5660/5666, splitting 6,6 MeV
    (rien d'observe dans cette zone ; B_J(5840)/B(5970) = candidats 2S,
    hors zone) ;
  - saturation Z3 (jamais de 4e famille) : aucun signe contraire.""")

    print("=" * 78)
    print("[C1] LE SHIFT ETRANGE DES BANDES BARYONIQUES (prediction S25)")
    print("=" * 78)
    # doublets P lambda-mode (diquark leger spin 0) : structure identique,
    # une substitution l->s entre Lambda_Q et Xi_Q -> Dn attendu +s
    lc_core, xc_core = MC + 2 * ML, MC + ML + MS
    lb_core, xb_core = MB + 2 * ML, MB + ML + MS
    data = {
        "Lambda_c": (lc_core, 2.28646,
                     [(2.59225, 0.5), (2.62811, 1.5)]),
        "Xi_c":     (xc_core, 2.46944,
                     [(2.7924, 0.5), (2.8167, 1.5)]),
        "Lambda_b": (lb_core, 5.61960,
                     [(5.91219, 0.5), (5.92009, 1.5)]),
        "Xi_b":     (xb_core, 5.7919,
                     [(6.0872, 0.5), (6.0954, 1.5)]),
    }
    dn = {}
    print("  %-9s %8s %8s %8s %8s" % ("etat", "n_fond", "n_P", "Dn(L=1)", ""))
    for nom, (core, mg, pstates) in data.items():
        ng = nval(mg, core)
        np_ = nval(cen(pstates), core)
        dn[nom] = np_ - ng
        print("  %-9s %8.3f %8.3f %8.3f" % (nom, ng, np_, dn[nom]))
    s_pred = 0.359
    d_c = dn["Xi_c"] - dn["Lambda_c"]
    d_b = dn["Xi_b"] - dn["Lambda_b"]
    print("""
  Dn(Xi_c) - Dn(Lambda_c) = %+.3f   (prediction s = %.3f : %+.0f %%)
  Dn(Xi_b) - Dn(Lambda_b) = %+.3f   (prediction s = %.3f : %+.0f %%)
  -> LE SIGNE EST CONFIRME (les bandes etranges sont decalees vers le
  HAUT, deux secteurs independants, +0,26/+0,32 tres loin de zero) et
  l'ORDRE est le bon ; les valeurs tombent 10-30 %% SOUS le s mesonique
  -- compatible avec la lecture identite (l'ecart = systematique des
  fonds/coeurs, ou une dilution baryonique partielle a examiner).
  VERDICT : prediction S25 SUPPORTEE (signe + ordre), decimale ouverte.
  NB : Omega_c/Omega_b (quintet/quartet, diquark ss spin 1, 5 etats)
  = structure de multiplet DIFFERENTE : pas de comparaison directe.""" % (
        d_c, s_pred, 100 * (d_c / s_pred - 1),
        d_b, s_pred, 100 * (d_b / s_pred - 1)))

    print("=" * 78)
    print("[C2] LE VERROU ss DES FONDS Omega (re-verification donnees LHCb/PDG)")
    print("=" * 78)
    oc_core, ob_core = MC + 2 * MS, MB + 2 * MS
    n_oc = nval(2.6952, oc_core)
    n_ob = nval(6.0458, ob_core)
    print("""  n(Omega_c fond) = %.4f  vs ladder 1+(d-1) = %.4f  (%+.2f %%)
  n(Omega_b fond) = %.4f  vs ladder 1+3(d-1) = %.4f  (%+.2f %%)
  -> le verrou ss (exces d-1 par unite, x3 par generation, NON borne
  par la saturation) tient sur les DONNEES aux deux generations.""" % (
        n_oc, 1 + (D - 1), 100 * (n_oc / (1 + (D - 1)) - 1),
        n_ob, 1 + 3 * (D - 1), 100 * (n_ob / (1 + 3 * (D - 1)) - 1)))

    print("=" * 78)
    print("[C3] VALEURS DE BANDE P BARYONIQUES : MESUREES (residu nomme)")
    print("=" * 78)
    r = dn["Lambda_b"] / dn["Lambda_c"]
    print("""  Dn(Lambda_c) = %.3f ; Dn(Lambda_b) = %.3f ; ratio b/c = %.2f --
  le MEME « ratio ~2 » que les bandes mesoniques avant decomposition
  (ou il CACHAIT le x3 derriere l'orbital aveugle). Candidats de
  decomposition Dn = b + 3^g X : aucun (b, X) canonique propre ne
  tombe (b=2 -> ratio 4,5 ; b=1 -> 2,7 ; b=3/2 -> 3,3) : la bande
  baryonique reste LE residu nomme (recouplage jonction Y complet) --
  desormais avec ses quatre valeurs MESUREES (Lambda_c/Xi_c/
  Lambda_b/Xi_b) + les multiplets Omega_c (quintet 2017/2023) et
  Omega_b (quartet 2020) en attente de la regle de multiplet.""" % (
        dn["Lambda_c"], dn["Lambda_b"], r))

    print("=" * 78)
    print("BILAN LHCb 2026")
    print("=" * 78)
    print("""  VERIFIE : Omega_cc ~3727 DANS la fourchette 3715-3788 (bord verrou
  +12 MeV ; chronologie declaree) -> ambiguite Xi_cc TRANCHEE (verrou),
  Omega_ccc affute a 4782 MeV (prediction stricte, plus de fourchette) ;
  verrou ss des fonds Omega re-verifie (+0,1/+1,2 %) ; SHIFT ETRANGE
  BARYONIQUE : signe et ordre confirmes sur 2 secteurs (+0,32/+0,26 vs
  +0,359), decimale ouverte. OUVERT : Xi_cc* 3659, Omega_b* 6067,
  B0*/B1' 5665, Omega_ccc 4782. CONTREDIT : rien.""")


if __name__ == "__main__":
    main()
