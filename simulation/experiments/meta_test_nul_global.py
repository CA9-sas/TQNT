"""Méta-test nul global — le facteur d'essais cumulé de la TQNT
(27/07/2026, réponse à l'avis critique : la critique n°2).

LA CRITIQUE : chaque résultat porteur passe son test nul local, mais
l'édifice a fait des DIZAINES de choix (identifications, jeux fermés,
balayages) — sans comptabilité globale, quelques coïncidences au ppm
deviennent moins improbables qu'il n'y paraît.

LA MÉTHODE (estimation d'ordre, PAS une statistique rigoureuse —
déclaré) : pour chaque résultat porteur,
  - epsilon = précision relative de l'accord (fenêtre bilatérale) ;
  - T = facteur d'essais ENCADRÉ [T_bas, T_haut] : T_bas = lecture
    « mécanisme accepté » (le mécanisme sélectionne la forme, seuls
    les choix résiduels comptent) ; T_haut = lecture « balayage nu »
    (toutes les formes comparables auraient été acceptées) ;
  - n = nombre de confrontations INDÉPENDANTES du même mécanisme
    (ex. : l'hyperfin est testé sur ~8 systèmes indépendants) ;
  - p par résultat ~ min(1, T (2 epsilon)^n) [fenêtre uniforme ; le
    facteur d'essais s'applique UNE fois au choix de la forme, les n
    confrontations indépendantes multiplient ensuite].
Combinaison : produit des p indépendants (classes d'indépendance
déclarées ; les résultats DÉRIVÉS d'un autre sont exclus), puis
correction de « regard ailleurs » global : multiplication par le
nombre de tentatives du programme, borné par le compte OBJECTIF des
scripts d'expérience du dépôt (proxy déclaré).

SORTIES : table des p encadrées ; le NOYAU DUR (résultats qui
survivent à la comptabilité PESSIMISTE, p_haut < 0,01) ; la
significativité globale encadrée ; le verdict honnête.
"""
import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# (nom, epsilon, T_bas, T_haut, n_indep, justification courte)
PORTEURS = [
    ("v = sqrt(2pi) a^8 M_Pl (1+a/4pi)", 2.8e-5, 1, 300, 1,
     "T_haut : formes sqrt(q pi) a^n M_Pl, n=6..10 x ~20 préfacteurs "
     "(le null du 19/07) ; T_bas : le croisement-ouvert sélectionne la forme"),
    ("Koide (A=rac2, delta, mu) -> m_mu, m_tau", 1e-4, 10, 5000, 2,
     "2 rapports prédits à ~1e-4 ; T : A dans le jeu des dimensions "
     "quantiques (~10) x delta rationnels (~50) x formes de mu (~10)"),
    ("G : (12/7) a^18 (lecture de M_Pl)", 5e-4, 3, 150, 1,
     "T_haut : préfacteurs rationnels ~20 x exposants 15-21 ; T_bas : "
     "18=c+1 motivé, 12/7 mécanisé (r_p/lambda_C x 3/7)"),
    ("k+2 = round(8pi/3 l(3_1)) = 137", 0.32, 1, 20, 1,
     "proximité d'entier 0,16/0,5 ; T_haut : ~20 formes (a pi^b/c) l ; "
     "epsilon EST la proximité (pas une précision fine) — faible seul"),
    ("oscillations K0/D0/B0/Bs0 (2 prédites + ratio)", 1.5e-3, 2, 30, 2,
     "2 paramètres calibrés -> 2 systèmes prédits <0,15 % + ratio 0,1 % ; "
     "T : variantes de la forme 3-corps essayées (V5.35a-d)"),
    ("hyperfin V-P = pi sigma_chi (universel)", 7e-3, 2, 5, 6,
     "~8 systèmes pi->Bs à 0,7 % ; n=6 indépendants retenus (prudent) ; "
     "T : le quantum dans {1/4, 1/2, 1, 2} tours"),
    ("échelle de recouplage (15 baryons, formes closes)", 5e-3, 10, 100, 6,
     "15 états à 0,23 % zéro ancre ; n=6 indépendants retenus ; T : "
     "base{0,1} x (3 vs m_b/m_c) x demi x (d-1 dans jeu ~5) x échelles"),
    ("Fermi P15 : g_topo = 1 (0,11 %) + tau_mu", 2e-3, 2, 20, 1,
     "l'unité à 0,11 % ; T : normalisations du vertex essayées"),
    ("E_complet = alpha m_l (-0,3 %)", 3e-3, 3, 30, 1,
     "contrôle croisé k=135 <-> framing ; T : autres égalités candidates "
     "qu'on aurait acceptées"),
    ("Omega_cc dans [3715, 3788] (prédiction datée)", 0.24, 1, 2, 1,
     "fenêtre 73 MeV sur une gamme plausible ~300 MeV ; prédiction, "
     "T minimal"),
    ("n_s = 0,965 (Starobinsky adopté)", 1e-3, 2, 20, 1,
     "forme R^2 adoptée (choix) ; l'accord fin ensuite sans réglage"),
    ("A_s = 1,89e-9 sans paramètre libre", 5e-2, 1, 5, 1,
     "accord à ~5 % ; chaîne sans réglage mais une seule confrontation"),
    ("eta_B (0,4 sigma)", 0.3, 1, 3, 1,
     "chaîne d'ordre ; faible individuellement, compté honnêtement"),
    ("m_e = Green retardée (+-3 %)", 3e-2, 3, 20, 1,
     "chaîne zéro paramètre mais 4 lectures essayées avant la bonne "
     "(consignées) — compté comme suggestif, PAS porteur"),
]

# résultats EXCLUS du produit (dépendants d'un autre, ou calibrés dessus) :
EXCLUS = [
    "m_base 938,2 (sert à la recalibration de sqrt(sigma))",
    "GPS/Mercure/GW150914 (dérivés de G : comptés via G)",
    "Delta m(n-p) (beta calibré dessus)",
    "89 états 0,13 % (assignations : compatibilités, pas porteurs)",
    "Omega_tot=1, w=-1, indiscernabilité (théorèmes : pas des accords "
    "numériques)",
]


def p_range(eps, t_lo, t_hi, n):
    # le facteur d'essais s'applique UNE fois (au choix de la forme) ;
    # les n confrontations indépendantes multiplient ensuite (2 eps)^n
    p_lo = min(1.0, t_lo * (2 * eps) ** n)
    p_hi = min(1.0, t_hi * (2 * eps) ** n)
    return p_lo, p_hi


def main():
    print(__doc__)
    print("=" * 78)
    print("TABLE DES PORTEURS (p encadrée par le facteur d'essais)")
    print("=" * 78)
    prod_opt = 1.0
    prod_pess = 1.0
    noyau = []
    for nom, eps, tlo, thi, n, just in PORTEURS:
        plo, phi = p_range(eps, tlo, thi, n)
        prod_opt *= plo
        prod_pess *= phi
        tag = ""
        if phi < 0.01:
            noyau.append((nom, phi))
            tag = "  <== NOYAU DUR (survit au pessimiste)"
        print("  %-46s p in [%.1e, %.1e]%s" % (nom[:46], plo, phi, tag))
        print("      (%s)" % just)
    print("\n  Exclus du produit (dépendances déclarées) :")
    for e in EXCLUS:
        print("    - %s" % e)

    print("=" * 78)
    print("LE COMPTE OBJECTIF DES TENTATIVES (regard ailleurs global)")
    print("=" * 78)
    exp_dir = pathlib.Path(__file__).resolve().parent
    n_scripts = len(list(exp_dir.glob("*.py")))
    n_ess_lo, n_ess_hi = n_scripts // 4, 3 * n_scripts
    print("  scripts d'expérience du dépôt : %d" % n_scripts)
    print("  tentatives « coïncidence de titre » estimées : [%d, %d]" % (
        n_ess_lo, n_ess_hi))
    print("  (proxy déclaré : chaque script = 0,25 à 3 confrontations "
          "qu'on aurait retenues si elles avaient réussi)")

    print("=" * 78)
    print("COMBINAISON GLOBALE")
    print("=" * 78)
    glob_opt = min(1.0, prod_opt * n_ess_lo)
    glob_pess = min(1.0, prod_pess * n_ess_hi)
    print("  produit des p (avant regard ailleurs) : optimiste %.1e / "
          "pessimiste %.1e" % (prod_opt, prod_pess))
    print("  significativité GLOBALE (x tentatives) : optimiste %.1e / "
          "PESSIMISTE %.1e" % (glob_opt, glob_pess))
    prod_noyau = 1.0
    for nom, phi in noyau:
        prod_noyau *= phi
    print("  NOYAU DUR seul (p pessimistes, x %d tentatives) : %.1e" % (
        n_ess_hi, min(1.0, prod_noyau * n_ess_hi)))
    print("""
VERDICT HONNÊTE :
  1. LE NOYAU DUR — les résultats STRUCTURAUX multi-systèmes
     (hyperfin V-P, échelle de recouplage, oscillations, Koide) —
     survit à TOUTE comptabilité d'essais plausible : même au
     pessimiste et après regard ailleurs, sa significativité reste
     écrasante. C'est lui, l'argument.
  2. Les résultats de CONSTANTE UNIQUE (v, G, k=137, E=alpha m_l,
     n_s) sont individuellement entre « suggestifs » et « forts »
     selon la sévérité du facteur d'essais : leur poids repose sur
     les MÉCANISMES déclarés, pas sur les seuls nombres — exactement
     le statut « conditionnel » du texte.
  3. Les items faibles (k=137 seul, A_s, eta_B, Omega_cc seule,
     m_e +-3 %) n'apportent presque rien statistiquement — les
     compter comme « porteurs » affaiblirait la présentation ;
     ils valent comme COHÉRENCES.
LIMITES (déclarées) : les T sont des jugements encadrés ; les
indépendances sont approximatives ; la fenêtre uniforme est grossière.
Ceci est une ESTIMATION D'ORDRE du facteur d'essais — sa conclusion
robuste est la HIÉRARCHIE (noyau dur / conditionnels / cohérences),
pas les valeurs numériques exactes des p.""")


if __name__ == "__main__":
    main()
