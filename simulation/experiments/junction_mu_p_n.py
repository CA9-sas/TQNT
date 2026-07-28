"""Jonction : moment de tube 3-composantes (absolu) <-> differentiel mu_p - mu_n (enlacement).

TENSION IDENTIFIEE : le moment de twist est Lk-AVEUGLE (Tw*, s, l = proprietes du type de
noeud ; p et n = memes noeuds, seul Lk differe) -> mu_tw(p) = mu_tw(n). Le differentiel doit
venir d'un AUTRE terme.

JONCTION (modele a deux termes) :
    mu = mu_tw(coeur, twist des tubes) + Q * (r_link/lambda_C) * mu_N
(la charge NETTE Q = Sigma chi + Lk circule au rayon d'enlacement a v=c -- exactement la
'reponse d'enlacement' deja etablie pour le differentiel).

RESULTATS (nucleon) :
  - NEUTRON Q=0 : le terme de circulation s'annule -> mu_n = mu_tw PUR -> le calcul
    3-composantes (d_tube = lambda_C(p) a 1%) est PRESERVE tel quel.
  - Differentiel : DeltaQ=1 -> r_link = 0.9896 fm (le 'peripheral edge' ~1 fm etabli).
  - TROISIEME coherence : E_K = hbar c/r_link = 199 MeV = l'echelle de seuil ~200 MeV etablie.
  - mu_p = mu_n + 4.706 = 2.7929 (mesure 2.79285) -- IDENTITE (2 params, 2 donnees) ;
    le contenu est la STRUCTURE (signes, Q=0 explique le neutron coeur-domine) et les
    3 coherences d'echelle, pas un fit de plus.

TESTS ZERO-PARAMETRE (hyperons, meme doublet = memes noeuds, DeltaLk connu) :
  - Sigma+ - Sigma- (DeltaQ=2) : kappa_Sigma = 2.30 vs kappa_N = 4.71 (ratio 2.05) -> PAS
    universel.
  - Xi0 - Xi- (DeltaQ=+1) : mesure -0.599 mu_N, prediction +3.35 -> SIGNE FAUX. ECHEC.
  => la circulation de charge au rayon d'enlacement est etablie AU NUCLEON ; elle n'est pas
     universelle (le secteur etrange circule autrement). OUVERT.
"""
LAM_C = 0.2103            # fm, hbar/(m_p c)
MU_P, MU_N_MEAS = 2.79285, -1.91304
M_P, M_SIGMA, M_XI = 938.272, 1193.2, 1318.3
MU_SP, MU_SM = 2.458, -1.160      # Sigma+/Sigma-
MU_XI0, MU_XIM = -1.250, -0.6507  # Xi0/Xi-


def run():
    print("=== jonction : mu = mu_tw(coeur) + Q*(r_link/lamC)*mu_N ===")
    u = MU_P - MU_N_MEAS
    r_link = u * LAM_C
    print(f"  neutron Q=0 -> mu_n = mu_tw pur (calcul d_tube=lamC PRESERVE)")
    print(f"  differentiel {u:.4f} mu_N -> r_link = {r_link:.4f} fm (etabli ~1 fm)")
    print(f"  E_K = hbar c/r_link = {197.327/r_link:.0f} MeV (echelle seuil ~200 MeV : coherent)")
    print(f"  mu_p = {MU_N_MEAS + u:.4f} (mesure {MU_P}) [identite ; contenu = structure + echelles]")

    print("\n=== tests zero-parametre (hyperons) ===")
    kap_S = (MU_SP - MU_SM) / (2 * (M_P / M_SIGMA))
    print(f"  Sigma (DeltaQ=2) : kappa_Sigma={kap_S:.2f} vs kappa_N={u:.2f} (ratio {u/kap_S:.2f}) -> pas universel")
    pred_xi = u * M_P / M_XI
    print(f"  Xi (DeltaQ=+1)   : mesure {MU_XI0-MU_XIM:+.3f}, prediction {pred_xi:+.2f} -> SIGNE FAUX")
    print("\nVERDICT : jonction coherente AU NUCLEON (3 coherences d'echelle) ;")
    print("          non universelle hors nucleon (secteur etrange) -> OUVERT.")


if __name__ == "__main__":
    run()
