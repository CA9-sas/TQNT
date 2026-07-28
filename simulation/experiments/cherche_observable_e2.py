"""Chercher l'observable en E^-2 a theorie fine (test decimal du croisement-ouvert).

Suite de verif_croisement_ouvert_oscillations.py (2026-07-21). Il faut
discriminer E_relax = 246,084 (croisement-ouvert) de v(G_F) = 246,220
(-0,055 %), soit 0,11 % sur G_F ~ 1/E^2.

[1] INVENTAIRE des observables faibles en E^-2 (exp + theorie TQNT) :
    l'obstacle n'est presque jamais l'experience -- c'est la theorie
    TQNT des ingredients qui accompagnent G_F.

[2] REFORMULATION : l'observable fine existe DEJA -- c'est l'identite
    interne elle-meme. v(G_F) est mesure a 5e-7 ; le membre droit
    sqrt(7pi/6)*(k+2)*m_base est fixe par la chaine sqrt(sigma_chi)
    (recalibration 2026-07-18, accord mutuel 7e-5 ; e/mu/tau/p <= 5e-5).
    Le residu +0,055 % est donc SIGNIFICATIF (~10x la precision des
    entrees) : l'identite NUE est exclue -- elle EXIGE un habillage
    radiatif, exactement comme g_topo = 0,99890 exige le delta_QED/4
    (precedent consigne : l'arbre TQNT recoit les habillages standards).

[3] OBSERVATION (post-hoc declare) : l'habillage +alpha/(4pi) place
    l'identite a +30 ppm -- DANS la precision de la chaine m_base.
    Look-elsewhere declare (~12 candidats O(alpha/n pi)) : p ~ 25 % ->
    OBSERVATION, pas derivation. Le mur de promotion devient UNIQUE et
    CALCULABLE : l'habillage a une boucle du croisement-ouvert
    (l'analogue du delta_QED de g_topo). Falsifiable des deux cotes :
    +alpha/(4pi) -> consolidation a 30 ppm ; 0 -> refutation.
"""
import math

V_MEAS = 246.21965           # GeV, de G_F (5e-7)
M_BASE = 0.938240            # GeV, chaine sqrt(sigma_chi) (~5e-5)
K_PLUS_2 = 137
ALPHA = 1.0 / 137.035999


def run():
    print(__doc__)
    print("=" * 72)
    print("[1] INVENTAIRE E^-2 (cible : theorie a 0,05 %)")
    print("=" * 72)
    rows = [
        ("tau_mu", "1e-6", "DEFINIT v(G_F) : consomme (c'est la cible)"),
        ("0+ -> 0+ superpermis (Ft)", "2e-4 exp",
         "x V_ud = sqrt(1-lam_Cab^2) ; lam_Cab = sqrt(m_d/m_s) ~ 2,5 % TQNT"),
        ("tau_n", "5e-4 exp", "idem + g_A (1e-3) : bloque"),
        ("Q_W (Cs, p)", "0,4 % th", "trop grossier"),
        ("Moller PV", "~1 % th", "trop grossier"),
        ("boites Delta m (K0/B0)", "~50 % th", "fait (E^-4) : ordre seulement"),
        ("poles EW (m_W...)", "0,7 % TQNT", "secteur entrelacs : trop grossier"),
    ]
    for name, prec, verdict in rows:
        print("  %-28s %-10s %s" % (name, prec, verdict))
    print("  -> AUCUNE observable externe n'atteint 0,05 % : l'obstacle est")
    print("     la theorie TQNT des cofacteurs (lam_Cab, g_A, entrelacs),")
    print("     pas l'experience.")

    print("\n" + "=" * 72)
    print("[2] L'OBSERVABLE FINE EST INTERNE : l'identite elle-meme")
    print("=" * 72)
    v_pred = math.sqrt(7 * math.pi / 6) * K_PLUS_2 * M_BASE
    res = V_MEAS / v_pred - 1
    print("  v(G_F)                   = %.5f GeV   (5e-7)" % V_MEAS)
    print("  sqrt(7pi/6)*137*m_base   = %.5f GeV   (chaine m_base ~5e-5)"
          % v_pred)
    print("  residu = %+.4f %%  ~ 10x la precision des entrees :" % (100 * res))
    print("  l'identite NUE est EXCLUE -- un habillage est REQUIS")
    print("  (precedent : g_topo = 0,99890, delta_QED/4 = 0,105 %).")

    print("\n" + "=" * 72)
    print("[3] L'HABILLAGE : observation alpha/(4pi), et le mur nomme")
    print("=" * 72)
    cands = [("alpha/pi", ALPHA / math.pi), ("alpha/2pi", ALPHA / (2 * math.pi)),
             ("alpha/4pi", ALPHA / (4 * math.pi)),
             ("3alpha/8pi", 3 * ALPHA / (8 * math.pi)),
             ("alpha/8pi", ALPHA / (8 * math.pi))]
    for name, d in cands:
        vd = v_pred * (1 + d)
        print("  (1 %+11s) -> %.4f GeV  (%+.4f %%)"
              % ("+ " + name, vd, 100 * (vd / V_MEAS - 1)))
    best = v_pred * (1 + ALPHA / (4 * math.pi))
    print("\n  v_pred x (1 + alpha/4pi) = %.4f GeV -> %+.1f ppm de v(G_F),"
          % (best, 1e6 * (best / V_MEAS - 1)))
    print("  DANS la precision de la chaine m_base (~50 ppm).")
    print("  Look-elsewhere declare : ~12 candidats O(alpha/n pi) x signe,")
    print("  residu couvert par le plus proche a ~5 % -> p ~ 25 % :")
    print("  OBSERVATION, PAS derivation.")
    print("\n  LE MUR DE PROMOTION, UNIQUE ET CALCULABLE :")
    print("  l'habillage a une boucle du croisement-ouvert (le sommet ouvert")
    print("  porte le couplage chi-EM alpha ; son auto-energie doit donner")
    print("  le facteur). Falsifiable des deux cotes :")
    print("    calcul -> +alpha/(4pi)  => consolidation a 30 ppm (10 -> 9) ;")
    print("    calcul -> 0 (ou autre)  => candidat REFUTE au niveau 10 sigma.")
    print("  D'ici la : candidat FORT non consolide, v au comptage (10+2).")


if __name__ == "__main__":
    run()
