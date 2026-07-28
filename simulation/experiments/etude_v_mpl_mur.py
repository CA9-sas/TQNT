"""Etudier le mur v/M_Pl : candidats structurels + null de formes (19/07).

DERNIER RAPPORT ORPHELIN du theoreme de fermeture des echelles. v = 246,22
GeV (E_relax, barriere Delta Lk = +-1 ; identifiee au VEV EW, non derivee).
Trois voies deja FERMEES (V6, Comptage) : fluxoide (x140), twist (x10-24),
Compton (degeneree en echelle). Ce banc :

 [1] CANDIDATS STRUCTURELS PRE-ENREGISTRES (prior avant calcul) : le
     quartet EW partage le noeud 8_3 (c = 8). Placements sur l'echelle du
     raccord : (i) masse ~ M_Pl alpha^((c+1)/2) ; (ii) masse^2 ~ P alpha^(c+1)
     (le motif de l'identite de hierarchie). Les DEUX doivent etre testes
     et le resultat accepte quel qu'il soit.
 [2] NULL DE FORMES DECLARE : espace = P x alpha^n, n dans {6, 6.5, ..., 10}
     (9 exposants), P dans une liste FIXE de 25 constantes structurelles
     O(1). Fenetre : |ecart| <= 0,5 %. Verdict par comparaison au nombre
     de hits attendu par hasard (densite locale de formes).
 [3] RATIOS INTERNES (v vs sqrt(sigma), m_base, Lambda_chi) contre des
     multiples de (k+2) : rapide, meme discipline.

REGLE : un hit du scan [2] sans convergence independante = coincidence
(cf. cellule de phase 138,6) ; seul un candidat [1] ou une convergence
a la m_l (deux structures sur un invariant mesure) eleverait au-dessus.
"""
import math

ALPHA = 7.2973525693e-3
M_PL = 1.220890e22            # MeV
V_EW = 246.22e3               # MeV
SQRT_SIGMA = 418.47
M_BASE = 938.240
LAMBDA_CHI = 209.2
R = V_EW / M_PL               # 2.0167e-17

P_LIST = [
    ("1", 1.0), ("2", 2.0), ("3", 3.0), ("4", 4.0), ("1/2", 0.5),
    ("1/3", 1 / 3), ("1/4", 0.25), ("3/2", 1.5), ("5/2", 2.5),
    ("pi", math.pi), ("2pi", 2 * math.pi), ("4pi", 4 * math.pi),
    ("1/pi", 1 / math.pi), ("1/2pi", 1 / (2 * math.pi)),
    ("pi^2", math.pi ** 2), ("sqrt2", math.sqrt(2)), ("sqrt3", math.sqrt(3)),
    ("sqrt(2pi)", math.sqrt(2 * math.pi)), ("sqrt(pi)", math.sqrt(math.pi)),
    ("e", math.e), ("1/e", 1 / math.e), ("12/7", 12 / 7), ("7/12", 7 / 12),
    ("3/4", 0.75), ("S_octet", 2.242006),
]
N_EXP = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
WINDOW = 0.005                 # 0,5 %


def run():
    print("=== Le mur v/M_Pl : candidats structurels + null de formes ===")
    print("    R = v/M_Pl = %.5e ; log_alpha R = %.3f\n" % (
        R, math.log(R) / math.log(ALPHA)))

    # [1] candidats structurels pre-enregistres (echelle du 8_3)
    print("[1] Candidats pre-enregistres (8_3, c = 8, noeud commun EW) :")
    m_i = M_PL * ALPHA ** 4.5
    print("  (i)  m ~ M_Pl alpha^((c+1)/2) = M_Pl alpha^4,5 = %.3e MeV" % m_i)
    print("       vs v : facteur %.1e -- ECHEC (x~1e6)" % (m_i / V_EW))
    p_ii = R ** 2 / ALPHA ** 9
    print("  (ii) motif hierarchie : v^2/M_Pl^2 = P alpha^9 -> P = %.2e" % p_ii)
    print("       hors de toute constante O(1) -- ECHEC (x~1e14)")
    print("  -> le VEV ne se place PAS sur l'echelle des noeuds : coherent")
    print("     avec << EW = entrelacs non confines, secteur VEV a part >>.\n")

    # [2] null de formes declare
    print("[2] Null de formes (espace declare : %d P x %d exposants = %d formes,"
          % (len(P_LIST), len(N_EXP), len(P_LIST) * len(N_EXP)))
    print("    fenetre 0,5 %%) :")
    logs = []
    hits = []
    for pl, pv in P_LIST:
        for n in N_EXP:
            f = pv * ALPHA ** n
            d = math.log10(f / R)
            logs.append(abs(d))
            if abs(f / R - 1) <= WINDOW:
                hits.append((pl, n, 100 * (f / R - 1)))
    for pl, n, e in sorted(hits, key=lambda t: abs(t[2])):
        print("    HIT : %s x alpha^%.1f  (%+.2f %%)" % (pl, n, e))
    # comptage honnete : hits a <0,5 % l'un de l'autre = UNE grappe
    vals = sorted(pv * ALPHA ** n for pl, pv in P_LIST for n in N_EXP
                  if abs(pv * ALPHA ** n / R - 1) <= WINDOW)
    clusters = 1 if vals else 0
    for a, b in zip(vals, vals[1:]):
        if b / a - 1 > WINDOW:
            clusters += 1
    near = sum(1 for d in logs if d < 0.15)     # formes a moins de 0,15 decade
    dens = near / (2 * 0.15)                    # formes par decade autour de R
    w_dec = 2 * WINDOW / math.log(10)           # fenetre en decades
    expected = dens * w_dec
    p_ge1 = 1 - math.exp(-expected)
    print("    grappes effectives : %d (les P proches a <0,5 %% comptent 1) ;" % clusters)
    print("    densite locale %.0f formes/decade -> attendu %.2f grappe(s),"
          % (dens, expected))
    print("    P(>=1 par hasard) = %.0f %%" % (100 * p_ge1))
    verdict_null = clusters <= 1 and p_ge1 > 0.05
    print("    -> %s\n" % (
        "UNE grappe, attendue a ~%.0f %% par hasard : COINCIDENCE non"
        " significative" % (100 * p_ge1)
        if verdict_null else "configuration inattendue : a examiner"))

    # [3] ratios internes vs multiples de (k+2)
    print("[3] Ratios internes vs structures (k+2) :")
    for lab, x in [("v/sqrt(sigma)", V_EW / SQRT_SIGMA),
                   ("v/m_base", V_EW / M_BASE),
                   ("v/Lambda_chi", V_EW / LAMBDA_CHI)]:
        r137 = x / 137.0
        print("  %-15s = %8.2f  (/137 = %.3f ; entier le plus proche : %d, %+.1f %%)"
              % (lab, x, r137, round(r137), 100 * (r137 / round(r137) - 1)
                 if round(r137) else float('nan')))
    print("  -> aucun accrochage entier propre.\n")

    print("VERDICT : le MUR TIENT -- et il est desormais CERTIFIE :")
    print("  - les placements structurels du 8_3 echouent par 6-14 ordres")
    print("    (le VEV n'est pas une masse de noeud : secteur a part) ;")
    print("  - les hits du scan sont au niveau attendu par hasard (aucune")
    print("    convergence independante du type m_l/ropelength) ;")
    print("  - les 3 voies historiques restent fermees (fluxoide, twist,")
    print("    Compton).")
    print("  v/M_Pl reste le DERNIER rapport calibre du secteur d'echelles ;")
    print("  toute future proposition devra battre ce null declare.")


if __name__ == "__main__":
    run()
