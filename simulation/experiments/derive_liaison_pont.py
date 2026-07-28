"""La liaison du pont derivee : le contact ferme la fenetre (22/07).

LA CIBLE : la fenetre [0, +19 MeV] du pont (S37) -- la longueur ell du
pont etait laissee libre dans [0, D]. La deriver = l'epingler.

LE THEOREME EN TROIS PIECES :
[1] LE PLANCHER (impenetrabilite, exact) : les jambes de J1 et les
    jambes de J2 sont des tubes NON ADJACENTS (ne partagent aucune
    jonction ; le chemin qui les relie passe par le pont) -> leurs
    axes sont a distance >= D (rayon D/2 chacun). Or les points de
    jonction J1, J2 sont SUR ces axes (les axes des jambes passent
    par leur jonction) => d(J1, J2) >= D. La fusion (contourner le
    plancher en coalescant) = le sommet 6-valent, EXCLU par Steiner
    (S37). Le plancher est une vraie frontiere.
[2] PAS D'ANCRE AU-DESSUS (lecture de la loi de masse) : dans le
    document principal, les jambes ont une longueur NULLE (la loi (somme m)^2 +
    torsion ne porte AUCUN terme de longueur de jambe -- le catalogue
    tient a 0,13 % de mediane sans lui ; les noeuds sont au contact de
    la jonction). Sans jambes etendues, aucune force ne retient les
    jonctions a ell > D (l'equilibre de Steiner a jambes etendues --
    l'evasement a arccos(-1/3) = 109,47 deg -- n'existe pas ici). La
    tension sigma_chi tire -> ell tombe au plancher : ell = D EXACT.
[3] LA COMPTABILITE (le pont compte en entier) : le pont est le SEUL
    tube nouveau du dibaryon par rapport a deux baryons libres (les
    m_i comptent noeuds + jonctions de chaque baryon). Un conduit qui
    relie deux points distants de D a une longueur >= D (egalite :
    pont droit, que la tension impose). => E_pont = sigma_chi x D.

CONCURRENTS EXCLUS : ell -> 0 (geometriquement impossible une fois le
plancher etabli : le conduit DOIT enjamber D -- le lire a zero, c'est
nier l'existence du conduit que la trialite exige) ; D/2 (l'estimation
centrale du S37 : AUCUN mecanisme -- regle du null) ; echelle de
Steiner aux terminaux (~0,4 x distance des noeuds >> D : exige des
jambes etendues, CONTREDIT la loi de masse sans jambes).
"""
import numpy as np

SIGMA = 0.175117                        # GeV^2
PS = np.pi * SIGMA
ML, MS_, MC, MB = 0.310, 0.4948, 1.557, 4.952
D1 = np.sqrt(2) - 1
HBARC = 0.19733                         # GeV.fm
D_TUBE = 0.0214                         # fm (diametre canonique)
E_PONT = SIGMA * D_TUBE / HBARC         # GeV


def seg_dist(p1, d1, p2, d2, L=5.0, n=400):
    """Distance min entre deux segments p+t*d, t in [0, L] (echantillonne)."""
    t = np.linspace(0.0, L, n)
    a = p1[None, :] + t[:, None] * d1[None, :]
    b = p2[None, :] + t[:, None] * d2[None, :]
    return np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2))


def check_floor(ell):
    """Verifie d(jambes J1, jambes J2) = ell pour les deux geometries."""
    res = {}
    for nom, zc in [("contact 120 deg (plan)", 0.0),
                    ("evasement Steiner (-1/3)", 1.0 / 3.0)]:
        dmin = np.inf
        for i in range(3):
            for j in range(3):
                th1 = 2 * np.pi * i / 3
                th2 = 2 * np.pi * j / 3 + np.pi / 3      # decalage 60 deg
                r = np.sqrt(1 - zc**2)
                d1 = np.array([r * np.cos(th1), r * np.sin(th1), -zc])
                d2 = np.array([r * np.cos(th2), r * np.sin(th2), +zc])
                dmin = min(dmin, seg_dist(
                    np.zeros(3), d1, np.array([0, 0, ell]), d2))
        res[nom] = dmin
    return res


def m_st(core, n, br):
    return np.sqrt((core + br)**2 + PS * n)


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] LE PLANCHER VERIFIE NUMERIQUEMENT (jambes vs jambes)")
    print("=" * 78)
    ell = 1.0
    for nom, dmin in check_floor(ell).items():
        print(f"  {nom:28s} : d_min(jambes J1, jambes J2) = "
              f"{dmin:.4f} x ell (attendu 1)")
    print("""  -> dans les DEUX geometries (contact baryon-standard et evasement
  libre), la distance min entre les deux systemes de jambes EST la
  separation des jonctions : le plancher d(J1,J2) >= D est exact et
  SATURE au contact -- aucune approche sous D sans chevauchement.""")

    print("=" * 78)
    print("[B] LA VALEUR")
    print("=" * 78)
    print(f"  E_pont = sigma_chi x D = {SIGMA:.6f} GeV^2 x "
          f"{D_TUBE / HBARC:.6f} GeV^-1 = {1e3 * E_PONT:.1f} MeV")
    print(f"  Sensibilite : delta E / E = delta D / D "
          f"(D a +-2 %% -> +-{1e3 * 0.02 * E_PONT:.1f} MeV, sous-dominant)")
    print("  LA FENETRE [0, +19] SE FERME SUR SON BORD HAUT : ell = D.")

    print("=" * 78)
    print("[C] LES QUATRE ETATS RE-CHIFFRES (pont = D, plus de fenetre)")
    print("=" * 78)
    etats = [
        ("H leger  0+", 2 * MS_ + 4 * ML, 2,
         [("Lambda Lambda", 2.23136)]),
        ("H_cc++   0+", 2 * MC + 4 * ML, 2,
         [("Xi_cc N", 4.5599), ("Lambda_c Lambda_c", 4.57292)]),
        ("H_cc++*  1+", 2 * MC + 4 * ML, 2 + 3 * D1,
         [("Xi_cc N", 4.5599), ("Lambda_c Lambda_c", 4.57292)]),
        ("H_bb0    0+", 2 * MB + 4 * ML, 2,
         [("Lambda_b Lambda_b", 11.23920)]),
        ("H_bb0*   1+", 2 * MB + 4 * ML, 2 + 9 * D1,
         [("Lambda_b Lambda_b", 11.23920), ("Xi_bb N", 11.2789)]),
    ]
    prev = {"H leger  0+": 2473, "H_cc++   0+": 4488, "H_cc++*  1+": 4563,
            "H_bb0    0+": 11203, "H_bb0*   1+": 11294}
    for nom, core, n, seuils in etats:
        m = m_st(core, n, E_PONT)
        print(f"  {nom} : M = {1e3 * m:.1f} MeV  "
              f"(ex-central {prev[nom]}, pont D/2)")
        for snom, sm in seuils:
            print(f"      vs {snom:20s} ({1e3 * sm:.1f}) : "
                  f"{1e3 * (m - sm):+.1f} MeV")
    m_ex = m_st(2 * MC + 4 * ML, 2 + 3 * D1, E_PONT)
    m_fo = m_st(2 * MC + 4 * ML, 2, E_PONT)
    print(f"""
  -> H_cc++ fondamental 0+ : 4497 +- 10 -- LIE PROFOND ({1e3 * (m_fo - 4.57292):+.0f}/
     {1e3 * (m_fo - 4.5599):+.0f}) : la carte (faible seule, c.tau ~30 um, vertex
     deplace, canaux) est INCHANGEE, la masse remonte de 9 MeV ;
  -> L'EXCITE 1+ TOMBE SUR LE SEUIL Lambda_c Lambda_c :
     {1e3 * m_ex:.1f} vs 4572,9 = {1e3 * (m_ex - 4.57292):+.1f} MeV (!) -- l'echo de la
     physique de seuil du document principal (T_cc a -0,4 du seuil D0 D*+, X(3872)
     a -0,0 de D Dbar*) : l'excite est EPINGLE au seuil a +-10 pres,
     et +12,5 au-dessus de Xi_cc N -> resonance etroite Xi_cc N juste
     sous Lambda_c Lambda_c ; le falsifiable « jamais de pic au-dessus
     de 4573 » devient EXACT au central ;
  -> doublet 0+/1+ = {1e3 * (m_ex - m_fo):.1f} MeV (inchange ~75) ;
  -> H_bb0 : 11212 +- 10, LIE (-27, marge reduite mais robuste) ;
  -> H leger : 2481, +250 -- ABSENT, robuste (le pont derive
     l'ELOIGNE du seuil : la postdiction se renforce).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  LA LIAISON DU PONT EST DERIVEE : ell = D par le theoreme du contact
  (plancher d'impenetrabilite exact + fusion Steiner-exclue + aucune
  ancre au-dessus [loi de masse sans jambes] + le conduit compte en
  entier) -> E_pont = sigma_chi x D = 19,0 MeV. La fenetre [0, +19]
  n'existe plus : c'etait un intervalle d'ignorance, le mecanisme la
  ferme sur son bord haut. Statut : theoreme conditionnel aux
  identifications canoniques (D = diametre canonique du tube ; jambes
  de longueur nulle = la lecture etablie de la loi (somme m)^2).
  Cartes mises a jour : H_cc++ 0+ = 4497 +- 10 (lie profond), excite
  1+ = 4572 +- 10 EPINGLE au seuil Lambda_c Lambda_c, doublet 75,
  H_bb0 = 11212 +- 10 (lie), H leger 2481 (absent renforce).""")


if __name__ == "__main__":
    main()
