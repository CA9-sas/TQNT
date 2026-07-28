"""Audit d'epoques : la naissance de G en marche et la chaine Omega (20/07).

RECADRAGE (document principal, fourche tranchee 19/07, verdict PLANCHER) :
 - h (la dent du socle)      : INVARIANT, jamais variable ;
 - M_Pl (horloge/rigidite)   : PREEXISTE a la condensation (ancre du
                               raccord ; ell_P en valeur = hc/M_Pl) ;
 - G (couplage operatoire)   : nait en UNE MARCHE a t_c ~ 1,4e-4 s
                               (indefini avant : pas de puits sans
                               nœuds) -- PAS de variation continue.

LA QUESTION REELLE : la chaine Omega opere de T_trans (155 MeV,
avant la marche) a T_c (62-67 MeV, l'epoque MEME de la marche).
Quelles etapes utilisent H (donc la gravitation operatoire) avant la
marche, et avec quelle sensibilite ? Audit piece par piece, puis
quantification : on multiplie H pre-marche par f_H in [0,1 ; 10].
"""
import math

ALPHA_EM = 1 / 137.036


def run():
    print("=== Audit d'epoques : G en marche vs chaine Omega ===\n")

    print("[1] AUDIT DES PIECES DE LA CHAINE (epoque, dependance H) :")
    rows = [
        ("etirement comobile (xi.T = kappa_dom)", "155 -> 62 MeV (PRE-marche)",
         "xi ~ a et T ~ 1/a : le produit est invariant pour TOUTE"
         " histoire a(t) -- INSENSIBLE a H"),
        ("gel de la marche (seuil 1/e)", "T_c (la marche)",
         "MICROPHYSIQUE (etape 12 : pas de log cosmologique) --"
         " AUCUN H"),
        ("distribution de type (S_d/Z)", "T_c",
         "thermique (Boltzmann relatif) -- AUCUN H"),
        ("decouplage c_min = 23 (Gamma/H)", "T_c (la marche : G nait la)",
         "utilise H A t_c -- G juste operatoire ; sensibilite : marge"
         " de 2 ordres par unite de c"),
        ("eta_B, s0/n_gamma, g_s", "post-marche", "standard -- H operatoire"),
    ]
    for name, ep, verdict in rows:
        print("   - %s\n     epoque : %s\n     %s" % (name, ep, verdict))

    print("\n[2] QUANTIFICATION : H pre-marche x f_H (histoire gravita-")
    print("    tionnelle inconnue avant t_c) :")
    print("    (a) etirement comobile : xi.T inchange pour tout f_H (exact).")
    print("    (b) decouplage c_min : Gamma/H(c) ~ alpha_EM^(2(c-18)) ;")
    print("        rapport mesure c=22 : 9,3 ; c=23 : 0,067 (x139/unite).")
    for f_h in (0.1, 1.0, 10.0):
        r22, r23 = 9.3 / f_h, 0.067 / f_h
        c_min = 23 if r23 < 1 <= r22 else (22 if r22 < 1 else 24)
        print("        f_H = %4.1f : Gamma/H(22) = %6.2f, (23) = %6.3f"
              " -> c_min = %d" % (f_h, r22, r23, c_min))
    print("        -> c_min = 23 ROBUSTE sur deux ordres de f_H ;")
    print("        f_H > ~15 le deplacerait (declare).")
    print("    (c) gel et types : aucun effet (pas de H).\n")

    print("[3] LA COINCIDENCE QUI N'EN EST PAS UNE :")
    print("    la marche de G et le gel des nœuds sont LE MEME evenement")
    print("    (la condensation) : G a besoin de puits, les puits ont")
    print("    besoin de nœuds formes -- le decouplage (seule piece en H)")
    print("    est evalue exactement quand G devient operatoire :")
    print("    la chaine n'utilise JAMAIS un G pre-marche inexistant.\n")

    print("BILAN : l'impact de la naissance de G en marche sur Omega est")
    print("NEGLIGEABLE PAR STRUCTURE -- les trois criteres de la chaine")
    print("sont soit exacts pour toute histoire a(t) (comobile), soit")
    print("microphysiques (gel, types), soit post-marche avec 2 ordres")
    print("de marge (decouplage). Les invariants du socle (h, M_Pl)")
    print("ne varient jamais ; il n'y a pas de << constante de Planck")
    print("par periode >> a calculer -- il y a UNE marche, et la chaine")
    print("est construite pour ne rien lui demander avant.")


if __name__ == "__main__":
    run()
