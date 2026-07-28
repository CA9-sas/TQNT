"""Depletion et compensation : G = amplitude du socle nu, CGQ = compensation.

Vision utilisateur (21/07, 3e raffinement) :
  << G est l'amplitude de l'onde du socle SANS particule. A l'emplacement
  de la particule, l'amplitude de l'onde se reduit considerablement, ce
  qui genere par effet compensatoire le CGQ. >>

Verification de coherence avec le document principal -- point par point :

[1] C'EST P20 QUI RECOIT SON MECANISME. Canon P20 : << la matiere
    repousse l'espace-temps ; la zone d'exclusion genere la courbure >>.
    Depletion = l'exclusion ; << genere par compensation >> = genere la
    courbure. La vision EST P20, desormais mecanise sur le socle.

[2] LA DEPLETION LOCALE EST BIEN << CONSIDERABLE >> : dans le tube du
    noeud, le fond libre est totalement exclu (le noeud OCCUPE le tube ;
    P16/P20). Et le noeud est FAIT d'onde socle confisquee : masse =
    frequence confinee (m/M_Pl = omega_0/omega_socle) -- le confine EST
    le preleve. Comptabilite exacte : deficit total = masse.

[3] POURQUOI LA GRAVITE EST FAIBLE MALGRE UNE DEPLETION LOCALE TOTALE :
    le budget d'amplitude du socle est planckien. La perturbation
    metrique se decompose EXACTEMENT en deux petitesses :
        Omega^2 - 1 (surface du proton) = 2 (m/M_Pl) (l_P/r_p)
    = 2 x (fraction confisquee du socle) x (taille de Planck/taille du
    noeud). Verifie numeriquement par DEUX voies independantes.

[4] LES SIGNES SONT COHERENTS : amplitude du socle REDUITE au noeud
    <=> facteur metrique Omega^2 AUGMENTE (les horloges y ralentissent :
    la dilatation gravitationnelle du temps, observee) -- ce sont deux
    quantites differentes reliees par la bi-conformite de P5. La formule
    operationnelle Omega^2 = 1 + kappa_g|Psi|^2 est INCHANGEE (|Psi|^2 =
    la densite confisquee).

[5] BONUS STRUCTUREL : une depletion ne peut qu'ATTIRER -- pas
    d'amplitude negative a prelever -> pas de masse negative, gravite
    toujours attractive : prediction de coherence gratuite de la vision.

[6] << G = amplitude du socle nu >> : G est le taux de conversion
    deficit -> metrique, G = kappa_g c^2/(8 pi m_base), et sa MARCHE a
    t_c devient tautologique : avant les noeuds, rien ne deplete -- G
    operationnellement indefini (le document principal dit deja exactement cela).

AUCUN CALCUL NE CHANGE : meme source (deficit total = masse), meme
kappa_g, meme chaine G ; seule la LECTURE du mecanisme s'affine
(S4' : depletion + compensation, consignable en V6 sur demande).
"""
import numpy as np

G_SI = 6.6710e-11          # G predit (convention document principal)
C = 299792458.0
HBAR = 1.054571817e-34
M_P = 1.67262192e-27       # kg (nucleon)
R_P = 0.841e-15            # m
M_PL = 2.176434e-8         # kg
L_P = 1.616255e-35         # m


def run():
    print(__doc__)
    print("=" * 72)
    print("[3] LA DOUBLE PETITESSE -- verification par deux voies")
    print("=" * 72)
    # voie A : directe
    omega2_a = 2 * G_SI * M_P / (R_P * C * C)
    # voie B : decomposition fraction confisquee x taille
    frac = M_P / M_PL
    omega2_b = 2 * frac * (L_P / R_P)
    print("  voie A (2Gm/rc^2)            : Omega^2 - 1 = %.3e" % omega2_a)
    print("  voie B (2 (m/M_Pl)(l_P/r_p)) : Omega^2 - 1 = %.3e  (ecart %.1e)"
          % (omega2_b, abs(omega2_a / omega2_b - 1)))
    print("  decomposition : fraction confisquee du socle m/M_Pl = %.2e" % frac)
    print("                  rapport de tailles l_P/r_p          = %.2e"
          % (L_P / R_P))
    print("  -> depletion LOCALE ~totale dans le tube (exclusion P20), mais")
    print("     la part du budget planckien confisquee est ~1e-19 et le")
    print("     noeud est ~1e-20 fois plus grand que la denture : produit")
    print("     ~3e-39 -- la gravite est faible parce que le socle est riche.")

    print("\n" + "=" * 72)
    print("[7] LA PARTICULE S'ASSOIT SUR BEAUCOUP DE DENTS")
    print("=" * 72)
    lam_c_e = HBAR / (9.1093837015e-31 * C)
    print("  nucleon : r_p/l_P      = %.1e dents enjambees" % (R_P / L_P))
    print("  electron : lambdabar_C/l_P = %.1e dents" % (lam_c_e / L_P))
    print("  << aucune particule ne tient dans un ecartement >> -- exact :")
    print("  tenir dans UNE denture exigerait lambdabar_C <= l_P, soit")
    m_limite = HBAR / (L_P * C)
    print("  m >= hbar/(l_P c) = %.3e kg = M_Pl (%.1e) : LA DENTURE LIMITE."
          % (m_limite, M_PL))
    print("  En action : le pas est h -- rien de physique ne porte moins")
    print("  d'un quantum par periode (E = hf identitaire) ; et le rapport")
    print("  l_P/r qui dilue le deficit sur les ~1e19 dents est PRECISEMENT")
    print("  le second facteur de la double petitesse [3].")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  La vision depletion+compensation NE CHANGE AUCUN CALCUL :")
    print("  source inchangee (deficit total = masse -> monopole), kappa_g")
    print("  inchange, chaine G inchangee, signes coherents (socle deplete")
    print("  <=> Omega^2 augmente <=> horloges ralenties -- bi-conformite")
    print("  P5). Elle APPORTE : le mecanisme de P20 (exclusion -> courbure")
    print("  par compensation), la gravite attractive-seulement (pas")
    print("  d'amplitude negative a prelever), la marche de G a t_c rendue")
    print("  tautologique, et la double petitesse expliquant la faiblesse")
    print("  de la gravite. Lecture S4' consignable en V6 sur demande.")


if __name__ == "__main__":
    run()
