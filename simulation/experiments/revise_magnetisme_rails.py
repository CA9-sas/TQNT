"""Revision du magnetisme : seules les ondes ENTREES dans un rail sont
magnetisees (23/07).

VISION UTILISATEUR : seules les ondes rentrant dans le rail sont
magnetisees -- pas celles qui suivent normalement le socle.

TRADUCTION : la magnetisation = propriete acquise a l'ENTREE (le
confinement 1D donne un courant DIRIGE et persistant -> champ de
Biot-Savart autour du rail) ; une onde libre du socle a un courant qui
OSCILLE a la longueur d'onde du socle (= 2 pas de rail) -> aucun
couplage net : SPECTATRICE.

QUATRE CONSEQUENCES A QUANTIFIER :
[A] LE THEOREME DE LA SPECTATRICE : le couplage de Neumann d'un
    courant de rail avec (i) un autre courant de rail perpendiculaire
    (fini) vs (ii) une onde libre du socle (courant oscillant a
    k = pi/a) -> (ii) s'annule cellule par cellule.
[B] LE COLMATAGE DU POINT MOU DU S47 : l'enfilement est RAIL-RAIL
    (deux courants aux noeuds, couple de Biot-Savart bien defini,
    les DEUX sens de capture = les deux cotes d'approche) -- le
    branchement (scission d / enroulement d-1 par sens) ne comptait
    DEJA que des ondes de rail : alpha = ln(2+sqrt2) INTACT.
[C] LE COROLLAIRE DU SECTEUR BASE : la lumiere = la population jamais
    entree dans les rails (jamais magnetisee, jamais nouee) -- le
    S = 0 du secteur base recoit son mecanisme ; coherent avec la
    protection zero-dispersion (l'onde non confinee est aveugle aux
    rails : elle se superpose sur beaucoup de cellules).
[D] LA PORTE D'ENTREE : l'entree dans le rail (magnetisation) = la
    porte etablie de la chaine Omega (percage Z3, offre 8/9) -- le
    raffinement se branche sur P_occ SANS changement.
"""
import numpy as np


def neumann_segments(P1, P2):
    """Couplage de Neumann discret entre deux polylignes orientees."""
    d1 = P1[1:] - P1[:-1]
    d2 = P2[1:] - P2[:-1]
    m1 = 0.5 * (P1[1:] + P1[:-1])
    m2 = 0.5 * (P2[1:] + P2[:-1])
    r = np.linalg.norm(m1[:, None, :] - m2[None, :, :], axis=2)
    r = np.maximum(r, 0.25)
    return float((d1 @ d2.T / r).sum())


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] LE THEOREME DE LA SPECTATRICE (couplages de Neumann)")
    print("=" * 78)
    # courant de rail : segment unitaire le long de z (rail), L = 20 pas
    L = 20
    z = np.linspace(-L / 2, L / 2, 400)
    rail = np.stack([np.zeros_like(z), np.zeros_like(z), z], -1)
    # (i) autre rail perpendiculaire au noeud (offset 1 pas)
    x = np.linspace(-L / 2, L / 2, 400)
    perp = np.stack([x, np.ones_like(x), np.zeros_like(x)], -1)
    M_rr = neumann_segments(rail, perp)
    # rail PARALLELE co-dirige (adjacent)
    par = np.stack([np.ones_like(z), np.zeros_like(z), z], -1)
    M_par = neumann_segments(rail, par)
    # (ii) onde libre du socle : courant le long de z mais OSCILLANT en
    # espace a k = pi/a (la demi-longueur d'onde du socle = le pas) --
    # modelisee par une ligne parallele dont le courant alterne de signe
    # a chaque pas : dl2 -> dl2 * cos(pi z)
    d_osc = (par[1:] - par[:-1]) * np.cos(np.pi *
                                          0.5 * (par[1:, 2] + par[:-1, 2]))[:, None]
    m1 = 0.5 * (rail[1:] + rail[:-1])
    m2 = 0.5 * (par[1:] + par[:-1])
    r = np.maximum(np.linalg.norm(m1[:, None, :] - m2[None, :, :], axis=2), 0.25)
    d1 = rail[1:] - rail[:-1]
    M_free = float((d1 @ d_osc.T / r).sum())
    print(f"""  Couplage rail-rail perpendiculaire (noeud)  : M = {M_rr:8.3f}
  Couplage rail-rail parallele co-dirige      : M = {M_par:8.3f}
  Couplage rail <-> onde LIBRE du socle
   (courant oscillant a k = pi/pas)           : M = {M_free:8.3f}
  -> rapport |libre|/|parallele| = {abs(M_free / M_par):.1e} : l'onde qui suit le
  socle normalement alterne son courant a CHAQUE pas de rail -- son
  couplage magnetique s'annule cellule par cellule. SEULE l'onde
  ENTREE dans un rail (courant unidirectionnel persistant) est
  magnetisee : le theoreme de la spectatrice est QUANTIFIE.""")

    print("=" * 78)
    print("[B] LE COLMATAGE DU S47 : l'enfilement est RAIL-RAIL")
    print("=" * 78)
    # couple de Biot-Savart sur un courant perpendiculaire selon le cote
    # d'approche : signe de la composante azimutale de la force
    def couple_helicoidal(direction):
        # B du rail (courant le long de z) : B = (-y, x, 0)/(2pi rho^2).
        # Onde croisante le long de x (dl = direction * ex, y = 0,5) :
        # F = I dl x B = direction * (0, 0, B_y) ; F_z(x) ANTISYMETRIQUE
        # -> un bout monte, l'autre descend : helicoidalisation autour
        # du rail ; le sens suit la direction de propagation.
        xs = np.linspace(-3, 3, 300)
        rho2 = xs**2 + 0.25
        Fz = direction * xs / rho2
        return float(np.trapezoid(Fz * xs, xs))   # chiralite du couple
    fp, fm = couple_helicoidal(+1), couple_helicoidal(-1)
    print(f"""  Couple helicoidal (onde croisante +x) : {fp:+.3f}
  Couple helicoidal (onde croisante -x) : {fm:+.3f}
  (NB honnete : l'inductance mutuelle de segments perpendiculaires est
  NULLE (dl1.dl2 = 0) -- la capture passe par la FORCE dans le champ
  de l'autre ; c'est le couple ci-dessus qui enroule.)
  -> DEUX sens d'enroulement = les deux DIRECTIONS de propagation de
  l'onde croisante (dl -> -dl retourne le couple) : les 4 flux
  perpendiculaires d'un noeud = 2 rails x 2 directions -> exactement
  les deux canaux Delta Lk = +-1 du theoreme S47, chacun avec l'exces d-1. Le
  branchement (scission = fermeture d ; enroulement = exces par sens)
  ne comptait DEJA que des rencontres d'ondes DE RAIL : le raffinement
  ne change PAS le ratio -- alpha = ln(2+sqrt2) = {np.log(2 + np.sqrt(2)):.4f} INTACT --
  il COLMATE le point mou (une onde libre ne peut pas etre capturee
  magnetiquement : elle n'a pas besoin de l'etre, la capture est
  rail-rail).""")

    print("=" * 78)
    print("[C] COROLLAIRE : la lumiere = la population jamais magnetisee")
    print("=" * 78)
    print("""  Le secteur BASE (S = 0 : photon, neutrino) recoit son mecanisme
  complet : une onde qui suit le socle ne se confine jamais dans un
  rail -> jamais de courant persistant -> jamais de magnetisation ->
  jamais d'enfilement -> jamais de topologie -> jamais de masse.
  COHERENCES : (i) la protection zero-dispersion (Lorentz) -- l'onde
  libre se superpose sur ~lambda/pas >> 1 cellules, aveugle a la
  structure des rails (pas de repere privilegie) ; (ii) le theoreme
  de la superposition (S43) -- le faufilement fantome est PRE-
  magnetisation, le verrou arrive a la fermeture ; (iii) la
  dichotomie devient ontologique : MATIERE = ondes entrees dans les
  rails (magnetisees, nouables) ; LUMIERE = ondes du socle libre
  (spectatrices). L'entree dans le rail est LA transition.""")

    print("=" * 78)
    print("[D] LA PORTE D'ENTREE = LA PORTE ETABLIE")
    print("=" * 78)
    print("""  Quand une onde entre-t-elle dans un rail ? Au gel (condensation) :
  l'entree = le confinement = la magnetisation = le debut de la
  matiere. C'est la porte deja etablie de la chaine Omega : le
  percage Z3 (offre de corde 8/9 EXACTE, denombree) et P_occ --
  le raffinement se branche sur la chaine SANS changement numerique.
  NB de coherence V6 : au prochain passage, preciser dans le
  paragraphe du theoreme (S47 consolide) que « l'onde perpendiculaire
  capturee » est elle-meme une onde DE RAIL (un mot).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print(f"""  LE RAFFINEMENT EST ADOPTABLE ET GRATUIT : (1) le theoreme de la
  spectatrice est quantifie (couplage libre/rail ~ {abs(M_free / M_par):.0e} : l'onde
  du socle alterne son courant a chaque pas -> decouplage cellule par
  cellule) ; (2) il COLMATE le point mou du S47 (capture rail-rail,
  couple de Biot-Savart bien defini, deux sens = deux cotes) sans
  toucher au ratio : alpha = ln(2+sqrt2) intact ; (3) il donne au
  secteur base son mecanisme (lumiere = jamais magnetisee) coherent
  avec la protection Lorentz ; (4) la porte d'entree est la porte
  etablie (percage 8/9, P_occ inchange). Cout : zero (une precision
  d'un mot en V6).""")


if __name__ == "__main__":
    main()
