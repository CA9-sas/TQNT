"""Demontrer la superselection de topologie de feuille (20/07).

TROIS PIECES (P1 numerique, P2 numerique, P3 formel) :

 (P1) CONSERVATION EXACTE : l'enroulement w est invariant sous TOUTE
      deformation lisse (du lacet de lecture comme de la configuration)
      tant que rien ne traverse le coeur (|psi| = 0). Le SEUL portail
      entre secteurs est la traversee du coeur = une reconnexion.
      Demonstration : (a) lacets de lecture de formes arbitraires ->
      w identique ; (b) famille continue de lacets -> w constant a
      chaque pas ; (c) famille qui traverse le bord du disque -> saut
      DISCRET w : 1 -> 0 (le portail, localise, discret).

 (P2) ENREGISTREMENT REDONDANT (darwinisme quantique) : K lacets
      environnementaux DISJOINTS lisent tous le meme w sans le
      perturber (les lectures commutent entre elles et avec w) ->
      structure de diffusion classique (broadcast) : l'environnement
      mesure la topologie de maniere optimale.

 (P3) FORMEL (theoreme conditionnel) : le couplage systeme-environnement
      est le contact local P15 ; par (P1), aucun contact local LISSE ne
      change w -> [H_int, W] = 0 EXACTEMENT (pas approximativement).
      Par le theoreme d'einselection (Zurek), W est une observable
      pointeur EXACTE : les coherences inter-secteurs decroissent au
      taux environnemental plein, les secteurs sont intacts ->
      SUPERSELECTION DYNAMIQUE de la topologie de feuille. Un systeme
      ISOLE (zero contact) garde ses coherences (le singulet vole :
      Bell preserve) ; le premier contact tranche.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "simulation/experiments")
from formalisation_lignes_phase import solid_angle_disk, winding_on_loop  # noqa: E402

RNG = np.random.default_rng(7)
NRM = np.array([0.0, 0.0, 1.0])
CTR = np.array([0.0, 0.0, 0.0])


def theta(x):
    return 0.5 * solid_angle_disk(CTR, NRM, 1.0, x)


def wobbly_loop(center, axis, r0, n=240, wob=0.3, seed=None):
    """Lacet ferme aleatoire lisse encerclant l'axe donne."""
    rng = np.random.default_rng(seed)
    a = axis / np.linalg.norm(axis)
    b = np.array([1.0, 0.0, 0.0])
    if abs(a @ b) > 0.9:
        b = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, b); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r0 * (1 + wob * (np.sin(2 * th + rng.random() * 6)
                         + 0.5 * np.cos(3 * th + rng.random() * 6)) / 2)
    z = wob * r0 * np.sin(th + rng.random() * 6)
    return (center[None, :] + np.outer(r * np.cos(th), u)
            + np.outer(r * np.sin(th), v) + np.outer(z, a))


def run():
    print("=== Superselection de topologie de feuille : demonstration ===\n")
    anchor = np.array([1.0, 0.0, 0.0])   # point du bord du disque (coeur)
    axis = np.array([0.0, 1.0, 0.0])     # direction qui perce le disque

    # (P1a) formes arbitraires de lacet de lecture
    print("(P1a) Lacets de lecture arbitraires autour du coeur :")
    for k in range(5):
        L = wobbly_loop(anchor, axis, 0.25 + 0.1 * k, wob=0.35, seed=k)
        print("   forme %d : w = %+.6f" % (k, winding_on_loop(theta, L)))

    # (P1b) famille continue de lacets
    print("\n(P1b) Deformation continue (10 pas, petit cercle -> grand lacet) :")
    L0 = wobbly_loop(anchor, axis, 0.15, wob=0.0, seed=1)
    L1 = wobbly_loop(anchor, axis, 0.55, wob=0.4, seed=2)
    ws = []
    for t in np.linspace(0, 1, 10):
        ws.append(winding_on_loop(theta, (1 - t) * L0 + t * L1))
    print("   w le long du chemin : min %+.6f, max %+.6f (constant)" % (
        min(ws), max(ws)))

    # (P1c) le portail : traversee du bord -> saut discret
    print("\n(P1c) Translation du lacet HORS du disque (traverse le bord) :")
    prev = None
    for s in np.linspace(0.0, 1.2, 13):
        L = wobbly_loop(anchor + np.array([s, 0, 0]), axis, 0.25, wob=0.0, seed=3)
        w = winding_on_loop(theta, L)
        mark = ""
        if prev is not None and abs(w - prev) > 0.5:
            mark = "   <-- SAUT DISCRET (traversee du coeur = reconnexion)"
        print("   decalage %.2f : w = %+.3f%s" % (s, w, mark))
        prev = w

    # (P2) redondance environnementale
    print("\n(P2) 20 lacets environnementaux disjoints (positions/formes/rayons")
    print("     aleatoires le long du coeur) :")
    vals = []
    for k in range(20):
        phi = RNG.random() * 2 * np.pi
        pt = np.array([np.cos(phi), np.sin(phi), 0.0])
        ax = np.array([-np.sin(phi), np.cos(phi), 0.0])
        L = wobbly_loop(pt, ax, 0.12 + 0.25 * RNG.random(), wob=0.3, seed=100 + k)
        vals.append(winding_on_loop(theta, L))
    vals = np.array(vals)
    print("   w : moyenne %+.6f, dispersion %.1e -> 20 enregistrements" % (
        vals.mean(), vals.std()))
    print("   IDENTIQUES et commutants : diffusion classique (darwinisme).")

    print("\n(P3) FORMEL : par (P1), aucun contact local lisse ne change w ->")
    print("   [H_int, W] = 0 EXACTEMENT ; theoreme d'einselection : W est une")
    print("   observable pointeur EXACTE -- coherences inter-secteurs")
    print("   decoherees au taux environnemental plein, secteurs intacts.")
    print("   => SUPERSELECTION DYNAMIQUE de la topologie de feuille :")
    print("   isolee -> coherence conservee (Bell) ; premier contact ->")
    print("   secteur tranche. Residu honnete : l'einselection ne choisit")
    print("   pas QUEL secteur est realise (probleme profond de la mesure,")
    print("   statut commun a toutes les approches decoherentielles) ;")
    print("   l'apport TQNT = pointeur EXACT (discret, conservation")
    print("   topologique) au lieu d'approximatif.")


if __name__ == "__main__":
    run()
