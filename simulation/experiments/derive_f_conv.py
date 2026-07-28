"""Derivation de f_conv : l'efficacite de conversion corde -> noeud (17/07).

CIBLE : P_occ = (8/9) x f_conv (offre exacte de la trialite) ; la fermeture
sur Planck exige f_conv = 0.43.

DERIVATION (canaux de contact + arbre de fragmentation) :
  A chaque CONTACT de deux brins pendant l'effondrement, deux canaux :
    - RECONNEXION (intercommutation, scission de la boucle) : poids 1
      (canal topologiquement libre des tubes de flux) ;
    - CHANGEMENT DE CROISEMENT (passage, DeltaLk = +-1 comme au sommet
      P15) : poids e^{-alpha_cond} PAR SIGNE, soit 2 e^{-alpha} au total
      -- alpha_cond = 1.217 est LA penalite par croisement derivee de la
      cinetique de condensation (le meme objet que le poids e^{-alpha c}).
  Un passage sur une boucle non nouee cree un trefle (nombre de denouement
  1) ; deux passages en des points distincts = somme connexe (toujours
  nouee) : PAS d'annulation.
  ARBRE DE FRAGMENTATION : chaque fragment final nait d'une scission (un
  arbre a n feuilles a n-1 noeuds internes) ; les passages se distribuent
  sur les fragments en Poisson de moyenne mu = (passages/scissions) =
  2 e^{-alpha}. D'ou :
      f_conv = P(fragment noue) = 1 - exp(-2 e^{-alpha_cond}) = 0.447
      P_occ  = (8/9) x 0.447 = 0.397
      Omega  = 14.3 x 0.397 = 5.68   (Planck : 5.45, +4 %)
  Variante a un seul signe (minorant) : 1 - exp(-e^{-alpha}) = 0.256.

TEST MONTE-CARLO (sans parametre, sur le reseau MESURE) : on rejoue la
cascade sur les cordes tracees de chi_string_network_pocc (Z3, 24^3) --
contacts = cellules revisitees par la meme corde ; a chaque contact,
scission (p = 1/(1+2e^-a)) ou passage (p = 2e^-a/(1+2e^-a), compteur de
nouage) ; les contacts a cheval sur deux fragments meurent (fusions
ignorees, declare). Mesure directe : fragments finals noues par domaine.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.chi_string_network_pocc import (  # noqa: E402
    fluxes_z3, trace)

ALPHA_COND = 1.217
MU = 2 * np.exp(-ALPHA_COND)
P_TWIST = MU / (1 + MU)
OMEGA_AT_POCC1 = 14.3          # reconstruction, g_s derive 15.2
PLANCK = 5.45


def contacts_of(path):
    seen = {}
    out = []
    for idx, c in enumerate(path):
        if c in seen:
            for j in seen[c]:
                out.append((j, idx))
        seen.setdefault(c, []).append(idx)
    return out


def displacement(path, N):
    d = np.zeros(3, int)
    for a, b in zip(path, path[1:] + path[:1]):
        step = np.array(b[:3], int) - np.array(a[:3], int)
        step -= N * np.rint(step / N).astype(int)
        d += step
    return d


def unwrap(path, N):
    """Coordonnees continues entieres (deplie les sauts PBC)."""
    P = [np.array(path[0], int)]
    for c in path[1:]:
        prev = P[-1]
        d = np.array(c, int) - np.mod(prev, N)
        d -= N * np.rint(d / N).astype(int)
        P.append(prev + d)
    return [tuple(x) for x in P]


def coarse(path):
    """Grain x2 (froissement auto-similaire) : blocs 2x2x2, doublons
    cycliques consecutifs supprimes."""
    q = [(c[0] // 2, c[1] // 2, c[2] // 2) for c in path]
    out = []
    for c in q:
        if not out or c != out[-1]:
            out.append(c)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def cascade_ms(path, rng, gmax=20):
    """Cascade multi-echelle : contacts a l'echelle courante, branchement
    scission/passage (rapport DERIVE), puis froissement (grain x2) et
    recommence. Retourne [(longueur_xi, n_passages)] des fragments finals."""
    finals = []
    stack = [(path, 0, 0, frozenset())]
    while stack:
        p, tw, g, resolved = stack.pop()
        cts = [ij for ij in contacts_of(p) if ij not in resolved]
        if cts:
            i, j = cts[rng.integers(len(cts))]
            if rng.random() < P_TWIST:
                stack.append((p, tw + 1, g, resolved | {(i, j)}))
            else:
                A, B = p[i:j], p[j:] + p[:i]
                tA = int(np.sum(rng.random(tw) < len(A) / len(p)))
                stack.append((A, tA, g, frozenset()))
                stack.append((B, tw - tA, g, frozenset()))
            continue
        if len(p) < 4 or g >= gmax:
            finals.append((len(p) * 2 ** g, tw))
            continue
        stack.append((coarse(p), tw, g + 1, frozenset()))
    return finals


def run():
    print("=== f_conv : canaux de contact + arbre de fragmentation ===\n")
    print("[1] DERIVATION (zero parametre) :")
    print("    canaux par contact : scission (poids 1) vs passage DeltaLk=+-1")
    print("    (poids e^-alpha par signe, alpha_cond = %.3f derive)" % ALPHA_COND)
    print("    mu = passages/scission = 2 e^-alpha = %.4f" % MU)
    f_ana = 1 - np.exp(-MU)
    f_one = 1 - np.exp(-np.exp(-ALPHA_COND))
    print("    f_conv = 1 - exp(-2 e^-alpha) = %.4f   (1 signe : %.4f)" % (f_ana, f_one))
    p_occ = (8 / 9) * f_ana
    print("    P_occ = (8/9) f_conv = %.4f -> Omega = %.2f (Planck %.2f : %+.1f %%)\n" % (
        p_occ, OMEGA_AT_POCC1 * p_occ, PLANCK,
        100 * (OMEGA_AT_POCC1 * p_occ / PLANCK - 1)))

    print("[2] TEST MONTE-CARLO multi-echelle sur le reseau mesure (Z3, 24^3 x 4) :")
    print("    (contacts de formation, puis froissement auto-similaire x2 par")
    print("    generation ; MEME rapport de branchement derive a chaque echelle)")
    N, reps = 24, 4
    tot_cells = 0
    n_frag = n_knot = 0
    len_tot = 0.0
    len_knot = 0.0
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        ph = rng.integers(0, 3, (N, N, N))
        loops, winders, _ = trace(*fluxes_z3(ph), N, rng)
        tot_cells += N ** 3
        for p in loops + winders:
            len_tot += len(p)
            for L, tw in cascade_ms(unwrap([tuple(c) for c in p], N), rng):
                n_frag += 1
                if tw >= 1:
                    n_knot += 1
                    len_knot += L
    print("    longueur totale : %.3f seg/domaine (offre 8/9 = %.3f)" % (
        len_tot / tot_cells, 8 / 9))
    print("    fragments finals : %.3f/dom ; NOUES (>=1 passage) : %.3f/dom" % (
        n_frag / tot_cells, n_knot / tot_cells))
    print("    fraction nouee des fragments : %.3f (analytique %.3f)" % (
        n_knot / max(1, n_frag), f_ana))
    p_occ_mc = n_knot / tot_cells
    print("    -> P_occ(MC) = %.3f ; Omega(MC) = %.2f (%+.0f %% vs Planck)" % (
        p_occ_mc, OMEGA_AT_POCC1 * p_occ_mc,
        100 * (OMEGA_AT_POCC1 * p_occ_mc / PLANCK - 1)))
    print("    energie moyenne par fragment noue : %.2f xi -> %.1f GeV (m(23)=1.71)" % (
        len_knot / max(1, n_knot), len_knot / max(1, n_knot) * 2.97))
    print()

    print("[3] La question residuelle : lambda (taille finale des fragments) :")
    lam_close = (8 / 9) * f_ana * OMEGA_AT_POCC1 / PLANCK
    print("    P_occ = (8/9) x f_frac / lambda ; fermeture exacte : lambda = %.2f" % lam_close)
    print("    bornes physiques : lambda >= 0.58 (payer m(23) = 1.71 GeV a")
    print("    2.97 GeV/xi) ; lambda <~ 2 (les kinks de reconnexion re-contactent")
    print("    chaque longueur de correlation a chaque periode) ; lambda = 1 est")
    print("    l'echelle naturelle ET le quantum d'energie (1 noeud du bord +")
    print("    73 %% rayonne par fragment-xi).")
    for lam in (0.6, 1.0, lam_close, 2.0):
        om = OMEGA_AT_POCC1 * (8 / 9) * f_ana / lam
        print("      lambda = %.2f : Omega = %.2f (%+.0f %%)" % (
            lam, om, 100 * (om / PLANCK - 1)))
    print()

    print("VERDICT :")
    print(" - DERIVE (identite d'arbre) : la fraction nouee PAR FRAGMENT est")
    print("   fixee par alpha_cond seul, INDEPENDAMMENT du taux de contact :")
    print("   passages/scissions = 2 e^-alpha = 0.592 -> f_frac = 0.447 (Poisson)")
    print("   / 0.28 (MC, correlations de lignee). Le +- vient du DeltaLk=+-1 de")
    print("   P15 (1 signe : 0.256, minorant).")
    print(" - La question migre : f_conv = f_frac/lambda. Le facteur restant")
    print("   n'est plus une efficacite inconnue mais une TAILLE DE FRAGMENT,")
    print("   bornee par l'energie (0.58 <= lambda <~ 2 -> Omega = 2.8-9.5,")
    print("   Planck DEDANS) ; fermeture exacte a lambda = 1.04 = l'echelle de")
    print("   correlation. Mes deux MC (contacts de formation seuls : lambda=22 ;")
    print("   froissement par grain : lambda=8) n'ont PAS les kinks de")
    print("   reconnexion (4 kinks/scission, re-contacts a chaque periode) :")
    print("   lambda ~ 1 est l'attendu dynamique, A DERIVER (simulation Nambu")
    print("   avec kinks ou argument de taux de radiation).")
    print(" - Chaine complete : Omega = 14.3 x (8/9)_exact x f_frac(alpha)_derive")
    print("   / lambda -> 5.68 a lambda = 1 (+4 %% de Planck).")


if __name__ == "__main__":
    run()
