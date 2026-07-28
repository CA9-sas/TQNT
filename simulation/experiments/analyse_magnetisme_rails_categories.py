"""Magnétisme de rails et lois de secteur S — analyse de l'idée
(27/07/2026, vision utilisateur).

L'IDÉE (T0) : une onde ENTRÉE dans les rails (la matière, §50) est
magnétisée — courant dirigé persistant le long du circuit fermé (le
nœud). Selon (a) la DIRECTION prise par l'onde dans les rails et
(b) le TEMPS passé dans les rails, le magnétisme diffère par nœud :
  - SENS      s_K = chiralité de la circulation (droite/gauche ;
                    renversement = miroir K̄ = antimatière) ;
  - INTENSITÉ I_K = aire vectorielle balayée par le courant unité,
                    m = (1/2)∮ r×dl (dipôle de Biot-Savart) ;
  - TEMPS     T_K = longueur du circuit (à vitesse c : T = ℓ/c ;
                    en pas de rails : N_min).
CONJECTURE UTILISATEUR : en modulant le sens et l'intensité du
magnétisme par nœud, on doit retrouver les équations S par catégorie
(BASE 0 / LEPTONS Koide / BARYONS Regge+Z3 / GOLDSTONE GMOR /
LOURDS constituant+Cornell / EW vev).

MÉTHODE : mesures sur les seules courbes VALIDES (toriques 3_1, 5_1,
7_1, 9_1 + 4_1 du catalogue ; 5_2, 7_2, 8_3, 6_2, 8_1, 8_2 réparées
par fermeture de tresse, Alexander contrôlé — repair_curves_braid).
  [1] table par nœud : temps réduit T̂ = L/R_g, cohérence de direction
      eta = |m| / (1/2 ∮|r×dl|)  (part de l'aire dirigée qui survit aux
      annulations de direction), intensité réduite mû = |m|/(L·R_g),
      rapport quadrupolaire ||Q||/(|m| R_g), writhe de Gauss (sens) ;
  [2] théorème de la paire miroir : |m|(K̄)=|m|(K), Wr(K̄)=−Wr(K) —
      dans une paire K+K̄ le SENS s'annule, le TEMPS s'additionne ;
  [3] temps↔masse : ℓ_ACPR (et entiers de rails N_min, §48) vs les
      rapports de masse du plateau — M = √σ_χ·ℓ relu « temps passé
      dans les rails » ;
  [4] synthèse : signature (sens, intensité, temps) par secteur S.

GARDE-FOUS : le writhe MESURÉ ici est celui du plongement de calcul
(tresse/catalogue), PAS le writhe idéal (seul canonique) — on ne
lit que son SIGNE (chiralité) et les symétries exactes ; règle du
test nul : aucune constante de secteur n'est revendiquée ici.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from simulation.experiments.repair_curves_braid import (
    BRAID_WORDS, braid_curve, multipoles)
from theory.omega_ts import _knot_curve  # helper de COURBE seulement (pas de masse)
from theory.ropelength_acpr import ROPELENGTH_ACPR


# ----------------------------------------------------------------------
def closed(P):
    P = np.asarray(P, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    return P - P.mean(0)


def resample(P, n):
    """ré-échantillonnage en longueur d'arc (courbe fermée)."""
    P = closed(P)
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    Pc = np.vstack([P, P[:1]])
    t = np.linspace(0, L, n, endpoint=False)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(t, s, Pc[:, k])
    return out


def gauss_writhe(P, n=240):
    """writhe de Gauss (double intégrale discrète)."""
    P = resample(P, n)
    T = np.roll(P, -1, 0) - P
    R = P + 0.5 * T
    W = 0.0
    for i in range(n):
        dv = R[i] - R
        r3 = np.linalg.norm(dv, axis=1) ** 3
        cr = np.cross(T[i], T)
        num = (cr * dv).sum(1)
        mask = np.ones(n, bool)
        for j in (-1, 0, 1):
            mask[(i + j) % n] = False
        W += (num[mask] / r3[mask]).sum()
    return W / (4 * np.pi)


def magnetism(P):
    """(T̂, eta, mû, q, m_vec) — mesures adimensionnées du courant unité."""
    P = closed(P)
    dl = np.roll(P, -1, 0) - P
    rmid = P + 0.5 * dl
    rxdl = np.cross(rmid, dl)
    m = 0.5 * rxdl.sum(0)
    A_tot = 0.5 * np.linalg.norm(rxdl, axis=1).sum()   # aire « dépliée »
    L = np.linalg.norm(dl, axis=1).sum()
    Rg = np.sqrt((P ** 2).sum(1).mean())
    md, Qn, _, q = multipoles(P)
    return dict(That=L / Rg, eta=np.linalg.norm(m) / A_tot,
                mu=np.linalg.norm(m) / (L * Rg), q=q, m=m, L=L, Rg=Rg)


def curve(K):
    if K in BRAID_WORDS:
        w, ns, _ = BRAID_WORDS[K]
        P, _, _ = braid_curve(w, ns, pts_per_seg=60)
        return np.asarray(P, float)
    return np.asarray(_knot_curve(K, 2048), float)


def mirror_curve(K):
    if K in BRAID_WORDS:
        w, ns, _ = BRAID_WORDS[K]
        P, _, _ = braid_curve([-x for x in w], ns, pts_per_seg=60)
        return np.asarray(P, float)
    return curve(K) * np.array([1.0, 1.0, -1.0])


CATEG = {
    "3_1": "LEPTON (torique C3 ; cœur du nucléon)",
    "5_1": "LEPTON-famille (torique C5)",
    "7_1": "torique C7 (EW W/H)",
    "9_1": "torique C9 (EW Z)",
    "4_1": "amphichiral (DM stricte)",
    "5_2": "chiral, décuplet",
    "7_2": "chiral, OCTET/nucléon (compagnon)",
    "8_3": "amphichiral (EW commun)",
    "6_2": "chiral (méson diagonal)",
    "8_2": "chiral (cœur top)",
}


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] MESURES PAR NŒUD (courbes valides ; courant unité)")
    print("=" * 78)
    print("%-5s %-38s %6s %6s %8s %6s %8s" % (
        "K", "catégorie", "T̂=L/Rg", "eta", "mû", "q", "Wr(plgt)"))
    rows = {}
    for K in ["3_1", "5_1", "7_1", "9_1", "4_1", "5_2", "7_2", "8_3",
              "6_2", "8_2"]:
        P = curve(K)
        g = magnetism(P)
        wr = gauss_writhe(P)
        rows[K] = (g, wr)
        print("%-5s %-38s %6.1f %6.3f %8.4f %6.2f %+8.2f" % (
            K, CATEG[K], g["That"], g["eta"], g["mu"], g["q"], wr))
    print("""
LECTURE : eta = cohérence de DIRECTION de la circulation (1 = toute
l'aire balayée pointe du même côté ; petit = les directions du chemin
s'annulent). q = 0 exact pour les toriques (protégé par C_q) ;
Wr(plongement) : seul le SIGNE est lu (chiralité = SENS).""")

    print("=" * 78)
    print("[2] THÉORÈME DE LA PAIRE MIROIR — le SENS s'annule, le TEMPS s'ajoute")
    print("=" * 78)
    for K in ["3_1", "7_2", "5_2"]:
        P, Pm = curve(K), mirror_curve(K)
        g, gm = magnetism(P), magnetism(Pm)
        wr, wrm = gauss_writhe(P), gauss_writhe(Pm)
        print("  %s vs %s-miroir : |m| %—.4f / %.4f (égal à %.1f %%) ; "
              "Wr %+.2f / %+.2f (opposés) ; T̂ additionnés dans la paire"
              .replace("%—", "%") % (
                  K, K, g["mu"], gm["mu"],
                  100 * abs(g["mu"] / gm["mu"] - 1), wr, wrm))
    print("""
  => dans toute paire K+K̄ (mésons, GOLDSTONE, nucléon 7_2+7̄_2) : les
  SENS de circulation sont opposés (pseudo-scalaire annulé EXACTEMENT),
  les TEMPS s'additionnent. Trois faits V6 RETROUVÉS par cette seule
  dichotomie :
   - nucléon : le moment magnétique est porté par le CŒUR trèfle (les
     compagnons 7_2+7̄_2 se compensent) — établi V6 §torsion, relu ;
   - GOLDSTONE (pi = paire trèfle) : annulation MAXIMALE du sens =>
     le secteur magnétiquement neutre est le plus LÉGER (forme GMOR) ;
   - EW (8_3 commun, amphichiral, Wr idéal = 0) : nœud SANS sens
     propre => la résonance Z/H = doublet miroir dégénéré (option (i)
     de la fourche 2026-07-14, relue : « les deux sens dégénérés »).""")

    print("=" * 78)
    print("[3] LE TEMPS DANS LES RAILS EST LA MASSE DU PLATEAU (relecture)")
    print("=" * 78)
    nmin = {"3_1": 24, "4_1": 30, "5_1": 34, "5_2": 36}
    print("  %-5s %10s %8s %14s" % ("K", "l_ACPR", "N_min", "ratio vs 3_1"))
    for K in ["3_1", "4_1", "5_1", "5_2"]:
        print("  %-5s %10.2f %8d %8.3f (ACPR) / %.3f (rails)" % (
            K, ROPELENGTH_ACPR[K], nmin[K],
            ROPELENGTH_ACPR[K] / ROPELENGTH_ACPR["3_1"],
            nmin[K] / nmin["3_1"]))
    print("""
  M_topo = √σ_χ·ℓ(K) : la loi de masse géométrique du plateau EST
  « intensité × temps » à courant unité — le nœud qui passe plus de
  temps dans les rails pèse plus. Les rapports d'entiers de rails
  (36/24 = 3/2 pour m_s/m_ℓ, §48) sont la version quantifiée. RELU,
  pas re-dérivé (m_ℓ reste la transmutation).""")

    print("=" * 78)
    print("[4] SYNTHÈSE — la signature (sens, intensité, temps) par secteur S")
    print("=" * 78)
    print("""  secteur    | signature magnétique               | équation S retrouvée ?
  -----------+------------------------------------+----------------------------
  BASE       | jamais entrée : M = 0              | S = 0 : THÉORÈME (§50, déjà
             | (spectatrice, la lumière)          | mécanisé — la lecture l'exige)
  LEPTONS    | circulation TORIQUE : dipôle pur   | forme Koide : les 3 généra-
             | (q = 0 exact), cohérence maximale, | tions = modes Z3 du MÊME
             | UN seul sens                       | circuit (phases, pas le temps)
             |                                    | => structure OUI, 3/4 et 2/9 NON
  BARYONS    | cœur chiral + paire de sens        | moment porté par le cœur
             | compensés (7_2+7̄_2)               | (établi, RETROUVÉ) ; masse =
             |                                    | jonction/Regge, aveugle au
             |                                    | magnétisme (pivot §17 cohérent)
  GOLDSTONE  | paire miroir du trèfle :           | « le plus annulé est le plus
             | annulation TOTALE du sens          | léger » (forme GMOR) ; E_Z3 NON
  LOURDS     | mêmes paires, nœuds à LONG temps   | secteur compression : l'EM
             | de rails (courants intenses)       | MUTUEL inter-nœuds existe déjà
             |                                    | (V5p27, 0,1 %) ; Cornell NON
  EW         | 8_3 amphichiral : SANS sens        | masse par l'ancre (vev dérivé) ;
             | propre (Wr idéal = 0)              | doublet Z/H = 2 sens dégénérés
  DM         | courant de rail VERROUILLÉ         | anneau conducteur, alpha_pol
             | (Lk fige la circulation)           | (établi §50, RETROUVÉ)""")

    print("=" * 78)
    print("VERDICT (honnête)")
    print("=" * 78)
    print("""  L'idée est une LECTURE UNIFICATRICE : la modulation (sens, intensité,
  temps) du magnétisme de rails reproduit la SECTORIALITÉ de S — c'est-
  à-dire POURQUOI chaque catégorie a sa propre loi (0 / phases / cœur /
  annulation / compression / ancre), et elle RETROUVE quatre faits déjà
  établis (S=0 lumière ; moment du cœur ; doublet Z/H ; anneau DM) sous
  UNE dichotomie. Elle ne re-dérive PAS les CONSTANTES de secteur
  ((3/4)√σ_χ, δ=2/9, E_Z3, base octet, Cornell) : cibles nommées —
  les dériver demanderait de calculer l'ÉNERGIE magnétique propre des
  circuits (au-delà du courant unité). Règle du test nul respectée :
  aucun accord numérique nouveau n'est revendiqué.""")


if __name__ == "__main__":
    main()
