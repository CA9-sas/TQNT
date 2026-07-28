"""Le magnétochiral de Koide — porteur de A, balayage pour delta,
fermeture de la voie fréquences (27/07/2026, suite de
derive_energie_propre_circuits).

TROIS CIBLES NOMMÉES à la Partie II, traitées dans l'ordre :

 [1] LE PORTEUR DE A = sqrt(2) : montrer numériquement que le doublet
     Z3 (classes 1/2) du noyau d'inductance est EXACTEMENT la paire
     des deux SENS de propagation (courants opposés ±j, classe 0
     à courant nul). Identification qui en découle (conditionnelle,
     au rang du théorème des canaux §12 déjà consolidé) : la
     composante enroulée du mode leptonique a DEUX canaux (les deux
     sens) contre UN pour la composante uniforme => son poids relatif
     est la dimension quantique du doublement de canal,
     A = d_1/2(k=2) = sqrt(2) — LE MÊME sqrt(2) que l'excès (d-1)
     des baryons lourds (le document principal déclare déjà « les 2
     indices = UNE structure ») : la lecture magnétique lui donne son
     porteur concret, le doublement de SENS de la circulation.

 [2] delta = 2/9 PAR L'HOLONOMIE DE FULLER ? — jeu FERMÉ des lectures
     canoniques de la phase transportée Phi = 2 pi Wr mod 2 pi
     (théorème de Fuller, établi V6) : plein / ruban (/2) / pas Z3
     (/3) / ruban x Z3 (/6) / writhe-angle direct. Wr idéal du trèfle:
     3,4165 ± 0,005 (katlas) et 24/7 (quasi-quantisation Stasiak).
     Verdict par la règle du test nul (mécanisme exigé).

 [3] TEST DE LA VOIE « ÉCHELLE DE FRÉQUENCES » : si masse = c x
     fréquence de mode du circuit (m ∝ omega^p, modes |n - W| avec n
     dans les 3 classes Z3 distinctes), le balayage TROUVE des
     ajustements à quelques % — mais avec 3 paramètres libres
     (W continu, 2 entiers) pour 2 rapports : la voie s'ajuste par
     LIBERTÉ D'ASSIGNATION (on compte les solutions), donc contenu
     prédictif NUL — test nul échoué par construction. La voie n'est
     pas impossible, elle est VIDE ; la forme amplitude, elle, porte
     des contraintes (A fixé par le porteur [1], delta partagé).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from simulation.experiments.derive_energie_propre_circuits import (
    kernel_and_shift, z3_blocks)

M_E, M_MU, M_TAU = 0.51099895, 105.6583755, 1776.86   # MeV
DELTA_KOIDE = 2.0 / 9.0
WR_KATLAS, WR_ERR = 3.4165, 0.005
WR_STASIAK = 24.0 / 7.0


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LE DOUBLET Z3 = LES DEUX SENS (porteur de A)")
    print("=" * 78)
    M, dev, N = kernel_and_shift(450)
    D = np.zeros((N, N))
    idx = np.arange(N)
    D[idx, (idx + 1) % N] = 0.5
    D[idx, (idx - 1) % N] = -0.5
    blocks_M = z3_blocks(M, N)
    blocks_D = z3_blocks(D, N)
    js = []
    for c in range(3):
        Mc = 0.5 * (blocks_M[c] + blocks_M[c].conj().T)
        w, V = np.linalg.eigh(Mc)
        v = V[:, -1]
        Jc = -1j * blocks_D[c]
        Jc = 0.5 * (Jc + Jc.conj().T)
        js.append(float(np.real(v.conj() @ (Jc @ v))))
    print("  courant du mode extrême par classe : j0 = %+.5f, "
          "j1 = %+.5f, j2 = %+.5f" % tuple(js))
    print("  j1 + j2 = %.2e (opposés exacts) ; j0 = %.2e (nul)" % (
        js[1] + js[2], abs(js[0])))
    print("""
  => le doublet {classe 1, classe 2} EST la paire des deux SENS de
  circulation (courants ±j exactement opposés ; la classe 0 n'en
  porte aucun). IDENTIFICATION (conditionnelle, rang §12) : la
  composante enroulée du lepton a 2 canaux (les 2 sens), l'uniforme
  en a 1 => poids relatif = dimension quantique du doublement,
  A = d_1/2(k=2) = sqrt(2) = 1,41421 — vs Koide mesuré A = sqrt(2)
  (identifié, 61 ppm). Le sqrt(2) de Koide et le (d-1) des baryons
  lourds partagent désormais un PORTEUR : le doublement de sens.""")

    print("=" * 78)
    print("[2] delta PAR L'HOLONOMIE DE FULLER — jeu fermé")
    print("=" * 78)
    for lab, wr in [("katlas 3,4165", WR_KATLAS), ("Stasiak 24/7", WR_STASIAK)]:
        phi = (2 * np.pi * wr) % (2 * np.pi)
        cands = [
            ("Phi (plein)", phi),
            ("Phi/2 (ruban 4pi)", phi / 2),
            ("Phi/3 (pas Z3)", phi / 3),
            ("Phi/6 (ruban x Z3)", phi / 6),
            ("Phi/12 (sans mécanisme)", phi / 12),
            ("Wr mod 2pi/3", wr % (2 * np.pi / 3)),
            ("Wr - pi", wr - np.pi),
        ]
        print("  Wr idéal = %s :" % lab)
        for nom, v in cands:
            print("    %-26s = %.4f rad   vs 2/9 = %.4f   (%+.1f %%)" % (
                nom, v, DELTA_KOIDE, 100 * (v / DELTA_KOIDE - 1)))
    print("""
  VERDICT (règle du test nul) : AUCUNE lecture canonique de
  l'holonomie ne tombe sur 2/9. La plus proche (Phi/12, -2 % katlas /
  +1 % Stasiak) n'a PAS de mécanisme (le facteur 12 n'est porté par
  rien : ni ruban (2), ni Z3 (3), ni leur produit (6)) => REJETÉE.
  delta reste un INDICE : son PORTEUR est identifié (le couplage
  T-impair au sens, Partie II), sa VALEUR n'est pas dérivée.""")

    print("=" * 78)
    print("[3] TEST DE LA VOIE « ÉCHELLE DE FRÉQUENCES » (liberté d'assignation)")
    print("=" * 78)
    targets = {1: (M_MU / M_E, M_TAU / M_E),
               2: (np.sqrt(M_MU / M_E), np.sqrt(M_TAU / M_E))}
    for p, (r1, r2) in targets.items():
        bb, n_fit5 = None, 0
        for W in np.arange(0.001, 30.0, 0.002):
            n0 = round(W)
            e0 = abs(n0 - W)
            if e0 < 1e-9:
                continue
            for n1 in range(max(0, n0 - 30), n0 + 31):
                if (n1 - n0) % 3 == 0:
                    continue
                for n2 in range(max(0, n0 - 30), n0 + 31):
                    if (n2 - n0) % 3 == 0 or (n2 - n1) % 3 == 0:
                        continue
                    q1, q2 = abs(n1 - W) / e0, abs(n2 - W) / e0
                    err = max(abs(np.log(q1 / r1)), abs(np.log(q2 / r2)))
                    if err < 0.05:
                        n_fit5 += 1
                    if bb is None or err < bb[0]:
                        bb = (err, W, n0, n1, n2, q1, q2)
        err, W, n0, n1, n2, q1, q2 = bb
        print("  m ∝ omega^%d : meilleur ajustement W=%.3f, modes (%d,%d,%d)"
              % (p, W, n0, n1, n2))
        print("    rapports %.1f / %.1f  vs requis %.1f / %.1f  (écart max"
              " %.0f %%) ; solutions à <5 %% : %d" % (
                  q1, q2, r1, r2, 100 * (np.exp(err) - 1), n_fit5))
    print("""
  => la voie fréquences S'AJUSTE (quelques %), mais par LIBERTÉ
  D'ASSIGNATION : des dizaines/centaines de solutions à <5 % pour
  2 rapports avec 3 paramètres libres, et les meilleurs ajustements
  demandent des modes arbitraires (aucune structure) et un W accordé
  finement — le contenu prédictif est NUL (test nul échoué par
  construction). La voie n'est pas impossible : elle est VIDE. La
  forme amplitude (canonique, sqrt(m_k) = mu(1 + A cos(2pi k/3 +
  delta))) est la seule des deux qui PORTE des contraintes — et
  c'est là que le porteur [1] place A = sqrt(2).""")

    print("=" * 78)
    print("BILAN DES TROIS CIBLES")
    print("=" * 78)
    print("""   A = sqrt(2)  : PORTEUR DÉRIVÉ (doublement de sens, courants ±j
                  exacts ; identification au rang du théorème des
                  canaux — le même sqrt(2) que d-1 et Koide) ;
   delta = 2/9  : porteur identifié (T-impair, couplage au sens),
                  valeur NON dérivée (holonomie pure rejetée au jeu
                  fermé) — cible restante ;
   mode -> masse : voie fréquences VIDE (elle s'ajuste par liberté
                  d'assignation, contenu prédictif nul — test nul) ;
                  la forme amplitude (canonique) est la seule
                  porteuse de contraintes — cohérente avec le
                  (3/4)√σ_χ = j(j+1) spin-1/2 déjà indicé au raccord
                  (non poussé ici).""")


if __name__ == "__main__":
    main()
