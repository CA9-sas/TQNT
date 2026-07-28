"""La sélection de m par les lignes de framing — le dernier maillon
de structure (27/07/2026, suite du menu des modes).

MÉCANISME CANDIDAT (au rang de l'incompressibilité) : le courant de
surface n'est pas une nappe libre — il est PORTÉ par les lignes de
framing (White), et chaque ligne porte un courant FIXE (pas de
source sur une ligne : la charge n'y accumule pas). Les excitations
physiques du faisceau de lignes sont les ondes de REGROUPEMENT
(les lignes se resserrent/s'écartent transversalement, chacune
gardant son courant). Une telle onde a son vecteur d'onde
PERPENDICULAIRE aux lignes. Sur le tore de surface (sigma, phi), la
direction des lignes est (1, Tw) (Tw tours d'azimut par tour de
circuit) et le mode (n=3, m) a le vecteur d'onde (3, m) :
    k perpendiculaire aux lignes  <=>  3 + m Tw = 0
    =>  m* = -3/Tw = 3/(Wr - Lk) = +1,24   (Tw = Lk - Wr = -2,42).
m* n'est pas entier (le regroupement parfait est FRUSTRÉ) ; le seul
entier à résidu SOUS-TOUR est m = +1 (|3 + Tw| = 0,58 ; tous les
autres >= 1,83) : LA SÉLECTION EST UNIQUE, et le SIGNE sort
automatiquement de la chiralité (Tw < 0 => m > 0) — c'est le
gagnant du menu (0,461 MeV, -9,7 %).

CE QUE ÇA FERME : la chaîne m_e complète —
    m_e = énergie de Green du mode de regroupement des lignes de
          framing (n=3, m=+1), sélectionné par la conservation du
          courant de ligne, sur le tube trèfle idéal — zéro
          paramètre, -9,7 %.
RESTE OUVERT : le résidu (-9,7 %) — candidats : la frustration
(m* = 1,24 vs 1 : dérive résiduelle 0,58 tour le long des lignes),
l'amplitude fine, la géométrie (~1 %). Règle du test nul.

VÉRIFICATIONS ICI : [1] table de sélection |3 + m Tw| (l'unicité) ;
[2] cohérence sélection/spectre (le mécanisme choisit m=+1 SANS
minimiser l'énergie — déclaré) ; [3] raffinement numérique de
E(m=+1) (résolution accrue).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from simulation.ideal_embedding import sono, resample
from theory.omega_ts import _knot_curve
from simulation.experiments.derive_menu_modes_surface import mode_energy

M_E = 0.51099895
WR_IDEAL = 3.4165
LK = 1
TW = LK - WR_IDEAL


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LA TABLE DE SÉLECTION (|3 + m·Tw|, en tours)")
    print("=" * 78)
    menu = {0: 0.678, 1: 0.461, -1: 0.856, 2: 0.417, -2: 0.396,
            3: 0.296, -3: 0.288}
    print("  m* exact = -3/Tw = %.3f (non entier : regroupement frustré)"
          % (-3 / TW))
    print("  %-4s %14s %12s %s" % ("m", "|3+m·Tw| (tours)", "E(m) MeV",
                                   ""))
    for m in [1, 2, 0, 3, -1, -2, -3]:
        r = abs(3 + m * TW)
        note = "   <== SEUL résidu sous-tour : SÉLECTIONNÉ" if r < 1 else ""
        print("  %-4d %10.2f %14.3f %s" % (m, r, menu[m], note))
    print("""
  => la sélection est UNIQUE (m = +1) et le signe sort de la
  chiralité (Tw < 0 => m > 0) — le dédoublement chiral du menu
  (0,461 vs 0,856) est l'ombre énergétique de cet alignement.
  NB : le mécanisme choisit m SANS minimiser l'énergie (m = ±3 sont
  moins chers mais mal alignés) — c'est une sélection cinématique
  (conservation du courant de ligne), pas énergétique.""")

    print("=" * 78)
    print("[3] RAFFINEMENT DE E(m=+1)")
    print("=" * 78)
    P0 = np.asarray(_knot_curve("3_1", 200), float)
    P_id, info = sono(P0, iters=20000)
    ell = info["ropelength"]
    for nres, nphi in [(240, 12), (320, 16)]:
        P = resample(P_id, nres)
        dl = np.roll(P, -1, 0) - P
        a = np.linalg.norm(dl, axis=1).sum() / ell
        E_ref, n2_ref, L = mode_energy(P, a, 0, n_phi=nphi,
                                       longitudinal_ref=True)
        CAL = 2.283 / (0.5 * E_ref)
        E_raw, n2, _ = mode_energy(P, a, 1, n_phi=nphi)
        E1 = 0.5 * E_raw * CAL * (n2_ref / n2)
        print("  n=%3d, n_phi=%2d : E(m=+1) = %.4f MeV  (%+.1f %%)" % (
            nres, nphi, E1, 100 * (E1 / M_E - 1)))
    print("""
  BILAN : la chaîne m_e est fermée en STRUCTURE —
    m_e = E_Green[mode de regroupement des lignes de framing,
          n=3, m=+1 sélectionné par la conservation du courant de
          ligne] sur le tube trèfle idéal,
  zéro paramètre, résidu ~-10 %. Le résidu est l'ouvert (frustration
  0,58 tour / amplitude fine / géométrie) — nommé, pas revendiqué.""")


if __name__ == "__main__":
    main()
