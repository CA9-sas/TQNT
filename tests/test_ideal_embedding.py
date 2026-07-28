"""Tests de l'outillage de plongement idéal (minimisation de ropelength, SONO)."""
import numpy as np
import pytest

from simulation.ideal_embedding import (
    sono, torus_knot, twist_knot_seed, ropelength, min_gap, resample, curve_length,
)


def test_twist_knot_seed_realizes_7_2():
    """L'amorce native (n_half=9, x_eye=0.10) réalise bien le 7_2 (Alexander)."""
    from simulation.experiments.repair_curves_braid import alexander
    P = twist_knot_seed(9, hand=+1, thread=+1, x_eye=0.10)
    assert tuple(int(x) for x in alexander(P)[0]) == (3, -5, 3)


def test_plat_seed_realizes_6_2():
    """Fermeture en plat du mot canonique [3,1,2] (b(11,3)) : Alexander du 6_2."""
    from simulation.ideal_embedding import plat_seed
    from simulation.experiments.repair_curves_braid import alexander
    P = plat_seed([2, 2, 2, -1, 2, 2])
    assert tuple(int(x) for x in alexander(P)[0]) == (1, -3, 3, -3, 1)


def test_resample_uniform_edges():
    P = torus_knot(2, 3, n=97)
    Q = resample(P, 160)
    d = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
    assert d.std() / d.mean() < 0.05
    assert abs(curve_length(Q) - curve_length(P)) / curve_length(P) < 0.02


def test_unknot_converges_to_thickness_limit():
    """Cercle : les deux contraintes (courbure, antipodes) se rejoignent à r=R,
    ropelength idéale = 2*pi."""
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    P0 = 3.0 * np.stack([np.cos(t), np.sin(t), 0 * t], 1)
    P, info = sono(P0, iters=4500)
    # cas dégénéré (borné par la seule courbure) : biais ~5% tolèré (seuil 1.05)
    assert 2 * np.pi * 0.93 < info["ropelength"] < 7.5


def test_hopf_link_ropelength_theorem():
    """Entrelacs de Hopf : ropelength idéale = 8*pi (théorème CKS 2002),
    deux cercles de rayon 2 ; Lk préservé par la relaxation."""
    from simulation.ideal_embedding import sono_link, linking_number, link_min_gap
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    A = np.stack([3 * np.cos(t), 3 * np.sin(t), 0 * t], 1)
    B = np.stack([3 + 3 * np.cos(t), 0 * t, 3 * np.sin(t)], 1)
    assert abs(linking_number(A, B)) == 1
    comps, info = sono_link([A, B], iters=4000)
    assert abs(info["ropelength"] - 8 * np.pi) / (8 * np.pi) < 0.03
    assert info["min_gap"] >= 1.95
    assert abs(linking_number(*comps)) == 1


def test_gauss_linking_number_unlinked():
    from simulation.ideal_embedding import linking_number
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    A = np.stack([np.cos(t), np.sin(t), 0 * t], 1)
    B = np.stack([8 + np.cos(t), np.sin(t), 0 * t], 1)
    assert linking_number(A, B) == 0


def test_trefoil_ropelength_near_published():
    """3_1 : ropelength publiée 32.74 (unités de rayon) ; SONO doit approcher a ~5%."""
    P0 = torus_knot(2, 3, n=140)
    P, info = sono(P0, iters=2500)
    assert 31.5 < info["ropelength"] < 34.5
    assert info["min_gap"] >= 1.95
    # la contrainte globale est active (nœud serré, pas seulement courbure)
    assert min_gap(P) == pytest.approx(2.0, abs=0.05)


def test_polish_gradient_improves_trefoil():
    """Le polisseur de gradient améliore strictement le minimum SONO (mou non
    uniforme) sans casser les contraintes (gap >= 2R a 2% pres)."""
    from simulation.ideal_embedding import polish_gradient
    P, info = sono(torus_knot(2, 3, n=140), iters=2500)
    Q, rho = polish_gradient(P, iters=200)
    assert rho <= info["ropelength"] + 1e-9
    assert min_gap(Q) >= 2.0 * 0.98
    # proche du publie (32.744) apres polissage
    assert rho < 33.3


def test_polish_gradient_link_hopf():
    """Version entrelacs : améliore le Hopf relaxé, préserve Lk et le gap."""
    from simulation.ideal_embedding import (sono_link, polish_gradient_link,
                                            linking_number)
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    A = np.stack([1.2 * np.cos(t), 1.2 * np.sin(t), 0 * t], 1)
    B = np.stack([1.2 + 1.2 * np.cos(t), 0 * t, 1.2 * np.sin(t)], 1)
    comps, info = sono_link([A, B], iters=2500)
    Q, rho = polish_gradient_link(comps, iters=200)
    assert rho <= info["ropelength"] + 1e-9
    assert abs(linking_number(*Q)) == 1
