import pytest
from theory.strand_charge import closure_permutation, n_components, is_single_knot
from theory.strand_charge import (
    strand_writhes, writhe_symmetric_invariants, conjugate, inverse,
)

W_52 = [1, 1, 1, 2, -1, 2]  # sigma1^3 sigma2 sigma1^-1 sigma2 ; cloture = noeud unique 5_2 (3-cycle)

def test_closure_permutation_is_3cycle():
    assert closure_permutation(W_52) == [2, 0, 1]

def test_52_is_single_knot():
    assert n_components(W_52) == 1
    assert is_single_knot(W_52) is True

def test_trefoil_on_3_strands_is_two_components():
    # sigma1^3 n'utilise que les brins 1-2 -> trefle + inconnu = 2 composantes
    assert n_components([1, 1, 1]) == 2
    assert is_single_knot([1, 1, 1]) is False


def test_strand_writhes_sum_is_twice_writhe():
    # W_52 a des signes +,+,+,+,-,+ -> writhe total = 1+1+1+1-1+1 = 4 ; Sigma n_i = 8
    assert sum(strand_writhes(W_52)) == 8

def test_inverse_and_conjugate():
    assert inverse([1, -2, 1]) == [-1, 2, -1]
    assert conjugate([1, 1], [2]) == [2, 1, 1, -2]

def test_porte0_symmetric_invariants_under_conjugation():
    # Porte 0 : les symetriques de {n_i} doivent etre INVARIANTES par conjugaison
    # (la cloture, donc le baryon physique, est inchangee). Teste sur plusieurs gamma.
    base = writhe_symmetric_invariants(W_52)
    gammas = [[2], [1, 2], [-1, 2, 2], [2, -1, 2]]
    results = {tuple(base): 1}
    for gamma in gammas:
        w = conjugate(W_52, gamma)
        assert is_single_knot(w), "la conjugaison doit preserver le noeud"
        inv = writhe_symmetric_invariants(w)
        results[tuple(inv)] = results.get(tuple(inv), 0) + 1
    # Ce test ENREGISTRE le verdict de la Porte 0. Il PASSE si les symetriques sont
    # invariantes (un seul tuple), et DOIT etre marque xfail sinon (voir Step 3bis).
    if len(results) > 1:
        pytest.xfail(f"Porte 0 CASSE (attendu possible) : symetriques non invariantes = {results} "
                     "-> charge de brin non topologique -> CG reste OUVERT")
    assert len(results) == 1, f"Porte 0 CASSE : symetriques non invariantes = {results}"
