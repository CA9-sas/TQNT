"""Charges de brin d'une tresse à 3 brins depuis l'holonomie ℤ₃ de la clôture.

Encodage d'un mot : list[int], ±k = σ_k^{±1} (k∈{1,2}), signe = signe du croisement.
Voir docs/superpowers/specs/2026-07-08-charge-brins-z3-design.md.
"""
from __future__ import annotations

N_STRANDS = 3


def closure_permutation(word: list[int]) -> list[int]:
    """perm[j] = étiquette du brin occupant la position j après la tresse (0-indexé).
    (Pour « où va le brin i », prendre l'inverse : perm.index(i).)
    Le nombre de composantes de la clôture ne dépend que de la structure de cycles,
    identique pour une permutation et son inverse."""
    perm = list(range(N_STRANDS))
    for g in word:
        k = abs(g) - 1          # σ_k échange les positions k et k+1
        perm[k], perm[k + 1] = perm[k + 1], perm[k]
    return perm


def n_components(word: list[int]) -> int:
    """Nombre de composantes de la clôture = nombre de cycles de la permutation."""
    perm = closure_permutation(word)
    seen: set[int] = set()
    cycles = 0
    for s in range(N_STRANDS):
        if s in seen:
            continue
        cycles += 1
        x = s
        while x not in seen:
            seen.add(x)
            x = perm[x]
    return cycles


def is_single_knot(word: list[int]) -> bool:
    """Vrai si la clôture est un nœud unique (permutation = 3-cycle)."""
    return n_components(word) == 1


def strand_writhes(word: list[int]) -> tuple[int, int, int]:
    """n_i = croisements signés où le brin d'identité i intervient (suivi par identité)."""
    pos_to_id = list(range(N_STRANDS))
    n = [0, 0, 0]
    for g in word:
        k = abs(g) - 1
        eps = 1 if g > 0 else -1
        id_a, id_b = pos_to_id[k], pos_to_id[k + 1]
        n[id_a] += eps
        n[id_b] += eps
        pos_to_id[k], pos_to_id[k + 1] = pos_to_id[k + 1], pos_to_id[k]
    return tuple(n)


def writhe_symmetric_invariants(word: list[int]) -> tuple[int, int, int]:
    """(Sigma n_i, Sigma n_i^2, Sigma_{i<j} n_i n_j) — seules combinaisons invariantes par permutation."""
    n = strand_writhes(word)
    s1 = sum(n)
    s2 = sum(x * x for x in n)
    s11 = n[0] * n[1] + n[0] * n[2] + n[1] * n[2]
    return (s1, s2, s11)


def inverse(word: list[int]) -> list[int]:
    return [-g for g in reversed(word)]


def conjugate(word: list[int], gamma: list[int]) -> list[int]:
    return list(gamma) + list(word) + inverse(gamma)
