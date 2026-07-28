"""Hiérarchie de masse des leptons : symétrie Z₃ du trèfle → formule de Koide.

Les leptons e, μ, τ sont le MÊME nœud 3₁ (trèfle, ponctuel) en trois GÉNÉRATIONS,
les trois modes de sa symétrie Z₃ (3 lobes, torus T(2,3)). Les amplitudes de masse
suivent la paramétrisation de Koide :

    √m_k = √m₀ · (1 + √2·cos(δ + 2πk/3)),   k = 0, 1, 2

DÉRIVATION de la relation de Koide depuis Z₃ : avec v_k = 1 + √2 cos(θ_k), θ_k aux
trois phases à 120°, on a Σv_k = 3 (Σcos = 0) et Σv_k² = 3 + 2·(3/2) = 6, donc

    Q = Σm / (Σ√m)² = Σv² / (Σv)² = 6/9 = 2/3.

Le facteur √2 + la symétrie Z₃ donnent EXACTEMENT Koide. Restent deux paramètres :
l'échelle m₀ et la phase δ (≈ 2/9 rad). Reproduit e, μ, τ à mieux de 0,5 %.
"""
import numpy as np
import math

M_E, M_MU, M_TAU = 0.51099895, 105.6583755, 1776.86
M_P = 938.272  # m_0 = m_p/3 (échelle = masse/composante du nucléon, à 0,35%)


def koide_Q(masses):
    s = sum(masses)
    sq = sum(math.sqrt(m) for m in masses)
    return s / sq**2


def lepton_masses(m0, delta):
    """les 3 masses depuis (m0, δ) + symétrie Z₃ (k=0→τ, 1→e, 2→μ)."""
    return [m0 * (1 + math.sqrt(2) * math.cos(delta + 2 * math.pi * k / 3))**2
            for k in range(3)]


def fit():
    """extrait m₀ et δ des masses mesurées."""
    sq = np.array([math.sqrt(M_E), math.sqrt(M_MU), math.sqrt(M_TAU)])
    m0 = (sq.sum() / 3)**2
    delta = math.acos(((sq / math.sqrt(m0)) - 1).max() / math.sqrt(2))
    return m0, delta


if __name__ == "__main__":
    Q = koide_Q([M_E, M_MU, M_TAU])
    print(f"Koide : Q = {Q:.6f}  (2/3 = {2/3:.6f})  → dérivé de Z₃ + √2")
    m0, delta = fit()
    print(f"m₀ = {m0:.1f} MeV ; δ = {delta:.4f} rad ({math.degrees(delta):.2f}°, ≈ 2/9 = {2/9:.4f})")
    mk = lepton_masses(m0, delta)
    for nm, mp, exp in zip(["τ", "e", "μ"], mk, [M_TAU, M_E, M_MU]):
        print(f"  {nm}: prédite {mp:9.3f} MeV  exp {exp:9.3f}  ({(mp/exp-1)*100:+.2f}%)")
