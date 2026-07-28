"""
Bilan détaillé statistique pour la condensation des nœuds TQNT (V5 §13.5).

Implémente le mécanisme de freeze-out :
  χ⁰  ⇌  K(c_n)
où χ⁰ est la phase pré-condensée (vide CS sans nœuds) et K(c_n) un nœud
fermé à c_n croisements. L'équilibre dynamique à T_c ≈ 67 MeV détermine
l'abondance reliquaire n_K / n_χ.

FORMALISME
----------
Le taux de formation est :
  Γ_+(χ⁰ → K) = σ_+ · n_χ · v_rel
            ∝ |⟨W_K⟩|² · Verlinde_count · n_χ
Le taux de destruction (décondensation thermique) :
  Γ_-(K → χ⁰) ∝ |⟨W_K⟩|² / Verlinde_count · n_K

À l'équilibre Γ_+ = Γ_- :
  n_K / n_χ = (Verlinde_count(K))² · e^(-ΔE_K / T)

où ΔE_K est l'énergie de formation (V5 chap. 14 : m(K) ≈ Ω_ts(K)).

EXTRACTION DE α
---------------
La V5 §13.5 paramétrise n_K(c_n) ~ N(c_n) · e^(-α c_n). En posant cette
forme à partir du bilan détaillé, on extrait :
  α_eff(c_n, c_n+1) = log[n_K(c_n) / n_K(c_n+1)]
                    - log[N(c_n) / N(c_n+1)]

Au point fixe asymptotique c_n → ∞, α_eff → constante (prédiction V5
qu'on cherche à dériver).

ENTRÉES
-------
T_c : température de condensation (≈ 67 MeV, V5 chap. 22 cinétique
       Big Bang).
k   : niveau Chern-Simons (=135, V5 §10.4).

Voir docs/TQNT_detaillee_V5_FR.tex §13.5 et chap. 22.
"""
from __future__ import annotations

import math
from functools import lru_cache

from theory.su2_modular import (
    V5_LEVEL, quantum_dim, total_dim_squared, conformal_weight,
)

T_C_MEV = 67.0
"""Température de condensation Big Bang TQNT (V5 chap. 22)."""

M_E_MEV = 0.51099895
"""Masse électron (référence V5 axiome 1)."""


def m_knot_simple(c: int) -> float:
    """Masse extrapolée d'un nœud à c croisements (V5 §13.5).

    Forme V5 : m(c) = m_e · (c/3)^6,65 (calibrée sur 3_1 = électron).
    Pour c=3 : m_e. Pour c=18 : ~32 GeV. Pour c=30 : ~5 TeV.
    """
    return M_E_MEV * (c / 3.0) ** 6.65


def verlinde_state_count(c: int, k: int = V5_LEVEL) -> float:
    """Nombre d'états Verlinde admissibles pour un nœud à c croisements.

    APPROXIMATION : nombre de fusions admissibles pour c lignes de Wilson
    fondamentales se croisant. La formule de Verlinde donne :
      N_states = Σ_j (d_j)^(c-1) / D^(2(c-1))
    pour c → grand. Pour c modeste (≤ 30), c'est un bon proxy de la
    complexité combinatoire.

    Approximation simple : N_states(c) ≈ d_{1/2}^(c-1)
    car les états dominants sont les chaînes de fusion par la fondamentale.
    """
    d_half = quantum_dim(0.5, k)
    return d_half ** (c - 1)


def wilson_loop_amplitude_sq(c: int, k: int = V5_LEVEL) -> float:
    """|⟨W_K⟩|² moyen sur les nœuds premiers à c croisements.

    APPROXIMATION : ⟨W_K⟩ ≈ d_{1/2} pour les nœuds simples (unknot-like).
    Pour les nœuds complexes, on prend |⟨W_K⟩|² ~ d_{1/2}² × (facteur Wess-Zumino).

    Cette approximation devra être raffinée avec un calcul explicite des
    polynômes de Jones pour les 19 nœuds catalogués (theory/jones.py).
    """
    d_half = quantum_dim(0.5, k)
    # Pour l'instant : |⟨W_K⟩|² ≈ d_{1/2}² indépendant de c
    # (approximation grossière qui sera raffinée)
    return d_half ** 2


def boltzmann_ratio(c: int, T_c: float = T_C_MEV,
                    k: int = V5_LEVEL) -> float:
    """Rapport d'équilibre n_K(c) / n_χ d'après bilan détaillé.

    Formule générale :
      n_K / n_χ = G(c, k) · e^(-m(c) / T_c)

    avec G(c, k) = facteur structural depuis SU(2)_k :
      G(c, k) = Verlinde_state_count(c) · |⟨W_K⟩|²

    Pour les c légers (m(c) << T_c) : G domine, n_K/n_χ ~ G(c,k).
    Pour les c lourds (m(c) >> T_c) : exp suppression écrase tout.
    Le seuil c* ~ 18 (V5) où la transition se produit.
    """
    m_c = m_knot_simple(c)
    G = verlinde_state_count(c, k) * wilson_loop_amplitude_sq(c, k)
    # Limite de sécurité numérique : exp(-50) → 0
    exponent = m_c / T_c
    if exponent > 700:
        return G * 0.0
    return G * math.exp(-exponent)


def alpha_from_boltzmann(c1: int, c2: int, T_c: float = T_C_MEV,
                         k: int = V5_LEVEL) -> float:
    """Coefficient α effectif entre c1 et c2 (paramétrisation V5 §13.5).

    n_K(c) ~ N(c) · e^(-α c)   (V5)
    bilan : n_K(c) = G(c, k) · e^(-m(c)/T_c)

    En identifiant N(c) ~ G(c, k) :
      α(c1, c2) = [log(G(c1)) - log(G(c2)) - (m(c1) - m(c2))/T_c] / (c2 - c1)
    plus précisément :
      log(n_K(c1)/n_K(c2)) = α (c2 - c1)
    donc :
      α = -[log(G(c1)) - log(G(c2)) - (m(c1) - m(c2))/T_c] / (c1 - c2)
    """
    n1 = boltzmann_ratio(c1, T_c, k)
    n2 = boltzmann_ratio(c2, T_c, k)
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    # ratio en log
    log_ratio = math.log(n1 / n2)
    return -log_ratio / (c1 - c2)


@lru_cache(maxsize=4)
def alpha_asymptotic(T_c: float = T_C_MEV, k: int = V5_LEVEL,
                     c_min: int = 30, c_max: int = 50) -> float:
    """Limite asymptotique α∞ = lim_{c→∞} α(c, c+1).

    Pour la condensation primordiale, α est attendu constant pour c grand.
    Cette fonction mesure α_eff sur la fenêtre [c_min, c_max] en moyennant.
    """
    alphas = []
    for c in range(c_min, c_max):
        a = alpha_from_boltzmann(c, c + 1, T_c, k)
        if math.isfinite(a):
            alphas.append(a)
    return sum(alphas) / len(alphas) if alphas else float("nan")


def alpha_per_crossing_structural(k: int = V5_LEVEL) -> dict:
    """Plusieurs hypothèses structurales V5.5 pour α depuis CS k=135.

    HYPOTHÈSES TESTÉES (5 candidats) :

    H1 : α = log(d_{1/2}²)
         Origine : chaque croisement ajoute 1 état fondamental
         Prédiction : log(d_{1/2}²) = log(4 cos²(π/137)) ≈ 1.386

    H2 : α = log(d_{1/2})
         Origine : chaque croisement = 1 fusion fondamentale
         Prédiction : log(d_{1/2}) ≈ 0.693

    H3 : α = 2π × h_{1/2}
         Origine : phase conforme par croisement
         Prédiction : 2π × 3/(8 × 137) ≈ 0.0344

    H4 : α = log(D²) / nombre de primaires
         Origine : entropie modulaire normalisée
         Prédiction : log(1.3e5) / 136 ≈ 0.0867

    H5 : α = log(d_{1/2}² · (k+2)/k) — correction de niveau
         Origine : H1 + facteur de level
         Prédiction : log(4 × 137/135) ≈ 1.401

    H6 (NOUVEAU) : α = log(d_{1/2}²) - log(N_HT_growth)
         où N_HT_growth ≈ 2.68 (taux croissance Hoste-Thistlethwaite)
         Prédiction : 1.386 - log(2.68/d_{1/2}²) ≈ 1.386 - 0.290 ≈ 1.096 ?
         (ou autre combinaison logique)

    H7 : α = log(d_{1/2}^{1.731})
         Origine : exposant ajusté pour reproduire 1.20
         Prédiction : 1.731 × log(d_{1/2}) = 1.200 par construction

    Le but : identifier laquelle des hypothèses (H1-H7 ou autre) reproduit
    le mieux α_V5 = 1.20 sans paramètre libre.
    """
    d_half = quantum_dim(0.5, k)
    h_half = conformal_weight(0.5, k)
    D2 = total_dim_squared(k)
    n_primaries = k + 1
    N_HT = 2.68  # taux asymptotique Hoste-Thistlethwaite

    hypotheses = {
        "H1: log(d_{1/2}²)": math.log(d_half ** 2),
        "H2: log(d_{1/2})": math.log(d_half),
        "H3: 2π h_{1/2}": 2 * math.pi * h_half,
        "H4: log(D²) / N_primaries": math.log(D2) / n_primaries,
        "H5: log(d_{1/2}² · (k+2)/k)": math.log(d_half ** 2 * (k + 2) / k),
        "H6: log(d_{1/2}²) - log(N_HT / d_{1/2})":
            math.log(d_half ** 2) - math.log(N_HT / d_half),
        "H7: log(d_{1/2}^{1.731}) (cal.)": 1.731 * math.log(d_half),
        "Calibration V5 §13.5": 1.20,
    }
    return hypotheses


if __name__ == "__main__":
    print("=" * 78)
    print(f"V5.5 — Bilan détaillé statistique χ⁰ ⇌ K(c_n)")
    print(f"  T_c = {T_C_MEV} MeV (V5 chap. 22)")
    print(f"  k   = {V5_LEVEL} (V5 §10.4)")
    print("=" * 78)

    # Rapport d'équilibre n_K / n_χ
    print(f"\n  {'c':>3} {'m(c) MeV':>12} {'Verlinde':>12} "
          f"{'|W|²':>8} {'n_K/n_χ':>15}")
    for c in [3, 5, 10, 15, 18, 20, 25, 30, 40, 50]:
        m = m_knot_simple(c)
        v = verlinde_state_count(c)
        w2 = wilson_loop_amplitude_sq(c)
        n_ratio = boltzmann_ratio(c)
        print(f"  {c:>3} {m:>12.3e} {v:>12.2e} {w2:>8.3f} "
              f"{n_ratio:>15.3e}")

    # α effectif depuis bilan détaillé
    print()
    print(f"\n  α effectif par paire (c, c+1) :")
    print(f"  {'c':>3} {'α(c, c+1)':>15}")
    for c in [3, 5, 10, 15, 18, 20, 25, 30, 35]:
        a = alpha_from_boltzmann(c, c + 1)
        print(f"  {c:>3} {a:>15.4f}")

    print()
    print(f"  α asymptotique (moyenne c ∈ [30, 50]) : "
          f"{alpha_asymptotic():.4f}")
    print(f"  V5 calibré §13.5 : 1.20")

    # Hypothèses structurales
    print()
    print("=" * 78)
    print("HYPOTHÈSES STRUCTURALES DEPUIS SU(2)_135")
    print("=" * 78)
    hypotheses = alpha_per_crossing_structural()
    for label, value in hypotheses.items():
        delta = (value - 1.20) / 1.20 * 100
        flag = " ←" if abs(delta) < 5 else ""
        print(f"  {label:50s} = {value:>8.4f}   "
              f"(vs V5 : {delta:>+7.1f}%){flag}")

    print()
    print("=" * 78)
    print("OBSERVATIONS")
    print("=" * 78)
    print(f"""
  H1 (log d_{{1/2}}² ≈ 1.386) est la plus proche de 1.20 (+15.5 %).
  H7 ajuste par construction l'exposant pour reproduire 1.20.

  La forme générale candidate :
    α = log(d_{{1/2}}^p) avec p à dériver structurellement.

  Pour V5.5 : déterminer p depuis les premiers principes de SU(2)_135 :
    - p = 2 : un état par crossing (trop)
    - p = 1 : une fusion par crossing (pas assez)
    - p = 1.73 : ratio 1.20 / log(d_{{1/2}}) — empirique

  Hypothèse de travail : p émerge de la dimension effective des canaux
  de fusion accessibles, mod la suppression Hoste-Thistlethwaite.
  L'étape suivante est de calculer p depuis une moyenne sur les nœuds
  du catalogue avec leurs polynômes de Jones explicits (theory/jones.py).
""")
