"""V5.36a — Tentative de dérivation rigoureuse de c_J(3_1) depuis SU(2)_135.

Contexte
========

V5.35d ferme les 4 oscillations chirales {K0, D0, B0, B_s0} à 0,1 % en Σ|log|
avec deux paramètres calibrés :

    β = -1,3000      (asymétrie de complexité, V5.35b)
    δ = -2,7667      (3-corps multiplicatif, V5.35d)

V5.36 (commit b5340e2) a identifié une conjecture **STRONG** :

    β ≈ -ln(c_J(3_1))            avec c_J(3_1) = 3,66  →  -1,2975  (0,20 %)

Le problème : le **code TQNT** définit
``c_J(K) = span(V_K) + γ·|V_K(-1)|`` avec γ=0,1 (cf. theory/mass_formula.py,
theory/jones.py), ce qui donne c_J(3_1) = 3 + 0,1·3 = **3,3** — pas 3,66 !

D'où l'écart résiduel : -ln(3,3) = -1,1939, soit **8,16 %** d'erreur sur β
si on prend la définition canonique du code.

Pour que β = -ln(c_J(3_1)) soit *first-principles*, il faudrait c_J = e^{1,30}
= 3,669. La question : peut-on dériver **rigoureusement** depuis SU(2)_135
une valeur de c_J(3_1) entre 3,66 et 3,67 ?

Définitions testées dans ce module
==================================

Toutes les définitions sont calculées sur ``V_{3_1}(t) = t + t^3 - t^4``
(convention KnotInfo, droitière, code ``theory/jones.py``).

* **M1 (code TQNT)**    c_J = span + γ·|V_K(-1)| = 3 + 0,1·3                = 3,3000
* **M2 (RT à q²)**       |V_K(q²)| à q = exp(2iπ/137)                       = 1,0247
* **M3 (RT normalisé)**  |V_K(q)| / d_{1/2}(k=135)                          = 0,5033
* **M4 (Mahler)**        Mesure de Mahler de V_K(t)/t                       = 1,4656
* **M5 (sum |a_n|)**     Σ|a_n| (norme L1 des coefficients)                 = 3,0000
* **M6 (sqrt sum a²)**   √Σa_n² (norme L2 des coefficients)                 = 1,7321
* **M7 (span+2/3)**      span + 2/3 (heuristique, écart 0,07 % à e^1,30)    = 3,6667
* **M8 (exp(13/10))**    exp(1,30) (tautologique, vérification numérique)   = 3,6693
* **M9 (V_K(q²)·k+2)**   |V_K(q²)| × √(k+2) / √(coefs)                      ≈ varie
* **M10 (Kashaev)**      Limite |J_N(K)| pour 3_1 (torus, vol=0 → triviale) ≈ 1

Verdict honnête V5.36a
======================

**Aucune définition first-principles SU(2)_135 ne reproduit c_J(3_1) = 3,669
à <1 %.**  Les meilleures candidates :

* **M1 (code, 3,30)** : β = -ln(3,30) = -1,194 → écart **8,2 %** à -1,30
* **M7 (span+2/3, 3,667)** : β = -ln(3,667) = -1,2993 → écart **0,05 %** mais
  le « 2/3 » est numérologique sans interprétation TQNT (c'est la fraction
  de charge up-quark, or les quarks ne sont pas fondamentaux en TQNT).

Conséquence pour V5.36a :

* La conjecture B3 de V5.36 reposait sur c_J = 3,66 **non documenté
  dans le code** — il s'agissait d'une supposition (probablement issue
  de la mémoire d'une version V3/V4 où ``γ`` aurait été ajusté différemment).
* Pour préserver la cohérence de la TQNT, β reste **calibré** (-1,30) en
  attendant une dérivation Chern-Simons à boucles ou une réinterprétation
  via les invariants de Vassiliev d'ordre 2 ou 3.

Pistes ouvertes :

* Augmenter γ de 0,1 à 0,223 → c_J(3_1) = 3 + 0,223·3 = 3,669 mais ce serait
  une calibration cachée.
* Définir c_J via ``span(V_K) + ln(det) × κ`` avec κ adapté.
* Définir c_J via la Mahler measure ou Alexander polynomial.
* Lien possible avec **dimension Kashaev quantique** au lieu de span+det.

Référence
=========
- theory/jones.py             : définition canonique c_J = span + γ·det
- theory/su2_modular.py       : structure SU(2)_135
- theory/v5p36_first_principles_beta_delta.py : conjectures V5.36
- docs/TQNT_detaillee_V5_FR.tex (chap. masse) : c_J(3_1) = 3,3
- docs/TQNT_V6_EN.tex                          : c_J(3_1) = 3,3 (glossaire)
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from theory.jones import (
    evaluate_jones,
    jones_determinant,
    jones_polynomial,
    jones_retraction_factor,
    jones_span,
)
from theory.mass_formula import GAMMA, c_j
from theory.su2_modular import (
    V5_LEVEL,
    conformal_weight,
    quantum_dim,
    q_parameter,
    wilson_loop_from_jones,
    wilson_loop_unknot,
)


# Cible imposée par la conjecture V5.36 (B3 : β = -ln c_J(3_1)) :
TARGET_C_J_3_1: float = math.exp(1.30)
"""Valeur ``c_J(3_1)`` requise pour que β = -ln(c_J) = -1,30 (V5.35b calibré)."""

BETA_CALIBRATED: float = -1.3000
"""β calibré sur la fermeture chirale V5.35b (commit 8c7ea0e)."""


@dataclass(frozen=True)
class CJCandidate:
    """Une définition candidate de ``c_J(3_1)``."""

    name: str
    """Identifiant court (M1, M2, ...)."""

    formula: str
    """Forme symbolique courte."""

    value: float
    """Valeur numérique."""

    interpretation: str
    """Justification TQNT/CS si elle existe (sinon ``heuristique``)."""

    is_first_principles: bool
    """True si l'expression repose uniquement sur k=135 + invariants de K."""

    @property
    def beta(self) -> float:
        """β résultant = -ln(c_J)."""
        if self.value <= 0:
            return float("nan")
        return -math.log(self.value)

    @property
    def relative_error_cJ(self) -> float:
        """|value - cible|/cible (cible = e^{1,30} ≈ 3,669)."""
        return abs(self.value - TARGET_C_J_3_1) / TARGET_C_J_3_1

    @property
    def relative_error_beta(self) -> float:
        """|β - β_calibré|/|β_calibré|."""
        if math.isnan(self.beta):
            return float("inf")
        return abs(self.beta - BETA_CALIBRATED) / abs(BETA_CALIBRATED)


# ---------------------------------------------------------------------------
# Méthodes de calcul de c_J(3_1)
# ---------------------------------------------------------------------------
def cJ_3_1_code_definition() -> float:
    """c_J(3_1) selon la définition canonique du code : span + γ·det.

    γ = 0,10 (theory/mass_formula.GAMMA, theory/jones.GAMMA via postulates).
    Résultat : 3 + 0,1·3 = **3,3000** (tabulé dans tous les .tex V3–V6).
    """
    return c_j("3_1")


def cJ_3_1_from_jones_at_q2(k: int = V5_LEVEL) -> float:
    """|V_{3_1}(q²)| avec q = exp(2iπ/(k+2)).

    Cette évaluation correspond à l'expectation Wilson loop en représentation
    fondamentale (spin 1/2) dans la théorie de Chern-Simons SU(2)_k.

    Convention : ⟨W_K⟩_{j=1/2} = V_K(q²) (Witten 1989).
    À k=135 : V_{3_1}(q²) ≈ 1,0247 (très proche de 1, car q² ≈ 1).
    """
    poly = jones_polynomial("3_1")
    q = q_parameter(k)
    q2 = q * q
    return abs(evaluate_jones(poly, q2))


def cJ_3_1_from_wilson_loop(k: int = V5_LEVEL) -> float:
    """Wilson loop ⟨W_{3_1}⟩ normalisé par ⟨W_unknot⟩ = d_{1/2}.

    ⟨W_K⟩ = V_K(q) avec q = exp(2iπ/(k+2)).

    Cette normalisation produit une valeur < 1 pour 3_1 (≈ 0,503), donc
    -ln(...) > 0 (β > 0), incompatible avec β = -1,30 sans inversion.
    """
    poly = jones_polynomial("3_1")
    W = wilson_loop_from_jones(poly, k)
    W_unknot = wilson_loop_unknot(k)
    return abs(W) / W_unknot


def cJ_3_1_mahler_measure() -> float:
    """Mesure de Mahler de V_{3_1}(t)/t = 1 + t² − t³.

    M(p) = |a_n| · ∏_i max(1, |z_i|) où z_i sont les racines complexes.
    Pour V_{3_1}(t)/t = -t³ + t² + 1, racines numériques.
    """
    # V_{3_1}(t) = t + t^3 - t^4 = t * (1 + t^2 - t^3)
    # Factor: (1 + t^2 - t^3) = -t^3 + t^2 + 1
    # Polynomial in numpy: coefficients high-to-low
    coeffs = [-1.0, 1.0, 0.0, 1.0]  # -t^3 + t^2 + 0·t + 1
    roots = np.roots(coeffs)
    leading = abs(coeffs[0])
    M = leading * np.prod([max(1.0, abs(r)) for r in roots])
    return float(M)


def cJ_3_1_sum_abs_coeffs() -> float:
    """Norme L1 des coefficients de V_{3_1}(t) : Σ|a_n| = |1| + |1| + |−1| = 3."""
    poly = jones_polynomial("3_1")
    return float(sum(abs(c) for c in poly.values()))


def cJ_3_1_sqrt_sum_sq() -> float:
    """Norme L2 des coefficients : √Σa_n² = √3 ≈ 1,732."""
    poly = jones_polynomial("3_1")
    return float(math.sqrt(sum(c**2 for c in poly.values())))


def cJ_3_1_span_plus_two_thirds() -> float:
    """span(V_K) + 2/3 — heuristique : écart 0,05 % à la cible.

    Pas d'interprétation TQNT propre (2/3 = charge de l'up-quark, mais
    les quarks ne sont pas fondamentaux en TQNT V5).  À retenir comme
    coïncidence numérique frappante uniquement.
    """
    return jones_span(jones_polynomial("3_1")) + 2.0 / 3.0


def cJ_3_1_target_tautological() -> float:
    """Tautologie : exp(-β_calibré) = exp(1,30) = 3,669.  Vérification numérique."""
    return math.exp(-BETA_CALIBRATED)


def cJ_3_1_span_quantum() -> float:
    """span × d_{1/2} / √3 = 3·d_{1/2}/√3 — test SU(2)_135 quantique."""
    d_half = quantum_dim(0.5, V5_LEVEL)
    return jones_span(jones_polynomial("3_1")) * d_half / math.sqrt(3.0)


def cJ_3_1_kashaev_torus() -> float:
    """Limite Kashaev pour 3_1 (torus knot, vol = 0).

    Pour un nœud torique T(2,3), exp(Vol/(2π)) = 1 (volume hyperbolique nul).
    Kashaev invariant ≈ 1 à toute l'ordre dominant.
    """
    return 1.0


def cJ_general(knot_name: str) -> float:
    """c_J(K) selon la définition canonique du code (alias de c_j)."""
    return c_j(knot_name)


# ---------------------------------------------------------------------------
# Collection des candidats
# ---------------------------------------------------------------------------
def all_candidates() -> list[CJCandidate]:
    """Retourne les 10 méthodes testées pour c_J(3_1)."""
    return [
        CJCandidate(
            name="M1",
            formula="span + γ·|V_K(-1)| = 3 + 0,1·3 = 3,3 (code TQNT canonique)",
            value=cJ_3_1_code_definition(),
            interpretation=(
                "Définition opérationnelle V3–V6 (theory/jones.py, "
                "theory/mass_formula.py) avec γ=0,10 calibré. "
                "Cette valeur 3,3 est tabulée dans tous les .tex et le "
                "xlsx V5 du catalogue. β résultant = -1,194 (écart 8,2 % à -1,30)."
            ),
            is_first_principles=True,
        ),
        CJCandidate(
            name="M2",
            formula="|V_{3_1}(q²)| à q = exp(2iπ/137) (RT spin-1/2)",
            value=cJ_3_1_from_jones_at_q2(),
            interpretation=(
                "Wilson loop normalisé en représentation fondamentale "
                "(Witten 1989). À k=135, q² ≈ 1 donc V_K(q²) ≈ 1 + O(1/k). "
                "Valeur ≪ 3,669, β résultant proche de 0."
            ),
            is_first_principles=True,
        ),
        CJCandidate(
            name="M3",
            formula="|V_K(q)| / d_{1/2}(k=135) (RT normalisé)",
            value=cJ_3_1_from_wilson_loop(),
            interpretation=(
                "Normalisation Reshetikhin-Turaev par ⟨W_unknot⟩ = d_{1/2}. "
                "Donne ≈ 0,503, incompatible avec β < 0 sans inversion."
            ),
            is_first_principles=True,
        ),
        CJCandidate(
            name="M4",
            formula="M(V_K(t)/t) = mesure de Mahler",
            value=cJ_3_1_mahler_measure(),
            interpretation=(
                "Produit des max(1,|z_i|) sur les racines complexes ; "
                "invariant entropique du polynôme de Jones. "
                "≈ 1,466 (golden-ratio-like), pas la cible."
            ),
            is_first_principles=True,
        ),
        CJCandidate(
            name="M5",
            formula="Σ|a_n| (norme L1 des coefficients de V_K)",
            value=cJ_3_1_sum_abs_coeffs(),
            interpretation=(
                "Norme L1 = 3 (= |V_K(1)| en valeur absolue). "
                "Coïncidence avec span(V_K). β résultant = -ln(3) = -1,098."
            ),
            is_first_principles=True,
        ),
        CJCandidate(
            name="M6",
            formula="√Σa_n² (norme L2 des coefficients)",
            value=cJ_3_1_sqrt_sum_sq(),
            interpretation=(
                "Norme L2 = √3 ≈ 1,732. Trop petite, β positif."
            ),
            is_first_principles=True,
        ),
        CJCandidate(
            name="M7",
            formula="span + 2/3 = 3 + 2/3 = 3,667",
            value=cJ_3_1_span_plus_two_thirds(),
            interpretation=(
                "Coïncidence numérique frappante (0,07 % d'écart). "
                "Mais 2/3 = charge up-quark, et les quarks ne sont PAS "
                "fondamentaux en TQNT — donc heuristique sans interprétation. "
                "À ignorer comme dérivation."
            ),
            is_first_principles=False,
        ),
        CJCandidate(
            name="M8",
            formula="exp(-β_calibré) = e^{1,30} (tautologique)",
            value=cJ_3_1_target_tautological(),
            interpretation=(
                "Vérification numérique : si on impose β = -ln(c_J), alors "
                "c_J = e^{1,30} = 3,669 par construction. Pas une dérivation."
            ),
            is_first_principles=False,
        ),
        CJCandidate(
            name="M9",
            formula="span · d_{1/2}(k=135) / √3",
            value=cJ_3_1_span_quantum(),
            interpretation=(
                "Dimension quantique du fondamental ; à k=135 d_{1/2} ≈ 2 "
                "donc valeur ≈ 3,46. Écart ≈ 5,6 %."
            ),
            is_first_principles=True,
        ),
        CJCandidate(
            name="M10",
            formula="Kashaev limit (torus knot 3_1, vol=0) → 1",
            value=cJ_3_1_kashaev_torus(),
            interpretation=(
                "Pour T(2,3), le volume hyperbolique = 0 donc Kashaev → 1. "
                "Pas la bonne échelle."
            ),
            is_first_principles=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Tri et rapport
# ---------------------------------------------------------------------------
def rank_candidates(
    candidates: list[CJCandidate] | None = None,
) -> list[CJCandidate]:
    """Trie les candidats par erreur relative sur c_J croissante."""
    if candidates is None:
        candidates = all_candidates()
    return sorted(candidates, key=lambda c: c.relative_error_cJ)


def best_first_principles(
    candidates: list[CJCandidate] | None = None,
) -> CJCandidate | None:
    """Retourne le meilleur candidat first-principles strict, sinon None."""
    if candidates is None:
        candidates = all_candidates()
    fp = [c for c in candidates if c.is_first_principles]
    if not fp:
        return None
    return min(fp, key=lambda c: c.relative_error_cJ)


def verdict(
    candidates: list[CJCandidate] | None = None,
    threshold_strong: float = 0.01,
    threshold_acceptable: float = 0.05,
) -> str:
    """STRONG (<1 %) / ACCEPTABLE (<5 %) / WEAK (≥5 %) / NONE."""
    best = best_first_principles(candidates)
    if best is None:
        return "NONE"
    if best.relative_error_cJ < threshold_strong:
        return "STRONG"
    if best.relative_error_cJ < threshold_acceptable:
        return "ACCEPTABLE"
    return "WEAK"


def summary_report() -> str:
    """Rapport texte synthétique pour CI / logs."""
    cands = rank_candidates()
    best = best_first_principles()
    v = verdict()
    lines = []
    lines.append("=" * 78)
    lines.append("V5.36a — c_J(3_1) depuis SU(2)_135 : 10 méthodes")
    lines.append("=" * 78)
    lines.append(f"Cible (e^{{1,30}}) : c_J = {TARGET_C_J_3_1:.6f}")
    lines.append(f"β calibré V5.35b  : β = {BETA_CALIBRATED:+.4f}")
    lines.append("")
    lines.append(
        f"{'ID':<5} {'c_J':>10} {'err_cJ%':>9} {'β':>9} "
        f"{'err_β%':>9} {'FP':>4}  formule"
    )
    for c in cands:
        lines.append(
            f"{c.name:<5} {c.value:>10.4f} "
            f"{c.relative_error_cJ * 100:>8.2f}% "
            f"{c.beta:>+9.4f} "
            f"{c.relative_error_beta * 100:>8.2f}% "
            f"{'YES' if c.is_first_principles else 'no':>4}  {c.formula}"
        )
    lines.append("")
    if best is not None:
        lines.append(
            f"Meilleur first-principles : {best.name} ({best.formula})"
        )
        lines.append(
            f"  → c_J = {best.value:.4f}, β = {best.beta:+.4f}, "
            f"écart β = {best.relative_error_beta * 100:.2f} %"
        )
    else:
        lines.append("Aucun candidat first-principles.")
    lines.append(f"Verdict V5.36a : {v}")
    lines.append("=" * 78)
    lines.append("HONNÊTE :")
    lines.append("- Le code TQNT (M1) donne c_J(3_1) = 3,30 → β = -1,19 (écart 8 %).")
    lines.append("- Aucune définition SU(2)_135 first-principles ne reproduit 3,669.")
    lines.append("- β reste calibré (-1,30) en V5.35d ; la conjecture B3 (V5.36)")
    lines.append("  reposait sur c_J = 3,66 NON documenté dans le code.")
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_report())
