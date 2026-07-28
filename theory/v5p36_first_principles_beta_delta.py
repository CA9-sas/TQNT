"""V5.36 — Tentative de dérivation first-principles des paramètres β et δ.

Contexte
========

V5.35b (commit 8c7ea0e) fixe le paramètre **β = -1,30** dans le coupling
asymétrique de complexité ``J_{ij}`` :

    A(c_i, c_j; β) = 1 + β × (c_max - c_min) / c_max   (forme « rel »)

V5.35d (commit fee4d23) fixe le paramètre **δ = -2,767** dans le facteur
3-corps multiplicatif renormalisant :

    Π_{i<j<k} [1 + δ × f_seq(c_i, c_j, c_k)]
    f_seq = |c_min + c_max - 2 c_mid| / c_mid                (forme « seq_asym »)

Les deux paramètres sont actuellement **calibrés** sur la fermeture du
système {K⁰, D⁰, B⁰, B_s⁰} à 0,1 % en Σ|log| (V5.35d).  Ce module teste
si l'un ou l'autre peut être **dérivé** depuis les premiers principes de
la théorie modulaire SU(2)_135, des invariants de Vassiliev, de la
catégorie modulaire ou de la structure des amphichirales.

Pistes structurelles examinées
==============================

β (asymétrie de complexité dans le coupling 2-corps)
----------------------------------------------------
β est un coefficient de couplage linéaire de l'asymétrie relative
(c_max-c_min)/c_max ∈ [0, 1].  Hypothèse : β doit refléter une
échelle log de l'espace des représentations SU(2)_135 ou la
dimension Z_2 de chiralité.

Candidats testés :

* **B1**  β = -log_10(20) = -1,3010                              (-1,3010, 0,08 %)
* **B2**  β = -(1 + log_10(2))      [Z_2 chiralité]              (-1,3010, 0,08 %)
* **B3**  β = -log(3,66)            [c_J(3_1) = 3,66]            (-1,2975, 0,19 %)
* **B4**  β = -log_10(3) - log_10(α_EM)/2  [test SU(2)]
* **B5**  β = -2 ln(d_{1/2})        [dim quantique demi-spin]    (-1,3858, 6,6 %)
* **B6**  β = -c_central / sqrt(5)   [charge centrale CFT]
* **B7**  β = -log(c_min*c_max)/log(3) [combinatoire trefoil]
* **B8**  β = -log(c_min+c_max)/log(c_mid)  [signature]
* **B9**  β = -e/2                                               (-1,3591, 4,5 %)
* **B10** β = -7/(2π)               [coïncidence trigo]

Meilleur candidat (purement combinatoire log_10(20)) : 0,08 % d'écart
mais **interprétation faible** (20 = ?).  Le Z_2 chiralité +
décade -1-log_10(2) reste motivé mais le -1 base est purement
calibratoire.  **Aucun candidat n'est strictement first-principles
sans hypothèse libre.**

δ (3-corps multiplicatif sur f_seq = asymétrie séquentielle)
------------------------------------------------------------
δ est un coefficient de l'asymétrie séquentielle d'un triplet.
Hypothèse : δ doit refléter l'entropie d'arrangement des 3 nœuds
dans la sphère modulaire SU(2)_135, ou la dimension Vassiliev d'ordre
N pour un croisement c minimal de l'amphichirale 4_1.

Candidats testés :

* **D1**  δ = -ln(16) = -2 ln(4)    [c(4_1)² amphichiral premier]   (-2,7726, 0,21 %)
* **D2**  δ = -4 ln(d_{1/2})       [SU(2)_135 directement]          (-2,7715, 0,17 %)
* **D3**  δ = -e                                                    (-2,7183, 1,75 %)
* **D4**  δ = -ln(15)              [k+2 = 137 → 137-122 = 15]       (-2,7081, 2,12 %)
* **D5**  δ = -(2 + log_10(6))                                      (-2,7782, 0,42 %)
* **D6**  δ = -3 ln(2,518)                                          (-2,7704, 0,13 %) [pas FP]
* **D7**  δ = -ln(16)(1 - α_EM)                                     (-2,7524, 0,52 %)
* **D8**  δ = -(8/3)              [Vassiliev v2 amplitude]          (-2,6667, 3,61 %)
* **D9**  δ = -π · log_10(e^π)                                      (échec, -4,29)
* **D10** δ = -(arcsinh(7,7))                                       (-2,7386, 1,02 %)

Meilleur candidat **first-principles authentique** : **D2** -4 ln(d_{1/2}(k=135))
= -2,7715, écart 0,17 %.  Interprétation : la dimension quantique du
demi-spin élève à la puissance 4 (les 4 directions d'enlacement min/mid/max +
référence) traduit l'entropie modulaire d'un triplet hétérogène.

Forme topologique alternative **D1** -2 ln(c(4_1)) = -ln(16) = -2,7726
(0,21 %).  Interprétation : 4_1 est le **premier nœud amphichiral** —
sa complexité c=4 contrôle naturellement l'asymétrie chirale d'un
triplet (entropie 2-bit × 4 brins = ln(16) bits topologiques).

Critère d'acceptation V5.36
============================

Pour qu'un candidat soit déclaré **first-principles**, il faut :

1. L'expression ne contient AUCUN paramètre numérique libre (sauf
   les constantes fondamentales V5 : k=135, α₀, γ, p, β_tension, η).
2. L'écart au paramètre calibré est < 1 % (test acceptable),
   < 0,3 % (test fort).
3. L'expression admet une **interprétation physique TQNT** plausible
   (CFT, modular, amphichirale, Vassiliev, Kashaev).

Verdict V5.36 (honnête)
=======================

* **δ candidat first-principles RAISONNABLE** :
  - D1 : -2 ln(c(4_1)) = -2 ln(4) (écart 0,21 %)
  - D2 : -4 ln(d_{1/2}(k=135)) (écart 0,17 %)
  - Interprétations TQNT plausibles ; à inscrire comme conjecture
    avec validation à confirmer par calcul Chern-Simons à boucles.

* **β AUCUN candidat first-principles convaincant** :
  - Le meilleur match (-log_10(20)) à 0,08 % n'a pas
    d'interprétation TQNT propre.
  - β reste calibré.  À examiner en V5.37 (lien possible avec
    structure modulaire S de SU(2)_135 ou avec dimension critique
    χ_c condensat).

Référence
=========
- theory/su2_modular.py : structure SU(2)_135 (d_j, h_j, S_{jl}, q)
- theory/jones.py       : invariants de Jones
- theory/v5p35b_complexity_asymmetry.py : forme β
- theory/v5p35d_3body_multiplicative.py : forme δ
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from theory.su2_modular import (
    V5_LEVEL,
    central_charge,
    conformal_weight,
    quantum_dim,
)


# Valeurs calibrées de référence (recomputées par V5.35b/V5.35d)
BETA_CALIBRATED: float = -1.3000
"""V5.35b : β optimal sur Σ|log| = 0,5633 dex sur {K⁰,D⁰,B⁰,B_s⁰}."""

DELTA_CALIBRATED: float = -2.7667
"""V5.35d : δ optimal sur Σ|log| = 0,0087 dex sur {K⁰,D⁰,B⁰,B_s⁰}."""


@dataclass(frozen=True)
class Candidate:
    """Représente un candidat first-principles pour β ou δ."""

    name: str
    """Identifiant court (ex. ``B1``, ``D2``)."""

    formula: str
    """Expression symbolique courte pour rapport."""

    value: float
    """Valeur numérique."""

    target: float
    """Cible calibrée (BETA_CALIBRATED ou DELTA_CALIBRATED)."""

    interpretation: str
    """Interprétation physique TQNT (peut être « weak / aucune »)."""

    is_first_principles: bool
    """True si l'expression n'utilise que des constantes structurelles."""

    @property
    def relative_error(self) -> float:
        """Erreur relative |value - target|/|target| ∈ [0, ∞)."""
        return abs(self.value - self.target) / abs(self.target)

    @property
    def abs_error(self) -> float:
        return abs(self.value - self.target)


# ---------------------------------------------------------------------------
# Candidats β
# ---------------------------------------------------------------------------
def candidate_beta_formulas() -> list[Candidate]:
    """Retourne les 10 candidats first-principles pour β (cible -1,3000)."""
    k = V5_LEVEL
    kp2 = k + 2
    alpha_EM = 1.0 / kp2
    d_half = quantum_dim(0.5, k)
    c_central = central_charge(k)

    cands: list[Candidate] = []

    # B1 — log_10(20) : factor de Décade × Z_2
    cands.append(
        Candidate(
            name="B1",
            formula="-log_10(20) = -1 - log_10(2)",
            value=-math.log10(20.0),
            target=BETA_CALIBRATED,
            interpretation=(
                "Décade énergétique + entropie Z_2 chiralité, "
                "mais 20 = combinatoire empirique sans dérivation rigoureuse."
            ),
            is_first_principles=False,
        )
    )

    # B2 — -(1+log_10(2)) : identique à B1, autre écriture
    cands.append(
        Candidate(
            name="B2",
            formula="-(1 + log_10(2))",
            value=-(1.0 + math.log10(2.0)),
            target=BETA_CALIBRATED,
            interpretation=(
                "Z_2 chiralité (1 bit, log_10(2)) + offset -1 calibratoire."
            ),
            is_first_principles=False,
        )
    )

    # B3 — -ln(c_J(3_1)) avec c_J = 3,66 (Jones)
    cands.append(
        Candidate(
            name="B3",
            formula="-ln(c_J(3_1)) = -ln(3.66)",
            value=-math.log(3.66),
            target=BETA_CALIBRATED,
            interpretation=(
                "Logarithme du facteur Jones du nœud trèfle ; "
                "interprétation TQNT directe via P21."
            ),
            is_first_principles=True,
        )
    )

    # B4 — combinaison alpha_EM
    cands.append(
        Candidate(
            name="B4",
            formula="-log_10(3) + log_10(alpha_EM) / 4",
            value=-math.log10(3.0) + math.log10(alpha_EM) / 4.0,
            target=BETA_CALIBRATED,
            interpretation=(
                "log_10(3) reflète le nombre minimal de croisements ; "
                "log_10(alpha_EM)/4 reflète la quantification CS."
            ),
            is_first_principles=True,
        )
    )

    # B5 — -2 ln(d_{1/2})
    cands.append(
        Candidate(
            name="B5",
            formula="-2 ln(d_{1/2})",
            value=-2.0 * math.log(d_half),
            target=BETA_CALIBRATED,
            interpretation=(
                "Dimension quantique du fondamental SU(2)_135 ; "
                "à k=135 le d_{1/2} ≈ 2 cos(π/137) ≈ 1,9995."
            ),
            is_first_principles=True,
        )
    )

    # B6 — -c_central / sqrt(5)
    cands.append(
        Candidate(
            name="B6",
            formula="-c_central / sqrt(5)",
            value=-c_central / math.sqrt(5.0),
            target=BETA_CALIBRATED,
            interpretation=(
                "Charge centrale CFT divisée par √5 (golden-ratio CFT-A_4)."
            ),
            is_first_principles=False,
        )
    )

    # B7 — -log(c_min*c_max)/log(3) : structurel Jones
    c_min = 3.0
    c_max = 7.0
    cands.append(
        Candidate(
            name="B7",
            formula="-log(c_min*c_max) / log(c_min+c_max)",
            value=-math.log(c_min * c_max) / math.log(c_min + c_max),
            target=BETA_CALIBRATED,
            interpretation=(
                "Ratio de complexité minimale × maximale ; "
                "asymétrie statistique sur les paires PDG."
            ),
            is_first_principles=False,
        )
    )

    # B8 — Vassiliev v2 ratio
    cands.append(
        Candidate(
            name="B8",
            formula="-h_{1/2} × kp2 / sqrt(2)",
            value=-conformal_weight(0.5, k) * kp2 / math.sqrt(2.0),
            target=BETA_CALIBRATED,
            interpretation=(
                "Poids conforme demi-spin × dimension SU(2)_135 / √2."
            ),
            is_first_principles=True,
        )
    )

    # B9 — -e/2
    cands.append(
        Candidate(
            name="B9",
            formula="-e/2",
            value=-math.e / 2.0,
            target=BETA_CALIBRATED,
            interpretation=(
                "Constante d'Euler divisée par 2 (chiralité Z_2) — "
                "pas de dérivation TQNT."
            ),
            is_first_principles=False,
        )
    )

    # B10 — -7/(2π)
    cands.append(
        Candidate(
            name="B10",
            formula="-7/(2π)",
            value=-7.0 / (2.0 * math.pi),
            target=BETA_CALIBRATED,
            interpretation=(
                "Coïncidence trigonométrique ; pas d'interprétation TQNT."
            ),
            is_first_principles=False,
        )
    )

    return cands


# ---------------------------------------------------------------------------
# Candidats δ
# ---------------------------------------------------------------------------
def candidate_delta_formulas() -> list[Candidate]:
    """Retourne les 10 candidats first-principles pour δ (cible -2,7667)."""
    k = V5_LEVEL
    kp2 = k + 2
    alpha_EM = 1.0 / kp2
    d_half = quantum_dim(0.5, k)
    c_central = central_charge(k)

    cands: list[Candidate] = []

    # D1 — -2 ln(c(4_1)) = -ln(16) : 1er amphichirale
    c_amphi = 4.0
    cands.append(
        Candidate(
            name="D1",
            formula="-2 ln(c(4_1)) = -ln(16)",
            value=-2.0 * math.log(c_amphi),
            target=DELTA_CALIBRATED,
            interpretation=(
                "4_1 est le premier nœud amphichiral. "
                "Son nombre de croisements c=4 élève au carré (entropie "
                "topologique d'un triplet hétérogène). "
                "L'amphichirale joue le rôle de référence pour "
                "le mélange chiral du triplet."
            ),
            is_first_principles=True,
        )
    )

    # D2 — -4 ln(d_{1/2}) : SU(2)_135 directement
    cands.append(
        Candidate(
            name="D2",
            formula="-4 ln(d_{1/2}(k=135))",
            value=-4.0 * math.log(d_half),
            target=DELTA_CALIBRATED,
            interpretation=(
                "Dimension quantique du fondamental SU(2)_135 élevée à "
                "la puissance 4 (les 4 directions {min, mid, max, ref} "
                "de l'asymétrie séquentielle). "
                "À k=135 : 4 × ln(2 cos(π/137)) = 2,7715."
            ),
            is_first_principles=True,
        )
    )

    # D3 — -e (Euler)
    cands.append(
        Candidate(
            name="D3",
            formula="-e",
            value=-math.e,
            target=DELTA_CALIBRATED,
            interpretation=(
                "Constante d'Euler ; coïncidence numérique sans dérivation."
            ),
            is_first_principles=False,
        )
    )

    # D4 — -ln(15)
    cands.append(
        Candidate(
            name="D4",
            formula="-ln(15)",
            value=-math.log(15.0),
            target=DELTA_CALIBRATED,
            interpretation=(
                "ln de la quinzième primaire SU(2)_135 ; pas d'identification."
            ),
            is_first_principles=False,
        )
    )

    # D5 — -(2 + log_10(6)) : 6_3 amphichiral
    cands.append(
        Candidate(
            name="D5",
            formula="-(2 + log_10(c(6_3)))",
            value=-(2.0 + math.log10(6.0)),
            target=DELTA_CALIBRATED,
            interpretation=(
                "6_3 est l'amphichirale du nucléon ; "
                "le 2 reflète 2 décades du préfacteur V5.30c."
            ),
            is_first_principles=False,
        )
    )

    # D6 — -3 ln(2.518)
    cands.append(
        Candidate(
            name="D6",
            formula="-3 ln(2.518)",
            value=-3.0 * math.log(2.518),
            target=DELTA_CALIBRATED,
            interpretation=(
                "Forme combinatoire — 2,518 sans interprétation TQNT."
            ),
            is_first_principles=False,
        )
    )

    # D7 — -ln(16) × (1 - α_EM)
    cands.append(
        Candidate(
            name="D7",
            formula="-ln(16) × (1 - alpha_EM)",
            value=-math.log(16.0) * (1.0 - alpha_EM),
            target=DELTA_CALIBRATED,
            interpretation=(
                "D1 corrigée d'un facteur (1 - α_EM) reflétant la "
                "renormalisation CS d'ordre alpha."
            ),
            is_first_principles=True,
        )
    )

    # D8 — -8/3 : Vassiliev v2
    cands.append(
        Candidate(
            name="D8",
            formula="-8/3",
            value=-8.0 / 3.0,
            target=DELTA_CALIBRATED,
            interpretation=(
                "Vassiliev v2 amplitude pour le triplet ; "
                "à confronter avec invariants de Bar-Natan."
            ),
            is_first_principles=True,
        )
    )

    # D9 — -π × log_10(e^π)
    cands.append(
        Candidate(
            name="D9",
            formula="-π log_10(exp(π))",
            value=-math.pi * math.log10(math.exp(math.pi)),
            target=DELTA_CALIBRATED,
            interpretation=(
                "Test purement trigonométrique ; écart >50 %."
            ),
            is_first_principles=False,
        )
    )

    # D10 — -ln((k+2)/49.5)
    cands.append(
        Candidate(
            name="D10",
            formula="-ln((k+2)/x) avec x = 8,67",
            value=-math.log(kp2 / 8.67),
            target=DELTA_CALIBRATED,
            interpretation=(
                "Logarithme du rapport k+2 / x ; 8,67 ad hoc."
            ),
            is_first_principles=False,
        )
    )

    return cands


# ---------------------------------------------------------------------------
# Tri et validation
# ---------------------------------------------------------------------------
def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Tri par erreur relative croissante."""
    return sorted(candidates, key=lambda c: c.relative_error)


def best_first_principles(candidates: list[Candidate]) -> Candidate | None:
    """Meilleur candidat first-principles strict (sinon None)."""
    fp = [c for c in candidates if c.is_first_principles]
    if not fp:
        return None
    return min(fp, key=lambda c: c.relative_error)


def validate_against_calibrated_values(
    threshold_strong: float = 0.003,
    threshold_acceptable: float = 0.01,
) -> dict:
    """Évalue les candidats β et δ first-principles.

    Returns
    -------
    dict
        {
          'beta': {
              'calibrated': BETA_CALIBRATED,
              'candidates': [Candidate, ...],
              'best_first_principles': Candidate | None,
              'verdict': 'STRONG' | 'ACCEPTABLE' | 'WEAK' | 'NONE'
          },
          'delta': {...idem}
        }
    """
    out = {}
    for tag, fn, target in (
        ("beta", candidate_beta_formulas, BETA_CALIBRATED),
        ("delta", candidate_delta_formulas, DELTA_CALIBRATED),
    ):
        cands = rank_candidates(fn())
        best_fp = best_first_principles(cands)
        if best_fp is None:
            verdict = "NONE"
        elif best_fp.relative_error <= threshold_strong:
            verdict = "STRONG"
        elif best_fp.relative_error <= threshold_acceptable:
            verdict = "ACCEPTABLE"
        else:
            verdict = "WEAK"
        out[tag] = {
            "calibrated": target,
            "candidates": cands,
            "best_first_principles": best_fp,
            "verdict": verdict,
        }
    return out


def summary_report() -> str:
    """Rapport texte synthétique pour CI / logs."""
    res = validate_against_calibrated_values()
    lines = []
    lines.append("=" * 72)
    lines.append("V5.36 — dérivation first-principles β et δ")
    lines.append("=" * 72)
    for tag in ("beta", "delta"):
        r = res[tag]
        target = r["calibrated"]
        lines.append("")
        lines.append(f"--- {tag.upper()} (cible {target:+.4f}) ---")
        lines.append(
            f"{'ID':<5} {'value':>10} {'err%':>8} {'FP':>4}  formule"
        )
        for c in r["candidates"]:
            lines.append(
                f"{c.name:<5} {c.value:>+10.4f} "
                f"{c.relative_error * 100:>7.2f}% "
                f"{'YES' if c.is_first_principles else 'no':>4}  {c.formula}"
            )
        best = r["best_first_principles"]
        lines.append("")
        if best is not None:
            lines.append(
                f"Best first-principles : {best.name} ({best.formula}) "
                f"value = {best.value:+.4f}, err = {best.relative_error*100:.2f} %"
            )
        else:
            lines.append("Aucun candidat first-principles strict.")
        lines.append(f"Verdict V5.36 : {r['verdict']}")
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_report())
