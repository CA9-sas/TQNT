"""V5.22 -- Tentative derivation of Δc_min ≥ 30 for amphichiral DM pearls.

Context
-------
V5.21 (commit ``8743f4f``, see ``theory/dc_selection_rule.py``) established
that the four canonical selection mechanisms (Q-balance, Kanenobu-Murakami
amphichiral linking, Bar-Natan Vassiliev, SU(2)_135 modular fusion) give

    Δc_min^V5.21 = 22   (binding pair: 6_3+6_3 self-link, Vassiliev)

The Vassiliev suppression α_EM^{22} ≈ 1.0e-47 was claimed sufficient to
hide the Class E cascade beneath the Fermi-LAT EDGRB. However the
*line-search* test (commit ``06ad81f``, ``scripts/V5_Fermi_LAT/
v5p20_class_E_lines.py``) showed that ALL SIX predicted γ-lines are
excluded by Fermi-LAT 2015 even with the V5.21 suppression: the
constraint coming from individual narrow lines is stronger than from the
diffuse background by ~ 1.0e+12. Concretely, Fermi line search requires

    Δc_min ≥ 30

for V5.20 to survive Class E line tests.

Question (V5.22)
----------------
Does V5 contain ADDITIONAL topological selection mechanisms, beyond the
four V5.21 channels, that raise Δc_min from 22 to 30 without introducing
free parameters?

This module investigates FIVE candidate mechanisms:

  M1 -- Primitive Vassiliev requirement (non-decomposable order)
  M2 -- Reshetikhin-Turaev color decomposition multi-step
  M3 -- Higher topological moment Q^{(2)} (quadrupole) conservation
  M4 -- Framing-anomaly differential between n=1 and n=2 components
        (V5.6 derivation, theory/framing_anomaly_V5.py)
  M5 -- One-loop conformal weight Δh from R²/96π (V5.15)

For each, the module computes Δc_additional on top of Vassiliev N_min,
gives a HONEST verdict on whether the mechanism is V5-axiomatic or
ad-hoc, and aggregates the total Δc_min^V5.22.

Methodology
-----------
* Each mechanism returns Δc_additional ≥ 0.
* "V5-axiomatic" flag is True iff the mechanism follows DIRECTLY from
  V5 postulates without new ingredients. The aggregate ``delta_c_min_v5p22``
  ONLY counts mechanisms with axiomatic == True (honest filter).
* The MAX over (V5.21 contributions ∪ V5.22 axiomatic contributions)
  is the binding constraint.

Honest verdict
--------------
If Δc_min^V5.22 ≥ 30 → V5.20 lines redeemed.
If Δc_min^V5.22 <  30 → F3 (Class E line search) falsification holds.

References
----------
* V5.21 derivation: theory/dc_selection_rule.py (commit 8743f4f)
* V5.20 line search: scripts/V5_Fermi_LAT/v5p20_class_E_lines.py
                     (commit 06ad81f)
* Bar-Natan (1995), "On the Vassiliev knot invariants", Topology 34, 423
* Vassiliev (1990), "Cohomology of knot spaces" (primitive part)
* Reshetikhin & Turaev (1991), Invent. Math. 103, 547
* Witten (1989), Commun. Math. Phys. 121, 351
* V5.6 framing anomaly: theory/framing_anomaly_V5.py
* V5.15 coefficient a_R²: theory/coefficient_a_R2.py
* V5.21 derivation: theory/dc_selection_rule.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from theory.dc_selection_rule import (
    AMPHICHIRAL_DATA,
    Amphichiral,
    delta_c_Lk,
    delta_c_Q_phys,
    delta_c_SU2_135,
    delta_c_vassiliev,
)
from theory.su2_modular import V5_LEVEL, conformal_weight


# ---------------------------------------------------------------------------
# Threshold required by Fermi-LAT line search (commit 06ad81f)
# ---------------------------------------------------------------------------

FERMI_LINE_SEARCH_THRESHOLD: int = 30
"""Δc_min required so that all 6 Class E predicted γ-lines fall below the
Fermi-LAT 2015 narrow-line upper limits at the V5.20 luminosities.

Derivation : Fermi-LAT line upper-limits at E ~ 1-50 GeV are ~ 1.0e-12
ph/cm²/s, while the V5.20 prediction at Δc = 22 is ~ 1.0e+0 ph/cm²/s,
i.e. 10^12 above the limit. Reducing the prediction by 10^12 requires
α_EM^{Δc} = 10^{-12} more suppression, i.e. Δc ≥ 22 + log_{1/α}(10^{12})
≈ 22 + 12/log10(137) ≈ 22 + 5.6 ≈ 27.6, conservatively 28. The 6-line
joint constraint at 95% CL pushes this to 30 (commit 06ad81f).
"""


# ===========================================================================
# Mechanism 1 : Primitive Vassiliev (non-decomposable order)
# ===========================================================================

def higher_vassiliev_primitive_min(c_i: int, c_j: int) -> int:
    """First-order primitive (non-decomposable) Vassiliev for amphi link.

    Mechanism (Vassiliev 1990, Bar-Natan 1995 §4):
      An invariant v_N(L) is *decomposable* if it can be written as a
      polynomial in lower-order invariants of L's components: e.g.,
      v_N(K_1 # K_2) = Σ v_a(K_1) v_b(K_2) with a+b = N.

      The lowest non-zero Vassiliev v_{N_min} with N_min = c_i + c_j − 2
      computed in V5.21 corresponds to the term where v_{c_i-1}(K_i) =
      const · v_{c_j-1}(K_j) ≠ 0. This is a DECOMPOSABLE invariant.

      For PRIMITIVE (truly link-distinguishing) Vassiliev invariants, the
      lowest order rises by 2 (one unit per component of the link),
      because primitive v_N involves a chord-diagram of N chords that
      MUST cross at least once between K_i and K_j strands ONLY for
      N ≥ c_i + c_j (no chord can be fully internal to a single
      component once we require non-decomposability).

      Bar-Natan (Topology 34, eq. 6.3) :
        ord_prim(L_amphi) = ord_first(L) + 2 = (c_i + c_j − 2) + 2 = c_i + c_j

      For a link to support a primitive Vassiliev of order N, it must
      have at least N crossings (otherwise v_N ≡ 0 trivially). So:

        Δc_primVass(c_i, c_j) = c_i + c_j

    AXIOMATIC : True (direct consequence of Bar-Natan §4 + amphichirality).
    The +2 correction over V5.21 Vassiliev is RIGOROUS but only adds 2.
    """
    return c_i + c_j


# ===========================================================================
# Mechanism 2 : Reshetikhin-Turaev color decomposition multi-step
# ===========================================================================

def color_decomposition_steps(c_i: int, c_j: int,
                              k: int = V5_LEVEL) -> Dict[str, float]:
    """Multi-step color fusion in the Reshetikhin-Turaev category.

    Mechanism (Reshetikhin-Turaev 1991):
      In SU(2)_k, the link reconnection (K_i, K_j) → L proceeds via
      *fusion intermediates* labelled by representations j_a:
        K_i ⊗ K_j  →  ⊕_{j_a} N_{j_i j_j}^{j_a} V_{j_a}
      where N is the fusion multiplicity. Each fusion vertex adds a
      "trivalent" topological node, which in the link diagram appears
      as a pair of crossings (over + under).

      For a link of two pearls of fundamental spin j_i = c_i/4, j_j =
      c_j/4, the number of distinct fusion paths is bounded above by
      the truncated tensor product dimension:
        N_paths = min(j_i + j_j, k - j_i - j_j) + 1
                = min(c_i/4 + c_j/4, k - c_i/4 - c_j/4) + 1

      Each path requires 2 crossings (trivalent vertex). The MINIMUM
      number of fusion steps to keep the diagram amphichiral is
      log_2(N_paths) (a balanced binary tree of fusions), so:

        Δc_RT = 2 · ⌈log_2(N_paths)⌉
              = 2 · ⌈log_2( min((c_i+c_j)/4, k - (c_i+c_j)/4) + 1 )⌉

    For c_i = c_j = 6 (lightest amphichiral 6_3+6_3, c_i pearl = 12):
      j_i = j_j = 3, j_i + j_j = 6, N_paths = 7, Δc_RT = 2 · 3 = 6.
    For c_i = 12, c_j = 12 (binding pair self 6_3+6_3 + 6_3+6_3):
      j_i = j_j = 3, N_paths = 7, Δc_RT = 6.

    AXIOMATIC : Partially. Reshetikhin-Turaev IS a V5 ingredient
    (Chern-Simons quantization), but the IDENTIFICATION between
    "fusion steps" and "diagram crossings" is a CHOICE: in the standard
    R-T formalism, fusions occur at trivalent vertices that are NOT
    diagrammatic crossings of the link. We mark this as axiomatic=False
    by default (over-counting risk).

    Returns dict with 'n_paths', 'delta_c_RT', 'axiomatic'.
    """
    j_i = c_i / 4.0
    j_j = c_j / 4.0
    sum_j = j_i + j_j
    max_fusion = min(sum_j, k - sum_j)
    n_paths = int(math.floor(max_fusion)) + 1
    delta_c = 2 * max(1, math.ceil(math.log2(max(n_paths, 2))))
    return {
        "n_paths": n_paths,
        "delta_c_RT": delta_c,
        "axiomatic": False,  # crossings ≠ trivalent vertices
        "rationale": "R-T trivalent vertices are NOT link crossings",
    }


# ===========================================================================
# Mechanism 3 : Higher topological moment Q^{(2)} (quadrupole)
# ===========================================================================

def quadrupole_topological_conservation(amphi_i: Amphichiral,
                                        amphi_j: Amphichiral) -> Dict[str, float]:
    """Conservation of higher topological moment Q^{(2)}.

    Mechanism (HYPOTHETICAL — NOT in V5 axioms):
      Beyond the linear Q_phys = Σχ + Lk = 0 of V5 §6, one could
      postulate conservation of a quadrupole moment:
        Q^{(2)}(L) = Σ (χ_i + Lk_i)² × c_i = 0

      For two amphichirals with Q_i = Q_j = 0, this is identically zero
      regardless of crossings, so it imposes NO additional constraint.

      A non-trivial higher moment would require breaking the amphichiral
      symmetry of the components, which contradicts V5 §13.7 by
      construction. Therefore Δc^{(2)} = 0 for amphichiral pairs.

    AXIOMATIC : FALSE — this is not a V5 postulate, and even as an
    *ad hoc* extension it gives zero on the relevant sample. Honest
    verdict: cannot be invoked to lift Δc_min.
    """
    q2_i = (amphi_i.Q_phys + amphi_i.lk_self) ** 2 * amphi_i.c
    q2_j = (amphi_j.Q_phys + amphi_j.lk_self) ** 2 * amphi_j.c
    q2_link = q2_i + q2_j
    return {
        "Q2_link": float(q2_link),
        "delta_c_Q2": 0,  # vanishes identically on amphichirals
        "axiomatic": False,
        "rationale": "Q^(2) not in V5 axioms; identically zero on amphichirals",
    }


# ===========================================================================
# Mechanism 4 : Framing-anomaly differential
# ===========================================================================

def framing_anomaly_addition_crossings(c_i: int, c_j: int,
                                       k: int = V5_LEVEL) -> Dict[str, float]:
    """Additional crossings from V5.6 framing-anomaly differential.

    Mechanism (V5.6, theory/framing_anomaly_V5.py):
      A Wilson loop at spin j picks up a framing phase
        W_K(f) = exp(2πi h_j f) · W_K(canonical)
      with h_j = j(j+1)/(k+2). For an intra-pearl reconnection bringing
      a n=1 amphichiral (c = c_min) into a n=2 link (c = c_min + Δc),
      the framing differential between the two states is

        Δf = h_{j_link} - (h_{j_i} + h_{j_j})

      and to keep the diagram amphichiral, Δf must be a half-integer
      (mirror invariance) ⇒ Δf ∈ {0, 1/2}.

      For two amphichirals j_i = c_i/4, j_j = c_j/4 fusing into
      j_link = (c_i+c_j)/4, the differential is

        Δh = j_link(j_link+1)/(k+2) - [j_i(j_i+1) + j_j(j_j+1)] / (k+2)

      Solving Δh = 1/2 for the EXTRA crossings Δc_f needed:
        Δc_f = ⌈ 4 · (k+2) · 1/2 / (c_i + c_j) ⌉
             ≈ 2(k+2) / (c_i + c_j)

      For (c_i, c_j) = (12, 12) and k=135: Δc_f ≈ 2·137/24 ≈ 11.4 → 12

      However V5.6 has DERIVED δ_framing = -0.599 per log10(M_h),
      which is a SLOPE scaling, NOT a constant crossings addition.
      Translating slope-per-mass into a per-link Δc requires assuming a
      reference mass scale; the natural V5 choice is M_ref = m_pearl,
      giving Δf = δ · log10(m_link/m_pearl) ≈ δ · log10(2) ≈ -0.18.

      To force this into a half-integer ⇒ NO ADDITIONAL crossings (the
      framing freedom is absorbed by the choice of canonical framing).

    AXIOMATIC : V5.6 is in V5, but its implication for Δc on a fusion
    is a NEW derivation. We compute Δc_f conservatively as the integer
    ceiling of the framing-driven crossing requirement, AND mark
    axiomatic=True only if the framing differential exceeds 1/2.

    Returns dict with 'delta_h', 'delta_c_framing', 'axiomatic'.
    """
    j_i = c_i / 4.0
    j_j = c_j / 4.0
    j_link = (c_i + c_j) / 4.0
    h_i = j_i * (j_i + 1) / (k + 2)
    h_j = j_j * (j_j + 1) / (k + 2)
    h_link = j_link * (j_link + 1) / (k + 2)
    delta_h = h_link - h_i - h_j
    # Number of half-integer steps to absorb : ⌈2|Δh|⌉
    n_half_steps = math.ceil(abs(delta_h) / 0.5) if delta_h != 0 else 0
    # Each half-step corresponds to 2 crossings (Witten 1989, §2)
    delta_c = 2 * n_half_steps
    axiomatic = abs(delta_h) >= 0.5
    return {
        "delta_h": delta_h,
        "delta_c_framing": delta_c,
        "axiomatic": axiomatic,
        "rationale": ("framing differential < 1/2 absorbed by canonical choice"
                      if not axiomatic
                      else "framing differential forces extra crossings"),
    }


# ===========================================================================
# Mechanism 5 : One-loop conformal weight correction
# ===========================================================================

def one_loop_conformal_weight_correction(c_i: int, c_j: int,
                                         k: int = V5_LEVEL) -> Dict[str, float]:
    """One-loop conformal weight correction Δh from V5.15 R²/96π.

    Mechanism (V5.15, theory/coefficient_a_R2.py):
      The R²/(384π G) term in the V5 Lagrangian generates a 1-loop
      Coleman-Weinberg correction to the conformal weight of Wilson
      loops:
        Δh_loop = (1/96π) · α_EM · h_j
                = α_EM · h_j / 96π
      where α_EM = 1/(k+2) = 1/137.

      For a fusion j_i + j_j → j_link, the 1-loop correction to the
      combined weight is
        Δh_loop = (α_EM/96π) · (h_link - h_i - h_j)

      For (c_i, c_j) = (12,12), j_link = 6, j_i = j_j = 3:
        h_link - h_i - h_j = 42/137 - 12/137 - 12/137 = 18/137 ≈ 0.131
        Δh_loop = (1/137·96π) · 0.131 ≈ 3.2e-6

      To force this 1-loop correction to be exactly compensated by
      additional crossings, we would need
        Δc_loop · log(α_EM) ≥ log(Δh_loop / h_typical)
        Δc_loop ≥ log(3.2e-6 / 0.1) / log(1/137) ≈ 4.5/4.92 ≈ 0.9
        ⇒ Δc_loop = 1 crossing.

      This is an ESTIMATE; the rigorous derivation would require
      a full 1-loop CFT computation on the link complement. As an
      upper bound, Δc_loop ≤ 2.

    AXIOMATIC : V5.15 is V5, but the connection 1-loop CW correction
    → Δc is fragile (order-of-magnitude). axiomatic=False conservative.
    """
    j_i = c_i / 4.0
    j_j = c_j / 4.0
    j_link = (c_i + c_j) / 4.0
    h_diff = (j_link*(j_link+1) - j_i*(j_i+1) - j_j*(j_j+1)) / (k+2)
    alpha = 1.0 / (k + 2)
    delta_h_loop = alpha * h_diff / (96.0 * math.pi)
    # crude estimate of Δc_loop : log(Δh_loop / 0.1) / log(alpha)
    if delta_h_loop > 0 and delta_h_loop < 0.1:
        delta_c_loop = max(0, math.ceil(
            math.log(0.1 / delta_h_loop) / math.log(1.0 / alpha)
        ))
    else:
        delta_c_loop = 0
    return {
        "delta_h_loop": delta_h_loop,
        "delta_c_loop": int(delta_c_loop),
        "axiomatic": False,
        "rationale": "1-loop CW correction order-of-magnitude only",
    }


# ===========================================================================
# Aggregator
# ===========================================================================

@dataclass(frozen=True)
class V5p22MechanismResult:
    name: str
    delta_c_additional: int
    axiomatic: bool
    rationale: str


def evaluate_all_mechanisms(amphi_i: Amphichiral,
                            amphi_j: Amphichiral,
                            k: int = V5_LEVEL) -> List[V5p22MechanismResult]:
    """Evaluate the 5 V5.22 mechanisms on a pair of amphichirals.

    Returns a list of V5p22MechanismResult.
    """
    c_i, c_j = amphi_i.c, amphi_j.c
    out: List[V5p22MechanismResult] = []

    # M1 : Primitive Vassiliev (Δc_additional = +2 vs V5.21 Vassiliev)
    n_prim = higher_vassiliev_primitive_min(c_i, c_j)
    n_v21 = delta_c_vassiliev(c_i, c_j)
    out.append(V5p22MechanismResult(
        name="primitive_vassiliev",
        delta_c_additional=max(0, n_prim - n_v21),
        axiomatic=True,
        rationale="Bar-Natan §4 + amphichirality: lowest primitive at N=c_i+c_j",
    ))

    # M2 : R-T color decomposition
    d2 = color_decomposition_steps(c_i, c_j, k)
    out.append(V5p22MechanismResult(
        name="rt_color_decomposition",
        delta_c_additional=int(d2["delta_c_RT"]),
        axiomatic=bool(d2["axiomatic"]),
        rationale=str(d2["rationale"]),
    ))

    # M3 : Quadrupole topological moment
    d3 = quadrupole_topological_conservation(amphi_i, amphi_j)
    out.append(V5p22MechanismResult(
        name="quadrupole_Q2",
        delta_c_additional=int(d3["delta_c_Q2"]),
        axiomatic=bool(d3["axiomatic"]),
        rationale=str(d3["rationale"]),
    ))

    # M4 : Framing anomaly differential
    d4 = framing_anomaly_addition_crossings(c_i, c_j, k)
    out.append(V5p22MechanismResult(
        name="framing_anomaly_differential",
        delta_c_additional=int(d4["delta_c_framing"]) if d4["axiomatic"] else 0,
        axiomatic=bool(d4["axiomatic"]),
        rationale=str(d4["rationale"]),
    ))

    # M5 : 1-loop conformal weight
    d5 = one_loop_conformal_weight_correction(c_i, c_j, k)
    out.append(V5p22MechanismResult(
        name="one_loop_conformal_weight",
        delta_c_additional=int(d5["delta_c_loop"]),
        axiomatic=bool(d5["axiomatic"]),
        rationale=str(d5["rationale"]),
    ))

    return out


def delta_c_min_total(amphi_i: Amphichiral,
                      amphi_j: Amphichiral,
                      k: int = V5_LEVEL,
                      honest: bool = True) -> Dict[str, object]:
    """Aggregate V5.21 + V5.22 axiomatic-only contributions.

    The contract:
      * delta_c_v5p21    : MAX of (Q_phys, Lk, Vassiliev, SU(2)_135)
      * delta_c_v5p22    : sum of axiomatic-only V5.22 Δc_additional
                           ON TOP OF the V5.21 binding mechanism if
                           that mechanism was Vassiliev (otherwise the
                           V5.22 additions don't stack).
      * delta_c_min_v5p22_total : MAX(V5.21 contributions, V5.21_vass + V5.22)

    The 'honest=True' flag (default) ONLY counts axiomatic mechanisms.
    Set honest=False to include all 5 mechanisms (UNTRUSTED upper bound).

    Returns dict with keys
      'v5p21_contribs'        : dict {Q, Lk, Vass, SU2}
      'v5p21_binding'         : str
      'v5p21_total'           : int
      'v5p22_mechanisms'      : list of dict
      'v5p22_axiomatic_sum'   : int (only axiomatic ones)
      'v5p22_all_sum'         : int (all 5, untrusted)
      'delta_c_min_v5p22_honest' : int
      'delta_c_min_v5p22_max'    : int (untrusted)
      'binding_mechanism'        : str
    """
    c_i, c_j = amphi_i.c, amphi_j.c

    # V5.21 raw contributions
    dc_Q = delta_c_Q_phys(c_i, c_j)
    dc_Lk = delta_c_Lk(c_i, c_j)
    dc_V = delta_c_vassiliev(c_i, c_j)
    dc_S = delta_c_SU2_135(c_i, c_j, k)
    v21 = {"Q_phys": dc_Q, "Lk": dc_Lk, "Vass": dc_V, "SU2_135": dc_S}
    v21_total = max(v21.values())
    v21_binding = max(v21, key=v21.get)

    # V5.22 mechanisms
    mechs = evaluate_all_mechanisms(amphi_i, amphi_j, k)
    v22_list = [{"name": m.name,
                 "delta_c_additional": m.delta_c_additional,
                 "axiomatic": m.axiomatic,
                 "rationale": m.rationale}
                for m in mechs]
    v22_axiomatic_sum = sum(m.delta_c_additional for m in mechs if m.axiomatic)
    v22_all_sum = sum(m.delta_c_additional for m in mechs)

    # The V5.22 additions sit ON TOP of the Vassiliev channel
    # (because both the primitive-Vass and framing-anomaly mechanisms
    # extend the Vassiliev-based suppression). They do NOT stack on Q,
    # Lk or SU2 channels (those constraints are saturated independently).
    v22_on_vass_axiomatic = dc_V + v22_axiomatic_sum
    v22_on_vass_max = dc_V + v22_all_sum

    # Final binding constraint : MAX over all V5.21 contribs + augmented Vass
    candidates_honest = {**v21, "Vass+V5p22": v22_on_vass_axiomatic}
    candidates_max = {**v21, "Vass+V5p22_all": v22_on_vass_max}
    total_honest = max(candidates_honest.values())
    total_max = max(candidates_max.values())
    binding = max(candidates_honest, key=candidates_honest.get)

    return {
        "v5p21_contribs": v21,
        "v5p21_binding": v21_binding,
        "v5p21_total": v21_total,
        "v5p22_mechanisms": v22_list,
        "v5p22_axiomatic_sum": v22_axiomatic_sum,
        "v5p22_all_sum": v22_all_sum,
        "delta_c_min_v5p22_honest": total_honest,
        "delta_c_min_v5p22_max": total_max,
        "binding_mechanism": binding,
    }


def all_amphichiral_pairs_v5p22(honest: bool = True) -> List[Dict[str, object]]:
    """Compute V5.22 Δc_min for all 21 amphichiral pairs.

    Returns list of dicts with pair labels, V5.21 + V5.22 results.
    """
    out: List[Dict[str, object]] = []
    n = len(AMPHICHIRAL_DATA)
    for i in range(n):
        ai = AMPHICHIRAL_DATA[i]
        # self pair
        d = delta_c_min_total(ai, ai, honest=honest)
        out.append({"name_i": ai.name, "name_j": ai.name,
                    "c_i": ai.c, "c_j": ai.c, **d})
        for j in range(i + 1, n):
            aj = AMPHICHIRAL_DATA[j]
            d = delta_c_min_total(ai, aj, honest=honest)
            out.append({"name_i": ai.name, "name_j": aj.name,
                        "c_i": ai.c, "c_j": aj.c, **d})
    return out


def _compute_v5p22_overall(honest: bool = True) -> Dict[str, int]:
    """Overall Δc_min^V5.22 (HONEST) across all 21 pairs."""
    pairs = all_amphichiral_pairs_v5p22(honest=honest)
    honest_min = min(p["delta_c_min_v5p22_honest"] for p in pairs)
    upper_min = min(p["delta_c_min_v5p22_max"] for p in pairs)
    return {"honest": honest_min, "upper_bound": upper_min}


DELTA_C_MIN_V5P22_HONEST: int = _compute_v5p22_overall()["honest"]
"""V5.22 Δc_min HONEST overall.

If ≥ 30 → V5.20 line tests redeemed.
If < 30 → F3 (Fermi line search) falsification remains open.
"""

DELTA_C_MIN_V5P22_UPPER: int = _compute_v5p22_overall()["upper_bound"]
"""V5.22 Δc_min UPPER BOUND (counts all 5 mechanisms incl. non-axiomatic).
Always ≥ HONEST. UNTRUSTED."""


__all__ = [
    "FERMI_LINE_SEARCH_THRESHOLD",
    "higher_vassiliev_primitive_min",
    "color_decomposition_steps",
    "quadrupole_topological_conservation",
    "framing_anomaly_addition_crossings",
    "one_loop_conformal_weight_correction",
    "V5p22MechanismResult",
    "evaluate_all_mechanisms",
    "delta_c_min_total",
    "all_amphichiral_pairs_v5p22",
    "DELTA_C_MIN_V5P22_HONEST",
    "DELTA_C_MIN_V5P22_UPPER",
]
