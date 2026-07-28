"""V5.26 — Tentative de derivation de Y_W (hypercharge faible) depuis A-polynome.

PROBLEME (V5.25 audit, commit b0d5a3a)
======================================
`theory/particle_classification.py:363 topological_hypercharge` retourne
plaquees Y_lepton = -1 et Y_quark = +1/3, indépendamment du nœud. Ce ne sont
pas des dérivations : ce sont des constantes du MS écrites en dur.

OBJECTIF V5.26
==============
Tester si la **structure** du A-polynome (racines, polytope de Newton, mesure
de Mahler, asymptotique, pentes) permet d'extraire de facon NON-CIRCULAIRE
la valeur Y_W pour chaque nœud individuel et pour des entrelacs (somme).

CONTRAINTES HONNETES
====================
1. La table `_A_POLY_TABLE` ne contient que 5 nœuds : 0_1, 3_1, 4_1, 5_1, 5_2.
   Pour les 15 autres, on utilise des **proxies bases sur le polynome de Jones**
   (degree-span, evaluation a t=-1, t=i...) clairement etiquetes comme proxies.
2. Le A-polynome ne contient **PAS** d'information de saveur (strangeness,
   charm, bottom). Par construction, il ne peut donc pas distinguer K+ (Y=1)
   de pi+ (Y=0) car les deux ont la meme topologie "u + d-bar".
   --> Pour les hadrons charmes/etranges/beaux, **aucun mecanisme topologique
       du A-polynome ne peut reproduire Y_PDG**. Resultat negatif attendu.
3. On documente honnetement chaque piste avec son verdict.

6 PISTES EVALUEES (toutes documentees, qu'elles marchent ou non)
================================================================
1. n_roots = degre en L du A-poly non-abelien
2. Polytope de Newton (vertices, aire, lattice points)
3. Mesure de Mahler M(A_K)
4. Asymptotique A_K(L=1, M) et A_K(L, M=1)
5. Pentes (slopes) — pour les nœuds toriques T(p,q)
6. Formule V5.26 finale apres analyse des pistes

VERDICT FINAL
=============
Voir `validate_against_pdg()` pour les statistiques. La conclusion est
qu'**aucune piste ne produit Y_W de premier principe** : l'information de
saveur (S, C, B') manque structurellement au A-polynome. On retient une
formule de meilleur ajustement honnete et on flag qu'elle est partielle.

References
----------
- Cooper, Culler, Gillet, Long, Shalen (1994) - definition A-polynome
- Garoufalidis (2004) - conjecture AJ
- Boyd (1998) - mesures de Mahler en topologie
- Champanerkar, Kofman (2005) - pentes des A-polynomes
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import sympy as sp

from theory.a_polynomial import (
    A_POLY_KNOT_NAMES,
    a_polynomial,
    a_polynomial_branches,
    a_polynomial_nonabelian,
)
from theory.jones import JONES_KNOT_NAMES, _JONES_TABLE

L_SYM, M_SYM = sp.symbols("L M", commutative=True)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _jones_features(knot_name: str) -> dict:
    """Proxy features from Jones polynomial for knots without A-poly entry.

    Honest disclaimer: these are NOT A-poly features. They are Jones-based
    surrogates used when the A-poly is unavailable. The downstream code is
    transparent about which knots use which source (see `feature_source`).
    """
    if knot_name not in _JONES_TABLE:
        raise ValueError(f"Unknown knot: {knot_name}")
    j = _JONES_TABLE[knot_name]
    if not j:
        return {
            "jones_span": 0,
            "jones_min": 0,
            "jones_max": 0,
            "jones_eval_t_minus_one": 0.0,  # determinant
            "jones_eval_t_i_abs": 0.0,
            "jones_n_terms": 0,
            "jones_coeff_sum": 0.0,
        }
    powers = sorted(j.keys())
    pmin, pmax = powers[0], powers[-1]
    # |V_K(-1)| = determinant
    det = abs(sum(c * ((-1) ** p) for p, c in j.items()))
    # |V_K(i)| relates to Arf invariant
    import cmath
    v_i = sum(c * (1j ** p) for p, c in j.items())
    v_i_abs = abs(v_i)
    return {
        "jones_span": pmax - pmin,
        "jones_min": pmin,
        "jones_max": pmax,
        "jones_eval_t_minus_one": float(det),
        "jones_eval_t_i_abs": float(v_i_abs),
        "jones_n_terms": len(j),
        "jones_coeff_sum": float(sum(j.values())),
    }


def _newton_polytope_features(expr: sp.Expr) -> dict:
    """Compute Newton polytope features of a 2-variable polynomial in (L, M).

    Returns vertices (as list of (deg_L, deg_M)), bounding box, area, and
    number of lattice points inside.
    """
    expanded = sp.expand(expr)
    # Build dict {(deg_L, deg_M): coeff}
    points: list[tuple[int, int]] = []
    if expanded == 0:
        return {"vertices": [], "n_vertices": 0, "bbox_area": 0, "n_lattice": 0,
                "deg_L_max": 0, "deg_L_min": 0, "deg_M_max": 0, "deg_M_min": 0}
    poly = sp.Poly(expanded, L_SYM, M_SYM)
    for monom, coeff in poly.terms():
        if coeff != 0:
            points.append((int(monom[0]), int(monom[1])))
    if not points:
        return {"vertices": [], "n_vertices": 0, "bbox_area": 0, "n_lattice": 0,
                "deg_L_max": 0, "deg_L_min": 0, "deg_M_max": 0, "deg_M_min": 0}
    deg_L = [p[0] for p in points]
    deg_M = [p[1] for p in points]
    bbox_area = (max(deg_L) - min(deg_L)) * (max(deg_M) - min(deg_M))
    # Convex hull via shoelace for 2D points (sympy)
    # Use a simple Graham-scan-like approach
    pts_sorted = sorted(set(points))
    if len(pts_sorted) < 3:
        hull = pts_sorted
        area = 0.0
    else:
        try:
            from scipy.spatial import ConvexHull
            import numpy as np
            arr = np.array(pts_sorted, dtype=float)
            hull_obj = ConvexHull(arr)
            hull = [tuple(arr[i]) for i in hull_obj.vertices]
            area = float(hull_obj.volume)  # 2D "volume" == area
        except Exception:
            hull = pts_sorted
            area = float(bbox_area)
    return {
        "vertices": hull,
        "n_vertices": len(hull),
        "bbox_area": int(bbox_area),
        "n_lattice": len(points),
        "deg_L_max": max(deg_L),
        "deg_L_min": min(deg_L),
        "deg_M_max": max(deg_M),
        "deg_M_min": min(deg_M),
    }


def _slope_features(expr: sp.Expr) -> dict:
    """Compute slope features from the A-polynomial.

    The "slope" of a boundary slope of a 2-bridge / torus knot is the
    quotient (deg_L / deg_M) for each edge of the Newton polytope.
    For T(p,q) torus knots, A = L + M^(2pq) gives slope 2pq.
    """
    expanded = sp.expand(expr)
    if expanded == 0:
        return {"slopes": [], "n_slopes": 0, "max_slope": 0.0, "min_slope": 0.0}
    poly = sp.Poly(expanded, L_SYM, M_SYM)
    points = []
    for monom, coeff in poly.terms():
        if coeff != 0:
            points.append((int(monom[0]), int(monom[1])))
    if len(points) < 2:
        return {"slopes": [], "n_slopes": 0, "max_slope": 0.0, "min_slope": 0.0}
    # Edges of convex hull
    try:
        from scipy.spatial import ConvexHull
        import numpy as np
        arr = np.array(sorted(set(points)), dtype=float)
        if len(arr) < 3:
            slopes = []
        else:
            hull = ConvexHull(arr)
            slopes: list[float] = []
            verts = arr[hull.vertices]
            n = len(verts)
            for i in range(n):
                a = verts[i]
                b = verts[(i + 1) % n]
                dL = b[0] - a[0]
                dM = b[1] - a[1]
                if abs(dL) > 1e-9:
                    slopes.append(float(dM / dL))
        if not slopes:
            return {"slopes": [], "n_slopes": 0, "max_slope": 0.0, "min_slope": 0.0}
        return {
            "slopes": slopes,
            "n_slopes": len(slopes),
            "max_slope": max(slopes),
            "min_slope": min(slopes),
        }
    except Exception:
        return {"slopes": [], "n_slopes": 0, "max_slope": 0.0, "min_slope": 0.0}


def _mahler_measure_approx(expr: sp.Expr, n_samples: int = 32) -> float:
    """Approximate logarithmic Mahler measure of a 2-variable polynomial.

    m(P) = (1/(2pi)^2) integral_{|L|=|M|=1} log|P(L, M)| dtheta_L dtheta_M

    We use a discrete Riemann sum on n_samples x n_samples grid on the
    bi-torus. Returns m(P) (the log Mahler measure).
    """
    import math
    expanded = sp.expand(expr)
    if expanded == 0:
        return 0.0
    P = sp.lambdify((L_SYM, M_SYM), expanded, modules=["numpy"])
    import numpy as np
    theta = np.linspace(0, 2 * math.pi, n_samples, endpoint=False)
    LL = np.exp(1j * theta)[:, None]
    MM = np.exp(1j * theta)[None, :]
    vals = np.abs(P(LL, MM))
    # Avoid log(0)
    vals = np.where(vals < 1e-15, 1e-15, vals)
    m = float(np.mean(np.log(vals)))
    return m


@lru_cache(maxsize=128)
def extract_a_poly_features(knot_name: str) -> dict:
    """Extract features from A-polynomial (or Jones proxy if A-poly unavailable).

    Returns a dict with the following keys:
        knot                : str — knot identifier
        feature_source      : 'a_polynomial' or 'jones_proxy'
        n_roots             : int — degree in L of A_na (number of non-abelian roots)
        newton_n_vertices   : int — vertices of Newton polytope
        newton_bbox_area    : int — bounding-box area
        newton_n_lattice    : int — lattice points
        newton_deg_L_max    : int
        newton_deg_M_max    : int
        n_slopes            : int
        max_slope           : float
        mahler_log          : float — log Mahler measure
        jones_span          : int — Jones polynomial span (always available)
        jones_det           : float — |V_K(-1)|
        jones_n_terms       : int

    For knots without A-polynomial entry, the A-poly-dependent fields
    are set to None / 0 and `feature_source = 'jones_proxy'`.
    """
    jones = _jones_features(knot_name)

    if knot_name in A_POLY_KNOT_NAMES:
        try:
            A_na = a_polynomial_nonabelian(knot_name)
            branches = a_polynomial_branches(knot_name)
            newton = _newton_polytope_features(A_na)
            slopes = _slope_features(A_na)
            mahler = _mahler_measure_approx(A_na)
            return {
                "knot": knot_name,
                "feature_source": "a_polynomial",
                "n_roots": int(branches["n_nonabelian_roots"]),
                "newton_n_vertices": newton["n_vertices"],
                "newton_bbox_area": newton["bbox_area"],
                "newton_n_lattice": newton["n_lattice"],
                "newton_deg_L_max": newton["deg_L_max"],
                "newton_deg_M_max": newton["deg_M_max"],
                "n_slopes": slopes["n_slopes"],
                "max_slope": slopes["max_slope"],
                "min_slope": slopes["min_slope"],
                "mahler_log": mahler,
                "jones_span": jones["jones_span"],
                "jones_det": jones["jones_eval_t_minus_one"],
                "jones_n_terms": jones["jones_n_terms"],
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            pass

    return {
        "knot": knot_name,
        "feature_source": "jones_proxy",
        "n_roots": None,
        "newton_n_vertices": 0,
        "newton_bbox_area": 0,
        "newton_n_lattice": 0,
        "newton_deg_L_max": 0,
        "newton_deg_M_max": 0,
        "n_slopes": 0,
        "max_slope": 0.0,
        "min_slope": 0.0,
        "mahler_log": 0.0,
        "jones_span": jones["jones_span"],
        "jones_det": jones["jones_eval_t_minus_one"],
        "jones_n_terms": jones["jones_n_terms"],
    }


# ---------------------------------------------------------------------------
# V5.26 candidate formulas
# ---------------------------------------------------------------------------
def _single_knot_y_v5p26(knot_name: str, sector_hint: str) -> float:
    """V5.26 hypercharge for an isolated knot.

    Mechanism retained after evaluation of the 6 pistes:
    --------------------------------------------------
    Piste 1 (n_roots) IS the only signal that survives correlation:
       n_roots(A_K) = 1 for torus T(2,q) (= leptonic / charged sector)
       n_roots(A_K) = 2 for hyperbolic 2-bridge (e.g. 4_1, 5_2)
       n_roots(A_K) = 0 for unknot

    For a single isolated knot we map (PDG convention Q = T_3 + Y/2 with
    doublet hypercharges Y_eL = -1, Y_qL = +1/3, calibrated to the V5
    catalog 3. Particules PDG sheet):
       n_roots = 0  --> Y_topo = 0       (unknot / pure wave / photon, Z, gluon)
       n_roots = 1  --> Y_topo = -1      (torus T(2,q) -> lepton-doublet Y)
       n_roots = 2  --> Y_topo = +1/3    (hyperbolic 2-bridge -> quark-doublet Y)
       n_roots >=3 --> Y_topo = +1       (highly composite -> hadron baseline)

    The map is calibrated so that the only Y values produced (0, -1, +1/3,
    +1) lie on the SM doublet grid and reproduce the PDG values for the
    isolated-knot particles (electron Y=-1, photon Y=0). Combining with
    the topological T_3 (sign of writhe) is consistent with GMN
    Q = T_3 + Y/2.

    HONESTY: This is a DISCRETE classifier on n_roots(A_K), not a continuous
    derivation. It only works for the 5 knots where A-poly is known
    (0_1, 3_1, 4_1, 5_1, 5_2). For the 15 other knots we use the Jones-span
    proxy mapped through the same n_roots-like discretisation.
    """
    feats = extract_a_poly_features(knot_name)
    if feats["feature_source"] == "a_polynomial":
        n = feats["n_roots"]
        if n is None or n == 0:
            return 0.0
        if n == 1:
            return -1.0
        if n == 2:
            return 1.0 / 3.0
        return 1.0
    # Proxy : detect torus T(2, q) vs hyperbolic 2-bridge from Jones structure.
    # Genuine T(2, q) torus knots have:
    #   - All Jones coefficients in {-1, +1}
    #   - Minimum Jones power > 0 (positive crossings only)
    # Hyperbolic 2-bridge knots have:
    #   - At least one |c| >= 2 in the Jones polynomial
    # The unknot (jones_span=0) returns 0.
    span = feats["jones_span"]
    if span == 0:
        return 0.0
    # Inspect raw Jones polynomial directly
    from theory.jones import _JONES_TABLE
    if knot_name in _JONES_TABLE:
        j = _JONES_TABLE[knot_name]
        max_abs = max(abs(c) for c in j.values())
        min_pow = min(j.keys())
        if max_abs == 1 and min_pow > 0:
            return -1.0  # torus T(2, q) like (-> lepton sector)
        return 1.0 / 3.0  # hyperbolic 2-bridge like (-> quark sector)
    return 1.0 / 3.0


def hypercharge_v5p26(knot_or_link: str, sector_hint: str = "auto") -> float:
    """V5.26 weak hypercharge Y_W from A-polynomial structure.

    Accepts either a single knot name (`"3_1"`) or a link configuration
    (`"3_1 + 7_2 + 7_2"`) and returns the topological Y_W prediction.

    For links (entrelacs), we sum single-knot Y_topo contributions. This
    matches the topological additivity of the U(1)_Y current along the
    component world-lines.

    Parameters
    ----------
    knot_or_link : str
        Single knot identifier (e.g. "3_1") or sum-of-knots (e.g.
        "3_1 + 7_2 + 7_2").
    sector_hint : str
        One of "auto", "lepton_L", "quark_L", "quark_R", "boson",
        "meson", "baryon". Used only for the V5.26 *single-knot* boson rule
        (n=0 -> 0). For multi-knot links the topological sum is returned
        unchanged.

    Returns
    -------
    Y : float
        Topological weak hypercharge (units of e).

    HONEST WARNING:
    - For boson_H (Higgs) the topological sum *typically* misses Y=+1/2
      because the Higgs doublet hypercharge has no purely topological
      analogue.
    - For mesons / baryons carrying strangeness or charm, Y_PDG depends on
      the FLAVOUR content (S, C, B'), which is NOT encoded in any
      topological invariant of the knot. Topological Y_V5.26 must
      systematically miss these cases.
    """
    import re
    pattern = re.compile(r"\d+_\d+")
    knots = pattern.findall(knot_or_link)
    if not knots:
        return 0.0
    if len(knots) == 1:
        return _single_knot_y_v5p26(knots[0], sector_hint)
    # Composite : sum topological Y contributions
    total = sum(_single_knot_y_v5p26(k, sector_hint) for k in knots)
    return total


# ---------------------------------------------------------------------------
# Pistes evaluation (for the validation script and the .tex)
# ---------------------------------------------------------------------------
@dataclass
class PisteResult:
    name: str
    description: str
    verdict: str  # "retenue", "rejetee", "partielle"
    correlation: Optional[float]
    notes: str


def evaluate_pistes() -> list[PisteResult]:
    """Return a structured summary of the 6 pistes evaluated for V5.26.

    The summary is hand-computed from the actual feature analysis (see
    docs/V5p26_hypercharge_a_polynomial.tex for the underlying tables).
    Re-running with new A-polynomials would change these verdicts.
    """
    return [
        PisteResult(
            name="P1_n_roots",
            description="n_roots(A_na) = degree-L of non-abelian A-poly factor",
            verdict="partielle",
            correlation=None,
            notes=(
                "Discrete signal: 0_1 -> 0 racine, torus T(2,q) -> 1 racine, "
                "hyperbolic 2-bridge -> 2 racines. Coincides with the "
                "sector split lepton/quark for the 5 knots with A-poly, "
                "but only via discrete classification, not first-principles "
                "derivation."
            ),
        ),
        PisteResult(
            name="P2_newton_polytope",
            description="Vertices, bounding-box area and lattice points of Newton polytope",
            verdict="rejetee",
            correlation=None,
            notes=(
                "Newton polytope area scales with knot complexity (genus, "
                "crossing number) — correlates with mass, not with Y_W. "
                "No mapping to the discrete 1/6 / 1/3 / 1/2 hypercharge "
                "grid emerges."
            ),
        ),
        PisteResult(
            name="P3_mahler_measure",
            description="Logarithmic Mahler measure m(A_K)",
            verdict="rejetee",
            correlation=None,
            notes=(
                "Mahler measure of A_K is bounded below by hyperbolic "
                "volume / 2pi (Boyd 1998). It tracks the hyperbolic volume, "
                "not Y_W. For torus knots m(A) = 0 (cyclotomic), which "
                "would collapse e- and quark on the same Y, contradiction."
            ),
        ),
        PisteResult(
            name="P4_asymptotics",
            description="Asymptotic A_K(L=1, M) and A_K(L, M=1)",
            verdict="rejetee",
            correlation=None,
            notes=(
                "A_K(L=1, M) is identically zero by construction (abelian "
                "factor). A_K(L, M=1) gives a 1D polynomial in L whose "
                "roots are units of finite order — no clean map to Y_W."
            ),
        ),
        PisteResult(
            name="P5_torus_slopes",
            description="Boundary slopes of A-polynomial Newton polytope",
            verdict="partielle",
            correlation=None,
            notes=(
                "For T(2,q), slope = 2q. Linear in q. Could encode T_3 "
                "(parity of q), but does not produce 1/6 or -1/2 "
                "directly. Used as auxiliary feature only."
            ),
        ),
        PisteResult(
            name="P6_combined",
            description="V5.26 retained formula: discrete n_roots classifier",
            verdict="retenue",
            correlation=None,
            notes=(
                "Y_V5.26(K) = f(n_roots(A_na)). Single-knot map: "
                "0 -> 0, 1 -> -1, 2 -> +1/3, >=3 -> +1 (PDG-doublet "
                "convention). Reproduces electron Y=-1 exact, fails on "
                "flavored hadrons by construction: A-poly has no flavour "
                "quantum number, so Y_meson and Y_baryon depend on "
                "(S, C, B') content invisible to topology."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Validation against PDG
# ---------------------------------------------------------------------------
def validate_against_pdg() -> dict:
    """Validate hypercharge_v5p26 against the V5 catalog PDG sheet.

    Loads `docs/TQNT_catalog_V5_diffusion.xlsx` and compares Y_V5.26 to the
    PDG reference (left-handed doublet convention Q = T_3 + Y/2).

    Returns
    -------
    stats : dict
        Keys: n, n_exact, n_within_1_6, n_lepton_match, n_quark_match,
        n_boson_match, n_hadron_match, per_particle (list of dicts).
    """
    import re
    from pathlib import Path
    import openpyxl

    # PDG table mirrored from scripts/V5_validation/v5p25_hypercharge_67_PDG.py
    PDG = {
        "e⁻": (-0.5, -1.0, -1.0, "lepton_L"),
        "μ⁻": (-0.5, -1.0, -1.0, "lepton_L"),
        "τ⁻": (-0.5, -1.0, -1.0, "lepton_L"),
        "γ":  (0.0, 0.0, 0.0, "boson"),
        "W±": (1.0, 0.0, 1.0, "boson"),
        "Z⁰": (0.0, 0.0, 0.0, "boson"),
        "H":  (-0.5, 1.0, 0.0, "boson"),
        "t":  (0.0, 4/3, 2/3, "quark_R"),
        # Mesons / baryons follow PDG convention
        "π⁰": (0.0, 0.0, 0.0, "meson"), "π±": (1.0, 0.0, 1.0, "meson"),
        "K±": (0.5, 1.0, 1.0, "meson"), "K⁰": (-0.5, 1.0, 0.0, "meson"),
        "η":  (0.0, 0.0, 0.0, "meson"), "η'": (0.0, 0.0, 0.0, "meson"),
        "η_c(1S)": (0.0, 0.0, 0.0, "meson"), "η_c(2S)": (0.0, 0.0, 0.0, "meson"),
        "J/ψ": (0.0, 0.0, 0.0, "meson"), "ψ(2S)": (0.0, 0.0, 0.0, "meson"),
        "χ_c0(1P)": (0.0, 0.0, 0.0, "meson"), "χ_c1(1P)": (0.0, 0.0, 0.0, "meson"),
        "χ_c2(1P)": (0.0, 0.0, 0.0, "meson"),
        "η_b(1S)": (0.0, 0.0, 0.0, "meson"), "Υ(1S)": (0.0, 0.0, 0.0, "meson"),
        "Υ(2S)": (0.0, 0.0, 0.0, "meson"),
        "χ_b0(1P)": (0.0, 0.0, 0.0, "meson"), "χ_b1(1P)": (0.0, 0.0, 0.0, "meson"),
        "χ_b2(1P)": (0.0, 0.0, 0.0, "meson"),
        "D⁰": (-0.5, 1.0, 0.0, "meson"), "D±": (0.5, 1.0, 1.0, "meson"),
        "D_s±": (0.0, 2.0, 1.0, "meson"),
        "D*(2007)⁰": (-0.5, 1.0, 0.0, "meson"),
        "D*(2010)±": (0.5, 1.0, 1.0, "meson"),
        "B±": (0.5, 1.0, 1.0, "meson"), "B⁰": (-0.5, 1.0, 0.0, "meson"),
        "B_s⁰": (0.0, 0.0, 0.0, "meson"), "B_c±": (0.0, 2.0, 1.0, "meson"),
        "ρ": (0.0, 0.0, 0.0, "meson"), "ω": (0.0, 0.0, 0.0, "meson"),
        "φ": (0.0, 0.0, 0.0, "meson"), "K*(892)": (-0.5, 1.0, 0.0, "meson"),
        "f₀(980)": (0.0, 0.0, 0.0, "meson"),
        "a₂(1320)": (1.0, 0.0, 1.0, "meson"), "f₂(1270)": (0.0, 0.0, 0.0, "meson"),
        "p": (0.5, 1.0, 1.0, "baryon"), "n": (-0.5, 1.0, 0.0, "baryon"),
        "Λ": (0.0, 0.0, 0.0, "baryon"),
        "Σ⁺": (1.0, 0.0, 1.0, "baryon"), "Σ⁰": (0.0, 0.0, 0.0, "baryon"),
        "Σ⁻": (-1.0, 0.0, -1.0, "baryon"),
        "Ξ⁰": (0.5, -1.0, 0.0, "baryon"), "Ξ⁻": (-0.5, -1.0, -1.0, "baryon"),
        "Δ": (0.5, 1.0, 1.0, "baryon"),
        "Σ*⁺": (1.0, 0.0, 1.0, "baryon"), "Ξ*⁰": (0.5, -1.0, 0.0, "baryon"),
        "Ω⁻": (0.0, -2.0, -1.0, "baryon"),
        "Λ_c⁺": (0.0, 2.0, 1.0, "baryon"), "Σ_c⁺⁺": (1.0, 2.0, 2.0, "baryon"),
        "Ξ_c⁺": (0.5, 1.0, 1.0, "baryon"), "Ω_c⁰": (0.0, 0.0, 0.0, "baryon"),
        "Λ_b⁰": (0.0, 0.0, 0.0, "baryon"), "Ξ_b⁰": (0.5, -1.0, 0.0, "baryon"),
        "Ω_b⁻": (0.0, -2.0, -1.0, "baryon"),
        "X(3872)": (0.0, 0.0, 0.0, "exotic"),
        "Z_c(3900)": (1.0, 0.0, 1.0, "exotic"),
        "X(6900)": (0.0, 0.0, 0.0, "exotic"),
        "P_c(4380)": (0.5, 1.0, 1.0, "exotic"),
        "P_c(4450)": (0.5, 1.0, 1.0, "exotic"),
    }

    root = Path(__file__).resolve().parents[1]
    catalog = root / "docs" / "TQNT_catalog_V5_diffusion.xlsx"
    wb = openpyxl.load_workbook(catalog, data_only=True)
    ws = wb["3. Particules PDG"]
    pattern = re.compile(r"\d+_\d+")

    rows: list[dict] = []
    for r in range(3, ws.max_row + 1):
        raw = [c.value for c in ws[r]]
        if not raw or not raw[0]:
            continue
        name = str(raw[0]).strip()
        if name not in PDG:
            continue
        cfg = str(raw[3]).strip() if raw[3] else ""
        knots = pattern.findall(cfg)
        if not knots:
            continue
        t3, y_pdg, q_pdg, sector = PDG[name]
        y_v5p26 = hypercharge_v5p26(cfg, sector)
        residual = y_v5p26 - y_pdg
        rows.append({
            "name": name,
            "config": cfg,
            "sector": sector,
            "t3_pdg": t3,
            "y_pdg": y_pdg,
            "q_pdg": q_pdg,
            "y_v5p26": y_v5p26,
            "residual": residual,
            "exact": abs(residual) < 1e-6,
            "within_1_6": abs(round(y_v5p26 * 6) - round(y_pdg * 6)) < 0.5,
        })

    n = len(rows)
    n_exact = sum(1 for r in rows if r["exact"])
    n_within = sum(1 for r in rows if r["within_1_6"])
    per_sector = {}
    for r in rows:
        s = r["sector"]
        per_sector.setdefault(s, {"n": 0, "exact": 0})
        per_sector[s]["n"] += 1
        if r["exact"]:
            per_sector[s]["exact"] += 1
    return {
        "n": n,
        "n_exact": n_exact,
        "n_within_1_6": n_within,
        "per_sector": per_sector,
        "per_particle": rows,
    }


__all__ = [
    "extract_a_poly_features",
    "hypercharge_v5p26",
    "evaluate_pistes",
    "validate_against_pdg",
    "PisteResult",
]
