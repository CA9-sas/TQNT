"""
Theorie Nodale -- Listing des noeuds simples jusqu'a 40 croisements

Enumere les noeuds toriques T(p,q) avec c = (p-1)*q <= 40 et combine
avec les 20 noeuds non-toriques du catalogue existant. Calcule les
proprietes physiques (masse, spin, charge, champs E/B) et applique
les regles P16 (exclusion) et P17 (consolidation).
"""
from __future__ import annotations

import math

import numpy as np

from theory.jones import (
    JONES_KNOT_NAMES,
    JonesPolynomial,
    jones_determinant,
    jones_polynomial,
    jones_power_center,
    jones_retraction_factor,
    jones_span,
)
from theory.knot_catalog import _torus_knot_curve, knot_curve
from theory.abraham import ETA as ETA_ABRAHAM
from theory.postulates import C, GAMMA, HBAR, R_0

# --- Constantes P15 (duree de vie) ---
_NU_WEAK = 7.75
_LAMBDA_F_WEAK = 5.0
_TAU_MU = 2.197e-6  # s (duree de vie du muon)

# Calibration tau_P_weak (lazy)
_TAU_P_WEAK: float | None = None


def _get_tau_p_weak() -> float:
    """Calibre tau_P du canal faible sur le muon (P15)."""
    global _TAU_P_WEAK
    if _TAU_P_WEAK is not None:
        return _TAU_P_WEAK
    from theory.multi_knot import cornell_equilibrium, SM_ASSIGNMENTS
    mu_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == "mu-")
    eq_mu = cornell_equilibrium(mu_cfg["knots"], mu_cfg["k"])
    n_mu = len(mu_cfg["knots"])
    _TAU_P_WEAK = (
        _TAU_MU * eq_mu["mass_raw"] ** _NU_WEAK
        / np.exp(-_LAMBDA_F_WEAK * n_mu)
    )
    return _TAU_P_WEAK


def bare_lifetime(rho_J: float) -> float:
    """Duree de vie d'un noeud isole (n=1, k=1, canal faible P15).

    tau = tau_P * exp(-lambda_f) / M_raw^nu
    avec M_raw = rho_J * (1 + F(1)/R_0) = 1.1 * rho_J.

    Retourne tau en secondes.
    """
    if rho_J <= 0:
        return float("inf")
    tau_P = _get_tau_p_weak()
    M_raw = rho_J * (1.0 + 1.0 / R_0)  # F(1) = 1
    return tau_P * np.exp(-_LAMBDA_F_WEAK) / M_raw ** _NU_WEAK


# ---------------------------------------------------------------------------
# Polynome de Jones pour T(2,n)
# ---------------------------------------------------------------------------

def torus_jones_2q(n: int) -> JonesPolynomial:
    """Polynome de Jones de T(2,n) pour n impair >= 3.

    Formule derivee du pattern verifie sur 3_1, 5_1, 7_1, 9_1 :
        V(t) = t^k + sum_{j=0}^{n-2} (-1)^j * t^{k+2+j}
    avec k = (n-1)/2.
    """
    if n < 3 or n % 2 == 0:
        raise ValueError(f"n doit etre impair >= 3, recu {n}")
    k = (n - 1) // 2
    poly: JonesPolynomial = {k: 1.0}
    for j in range(n - 1):
        poly[k + 2 + j] = (-1.0) ** j
    return poly


# ---------------------------------------------------------------------------
# Determinant des noeuds toriques (formule analytique)
# ---------------------------------------------------------------------------

def torus_alexander_det(p: int, q: int) -> int:
    """Determinant du noeud torique T(p,q).

    Resultat analytique de |Delta_{T(p,q)}(-1)| :
    - p pair, q impair : det = q
    - p impair, q pair : det = p
    - p et q impairs   : det = 1
    """
    if p % 2 == 0:
        return q
    elif q % 2 == 0:
        return p
    else:
        return 1


# ---------------------------------------------------------------------------
# rho_J pour T(p,q) general
# ---------------------------------------------------------------------------

def torus_rho_j(p: int, q: int) -> float:
    """Facteur de retractation rho_J pour T(p,q).

    rho_J = span + GAMMA * det
    avec span = (p-1)*q (nombre de croisements = span pour noeuds toriques)
    et det = torus_alexander_det(p,q).
    """
    span = (p - 1) * q
    det = torus_alexander_det(p, q)
    return float(span) + GAMMA * float(det)


# ---------------------------------------------------------------------------
# Charge topologique pour T(2,n) (analytique)
# ---------------------------------------------------------------------------

def _torus_2q_charge(n: int) -> float:
    """Charge topologique Q = (p_min + p_max) / span pour T(2,n).

    Pour T(2,n) : p_min = k = (n-1)/2, p_max = k+n = (3n-1)/2, span = n.
    => Q = (2n-1)/n.
    """
    return (2 * n - 1) / n


# ---------------------------------------------------------------------------
# Enumeration des noeuds toriques
# ---------------------------------------------------------------------------

def enumerate_torus_knots(max_crossing: int = 40) -> list[dict]:
    """Enumere tous les T(p,q) avec 2 <= p < q, gcd(p,q) = 1,
    c = (p-1)*q <= max_crossing.

    Retourne liste de dicts : {p, q, name, crossing_number}.
    """
    knots = []
    for p in range(2, max_crossing + 1):
        for q in range(p + 1, max_crossing + 1):
            if math.gcd(p, q) != 1:
                continue
            c = (p - 1) * q
            if c > max_crossing:
                break
            knots.append({
                "p": p,
                "q": q,
                "name": f"T({p},{q})",
                "crossing_number": c,
            })
    return knots


# ---------------------------------------------------------------------------
# Geometrie de Frenet a partir d'une courbe (sans passer par knot_curve)
# ---------------------------------------------------------------------------

def _frenet_from_curve(pts: np.ndarray) -> dict:
    """Calcule kappa(t) et tau(t) via le repere de Frenet-Serret.

    Meme algorithme que abraham.frenet_geometry mais prend des points
    directement au lieu d'un nom de noeud.
    """
    # Supprimer le point final (doublon pour courbe fermee)
    pts = pts[:-1]
    N = len(pts)
    dt = 2 * np.pi / N

    p_next = np.roll(pts, -1, axis=0)
    p_prev = np.roll(pts, 1, axis=0)
    p_next2 = np.roll(pts, -2, axis=0)
    p_prev2 = np.roll(pts, 2, axis=0)

    dr = (p_next - p_prev) / (2 * dt)
    ddr = (p_next - 2 * pts + p_prev) / dt**2
    dddr = (p_next2 - 2 * p_next + 2 * p_prev - p_prev2) / (2 * dt**3)

    speed = np.linalg.norm(dr, axis=1)

    cross = np.cross(dr, ddr)
    cross_norm = np.linalg.norm(cross, axis=1)
    kappa = cross_norm / (speed**3 + 1e-30)

    triple = np.sum(cross * dddr, axis=1)
    tau = triple / (cross_norm**2 + 1e-30)

    ds = speed * dt
    arc_length = np.concatenate([[0.0], np.cumsum(ds[:-1])])

    return {
        "kappa": kappa,
        "tau": tau,
        "speed": speed,
        "arc_length": arc_length,
        "z": pts[:, 2],
        "ds": ds,
    }


def _geometric_integrals_from_curve(pts: np.ndarray) -> dict:
    """Integrales geometriques a partir d'une courbe."""
    geo = _frenet_from_curve(pts)
    kappa = geo["kappa"]
    tau = geo["tau"]
    ds = geo["ds"]

    L = float(np.sum(ds))
    int_kappa = float(np.sum(kappa * ds))
    int_abs_tau = float(np.sum(np.abs(tau) * ds))

    return {"L": L, "int_kappa": int_kappa, "int_abs_tau": int_abs_tau}


def _sigma_phi_from_curve(pts: np.ndarray) -> dict:
    """Tension EM entre extrema de z(s), a partir d'une courbe."""
    geo = _frenet_from_curve(pts)
    z = geo["z"]
    kappa = geo["kappa"]
    tau = geo["tau"]
    ds = geo["ds"]

    z_prev = np.roll(z, 1)
    z_next = np.roll(z, -1)
    is_max = (z > z_prev) & (z > z_next)
    is_min = (z < z_prev) & (z < z_next)
    extrema_idx = np.where(is_max | is_min)[0]

    n_osc = len(extrema_idx)
    if n_osc < 2:
        return {"sigma_phi_E": 0.0, "sigma_phi_B": 0.0, "n_osc": n_osc}

    kappa_ds = kappa * ds
    abs_tau_ds = np.abs(tau) * ds

    sp_E = 0.0
    sp_B = 0.0
    for i in range(len(extrema_idx) - 1):
        i0 = extrema_idx[i]
        i1 = extrema_idx[i + 1]
        sp_E += np.sum(kappa_ds[i0:i1])
        sp_B += np.sum(abs_tau_ds[i0:i1])

    return {
        "sigma_phi_E": float(sp_E),
        "sigma_phi_B": float(sp_B),
        "n_osc": n_osc,
    }


# ---------------------------------------------------------------------------
# Proprietes d'un noeud
# ---------------------------------------------------------------------------

# Noms des noeuds toriques dans le catalogue existant (pour identification)
_CATALOG_TORUS = {
    "3_1": (2, 3),
    "5_1": (2, 5),
    "7_1": (2, 7),
    "9_1": (2, 9),
}


def _get_curve(name: str, p: int | None, q: int | None,
               n_points: int) -> np.ndarray:
    """Obtient la courbe 3D pour un noeud (catalogue ou torique)."""
    if name in JONES_KNOT_NAMES:
        return knot_curve(name, n_points)
    if p is not None and q is not None:
        return _torus_knot_curve(p, q, n_points)
    raise ValueError(f"Pas de courbe disponible pour {name}")


def compute_knot_properties(
    name: str,
    p: int | None = None,
    q: int | None = None,
    n_points: int = 5000,
) -> dict:
    """Calcule les proprietes physiques d'un noeud.

    Parameters
    ----------
    name : nom du noeud (ex. "3_1", "T(2,11)")
    p, q : parametres toriques (si applicable)
    n_points : points de discretisation pour la geometrie

    Returns
    -------
    dict avec cles : name, crossing_number, p, q, is_torus,
        rho_J, mass_raw, mass_mev, charge, spin, braid_index,
        sigma_E, sigma_B, int_kappa, int_abs_tau, L,
        r_excl, has_jones
    """
    is_torus = p is not None and q is not None
    is_catalog = name in JONES_KNOT_NAMES

    # --- Nombre de croisements ---
    if is_torus:
        crossing = (p - 1) * q
    elif is_catalog:
        crossing = int(name.split("_")[0])
    else:
        crossing = 0

    # --- Jones polynomial et invariants ---
    has_jones = False
    rho_J = 0.0
    charge = float("nan")
    det = 0

    if is_catalog:
        poly = jones_polynomial(name)
        rho_J = jones_retraction_factor(poly)
        charge = jones_power_center(poly)
        det = jones_determinant(poly)
        has_jones = True
    elif is_torus and p == 2:
        # T(2,q) : formule analytique exacte
        poly = torus_jones_2q(q)
        rho_J = jones_retraction_factor(poly)
        charge = jones_power_center(poly)
        det = jones_determinant(poly)
        has_jones = True
    elif is_torus:
        # T(p,q) avec p >= 3 : pas de Jones exact, calcul direct de rho_J
        rho_J = torus_rho_j(p, q)
        det = torus_alexander_det(p, q)
        charge = float("nan")
        has_jones = False

    # --- Spin et braid index ---
    if is_catalog:
        from theory.jones import braid_index
        bi = braid_index(name)
    elif is_torus:
        bi = min(p, q)
    else:
        bi = 1
    spin = (bi - 1) / 2.0

    # --- Masse d'Abraham nodale ---
    # m_K = eta * rho_J avec eta = m_e / rho_J(3_1) (formule tendue)
    mass_raw = rho_J  # ALPHA = 1.0
    mass_mev = ETA_ABRAHAM * rho_J

    # --- Geometrie (courbure, torsion, tensions EM) ---
    sigma_E = 0.0
    sigma_B = 0.0
    int_kappa = 0.0
    int_abs_tau = 0.0
    L = 0.0

    try:
        pts = _get_curve(name, p, q, n_points)
        gi = _geometric_integrals_from_curve(pts)
        sp = _sigma_phi_from_curve(pts)

        L = gi["L"]
        int_kappa = gi["int_kappa"]
        int_abs_tau = gi["int_abs_tau"]
        sigma_E = sp["sigma_phi_E"]
        sigma_B = sp["sigma_phi_B"]
    except Exception:
        pass

    # --- Rayon d'exclusion P16 ---
    r_excl = (HBAR * C / rho_J) if rho_J > 0 else float("inf")

    # --- Duree de vie P15 (noeud isole, n=1, k=1) ---
    tau = bare_lifetime(rho_J)
    log10_tau = math.log10(tau) if tau > 0 and not math.isinf(tau) else float("inf")

    return {
        "name": name,
        "crossing_number": crossing,
        "p": p,
        "q": q,
        "is_torus": is_torus,
        "rho_J": rho_J,
        "determinant": det,
        "mass_raw": mass_raw,
        "mass_mev": mass_mev,
        "charge": charge,
        "spin": spin,
        "braid_index": bi,
        "sigma_E": sigma_E,
        "sigma_B": sigma_B,
        "int_kappa": int_kappa,
        "int_abs_tau": int_abs_tau,
        "L": L,
        "r_excl": r_excl,
        "tau_bare": tau,
        "log10_tau": log10_tau,
        "has_jones": has_jones,
    }


# ---------------------------------------------------------------------------
# P16 : exclusion topologique
# ---------------------------------------------------------------------------

def apply_p16_exclusion(
    knots: list[dict],
    threshold: float = 20.0,
) -> list[dict]:
    """Ajoute un flag 'excluded' base sur P16.

    Un noeud est exclu si son rayon d'exclusion r_excl est inferieur
    au seuil (resolution topologique minimale hbar*c ~ 20 unites maillage).
    """
    for k in knots:
        k["excluded"] = k["r_excl"] < threshold
    return knots


# ---------------------------------------------------------------------------
# P17 : doublets consolides
# ---------------------------------------------------------------------------

def find_p17_doublets(
    knots: list[dict],
    rel_threshold: float = 0.05,
) -> list[tuple]:
    """Identifie les paires quasi-degenerees en masse.

    Retourne liste de (knot_i, knot_j, relative_gap) pour les paires
    avec |m_i - m_j| / m_i < rel_threshold.
    """
    doublets = []
    # Filtrer les noeuds avec masse > 0
    valid = [k for k in knots if k["mass_mev"] > 0]
    valid.sort(key=lambda k: k["mass_mev"])

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            mi = valid[i]["mass_mev"]
            mj = valid[j]["mass_mev"]
            gap = abs(mj - mi) / mi
            if gap < rel_threshold:
                doublets.append((valid[i]["name"], valid[j]["name"], gap))
    return doublets


# ---------------------------------------------------------------------------
# Listing complet
# ---------------------------------------------------------------------------

def generate_full_listing(
    max_crossing: int = 40,
    n_points: int = 5000,
) -> list[dict]:
    """Genere le listing complet des noeuds jusqu'a max_crossing croisements.

    Combine :
    1. Noeuds du catalogue existant (20 noeuds, 0_1 a 10_1)
    2. Noeuds toriques T(p,q) avec c <= max_crossing (hors doublons)
    3. Proprietes physiques pour chaque noeud
    4. Flags P16 (exclusion) et P17 (doublets)

    Retourne la liste triee par nombre de croisements puis masse.
    """
    seen_names: set[str] = set()
    all_knots: list[dict] = []

    # 1. Noeuds du catalogue existant
    for name in JONES_KNOT_NAMES:
        torus_pq = _CATALOG_TORUS.get(name)
        p_val = torus_pq[0] if torus_pq else None
        q_val = torus_pq[1] if torus_pq else None
        props = compute_knot_properties(name, p_val, q_val, n_points)
        all_knots.append(props)
        seen_names.add(name)

    # 2. Noeuds toriques non deja dans le catalogue
    torus_list = enumerate_torus_knots(max_crossing)
    for tk in torus_list:
        # Verifier si c'est deja dans le catalogue
        catalog_name = None
        for cname, (cp, cq) in _CATALOG_TORUS.items():
            if tk["p"] == cp and tk["q"] == cq:
                catalog_name = cname
                break
        if catalog_name and catalog_name in seen_names:
            continue

        name = tk["name"]
        if name in seen_names:
            continue
        props = compute_knot_properties(name, tk["p"], tk["q"], n_points)
        all_knots.append(props)
        seen_names.add(name)

    # 3. P16 et P17
    apply_p16_exclusion(all_knots)
    doublets = find_p17_doublets(all_knots)

    # Trier par crossing_number puis mass_mev
    all_knots.sort(key=lambda k: (k["crossing_number"], k["mass_mev"]))

    # Attacher les doublets comme metadata
    for k in all_knots:
        k["p17_doublets"] = [
            (a, b, g) for a, b, g in doublets
            if a == k["name"] or b == k["name"]
        ]

    return all_knots
