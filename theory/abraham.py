"""
Théorie Nodale — Formule d'Abraham nodale

Calcul des quantités géométriques (courbure κ, torsion τ) du repère de
Frenet-Serret et formule de masse d'Abraham :

    m_K = C_K × α × ρ_J × G_K

où α = π/(k+2) = π/431 ≈ 1/137.25 (constante de structure fine CS),
ρ_J le facteur de rétractation de Jones, et G_K une base géométrique
(ΣΦ, ΣΦ_E, ΣΦ_B, ∮κds, ∮|τ|ds, L, √U).
"""
from __future__ import annotations

import numpy as np

from theory.knot_catalog import knot_curve
from theory.jones import jones_polynomial, bridge_number
from theory.retraction import retraction_factor
from theory.postulates import K_CS

# --- Constantes ---
ALPHA_CS = 1.0 / (K_CS + 2)  # 1/137 (normalisation standard CS)

# --- Table d'Abraham (19 nœuds non-triviaux) ---
# C_K : (p, q) fraction rationnelle telle que m_K = (p/q) × α × ρ_J × ΣΦ
# best_base : base géométrique minimisant l'erreur
ABRAHAM_TABLE = {
    "3_1":  {"C_K": (4, 3),   "best_base": "sigma_phi_B"},
    "4_1":  {"C_K": (17, 20), "best_base": "sigma_phi_B"},
    "5_1":  {"C_K": (5, 11),  "best_base": "sqrt_U"},
    "5_2":  {"C_K": (21, 17), "best_base": "sigma_phi_B"},
    "6_1":  {"C_K": (14, 17), "best_base": "int_abs_tau"},
    "6_2":  {"C_K": (5, 6),   "best_base": "sigma_phi_B"},
    "6_3":  {"C_K": (7, 10),  "best_base": "sigma_phi_B"},
    "7_1":  {"C_K": (7, 20),  "best_base": "int_kappa"},
    "7_2":  {"C_K": (13, 15), "best_base": "sigma_phi_B"},
    "7_3":  {"C_K": (12, 17), "best_base": "sigma_phi_B"},
    "7_4":  {"C_K": (11, 16), "best_base": "sqrt_U"},
    "7_5":  {"C_K": (2, 3),   "best_base": "int_kappa"},
    "7_6":  {"C_K": (5, 8),   "best_base": "sigma_phi_B"},
    "7_7":  {"C_K": (7, 9),   "best_base": "sigma_phi_B"},
    "8_1":  {"C_K": (13, 18), "best_base": "sqrt_U"},
    "8_2":  {"C_K": (10, 17), "best_base": "sigma_phi_E"},
    "8_3":  {"C_K": (2, 3),   "best_base": "sigma_phi_B"},
    "9_1":  {"C_K": (2, 7),   "best_base": "sqrt_U"},
    "10_1": {"C_K": (7, 16),  "best_base": "sqrt_U"},
}

# Valeurs de référence à N=50,000 (document LaTeX, section 17bis)
REFERENCE_N50K = {
    "3_1":  {"L": 28.83, "int_kappa": 13.95, "int_abs_tau":  5.05, "sqrt_U":  3.01, "sigma_phi_E": 12.06, "sigma_phi_B":  3.85},
    "4_1":  {"L": 42.97, "int_kappa": 20.77, "int_abs_tau":  7.13, "sqrt_U":  4.08, "sigma_phi_E": 18.03, "sigma_phi_B":  6.84},
    "5_1":  {"L": 40.84, "int_kappa": 29.57, "int_abs_tau": 20.32, "sqrt_U":  6.53, "sigma_phi_E": 26.50, "sigma_phi_B": 19.84},
    "5_2":  {"L": 40.00, "int_kappa": 18.68, "int_abs_tau":  4.25, "sqrt_U":  3.17, "sigma_phi_E": 13.37, "sigma_phi_B":  3.71},
    "6_1":  {"L": 32.59, "int_kappa": 16.05, "int_abs_tau": 11.10, "sqrt_U":  4.89, "sigma_phi_E": 15.86, "sigma_phi_B":  9.86},
    "6_2":  {"L": 28.57, "int_kappa": 18.23, "int_abs_tau": 15.68, "sqrt_U":  7.71, "sigma_phi_E": 14.23, "sigma_phi_B": 11.10},
    "6_3":  {"L": 28.16, "int_kappa": 18.19, "int_abs_tau": 15.23, "sqrt_U":  8.41, "sigma_phi_E": 15.05, "sigma_phi_B": 15.15},
    "7_1":  {"L": 51.25, "int_kappa": 42.18, "int_abs_tau": 22.08, "sqrt_U":  7.23, "sigma_phi_E": 39.13, "sigma_phi_B": 21.65},
    "7_2":  {"L": 30.96, "int_kappa": 16.62, "int_abs_tau": 16.43, "sqrt_U":  5.82, "sigma_phi_E": 12.83, "sigma_phi_B": 11.56},
    "7_3":  {"L": 29.63, "int_kappa": 17.71, "int_abs_tau": 21.52, "sqrt_U":  7.80, "sigma_phi_E": 14.36, "sigma_phi_B": 15.56},
    "7_4":  {"L": 27.77, "int_kappa": 20.74, "int_abs_tau": 18.78, "sqrt_U": 11.67, "sigma_phi_E": 18.25, "sigma_phi_B": 12.50},
    "7_5":  {"L": 33.07, "int_kappa": 18.86, "int_abs_tau": 16.42, "sqrt_U":  7.00, "sigma_phi_E": 16.46, "sigma_phi_B": 14.83},
    "7_6":  {"L": 25.20, "int_kappa": 19.94, "int_abs_tau": 16.91, "sqrt_U": 10.88, "sigma_phi_E": 19.18, "sigma_phi_B": 14.92},
    "7_7":  {"L": 33.01, "int_kappa": 20.26, "int_abs_tau": 10.94, "sqrt_U":  7.55, "sigma_phi_E": 17.68, "sigma_phi_B":  9.65},
    "8_1":  {"L": 31.76, "int_kappa": 18.16, "int_abs_tau": 24.84, "sqrt_U":  7.74, "sigma_phi_E": 13.32, "sigma_phi_B": 16.05},
    "8_2":  {"L": 28.36, "int_kappa": 19.68, "int_abs_tau": 20.61, "sqrt_U": 10.81, "sigma_phi_E": 18.46, "sigma_phi_B": 17.43},
    "8_3":  {"L": 30.40, "int_kappa": 23.62, "int_abs_tau": 13.55, "sqrt_U": 38.63, "sigma_phi_E": 20.34, "sigma_phi_B": 11.07},
    "9_1":  {"L": 62.42, "int_kappa": 54.93, "int_abs_tau": 23.05, "sqrt_U":  7.94, "sigma_phi_E": 51.88, "sigma_phi_B": 22.67},
    "10_1": {"L": 32.86, "int_kappa": 19.81, "int_abs_tau": 38.79, "sqrt_U": 17.67, "sigma_phi_E": 17.36, "sigma_phi_B": 31.02},
}

# --- Échelle de masse dérivée (Abraham + Fáry-Milnor) ---
# C_tight(br=2) = C_K(3_1) × ΣΦ(3_1) / (4π)  [purement géométrique]
# Pour un plongement tendu, ∮κds → 4π, d'où η = C_tight × α × 4π.
_C_K_31 = ABRAHAM_TABLE["3_1"]["C_K"]
_SIGMA_PHI_31 = (REFERENCE_N50K["3_1"]["sigma_phi_E"]
                 + REFERENCE_N50K["3_1"]["sigma_phi_B"])
C_TIGHT_BR2 = (_C_K_31[0] / _C_K_31[1]) * _SIGMA_PHI_31 / (4 * np.pi)
ETA = C_TIGHT_BR2 * ALPHA_CS * 4 * np.pi  # ≈ 0.1548 MeV

# --- Route gravitationnelle (vérification) ---
# η = m_p / (ρ_nuc × A_18) ≈ 938.3 / (25.7 × 235.8) ≈ 0.1548 MeV
# Via P19 : m_p = κ_g·c²/(8πG), donc η ∝ κ_g ∝ 8πGm_p/c²
_RHO_NUC = (retraction_factor(jones_polynomial("9_1"))
            + retraction_factor(jones_polynomial("7_4"))
            + retraction_factor(jones_polynomial("6_3")))
_F_18 = 2584  # fibonacci(18)
_A_18 = (1 + _F_18 / 10.0) / (1 + 1 / 10.0)  # 259.4 / 1.1 = 235.8
M_PROTON = 938.272  # MeV (valeur expérimentale)
ETA_GRAV = M_PROTON / (_RHO_NUC * _A_18)  # ≈ 0.1548 MeV


def frenet_geometry(knot_name: str, n_points: int = 10000) -> dict:
    """Calcule κ(t) et τ(t) via le repère de Frenet-Serret.

    Dérivées par différences finies centrales (périodiques).

    Retourne
    --------
    dict avec clés : t, kappa, tau, speed, arc_length, z
    """
    pts = knot_curve(knot_name, n_points)
    # Supprimer le point final (doublon du premier pour courbe fermée)
    pts = pts[:-1]
    N = len(pts)
    dt = 2 * np.pi / N

    # Différences finies centrales (conditions périodiques)
    p_next = np.roll(pts, -1, axis=0)
    p_prev = np.roll(pts, 1, axis=0)
    p_next2 = np.roll(pts, -2, axis=0)
    p_prev2 = np.roll(pts, 2, axis=0)

    dr = (p_next - p_prev) / (2 * dt)
    ddr = (p_next - 2 * pts + p_prev) / dt**2
    dddr = (p_next2 - 2 * p_next + 2 * p_prev - p_prev2) / (2 * dt**3)

    speed = np.linalg.norm(dr, axis=1)

    # κ = |r' × r''| / |r'|³
    cross = np.cross(dr, ddr)
    cross_norm = np.linalg.norm(cross, axis=1)
    kappa = cross_norm / (speed**3 + 1e-30)

    # τ = (r' × r'') · r''' / |r' × r''|²
    triple = np.sum(cross * dddr, axis=1)
    tau = triple / (cross_norm**2 + 1e-30)

    # Longueur d'arc cumulée
    ds = speed * dt
    arc_length = np.concatenate([[0.0], np.cumsum(ds[:-1])])

    t = np.linspace(0, 2 * np.pi, N, endpoint=False)

    return {
        "t": t,
        "kappa": kappa,
        "tau": tau,
        "speed": speed,
        "arc_length": arc_length,
        "z": pts[:, 2],
    }


def geometric_integrals(knot_name: str, n_points: int = 10000) -> dict:
    """Intégrales de ligne sur la courbe fermée (quadrature trapézoïdale).

    Retourne
    --------
    dict avec clés : L, int_kappa, int_abs_tau, U_kappa, U_tau, U, sqrt_U
    """
    geo = frenet_geometry(knot_name, n_points)
    kappa = geo["kappa"]
    tau = geo["tau"]
    speed = geo["speed"]
    N = len(kappa)
    dt = 2 * np.pi / N

    # Pour une fonction périodique, la règle du rectangle ≡ trapèze
    ds = speed * dt

    L = np.sum(ds)
    int_kappa = np.sum(kappa * ds)
    int_abs_tau = np.sum(np.abs(tau) * ds)
    U_kappa = np.sum(kappa**2 * ds)
    U_tau = np.sum(tau**2 * ds)
    U = U_kappa + U_tau
    sqrt_U = np.sqrt(U)

    return {
        "L": float(L),
        "int_kappa": float(int_kappa),
        "int_abs_tau": float(int_abs_tau),
        "U_kappa": float(U_kappa),
        "U_tau": float(U_tau),
        "U": float(U),
        "sqrt_U": float(sqrt_U),
    }


def sigma_phi(knot_name: str, n_points: int = 10000) -> dict:
    """Tension EM entre creux et crêtes de z(s).

    1. Trouver les extrema locaux de z(t)
    2. Entre extrema consécutifs, intégrer κ(s)ds et |τ(s)|ds
    3. Sommer : ΣΦ_E = Σ ΔΦ_E, ΣΦ_B = Σ ΔΦ_B

    Retourne
    --------
    dict avec clés : sigma_phi_E, sigma_phi_B, sigma_phi, n_osc
    """
    geo = frenet_geometry(knot_name, n_points)
    z = geo["z"]
    kappa = geo["kappa"]
    tau = geo["tau"]
    speed = geo["speed"]
    N = len(z)
    dt = 2 * np.pi / N
    ds = speed * dt

    # Extrema locaux de z (conditions périodiques)
    z_prev = np.roll(z, 1)
    z_next = np.roll(z, -1)
    is_max = (z > z_prev) & (z > z_next)
    is_min = (z < z_prev) & (z < z_next)
    extrema_idx = np.where(is_max | is_min)[0]

    n_osc = len(extrema_idx)

    if n_osc < 2:
        return {
            "sigma_phi_E": 0.0,
            "sigma_phi_B": 0.0,
            "sigma_phi": 0.0,
            "n_osc": n_osc,
        }

    # Intégrales entre extrema consécutifs (sans bouclage)
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
        "sigma_phi": float(sp_E + sp_B),
        "n_osc": n_osc,
    }


def _get_geometric_base(knot_name: str, base: str, n_points: int) -> float:
    """Retourne la valeur de la base géométrique choisie."""
    if base in ("sigma_phi", "sigma_phi_E", "sigma_phi_B"):
        sp = sigma_phi(knot_name, n_points)
        return sp[base]
    gi = geometric_integrals(knot_name, n_points)
    return gi[base]


def abraham_mass(
    knot_name: str,
    C_K: tuple[int, int] | None = None,
    base: str = "sigma_phi",
    n_points: int = 10000,
) -> dict:
    """Masse d'Abraham : m_K = C_K × α × ρ_J × G_K.

    Paramètres
    ----------
    knot_name : nom du nœud (ex. "3_1")
    C_K : (p, q) fraction rationnelle. Si None, utilise ABRAHAM_TABLE.
    base : base géométrique ("sigma_phi", "sigma_phi_E", "sigma_phi_B",
           "int_kappa", "int_abs_tau", "L", "sqrt_U")
    n_points : nombre de points de discrétisation

    Retourne
    --------
    dict avec clés : mass_predicted, alpha, rho_J, G_K, C_K, error_pct, m_ref
    """
    if C_K is None:
        C_K = ABRAHAM_TABLE[knot_name]["C_K"]

    alpha = ALPHA_CS
    poly = jones_polynomial(knot_name)
    rho_J = retraction_factor(poly)
    G_K = _get_geometric_base(knot_name, base, n_points)

    c_k_val = C_K[0] / C_K[1]
    mass_predicted = c_k_val * alpha * rho_J * G_K

    # Masse de référence (rétractation) : m_ref = η × ρ_J
    m_ref = ETA * rho_J

    error_pct = abs(mass_predicted - m_ref) / m_ref * 100

    return {
        "mass_predicted": float(mass_predicted),
        "alpha": float(alpha),
        "rho_J": float(rho_J),
        "G_K": float(G_K),
        "C_K": C_K,
        "error_pct": float(error_pct),
        "m_ref": float(m_ref),
    }


def all_knots_abraham(n_points: int = 10000) -> list[dict]:
    """Calcule la masse Abraham pour les 19 nœuds non-triviaux.

    Retourne une liste de dicts triée par ρ_J avec erreur par nœud
    et erreur moyenne.
    """
    results = []
    for knot_name, info in ABRAHAM_TABLE.items():
        res = abraham_mass(knot_name, info["C_K"], "sigma_phi", n_points)
        res["knot"] = knot_name
        results.append(res)

    # Erreur moyenne
    errors = [r["error_pct"] for r in results]
    mean_error = sum(errors) / len(errors)
    for r in results:
        r["mean_error_pct"] = mean_error

    return results


def convergence_study(
    knot_name: str,
    n_points_list: list[int] | None = None,
) -> dict:
    """Étudie la convergence numérique en fonction de n_points.

    Retourne l'évolution de chaque quantité géométrique et l'erreur
    relative par rapport à la valeur au N maximal.
    """
    if n_points_list is None:
        n_points_list = [100, 500, 1000, 2000, 5000, 10000, 50000]

    quantities = ["L", "int_kappa", "int_abs_tau", "sqrt_U", "sigma_phi"]
    data = {q: [] for q in quantities}

    for n in n_points_list:
        gi = geometric_integrals(knot_name, n)
        sp = sigma_phi(knot_name, n)
        data["L"].append(gi["L"])
        data["int_kappa"].append(gi["int_kappa"])
        data["int_abs_tau"].append(gi["int_abs_tau"])
        data["sqrt_U"].append(gi["sqrt_U"])
        data["sigma_phi"].append(sp["sigma_phi"])

    # Erreurs relatives vs N maximal
    errors = {}
    for q in quantities:
        ref = data[q][-1]
        if abs(ref) > 1e-30:
            errors[q] = [abs(v - ref) / abs(ref) * 100 for v in data[q]]
        else:
            errors[q] = [0.0] * len(n_points_list)

    return {
        "n_points": n_points_list,
        "values": data,
        "errors": errors,
    }


# ============================================================
# Analyse de Fáry-Milnor — nombres de ponts et plongements tendus
# ============================================================

def fary_milnor_ratio(knot_name: str, n_points: int = 50000) -> dict:
    """Ratio de Fáry-Milnor pour un nœud.

    Le théorème de Fáry-Milnor stipule ∮κds ≥ 2π × br(K).
    Le ratio f_K = ∮κds / (2π × br) mesure la « non-tension » du plongement.
    Un plongement tendu (tight) aurait f_K = 1.

    Retourne
    --------
    dict avec clés : knot, br, int_kappa, fm_bound, f_K, fm_satisfied
    """
    br = bridge_number(knot_name)
    gi = geometric_integrals(knot_name, n_points)
    int_kappa = gi["int_kappa"]
    fm_bound = 2 * np.pi * br
    f_K = int_kappa / fm_bound

    return {
        "knot": knot_name,
        "br": br,
        "int_kappa": float(int_kappa),
        "fm_bound": float(fm_bound),
        "f_K": float(f_K),
        "fm_satisfied": f_K >= 1.0 - 0.005,  # tolérance 0.5% pour erreur numérique
    }


def tight_c_factor(br: int) -> float:
    """Facteur C_K^tight pour un plongement tendu à br ponts.

    Relation : C_K^tight(br) = C_TIGHT_BR2 × 2 / br
    Vérifié via η = C_TIGHT_BR2 × α × 4π.

    Valeurs : br=1 → ≈ 3.381, br=2 → ≈ 1.691, br=3 → ≈ 1.127

    Raises ValueError si br < 1.
    """
    if br < 1:
        raise ValueError(f"Bridge number must be ≥ 1, got {br}")
    return ETA / (ALPHA_CS * 2 * np.pi * br)


def fary_milnor_analysis(n_points: int = 50000) -> list[dict]:
    """Analyse de Fáry-Milnor pour les 19 nœuds non-triviaux.

    Pour chaque nœud, calcule :
    - Ratio f_K et borne FM
    - C_K actuel vs C_K^tight (plongement tendu)
    - C_K_ratio = C_K_actual / C_K^tight (mesure de non-tension)

    Retourne une liste triée par f_K croissant.
    """
    results = []
    for knot_name, info in ABRAHAM_TABLE.items():
        fm = fary_milnor_ratio(knot_name, n_points)
        c_k_actual = info["C_K"][0] / info["C_K"][1]
        c_k_tight = tight_c_factor(fm["br"])
        results.append({
            "knot": knot_name,
            "br": fm["br"],
            "int_kappa": fm["int_kappa"],
            "fm_bound": fm["fm_bound"],
            "f_K": fm["f_K"],
            "fm_satisfied": fm["fm_satisfied"],
            "C_K_actual": float(c_k_actual),
            "C_K_tight": float(c_k_tight),
            "C_K_ratio": float(c_k_actual / c_k_tight),
        })

    results.sort(key=lambda r: r["f_K"])
    return results


def fary_milnor_summary(n_points: int = 50000) -> dict:
    """Résumé statistique de l'analyse de Fáry-Milnor.

    Retourne
    --------
    dict avec clés :
    - results : liste triée par f_K
    - violations : nombre de nœuds violant la borne FM
    - near_tight : nœuds avec f_K < 1.2
    - f_K_mean, f_K_std : statistiques du ratio
    - c_tight_table : {br: C_K^tight} pour br = 1, 2, 3
    - eta_over_alpha : η/α (≈ 21.24)
    """
    results = fary_milnor_analysis(n_points)
    f_K_vals = [r["f_K"] for r in results]
    violations = sum(1 for r in results if not r["fm_satisfied"])
    near_tight = [r["knot"] for r in results if r["f_K"] < 1.2]

    return {
        "results": results,
        "violations": violations,
        "near_tight": near_tight,
        "f_K_mean": float(np.mean(f_K_vals)),
        "f_K_std": float(np.std(f_K_vals)),
        "c_tight_table": {br: tight_c_factor(br) for br in [1, 2, 3]},
        "eta_over_alpha": float(ETA / ALPHA_CS),
    }
