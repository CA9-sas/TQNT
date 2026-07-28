"""
Theorie Nodale -- Probabilite de formation des noeuds

Deux approches :
  A) Distribution de Boltzmann topologique (P18)
     P(K) propto exp(-beta * rho_J(K))
  B) Simulation Monte Carlo de polygones aleatoires
     Detection de nouage par determinant d'Alexander (Wirtinger)

References :
  - P18 (postulates.py) : matiere noire nodale
  - Sundberg-Thistlethwaite (2001) : N(c) ~ alpha_ST^c / c
  - Deguchi-Tsurusaki (1997) : P(knot) ~ 1 - exp(-N/N0)
"""
from __future__ import annotations

import math

import numpy as np

from theory.postulates import BETA_TOPO, C_STAR, GAMMA


# ===================================================================
# Constantes
# ===================================================================

ALPHA_ST = 2.68        # Sundberg-Thistlethwaite : N(c) ~ alpha_ST^c / c
_RHO_PER_CROSSING = 1.1  # rho_J ~ 1.1 * c (approximation pour noeuds alernes)
ALPHA_EFF = ALPHA_ST * math.exp(-BETA_TOPO * _RHO_PER_CROSSING)  # ~ 0.980


# ===================================================================
# Section A : Boltzmann topologique (P18)
# ===================================================================

def boltzmann_weight(rho_J: float, beta: float = BETA_TOPO) -> float:
    """Poids de Boltzmann topologique w(K) = exp(-beta * rho_J)."""
    return math.exp(-beta * rho_J)


def listing_partition_function(listing: list[dict], beta: float = BETA_TOPO) -> dict:
    """Fonction de partition sur le listing fini de noeuds.

    Parameters
    ----------
    listing : liste de dicts avec cle 'rho_J' (de generate_full_listing)
    beta : temperature topologique inverse

    Returns
    -------
    dict avec weights (list), Z_listing (float), probabilities (list)
    """
    weights = [boltzmann_weight(k["rho_J"], beta) for k in listing]
    Z = sum(weights)
    probs = [w / Z for w in weights]
    return {
        "weights": weights,
        "Z_listing": Z,
        "probabilities": probs,
    }


def asymptotic_partition_function(
    beta: float = BETA_TOPO,
    c_max: int = 500,
) -> dict:
    """Fonction de partition asymptotique Z_total via N(c) ~ alpha_ST^c / c.

    Calcul en log-espace pour eviter overflow.
    rho_J(c) ~ 1.1 * c (approximation alternante).

    Returns
    -------
    dict avec Z_total, Z_by_crossing (array), alpha_eff, c_dominant
    """
    # log(terme(c)) = c*log(alpha_ST) - log(c) - beta * 1.1 * c
    #               = c * (log(alpha_ST) - beta*1.1) - log(c)
    #               = c * log(alpha_eff) - log(c)
    log_alpha_eff = math.log(ALPHA_ST) - beta * _RHO_PER_CROSSING
    alpha_eff = math.exp(log_alpha_eff)

    # Calcul en log-espace
    crossings = np.arange(1, c_max + 1, dtype=float)
    log_terms = crossings * log_alpha_eff - np.log(crossings)

    # Conversion stable : Z = sum(exp(log_terms))
    # Utiliser logsumexp manuellement
    log_max = np.max(log_terms)
    Z_total = math.exp(log_max) * float(np.sum(np.exp(log_terms - log_max)))

    # Z par croisement (valeurs reelles)
    Z_by_crossing = np.exp(log_terms)

    # c dominant = argmax du terme
    c_dominant = int(np.argmax(log_terms) + 1)

    return {
        "Z_total": Z_total,
        "Z_by_crossing": Z_by_crossing,
        "alpha_eff": alpha_eff,
        "c_dominant": c_dominant,
    }


def normalized_probabilities(
    listing: list[dict],
    beta: float = BETA_TOPO,
    c_max: int = 500,
) -> dict:
    """Probabilites normalisees P(K) = w(K) / Z_total.

    Returns
    -------
    dict avec probabilities (list), coverage = Z_listing/Z_total,
    P_visible (c <= C_STAR), P_dark (c > C_STAR)
    """
    lp = listing_partition_function(listing, beta)
    ap = asymptotic_partition_function(beta, c_max)

    Z_total = ap["Z_total"]
    probs = [w / Z_total for w in lp["weights"]]
    coverage = lp["Z_listing"] / Z_total

    P_visible = sum(
        w / Z_total for w, k in zip(lp["weights"], listing)
        if k["crossing_number"] <= C_STAR
    )
    P_dark = 1.0 - P_visible

    return {
        "probabilities": probs,
        "coverage": coverage,
        "P_visible": P_visible,
        "P_dark": P_dark,
    }


def dark_matter_ratio(
    beta: float = BETA_TOPO,
    c_max: int = 500,
) -> dict:
    """Ratio Omega_DM / Omega_b = M_dark / M_visible.

    M(c) = N(c) * exp(-beta * 1.1 * c) * 1.1 * c
         = (alpha_ST^c / c) * exp(-beta * 1.1 * c) * 1.1 * c
         = 1.1 * alpha_eff^c

    Returns
    -------
    dict avec ratio, M_visible, M_dark, c_star
    """
    log_alpha_eff = math.log(ALPHA_ST) - beta * _RHO_PER_CROSSING

    crossings = np.arange(1, c_max + 1, dtype=float)
    # log(M(c)) = log(1.1) + c * log(alpha_eff)
    log_M = math.log(_RHO_PER_CROSSING) + crossings * log_alpha_eff

    log_max = np.max(log_M)
    M_terms = np.exp(log_M - log_max)

    # Visible : c <= C_STAR, Dark : c > C_STAR
    M_visible = float(np.sum(M_terms[:C_STAR]))
    M_dark = float(np.sum(M_terms[C_STAR:]))

    ratio = M_dark / M_visible if M_visible > 0 else float("inf")

    return {
        "ratio": ratio,
        "M_visible": M_visible * math.exp(log_max),
        "M_dark": M_dark * math.exp(log_max),
        "c_star": C_STAR,
    }


# ===================================================================
# Section B : Monte Carlo polygones aleatoires
# ===================================================================

def random_closed_polygon(
    n_edges: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Polygone ferme aleatoire en R^3.

    Methode : pas gaussiens + soustraction de la derive moyenne.
    Shape (n_edges+1, 3), dernier point = premier.
    """
    if rng is None:
        rng = np.random.default_rng()

    steps = rng.standard_normal((n_edges, 3))
    # Soustraire la derive pour que la somme des pas = 0
    steps -= steps.mean(axis=0)
    # Construire le polygone par sommes cumulees
    vertices = np.zeros((n_edges + 1, 3))
    vertices[1:] = np.cumsum(steps, axis=0)
    # Fermer exactement : distribuer le residu
    residual = vertices[-1]
    correction = np.outer(np.linspace(0, 1, n_edges + 1), residual)
    vertices -= correction
    return vertices


def project_and_find_crossings(polygon: np.ndarray) -> list[dict]:
    """Projection xy et detection d'intersections de segments non-adjacents.

    Version vectorisee : toutes les paires (i, j) avec j >= i+2 sont
    testees en bloc numpy pour eviter la double boucle Python.
    Pour chaque croisement : indices i, j, parametres t, s,
    signe (over/under via z).
    """
    n = len(polygon) - 1  # nombre d'aretes

    # Paires (i, j) non-adjacentes : j >= i+2 et pas (0, n-1)
    # Construire toutes les paires avec triu_indices puis filtrer
    all_i, all_j = np.triu_indices(n, k=2)
    # Exclure la paire (0, n-1) qui sont adjacentes (polygone ferme)
    mask = ~((all_i == 0) & (all_j == n - 1))
    idx_i = all_i[mask]
    idx_j = all_j[mask]
    if len(idx_i) == 0:
        return []

    # Points de depart et direction de chaque segment (projection xy)
    p1 = polygon[idx_i, :2]       # (M, 2)
    p2 = polygon[idx_i + 1, :2]
    p3 = polygon[idx_j, :2]
    p4 = polygon[idx_j + 1, :2]

    d1 = p2 - p1  # (M, 2)
    d2 = p4 - p3
    dp = p3 - p1

    denom = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    # Eviter division par zero
    safe = np.abs(denom) > 1e-12

    t = np.full(len(denom), -1.0)
    s = np.full(len(denom), -1.0)
    t[safe] = (dp[safe, 0] * d2[safe, 1] - dp[safe, 1] * d2[safe, 0]) / denom[safe]
    s[safe] = (dp[safe, 0] * d1[safe, 1] - dp[safe, 1] * d1[safe, 0]) / denom[safe]

    # Filtre : 0 < t < 1 et 0 < s < 1
    hit = safe & (t > 0) & (t < 1) & (s > 0) & (s < 1)
    hit_idx = np.where(hit)[0]

    crossings = []
    for k in hit_idx:
        i_val = int(idx_i[k])
        j_val = int(idx_j[k])
        t_val = float(t[k])
        s_val = float(s[k])
        z_i = polygon[i_val, 2] + t_val * (polygon[i_val + 1, 2] - polygon[i_val, 2])
        z_j = polygon[j_val, 2] + s_val * (polygon[j_val + 1, 2] - polygon[j_val, 2])
        crossings.append({
            "i": i_val,
            "j": j_val,
            "t": t_val,
            "s": s_val,
            "z_over": z_i > z_j,
            "sign": 1 if z_i > z_j else -1,
        })

    return crossings


def knot_determinant_from_crossings(
    crossings: list[dict],
    n_edges: int,
) -> int:
    """Determinant du noeud via presentation de Wirtinger evaluee a t = -1.

    Etiquetage des arcs :
    - Chaque passage SOUS un croisement decoupe le polygone en arcs.
    - n croisements -> n arcs (numerotes cycliquement 0..n-1).
    - L'arc k va de l'under-pass k a l'under-pass (k+1) % n.

    Matrice d'Alexander a t = -1 :
    - Croisement positif : -under_in + 2*over - under_out = 0
    - Croisement negatif :  under_in - 2*over + under_out = 0
    - Supprimer une ligne et une colonne -> |det(A')| = determinant du noeud.
    """
    if not crossings:
        return 1  # unknot

    n_crossings = len(crossings)

    # --- Collecter les under-passes ---
    # Pour chaque croisement, identifier le segment qui passe en dessous
    # et sa position parametrique sur le polygone.
    # Position globale = edge_index + parameter_t (pour le tri sequentiel)
    under_info: list[tuple[float, int]] = []  # (global_pos, crossing_index)
    over_info: list[tuple[float, int]] = []   # (global_pos, crossing_index)

    for ci, c in enumerate(crossings):
        if c["z_over"]:
            # segment i au-dessus, segment j en dessous
            under_info.append((c["j"] + c["s"], ci))
            over_info.append((c["i"] + c["t"], ci))
        else:
            # segment j au-dessus, segment i en dessous
            under_info.append((c["i"] + c["t"], ci))
            over_info.append((c["j"] + c["s"], ci))

    # Trier les under-passes par position sur le polygone
    under_info.sort(key=lambda x: x[0])

    # --- Arcs cycliques ---
    # n under-passes divisent le polygone en n arcs.
    # Arc k : du under-pass k au under-pass (k+1) % n.
    n_arcs = n_crossings

    if n_arcs < 2:
        return 1

    # under_pass_order[k] = crossing_index du k-eme under-pass
    under_pass_order = [ci for _, ci in under_info]

    # Pour chaque croisement, determiner under_in et under_out :
    # Si le croisement ci est le k-eme under-pass,
    #   under_in = arc k-1 (mod n) = arc arrivant a ce under-pass
    #   under_out = arc k = arc partant de ce under-pass
    crossing_to_under_rank = {}
    for k, ci in enumerate(under_pass_order):
        crossing_to_under_rank[ci] = k

    # Pour chaque croisement, determiner l'over-arc :
    # L'over-arc est l'arc contenant le point de passage superieur.
    # Le over-pass a position globale over_pos.
    # L'arc k couvre de under_pos[k] a under_pos[(k+1)%n] (cycliquement).
    under_positions = [pos for pos, _ in under_info]

    def _find_arc(pos: float) -> int:
        """Trouve l'arc contenant la position globale pos."""
        for k in range(n_arcs):
            start = under_positions[k]
            end = under_positions[(k + 1) % n_arcs]
            if start < end:
                if start < pos < end:
                    return k
            else:
                # L'arc traverse la fermeture (boucle)
                if pos > start or pos < end:
                    return k
        # Fallback : position exactement sur une frontiere
        return 0

    # --- Construction de la matrice ---
    A = np.zeros((n_crossings, n_arcs))

    for ci, c in enumerate(crossings):
        k = crossing_to_under_rank[ci]
        u_in = (k - 1) % n_arcs   # arc arrivant au under-pass
        u_out = k                   # arc partant du under-pass

        # Position du over-pass
        if c["z_over"]:
            over_pos = c["i"] + c["t"]
        else:
            over_pos = c["j"] + c["s"]
        o = _find_arc(over_pos)

        sign = c["sign"]
        if sign == 1:
            A[ci, u_in] += -1
            A[ci, o] += 2
            A[ci, u_out] += -1
        else:
            A[ci, u_in] += 1
            A[ci, o] += -2
            A[ci, u_out] += 1

    # Supprimer une ligne et une colonne -> matrice carree (n-1) x (n-1)
    A_reduced = A[:-1, :-1]
    if A_reduced.shape[0] == 0:
        return 1
    try:
        det_val = abs(round(np.linalg.det(A_reduced)))
    except np.linalg.LinAlgError:
        return 1
    return max(det_val, 1)


# Table de lookup : determinant -> nom du noeud
# Construite a partir des 20 noeuds du catalogue Jones
_DET_TO_KNOT: dict[int, str] = {
    1: "0_1",     # unknot
    3: "3_1",     # trefoil
    5: "4_1",     # figure-eight
    25: "5_1",    # torus T(2,5)
    7: "5_2",
    9: "6_1",
    11: "6_2",
    13: "6_3",
    49: "7_1",    # torus T(2,7)
    15: "7_2",
    17: "7_3",
    19: "7_4",
    21: "7_5",
    23: "7_6",
    29: "7_7",
}


def classify_knot(determinant: int, n_crossings: int) -> str:
    """Classification du noeud par determinant.

    Le determinant seul ne distingue pas tous les noeuds
    (ex: 7_3 et 8_1 ont det=17). Acceptable pour les statistiques globales.
    """
    if determinant == 1 and n_crossings == 0:
        return "0_1"
    if determinant == 1:
        return "0_1"  # Probablement unknot
    return _DET_TO_KNOT.get(determinant, f"?_det={determinant}")


def monte_carlo_knot_statistics(
    n_edges: int = 100,
    n_trials: int = 10000,
    seed: int = 42,
) -> dict:
    """Statistiques de nouage par Monte Carlo.

    Genere n_trials polygones fermes aleatoires de n_edges aretes,
    calcule le determinant pour chacun, classifie.

    Returns
    -------
    dict avec unknot_fraction, knotted_fraction, knot_counts, knot_fractions
    """
    rng = np.random.default_rng(seed)
    knot_counts: dict[str, int] = {}
    unknot_count = 0
    total = 0

    for _ in range(n_trials):
        poly = random_closed_polygon(n_edges, rng)
        crossings = project_and_find_crossings(poly)
        n_cross = len(crossings)
        det = knot_determinant_from_crossings(crossings, n_edges)
        name = classify_knot(det, n_cross)
        knot_counts[name] = knot_counts.get(name, 0) + 1
        if name == "0_1":
            unknot_count += 1
        total += 1

    unknot_frac = unknot_count / total
    knot_fracs = {k: v / total for k, v in sorted(knot_counts.items(), key=lambda x: -x[1])}

    return {
        "unknot_fraction": unknot_frac,
        "knotted_fraction": 1.0 - unknot_frac,
        "knot_counts": knot_counts,
        "knot_fractions": knot_fracs,
        "n_trials": total,
        "n_edges": n_edges,
    }


def knotting_probability_vs_length(
    n_edges_list: list[int] | None = None,
    n_trials: int = 2000,
    seed: int = 42,
) -> dict:
    """P(noue) vs N pour differentes longueurs de polygone.

    Reference : Deguchi-Tsurusaki P(knot) ~ 1 - exp(-N/N0) avec N0 ~ 300.

    Returns
    -------
    dict avec n_edges, p_knotted, fit_N0
    """
    if n_edges_list is None:
        n_edges_list = [30, 50, 100, 200, 500]

    p_knotted = []
    for n in n_edges_list:
        stats = monte_carlo_knot_statistics(n, n_trials, seed)
        p_knotted.append(stats["knotted_fraction"])

    # Fit exponentiel : P(knot) = 1 - exp(-N/N0)
    # => -N/N0 = log(1 - P) => N0 = -N / log(1 - P)
    N0_estimates = []
    for n, p in zip(n_edges_list, p_knotted):
        if 0 < p < 1:
            N0_estimates.append(-n / math.log(1 - p))

    fit_N0 = float(np.median(N0_estimates)) if N0_estimates else float("inf")

    return {
        "n_edges": n_edges_list,
        "p_knotted": p_knotted,
        "fit_N0": fit_N0,
    }
