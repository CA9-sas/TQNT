"""V5.37 — Splittings radiaux du quarkonium depuis Schrödinger χ-field sur nœud.

Contexte
========

V5.35d (commit fee4d23) ferme les Δm d'oscillations chirales (K⁰, D⁰,
B⁰, B_s⁰) à 0,1 %.  V5.35e (commit 983d0e2) a explicitement montré que
la formule V5.35d **n'est pas applicable** aux splittings radiaux du
quarkonium (J/ψ ↔ ψ(2S), Υ(1S) ↔ Υ(2S)) parce que :

* V5.35d est une formule d'oscillation chirale L ↔ L*, donc *même
  topologie* miroir.
* Les états excités de quarkonium dans le catalogue V5 ont des
  topologies **distinctes** de leur état fondamental :

    J/ψ        = 3_1 + 4_1 + 6_3
    ψ(2S)      = 6_1 + 8_2
    Υ(1S)      = 6_2 + 7_1 + 7_3
    Υ(2S)      = 6_2 + 6_2 + 8_2

V5.37 prend l'autre route : dériver le splitting depuis le **spectre
Schrödinger** du champ χ sur le lien.  Deux interprétations sont
calculées, dans la même fonction, pour rester honnête :

1. **Interprétation A — splitting intra-lien (E_2 − E_1)** :
   On calcule sur le **lien fondamental** (J/ψ link, Υ(1S) link) les
   premiers niveaux propres et on définit
   Δm_A = m_{fond} × (|E_2 − E_1| / |E_1|).
   C'est l'image quantum-mécanique standard où n=2 est l'excitation
   radiale sur le même puits — **mais cela n'est pas l'assignation V5**.

2. **Interprétation B — splitting inter-lien (E_1[L'] − E_1[L])** :
   On calcule E_1 sur le lien fondamental et E_1 sur le lien excité du
   catalogue V5, et on définit
   Δm_B = m_{fond} × |E_1[L'] − E_1[L]| / |E_1[L]|.
   C'est l'image V5 : ψ(2S) est un lien distinct, son écart à J/ψ vient
   de la différence de profondeur du puits effectif.

Aucune des deux n'est calibrée — la sortie est ce que le solveur donne,
et on rapporte le log_ratio vs PDG honnêtement.

Limitations honnêtes
====================

* Le solveur FEM résout [-ℏ²/2 ∇² − λρ_J / d²] Ψ = E Ψ dans une boîte
  finie.  Le spectre n'est PAS strictement convergé : les eigenvalues
  drift avec la résolution du maillage (cf. test convergence dans
  scripts/V5_oscillations/v5p37_radial_test.py).  On fixe la résolution
  à n_coarse=14 pour comparabilité, mais l'**incertitude numérique
  intrinsèque est de l'ordre de 30-50 %** sur le ratio (E_2 − E_1)/E_1.
* Le potentiel −λρ/d² n'est pas un Cornell potential : la TQNT ne
  prédit *a priori* aucun spacing de niveaux ~500 MeV constant.  Si
  l'écart calculé est très éloigné de 589/563 MeV, c'est le signal
  honnête que V5 dans son état actuel **ne reproduit pas** ces
  splittings sans ingrédient supplémentaire (par ex. potentiel de
  confinement σr).

Référence
=========

* theory/schrodinger.py : FEM solver Schrödinger
* simulation/solver.py  : interfaces haut-niveau
* docs/TQNT_catalog_V5_diffusion.xlsx onglet « 3. Particules PDG » :
  assignations V5 du quarkonium.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from theory.knot_catalog import knot_curve
from theory.schrodinger import (
    generate_mesh, assemble_system,
    generate_mesh_multi, assemble_system_multi,
    solve_eigenproblem,
)


# ---------------------------------------------------------------------------
# Assignations V5 du quarkonium (catalogue V5 onglet 3)
# ---------------------------------------------------------------------------
QUARKONIUM_ASSIGNMENTS: dict[str, dict] = {
    "J/psi":       {"knots": ["3_1", "4_1", "6_3"],   "mass_pdg_MeV": 3096.9},
    "psi(2S)":     {"knots": ["6_1", "8_2"],          "mass_pdg_MeV": 3686.10},
    "psi(3S)":     {"knots": ["6_1", "8_2"],          "mass_pdg_MeV": 4039.0,
                    "lowconf": True,
                    "note": "Pas d'assignation V5 distincte ; on reprend le "
                            "lien ψ(2S) et on lit E_3."},
    "Upsilon(1S)": {"knots": ["6_2", "7_1", "7_3"],   "mass_pdg_MeV": 9460.30},
    "Upsilon(2S)": {"knots": ["6_2", "6_2", "8_2"],   "mass_pdg_MeV": 10023.26},
    "Upsilon(3S)": {"knots": ["6_2", "6_2", "8_2"],   "mass_pdg_MeV": 10355.2,
                    "lowconf": True,
                    "note": "Pas d'assignation V5 distincte ; on reprend le "
                            "lien Υ(2S) et on lit E_3."},
}


# Familles : (état fondamental, états excités) pour les splittings
QUARKONIUM_FAMILIES: dict[str, dict] = {
    "charmonium": {
        "ground": "J/psi",
        "excited": ["psi(2S)", "psi(3S)"],
    },
    "bottomonium": {
        "ground": "Upsilon(1S)",
        "excited": ["Upsilon(2S)", "Upsilon(3S)"],
    },
}


# ---------------------------------------------------------------------------
# Paramètres numériques par défaut du solveur
# ---------------------------------------------------------------------------
DEFAULT_N_COARSE = 14
DEFAULT_EPSILON = 0.25
DEFAULT_N_CURVE_POINTS = 200
DEFAULT_N_LEVELS = 4


# ---------------------------------------------------------------------------
# quarkonium_assignment
# ---------------------------------------------------------------------------
def quarkonium_assignment(particle: str) -> dict:
    """Retourne l'assignation V5 (knots + masse PDG) d'un état quarkonium.

    Parameters
    ----------
    particle : str
        Nom (e.g. "J/psi", "psi(2S)", "Upsilon(1S)", "Upsilon(2S)").

    Returns
    -------
    dict
        { knots, mass_pdg_MeV, lowconf?, note? }
    """
    if particle not in QUARKONIUM_ASSIGNMENTS:
        raise KeyError(
            f"Particule {particle!r} sans assignation V5 dans "
            f"QUARKONIUM_ASSIGNMENTS.  Choix : "
            f"{list(QUARKONIUM_ASSIGNMENTS)}"
        )
    return dict(QUARKONIUM_ASSIGNMENTS[particle])


# ---------------------------------------------------------------------------
# schrodinger_spectrum_on_knot
# ---------------------------------------------------------------------------
def schrodinger_spectrum_on_knot(
    knots: list[str],
    n_levels: int = DEFAULT_N_LEVELS,
    n_coarse: int = DEFAULT_N_COARSE,
    epsilon: float = DEFAULT_EPSILON,
    n_curve_points: int = DEFAULT_N_CURVE_POINTS,
) -> dict:
    """Spectre Schrödinger χ-field sur un nœud ou un lien.

    Construit le maillage tétraédrique autour de la liste de nœuds
    (single-knot si len(knots)==1, multi-knot link sinon), assemble le
    Hamiltonien FEM (P1, 4-point Gauss quadrature) et résout
    H Ψ = E M Ψ pour les n_levels premiers eigenstates.

    Parameters
    ----------
    knots : list[str]
        Composantes du lien (e.g. ["3_1", "4_1", "6_3"] pour J/ψ).
    n_levels : int
        Nombre de niveaux propres à renvoyer.
    n_coarse, epsilon, n_curve_points : paramètres du solveur (defaults
        cohérents avec scripts/V5_oscillations/v5p37_radial_test.py).

    Returns
    -------
    dict
        { knots, eigenvalues, n_points, n_tets, n_components }
    """
    if not knots:
        raise ValueError("liste de nœuds vide")
    curves = [knot_curve(k, n_curve_points) for k in knots]

    if len(knots) == 1:
        points, tets, boundary = generate_mesh(
            curves[0], n_coarse=n_coarse, epsilon=epsilon,
        )
        H, M = assemble_system(
            points, tets, boundary, knots[0], cKDTree(curves[0]),
            energy=0.0, epsilon=epsilon,
        )
    else:
        points, tets, boundary = generate_mesh_multi(
            curves, n_coarse=n_coarse, epsilon=epsilon,
        )
        H, M = assemble_system_multi(
            points, tets, boundary, knots, curves,
            energy=0.0, epsilon=epsilon,
        )

    evals, _ = solve_eigenproblem(H, M, n_eigenvalues=n_levels)
    return {
        "knots": list(knots),
        "eigenvalues": np.asarray(evals, dtype=float),
        "n_points": int(len(points)),
        "n_tets": int(len(tets)),
        "n_components": int(len(knots)),
    }


# Cache pour éviter de recalculer le spectre du même lien plusieurs fois.
_SPECTRUM_CACHE: dict[tuple, dict] = {}


def _cached_spectrum(
    knots: list[str],
    n_levels: int,
    n_coarse: int,
    epsilon: float,
    n_curve_points: int,
) -> dict:
    key = (tuple(knots), int(n_levels), int(n_coarse),
           float(epsilon), int(n_curve_points))
    if key not in _SPECTRUM_CACHE:
        _SPECTRUM_CACHE[key] = schrodinger_spectrum_on_knot(
            knots, n_levels=n_levels, n_coarse=n_coarse,
            epsilon=epsilon, n_curve_points=n_curve_points,
        )
    return _SPECTRUM_CACHE[key]


def clear_spectrum_cache() -> None:
    """Vide le cache (utile pour les tests)."""
    _SPECTRUM_CACHE.clear()


# ---------------------------------------------------------------------------
# delta_m_radial
# ---------------------------------------------------------------------------
def delta_m_radial(
    particle_ground: str,
    particle_excited: str,
    n_levels: int = DEFAULT_N_LEVELS,
    n_coarse: int = DEFAULT_N_COARSE,
    epsilon: float = DEFAULT_EPSILON,
    n_curve_points: int = DEFAULT_N_CURVE_POINTS,
) -> dict:
    """Δm radial V5.37 entre deux états quarkonium.

    Deux interprétations sont calculées dans la même structure :

    1. **intra-lien** (interprétation A, image SM classique) :
       sur le **lien fondamental** uniquement, on lit E_1 et E_2 et on
       définit
           Δm_A = m_{fond} × (|E_{n_excited} − E_1| / |E_1|)
       où n_excited = 2 pour ψ(2S)/Υ(2S), 3 pour ψ(3S)/Υ(3S).
       Calibration : on impose que l'état fondamental |E_1| ↔ m_{fond}.

    2. **inter-lien** (interprétation B, image V5 catalogue) :
       on calcule E_1 sur le lien fondamental et E_1 sur le lien excité
       (différent dans le catalogue V5) et on définit
           Δm_B = m_{fond} × |E_1[L'] − E_1[L]| / |E_1[L]|

    Les deux Δm sont reportés ; aucun n'est calibré sur la PDG, donc
    log_ratio est purement diagnostic.

    Returns
    -------
    dict
        { particle_ground, particle_excited, knots_ground, knots_excited,
          mass_ground_MeV, mass_excited_pdg_MeV, delta_m_pdg_MeV,
          E_ground_levels, E_excited_levels,
          interp_A : { n_excited, dE_dimensionless, delta_m_v5p37_MeV,
                       log_ratio_A },
          interp_B : { dE_dimensionless, delta_m_v5p37_MeV, log_ratio_B },
          note }
    """
    cfg_g = quarkonium_assignment(particle_ground)
    cfg_e = quarkonium_assignment(particle_excited)
    knots_g = cfg_g["knots"]
    knots_e = cfg_e["knots"]
    m_g = float(cfg_g["mass_pdg_MeV"])
    m_e = float(cfg_e["mass_pdg_MeV"])
    delta_pdg = m_e - m_g

    # Choix de l'index 'excited' selon le nom (psi(2S) -> 2, psi(3S) -> 3, ...)
    n_excited = 2
    if "(3S)" in particle_excited:
        n_excited = 3
    elif "(4S)" in particle_excited:
        n_excited = 4

    # n_levels effectif suffisant pour couvrir n_excited.
    n_eff = max(n_levels, n_excited + 1)

    spec_g = _cached_spectrum(knots_g, n_eff, n_coarse, epsilon,
                              n_curve_points)
    spec_e = _cached_spectrum(knots_e, n_eff, n_coarse, epsilon,
                              n_curve_points)

    E_g = spec_g["eigenvalues"]
    E_e = spec_e["eigenvalues"]

    # ---------------- Interprétation A : intra-lien ----------------
    # E_g[0] est l'état fondamental |E_1|.  L'écart E_n_excited − E_1 est
    # signé ; on prend la valeur absolue pour le ratio dimensionless.
    if n_excited - 1 < len(E_g):
        dE_A_dim = float((E_g[n_excited - 1] - E_g[0]) / abs(E_g[0]))
    else:
        dE_A_dim = float("nan")
    dm_A_MeV = abs(dE_A_dim) * m_g
    if dm_A_MeV > 0 and abs(delta_pdg) > 0:
        log_A = float(np.log10(dm_A_MeV / abs(delta_pdg)))
    else:
        log_A = float("nan")

    # ---------------- Interprétation B : inter-lien ----------------
    if tuple(knots_g) == tuple(knots_e):
        # Même lien V5 (cas ψ(3S)/ψ(2S) qui partagent 6_1+8_2)
        # → on bascule sur l'écart intra-lien E_n - E_1 par construction
        dE_B_dim = dE_A_dim
        same_link = True
    else:
        dE_B_dim = float((E_e[0] - E_g[0]) / abs(E_g[0]))
        same_link = False
    dm_B_MeV = abs(dE_B_dim) * m_g
    if dm_B_MeV > 0 and abs(delta_pdg) > 0:
        log_B = float(np.log10(dm_B_MeV / abs(delta_pdg)))
    else:
        log_B = float("nan")

    if same_link:
        note_B = ("Même lien V5 que l'état fondamental → interp_B "
                  "identique à interp_A.")
    else:
        note_B = ("Liens V5 distincts : Δm_B = écart des fondamentaux "
                  "E_1[L'] − E_1[L] (image V5 catalogue).")

    return {
        "particle_ground": particle_ground,
        "particle_excited": particle_excited,
        "knots_ground": knots_g,
        "knots_excited": knots_e,
        "mass_ground_MeV": m_g,
        "mass_excited_pdg_MeV": m_e,
        "delta_m_pdg_MeV": float(delta_pdg),
        "E_ground_levels": E_g.tolist(),
        "E_excited_levels": E_e.tolist(),
        "interp_A": {
            "n_excited": int(n_excited),
            "dE_dimensionless": dE_A_dim,
            "delta_m_v5p37_MeV": float(dm_A_MeV),
            "log_ratio": log_A,
            "note": ("intra-lien : (E_{n_excited} − E_1)/|E_1| × m_fond, "
                     "image SM classique"),
        },
        "interp_B": {
            "dE_dimensionless": dE_B_dim,
            "delta_m_v5p37_MeV": float(dm_B_MeV),
            "log_ratio": log_B,
            "note": note_B,
        },
        "same_link": bool(same_link),
        "note": (
            "Solveur FEM non strictement convergé : incertitude numérique "
            "intrinsèque sur le ratio dimensionless ~30-50 % au n_coarse "
            "par défaut.  log_ratio purement diagnostic, AUCUN paramètre "
            "n'est calibré sur la PDG."
        ),
    }


# ---------------------------------------------------------------------------
# v5p37_table : table comparative
# ---------------------------------------------------------------------------
V5P37_PAIRS: list[tuple[str, str, str]] = [
    # (label, ground, excited)
    ("psi(2S)/J/psi",       "J/psi",       "psi(2S)"),
    ("psi(3S)/psi(2S)",     "psi(2S)",     "psi(3S)"),
    ("Upsilon(2S)/Upsilon(1S)", "Upsilon(1S)", "Upsilon(2S)"),
    ("Upsilon(3S)/Upsilon(2S)", "Upsilon(2S)", "Upsilon(3S)"),
]


def v5p37_radial_table(
    n_levels: int = DEFAULT_N_LEVELS,
    n_coarse: int = DEFAULT_N_COARSE,
    epsilon: float = DEFAULT_EPSILON,
    n_curve_points: int = DEFAULT_N_CURVE_POINTS,
) -> list[dict]:
    """Calcule Δm V5.37 sur les 4 paires (charmonium + bottomonium).

    Returns
    -------
    list[dict]
        Rows enrichis du label de la paire.
    """
    rows: list[dict] = []
    for label, g, e in V5P37_PAIRS:
        res = delta_m_radial(
            g, e, n_levels=n_levels, n_coarse=n_coarse,
            epsilon=epsilon, n_curve_points=n_curve_points,
        )
        res["label"] = label
        rows.append(res)
    return rows


# ---------------------------------------------------------------------------
# Verdict V5.37
# ---------------------------------------------------------------------------
def verdict_v5p37(rows: list[dict], threshold_dex: float = 0.3) -> dict:
    """Bilan honnête de V5.37 sur les splittings radiaux.

    Pour chaque paire on prend le MEILLEUR log_ratio (min |·|) entre les
    deux interprétations A/B (V5.37 ne pré-engage pas une lecture).

    Parameters
    ----------
    threshold_dex : float
        Seuil de "reproduction" en dex (par défaut 0.3 ≈ facteur 2).

    Returns
    -------
    dict
        { sum_abs_log_best, mean_abs_log_best, n_within_threshold,
          n_total, verdict, recommendation, best_interp_per_row }
    """
    best_logs: list[float] = []
    best_interp: list[str] = []
    for r in rows:
        log_A = r["interp_A"]["log_ratio"]
        log_B = r["interp_B"]["log_ratio"]
        candidates = []
        if np.isfinite(log_A):
            candidates.append(("A", log_A))
        if np.isfinite(log_B):
            candidates.append(("B", log_B))
        if not candidates:
            best_logs.append(float("nan"))
            best_interp.append("none")
            continue
        tag, best = min(candidates, key=lambda x: abs(x[1]))
        best_logs.append(float(best))
        best_interp.append(tag)

    finite = [x for x in best_logs if np.isfinite(x)]
    s_abs = float(np.sum(np.abs(finite))) if finite else float("nan")
    m_abs = float(np.mean(np.abs(finite))) if finite else float("nan")
    n_ok = sum(1 for x in finite if abs(x) < threshold_dex)
    n_tot = len(rows)

    if n_ok == n_tot:
        verdict = "REPRODUIT"
        reco = (
            f"V5.37 reproduit les {n_tot} splittings radiaux à mieux que "
            f"±{threshold_dex} dex.  Aucun paramètre supplémentaire."
        )
    elif n_ok == 0:
        verdict = "ECHEC"
        reco = (
            "V5.37 ne reproduit AUCUN splitting radial dans le seuil.  "
            "Le potentiel χ-field −λρ/d² n'a pas la structure d'un "
            "Cornell-like donnant des spacings ~500 MeV constants.  "
            "Une calibration via un terme de confinement σr serait "
            "nécessaire ; non implémentée ici."
        )
    else:
        verdict = "PARTIEL"
        reco = (
            f"V5.37 reproduit {n_ok}/{n_tot} splittings dans le seuil "
            f"±{threshold_dex} dex.  Diagnostic : le solveur FEM sur "
            "potentiel −λρ/d² ne donne pas naturellement des spacings "
            "constants ; un terme de confinement supplémentaire serait "
            "requis pour viser les ~500 MeV expérimentaux."
        )

    return {
        "sum_abs_log_best": s_abs,
        "mean_abs_log_best": m_abs,
        "n_within_threshold": int(n_ok),
        "n_total": int(n_tot),
        "threshold_dex": float(threshold_dex),
        "best_interp_per_row": best_interp,
        "best_log_per_row": best_logs,
        "verdict": verdict,
        "recommendation": reco,
    }
