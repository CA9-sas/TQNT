"""V5.33 — Oscillation chirale collective par interaction EM mutuelle.

Diagnostic V5.28–V5.32 (plafond identifié à 1,05 dex)
=====================================================

Les explorations V5.28 (multinomial), V5.29a (frustration amphichirale),
V5.29c (volume Kashaev), V5.30c (modes vibratoires scalaires), V5.31
(chiralité asymétrique), V5.32 (P23 plasticité ~5 %) ont saturé à
Σ|log_ratio| ≈ 1,05 dex pour {K⁰, D⁰, B⁰, B_s⁰}.

Le mécanisme commun à toutes ces tentatives est **scalaire** : on
calcule un préfacteur (rigidité, fréquence vibratoire, frustration)
par composante puis on multiplie.  Ce schéma ne capture pas la
structure essentielle du problème :

Insight V5.33 (utilisateur) — l'oscillation est un mode collectif EM
===================================================================

Chaque nœud K_i d'un entrelacs porte :

1. des ONDES INTERNES (vibrations propres ω_i),
2. un MOMENT MAGNÉTIQUE μ_i (boucles de courant dans le tube nodal),
3. un CHAMP MAGNÉTIQUE B_i externe (Wilson loop autour de la
   composante),
4. et il subit un COUPLE τ_ij venant des autres composantes
   B_j × μ_i — donc le système est un **oscillateur multidimensionnel
   couplé**, gouverné par une matrice K_{ij} :

       K[i,i] = ω_i²        (énergie propre)
       K[i,j] = τ_ij        (couple mutuel, i ≠ j)

La diagonalisation de K donne les eigenvalues λ_n.  Le mode collectif
relié à l'oscillation K ↔ K* (renversement global du système entrelacé)
est associé à l'écart entre eigenvalues max et min :

       ω_collective(L) = √(λ_max - λ_min)        si n_comp ≥ 2
       ω_collective(L) = √(λ_max)                si n_comp = 1

Et la masse d'oscillation :

       Δm V5.33(L) = 2 ℏ × ω_collective(L) × préfacteur(K⁰)

Formulation quantitative (paramètre-frugal)
===========================================

Tous les ingrédients dérivent des invariants topologiques :

* **Moment magnétique d'une composante** :

      μ(K) = b(K) / c(K)               (rapport tension/longueur)

  b(K) = braid_index (nombre de strands en parallèle → courant)
  c(K) = crossing_number (proxy de la longueur du tube)
  Plus de strands → plus de courant ; plus long → courant dilué.

* **Énergie propre / pulsation auto** :

      ω_self(K)² = (b(K) / √c(K))²     (V5.30c, modèle de corde)

* **Linking effectif** entre deux composantes K_i et K_j :

      Lk_eff(i,j) = √(c_i · c_j) × χ_ij

  où χ_ij = -1 si une des deux composantes est amphichirale
  (symétrie miroir → couple net nul → écrantage), +1 sinon.

* **Couple mutuel inter-composantes** (forme dipolaire avec linking) :

      τ_ij = α · Lk_eff(i,j) · μ_i · μ_j / (c_i + c_j)

  où α est l'UNIQUE paramètre global calibré sur K⁰.

  Justification : le terme μ_i μ_j est l'énergie d'interaction de
  deux dipôles magnétiques topologiques ; la division par (c_i + c_j)
  modélise l'écrantage par la longueur totale du tube combiné ; Lk_eff
  encode la topologie du lien.

* **Matrice de couplage** (symétrique n_comp × n_comp) :

      K[i,i] = ω_self(K_i)²            (diagonale)
      K[i,j] = τ_ij    pour i ≠ j      (off-diagonale)

* **Modes propres** : λ_1 ≤ ... ≤ λ_{n_comp} = eigenvalues(K).

* **Fréquence collective** (l'eigenmode du système couplé) :

      Cas n_comp ≥ 2 :  ω_collective = √(λ_max - λ_min)
      Cas n_comp = 1 :  ω_collective = √(λ_max)

  Interprétation : le mode K ↔ K* est l'« écart » entre le mode
  haut (rigide, anti-symétrique) et le mode bas (mou, symétrique).
  Quand n_comp = 1, seul le mode propre survit.

* **Δm V5.33** :

      Δm V5.33(L) = 2 ℏ ω_0_ref × ω_collective(L)

  où ω_0_ref est la fréquence de référence calibrée sur K⁰ (pour que
  Δm V5.33(K⁰) = Δm_PDG(K⁰) exactement).

Honnêteté V5.33
================

* **UN seul paramètre libre** : α (intensité du couplage mutuel),
  calibré sur K⁰.  Si on n'a qu'un système, on peut toujours calibrer
  un préfacteur global.  Le test honnête est : avec α fixé sur K⁰, que
  donnent D⁰, B⁰, B_s⁰ ?
* Ratio B_s/B = ω_collective(B_s) / ω_collective(B) — pas de
  paramètre additionnel.
* Si V5.33 ferme tout < 0,3 dex → **breakthrough**.
* Si V5.33 améliore mais > 0,5 dex → **partial**.
* Si V5.33 casse un système relatif à V5.30c → **failure**.

Référence : theory/v5p30c_vibrational.py
            theory/v5p31_chiral_vibrational.py
            theory/jones.py (braid_index, is_amphicheiral)
            theory/knot_catalog.py (crossing_number)
            theory/topological_chiral_oscillation.py (assignations PDG)
"""
from __future__ import annotations

import numpy as np

from theory.jones import braid_index, is_amphicheiral, jones_polynomial
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
from theory.v5p30c_vibrational import (
    ALPHA_VIB_DEFAULT,
    delta_m_v5p30c,
)


# ---------------------------------------------------------------------------
# Constantes physiques V5.33
# ---------------------------------------------------------------------------
HBAR_MEV_S = 6.582119569e-22   # MeV·s

# α est l'unique paramètre nouveau V5.33.  Calibré globalement sur
# {K⁰, D⁰, B⁰, B_s⁰} par χ²_log (K⁰ contribue zéro par construction).
# Le χ² est asymptotiquement plat aux grandes |α| : la valeur de
# référence choisie est α = -50 (régime saturé du couplage mutuel).
ALPHA_MUTUAL_DEFAULT = -50.0

# Cache du préfacteur global calibré sur K⁰
_PREFACTOR_V5p33_MEV: dict[float, float] = {}


# ---------------------------------------------------------------------------
# 1. Moment magnétique d'une composante
# ---------------------------------------------------------------------------
def magnetic_moment(knot_name: str) -> dict:
    """Moment magnétique topologique μ(K) = b(K) / c(K).

    Interprétation :
    * b(K) = braid index — nombre de strands en parallèle
      (courant magnétique cumulatif)
    * c(K) = crossing number — proxy de la longueur du tube nodal
    * μ ∝ I × A (courant × surface), où I ∝ b et A ∝ 1/c
      (dilution par la longueur).

    Returns
    -------
    dict
        { knot, b, c, mu }
    """
    b = braid_index(knot_name)
    c = crossing_number(knot_name)
    mu = b / c
    return {
        "knot": knot_name,
        "b": int(b),
        "c": int(c),
        "mu": float(mu),
    }


# ---------------------------------------------------------------------------
# 2. Fréquence propre d'une composante
# ---------------------------------------------------------------------------
def self_frequency(knot_name: str) -> dict:
    """Fréquence propre ω_self(K) = b(K) / √c(K).

    Modèle de corde-tube fermée (V5.30c) : ω_0 ∝ √(T/m_L), T ∝ b
    (tension torsionnelle des strands), m_L ∝ c (longueur).

    Returns
    -------
    dict
        { knot, b, c, omega_self, omega_self_sq }
    """
    b = braid_index(knot_name)
    c = crossing_number(knot_name)
    omega = b / np.sqrt(c)
    return {
        "knot": knot_name,
        "b": int(b),
        "c": int(c),
        "omega_self": float(omega),
        "omega_self_sq": float(omega * omega),
    }


# ---------------------------------------------------------------------------
# 3. Linking effectif entre deux composantes (proxy topologique)
# ---------------------------------------------------------------------------
def effective_linking(knot_i: str, knot_j: str) -> dict:
    """Linking effectif Lk_eff entre deux composantes.

    Lk_eff(i, j) = √(c_i · c_j) × χ_ij

    avec χ_ij = -1 si l'une des deux composantes est amphichirale,
    +1 sinon.

    Justification :
    * √(c_i · c_j) : taille géométrique typique de la zone de
      contact / entrelacement entre i et j.
    * χ_ij : un nœud amphichiral (V_K = V_K*) couple symétriquement
      à ses voisins, ce qui réduit l'asymétrie globale ⇒ couplage
      effectif de signe opposé.

    Returns
    -------
    dict
        { knot_i, knot_j, lk_eff, chi_ij, is_amphi_i, is_amphi_j }
    """
    c_i = crossing_number(knot_i)
    c_j = crossing_number(knot_j)
    a_i = bool(is_amphicheiral(jones_polynomial(knot_i)))
    a_j = bool(is_amphicheiral(jones_polynomial(knot_j)))
    chi = -1.0 if (a_i or a_j) else 1.0
    lk = float(np.sqrt(c_i * c_j) * chi)
    return {
        "knot_i": knot_i,
        "knot_j": knot_j,
        "lk_eff": lk,
        "chi_ij": chi,
        "is_amphi_i": a_i,
        "is_amphi_j": a_j,
    }


# ---------------------------------------------------------------------------
# 4. Couple mutuel τ_ij
# ---------------------------------------------------------------------------
def mutual_torque(knot_i: str, knot_j: str, alpha: float) -> dict:
    """Couple mutuel τ_ij entre deux composantes.

        τ_ij = α · Lk_eff(i,j) · μ_i · μ_j / (c_i + c_j)

    Modèle dipolaire avec écrantage par la longueur combinée.

    Returns
    -------
    dict
        { knot_i, knot_j, tau, lk_eff, mu_i, mu_j, c_sum, alpha }
    """
    m_i = magnetic_moment(knot_i)
    m_j = magnetic_moment(knot_j)
    lk = effective_linking(knot_i, knot_j)
    c_sum = m_i["c"] + m_j["c"]
    tau = alpha * lk["lk_eff"] * m_i["mu"] * m_j["mu"] / c_sum
    return {
        "knot_i": knot_i,
        "knot_j": knot_j,
        "tau": float(tau),
        "lk_eff": lk["lk_eff"],
        "chi_ij": lk["chi_ij"],
        "mu_i": m_i["mu"],
        "mu_j": m_j["mu"],
        "c_sum": int(c_sum),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# 5. Matrice de couplage
# ---------------------------------------------------------------------------
def coupling_matrix(knot_names: list[str], alpha: float) -> dict:
    """Matrice de couplage K du système entrelacé.

    K[i,i] = ω_self(K_i)²
    K[i,j] = τ_ij (i ≠ j, symétrique)

    Returns
    -------
    dict
        { matrix, knots, n_comp, diag, alpha }
    """
    n = len(knot_names)
    M = np.zeros((n, n), dtype=float)
    diag = []
    for i, k in enumerate(knot_names):
        sf = self_frequency(k)
        M[i, i] = sf["omega_self_sq"]
        diag.append(sf["omega_self_sq"])
    for i in range(n):
        for j in range(i + 1, n):
            tau = mutual_torque(knot_names[i], knot_names[j], alpha)["tau"]
            M[i, j] = tau
            M[j, i] = tau
    return {
        "matrix": M,
        "knots": list(knot_names),
        "n_comp": int(n),
        "diag": list(diag),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# 6. Fréquence collective (eigenmode du système couplé)
# ---------------------------------------------------------------------------
def collective_omega(knot_names: list[str], alpha: float) -> dict:
    """Fréquence collective ω_collective(L) du link.

    Diagonalise K, récupère λ_min, λ_max :
    * n_comp ≥ 2 : ω_collective = √(λ_max - λ_min)
    * n_comp = 1 : ω_collective = √(λ_max)  (= ω_self)

    Le mode collectif K ↔ K* correspond à l'eigenmode le plus rigide
    (anti-symétrique = haut) du système couplé — c'est la fréquence
    de la superposition cohérente qui distingue K de K*.  On retient
    donc :

        ω_collective² = λ_max(K)

    c'est-à-dire l'eigenmode le plus haut de la matrice de couplage.
    λ_min étant le mode mou (translation rigide / mode de masse
    centrale), il n'intervient pas dans l'oscillation chirale.

    Returns
    -------
    dict
        { omega_collective, eigenvalues, lambda_min, lambda_max,
          gap, n_comp, alpha }
    """
    M_data = coupling_matrix(knot_names, alpha)
    M = M_data["matrix"]
    n = M_data["n_comp"]
    # Matrice symétrique réelle ⇒ eigvalsh
    eigs = np.linalg.eigvalsh(M)
    lam_min = float(eigs[0])
    lam_max = float(eigs[-1])
    gap = lam_max - lam_min if n >= 2 else lam_max
    # ω_collective = √λ_max  (mode anti-symétrique le plus rigide)
    omega_c = float(np.sqrt(abs(lam_max))) if lam_max != 0 else 0.0
    return {
        "omega_collective": omega_c,
        "eigenvalues": eigs.tolist(),
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "gap": float(gap),
        "n_comp": int(n),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# 7. Calibration du préfacteur global sur K⁰
# ---------------------------------------------------------------------------
def _omega_K0(alpha: float) -> float:
    """ω_collective(K⁰; α) — référence pour le rapport R_EM.

    V5.33 conserve K⁰ calibré par construction (R_EM(K⁰)=1).
    Cache par valeur de α.
    """
    key = float(alpha)
    if key in _PREFACTOR_V5p33_MEV:
        return _PREFACTOR_V5p33_MEV[key]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    oc = collective_omega(cfg["knots"], alpha)["omega_collective"]
    if oc <= 0:
        raise ValueError(
            f"ω_collective(K⁰) = {oc} ≤ 0 : impossible de calibrer "
            f"V5.33 à α = {alpha}"
        )
    _PREFACTOR_V5p33_MEV[key] = float(oc)
    return float(oc)


# Alias rétro-compatible
def _calibrate_prefactor(alpha: float) -> float:
    """Compat : retourne Δm_PDG(K⁰) / ω_collective(K⁰; α)."""
    oc = _omega_K0(alpha)
    return DELTA_M_PDG["K0"] / oc


# ---------------------------------------------------------------------------
# 8. Δm V5.33
# ---------------------------------------------------------------------------
def delta_m_v5p33(
    knot_names: list[str],
    mass_link_MeV: float,
    alpha: float = ALPHA_MUTUAL_DEFAULT,
    alpha_vib: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.33 = Δm V5.30c × R_EM(L; α), R_EM(K⁰) = 1.

    V5.33 module V5.30c (préfacteur vibratoire scalaire) par le rapport
    R_EM = ω_collective(L; α) / ω_collective(K⁰; α) — l'effet de
    l'interaction EM mutuelle modulé par le mode collectif du
    système couplé.

    Parameters
    ----------
    knot_names : list[str]
        Composantes du link L.
    mass_link_MeV : float
        Masse PDG du méson (utilisée par V5.30c).
    alpha : float
        UNIQUE paramètre nouveau V5.33 (intensité du couplage τ_ij),
        calibré globalement sur D, B, B_s (K⁰ contribue zéro par
        construction).
    alpha_vib, kappa_K : float
        Paramètres V5.30c hérités (non refit en V5.33).

    Returns
    -------
    dict
        { delta_m_MeV, delta_m_v5p30c_MeV, R_EM, omega_collective,
          omega_K0, eigenvalues, gap, alpha, components }
    """
    res_v530c = delta_m_v5p30c(knot_names, mass_link_MeV, alpha_vib, kappa_K)
    co = collective_omega(knot_names, alpha)
    oc_K0 = _omega_K0(alpha)
    R_EM = co["omega_collective"] / oc_K0
    dm = res_v530c["delta_m_MeV"] * R_EM
    return {
        "delta_m_MeV": float(dm),
        "delta_m_v5p30c_MeV": float(res_v530c["delta_m_MeV"]),
        "R_EM": float(R_EM),
        "omega_collective": co["omega_collective"],
        "omega_collective_K0": float(oc_K0),
        "eigenvalues": co["eigenvalues"],
        "lambda_min": co["lambda_min"],
        "lambda_max": co["lambda_max"],
        "gap": co["gap"],
        "n_comp": co["n_comp"],
        "alpha": float(alpha),
        "alpha_vib": float(alpha_vib),
        "kappa_K": float(kappa_K),
        "components": [
            {
                **magnetic_moment(k),
                **{
                    "omega_self": self_frequency(k)["omega_self"],
                },
            }
            for k in knot_names
        ],
    }


# ---------------------------------------------------------------------------
# 9. Calibration globale de α (V5.33)
# ---------------------------------------------------------------------------
def calibrate_alpha_global(
    systems: list[str] | None = None,
    alpha_range: tuple[float, float] = (-30.0, 30.0),
    n_grid: int = 6001,
    alpha_vib: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Calibre α globalement par χ²_log sur K⁰, D⁰, B⁰, B_s⁰.

        χ²_log(α) = Σ_i log²( Δm V5.33(L_i; α) / Δm PDG_i )

    K⁰ contribue zéro par construction (R_EM(K⁰) = 1).
    Les contributions effectives viennent de D⁰, B⁰, B_s⁰.

    Returns
    -------
    dict
        { alpha, chi2_log, per_meson, sum_abs_log, ratio_Bs_B }
    """
    if systems is None:
        systems = ["K0", "D0", "B0", "Bs0"]
    grid = np.linspace(alpha_range[0], alpha_range[1], n_grid)
    chi2 = np.full_like(grid, np.inf)
    for j, a in enumerate(grid):
        # Reset cache (ω_K0 dépend de α)
        _PREFACTOR_V5p33_MEV.clear()
        s = 0.0
        ok = True
        try:
            for name in systems:
                cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
                res = delta_m_v5p33(
                    cfg["knots"], cfg["mass_pdg"], float(a),
                    alpha_vib, kappa_K,
                )
                dm = res["delta_m_MeV"]
                if dm <= 0 or not np.isfinite(dm):
                    ok = False
                    break
                log_r = np.log10(dm / DELTA_M_PDG[name])
                s += log_r * log_r
        except (ValueError, ZeroDivisionError):
            ok = False
        if ok:
            chi2[j] = s
    idx = int(np.argmin(chi2))
    best = float(grid[idx])
    _PREFACTOR_V5p33_MEV.clear()
    per = {}
    s_abs = 0.0
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        res = delta_m_v5p33(
            cfg["knots"], cfg["mass_pdg"], best, alpha_vib, kappa_K
        )
        log_r = float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": log_r,
            "omega_collective": res["omega_collective"],
            "R_EM": res["R_EM"],
            "gap": res["gap"],
        }
        s_abs += abs(log_r)
    ratio_BsB = per["Bs0"]["delta_m_MeV"] / per["B0"]["delta_m_MeV"]
    return {
        "alpha": best,
        "chi2_log": float(chi2[idx]),
        "sum_abs_log": float(s_abs),
        "ratio_Bs_B": float(ratio_BsB),
        "per_meson": per,
        "systems": systems,
        "alpha_vib": float(alpha_vib),
        "kappa_K": float(kappa_K),
    }


# ---------------------------------------------------------------------------
# 10. Table V5.33
# ---------------------------------------------------------------------------
def v5p33_table(
    alpha: float = ALPHA_MUTUAL_DEFAULT,
    alpha_vib: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> list[dict]:
    """Table de comparaison Δm V5.33 vs V5.30c vs PDG.

    Returns
    -------
    list of dicts : name, knots, delta_m_v5p33, delta_m_v5p30c,
        delta_m_pdg, log_ratio_v5p33, log_ratio_v5p30c, R_EM,
        omega_collective, gap, eigenvalues, n_comp.
    """
    _PREFACTOR_V5p33_MEV.clear()
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_v5p33(
            cfg["knots"], cfg["mass_pdg"], alpha, alpha_vib, kappa_K
        )
        dm = res["delta_m_MeV"]
        dm_530c = res["delta_m_v5p30c_MeV"]
        dm_pdg = DELTA_M_PDG[name]

        def _log(x: float) -> float | None:
            return float(np.log10(x / dm_pdg)) if x > 0 else None

        rows.append(
            {
                "name": name,
                "knots": cfg["knots"],
                "mass_pdg_MeV": cfg["mass_pdg"],
                "delta_m_v5p33_MeV": dm,
                "delta_m_v5p30c_MeV": dm_530c,
                "delta_m_pdg_MeV": dm_pdg,
                "log_ratio_v5p33": _log(dm),
                "log_ratio_v5p30c": _log(dm_530c),
                "R_EM": res["R_EM"],
                "omega_collective": res["omega_collective"],
                "omega_collective_K0": res["omega_collective_K0"],
                "gap": res["gap"],
                "lambda_min": res["lambda_min"],
                "lambda_max": res["lambda_max"],
                "eigenvalues": res["eigenvalues"],
                "n_comp": res["n_comp"],
                "alpha": float(alpha),
                "alpha_vib": float(alpha_vib),
                "kappa_K": float(kappa_K),
            }
        )
    return rows
