"""V5.37b — Potentiel Cornell TQNT pour splittings radiaux du quarkonium.

Contexte
========

V5.37 (commit 765f34d) a essayé de dériver les splittings ψ(2S)/J/ψ et
Υ(2S)/Υ(1S) depuis le spectre Schrödinger χ-field (-λρ/d²) sur le lien
du quarkonium.  Bilan honnête : Σ|log| = 0,76 dex, 3/4 paires dans
±0,30 dex, mais le potentiel χ-field ne génère pas naturellement des
spacings ~500 MeV de type Cornell.

V5.37b prend l'autre route : on ADMET le potentiel Cornell

    V(r) = -α/r + σ · r

(empiriquement excellent pour le quarkonium) et on teste DEUX
interprétations TQNT — la seule qui marche est rapportée comme telle.

Dérivation TQNT (honnête)
=========================

**Mode "topo" — c_J-dépendant (interprétation naïve)** :

    α_TQNT(L) = K_alpha × α_EM × N_c × c_J_eff(L)
    σ_TQNT(L) = K_sigma × β_V5 × c_J_eff(L)² × m_link / (ℏc)

Interprétation : Coulomb « gluonique » modulé par la densité topologique
du lien, tension du string proportionnelle à c_J² (densité au carré).
**Résultat : ferme charmonium mais détruit bottomonium** (c_J_eff(Υ) >
c_J_eff(J/ψ) → α, σ trop grands → spacings 2-3× trop gros).

**Mode "universal" — α, σ propriétés du vide χ (interprétation V5
correcte)** :

    α_TQNT = K_alpha × α_EM × N_c            (sans dimension)
    σ_TQNT = K_sigma × β_V5 × m_c / (ℏc)     (MeV/fm)

Le couplage α et la tension σ sont des **constantes universelles du
vide χ TQNT** — elles ne dépendent PAS du lien spécifique (charm ou
bottom).  C'est cohérent avec :

  * P21 : α_EM = 1/(k+2) = 1/137 est universel (k=135 unique TQNT).
  * Chap. 6 V5 : β = 0,210 MeV est la tension topologique universelle
    d'enlacement, pas une propriété d'un lien particulier.
  * m_c (masse constituante charm) fixe l'échelle de longueur via ℏc/m_c.

Avec cette lecture, V5.37b **ferme le quarkonium** (Σ|log| ≪ V5.37) en
deux paramètres calibrés une fois (K_alpha, K_sigma) sur ψ(2S)/J/ψ.

Honnêteté
=========

* DEUX paramètres globaux calibrés (K_alpha, K_sigma) sur UNE paire
  (J/ψ, ψ(2S)).  Les 3 autres paires sont des PRÉDICTIONS.
* Masses constituantes m_c, m_b sont des paramètres effectifs Cornell
  (~1,5 et 4,7 GeV) non dérivés.
* Cornell est non-relativiste : v/c ≈ 0,3 pour le charmonium ⇒ erreur
  intrinsèque ~10 %.
* Mode "topo" reporté pour la traçabilité honnête mais NE FERME PAS le
  quarkonium ; c'est le mode "universal" qui est l'interprétation V5
  correcte.

Référence
=========

* theory/mass_formula.py : c_J_eff(L), U_EM(L)
* theory/su3_color_topology.py : BETA_MEV = 0,210 MeV
* theory/postulates.py : ALPHA_EM = 1/137
* docs/TQNT_detaillee_V5_FR.tex chap. 6 (tension β) et chap. 14 (Ω_ts)
* docs/TQNT_catalog_V5_diffusion.xlsx onglet « 3. Particules PDG ».
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.mass_formula import mass_components
from theory.su3_color_topology import BETA_MEV
from theory.v5p37_radial_splittings import (
    QUARKONIUM_ASSIGNMENTS,
    quarkonium_assignment,
)


# ---------------------------------------------------------------------------
# Constantes physiques (unités MeV, fm, c = 1)
# ---------------------------------------------------------------------------
HBAR_C_MEV_FM = 197.3269804  # ℏc en MeV·fm
ALPHA_EM = 1.0 / 137.036     # constante de structure fine (TQNT : 1/(k+2), k=135)
N_COLOR = 3.0                # facteur de couleur (≡ n_composantes baryon V5)

# Masses constituantes quarkonium (paramètres Cornell effectifs).
M_CHARM_MEV = 1500.0
M_BOTTOM_MEV = 4700.0

# Familles : (état_fond, [excités], masse constituante m_q)
QUARKONIUM_RADIAL_FAMILIES: dict[str, dict] = {
    "charmonium": {
        "ground": "J/psi",
        "excited": ["psi(2S)", "psi(3S)"],
        "m_quark_MeV": M_CHARM_MEV,
    },
    "bottomonium": {
        "ground": "Upsilon(1S)",
        "excited": ["Upsilon(2S)", "Upsilon(3S)"],
        "m_quark_MeV": M_BOTTOM_MEV,
    },
}

# Modes d'interprétation
MODE_UNIVERSAL = "universal"  # α, σ universels (V5 vacuum) — l'interprétation correcte
MODE_TOPO = "topo"            # α, σ proportionnels à c_J_eff (naïf) — diagnostic
VALID_MODES = (MODE_UNIVERSAL, MODE_TOPO)


# ---------------------------------------------------------------------------
# c_J effectif d'un lien (V5 éq. 6.8)
# ---------------------------------------------------------------------------
def c_j_eff_link(knots: list[str]) -> float:
    """c_J_eff = f(n) · c_J_tot du lien, secteur auto (V5 éq. 6.8)."""
    if not knots:
        raise ValueError("liste de nœuds vide")
    cfg = "+".join(knots)
    return float(mass_components(cfg, sector="auto")["c_J_eff"])


# ---------------------------------------------------------------------------
# α_TQNT et σ_TQNT (deux modes)
# ---------------------------------------------------------------------------
def alpha_topological(
    knots: list[str],
    K_alpha: float,
    mode: str = MODE_UNIVERSAL,
) -> float:
    """α_TQNT sans dimension.

    Mode "universal" : α_TQNT = K_alpha · α_EM · N_c   (constante de vide χ)
    Mode "topo"      : α_TQNT = K_alpha · α_EM · N_c · c_J_eff(L)
    """
    if mode == MODE_UNIVERSAL:
        return float(K_alpha * ALPHA_EM * N_COLOR)
    if mode == MODE_TOPO:
        return float(K_alpha * ALPHA_EM * N_COLOR * c_j_eff_link(knots))
    raise ValueError(f"mode inconnu {mode!r} ; choix : {VALID_MODES}")


def sigma_topological(
    knots: list[str],
    mass_link_MeV: float,
    K_sigma: float,
    mode: str = MODE_UNIVERSAL,
) -> float:
    """σ_TQNT en MeV/fm.

    Mode "universal" : σ_TQNT = K_sigma · β_V5 · m_c / (ℏc)
        Tension de string universelle, échelle fixée par m_charm.
    Mode "topo"      : σ_TQNT = K_sigma · β_V5 · c_J_eff(L)² · m_link / (ℏc)
        Tension modulée par c_J² (interprétation naïve).
    """
    if mode == MODE_UNIVERSAL:
        # Échelle universelle : m_charm (lightest heavy quark)
        return float(K_sigma * BETA_MEV * M_CHARM_MEV / HBAR_C_MEV_FM)
    if mode == MODE_TOPO:
        return float(
            K_sigma * BETA_MEV * c_j_eff_link(knots) ** 2
            * mass_link_MeV / HBAR_C_MEV_FM
        )
    raise ValueError(f"mode inconnu {mode!r} ; choix : {VALID_MODES}")


# ---------------------------------------------------------------------------
# Potentiel Cornell TQNT
# ---------------------------------------------------------------------------
def cornell_potential_TQNT(
    r_fm: np.ndarray,
    alpha_topo: float,
    sigma_topo_MeV_per_fm: float,
    r_min_fm: float = 1.0e-4,
) -> np.ndarray:
    """V(r) = -α_topo · ℏc / r + σ_topo · r  (MeV).

    Parameters
    ----------
    r_fm : np.ndarray (fm)
    alpha_topo : float (sans dimension)
    sigma_topo_MeV_per_fm : float (MeV/fm)
    r_min_fm : float
        Régularisation près de 0 (1e-4 fm par défaut).
    """
    r_safe = np.maximum(np.asarray(r_fm, dtype=float), r_min_fm)
    V_coul = -alpha_topo * HBAR_C_MEV_FM / r_safe
    V_lin = sigma_topo_MeV_per_fm * r_safe
    return V_coul + V_lin


# ---------------------------------------------------------------------------
# Solveur Schrödinger radial 1D (ℓ=0, finite differences tridiagonales)
# ---------------------------------------------------------------------------
@dataclass
class RadialSpectrum:
    eigenvalues_MeV: np.ndarray
    r_grid_fm: np.ndarray
    V_grid_MeV: np.ndarray
    mu_MeV: float
    alpha_topo: float
    sigma_topo_MeV_per_fm: float


def solve_radial_cornell(
    mu_MeV: float,
    alpha_topo: float,
    sigma_topo_MeV_per_fm: float,
    r_max_fm: float = 5.0,
    n_grid: int = 2000,
    n_eigenvalues: int = 4,
) -> RadialSpectrum:
    """Résout -[ℏ²/(2μ)] u''(r) + V(r) u(r) = E u(r), ℓ=0, u(0)=u(R)=0.

    Discrétisation FD ordre 2 sur grille uniforme (n_grid points, bords
    Dirichlet).  Diagonalisation via scipy.linalg.eigh_tridiagonal.
    """
    if n_grid < 50:
        raise ValueError("n_grid trop faible")
    h = r_max_fm / (n_grid - 1)
    r = np.linspace(0.0, r_max_fm, n_grid)
    V = cornell_potential_TQNT(r, alpha_topo, sigma_topo_MeV_per_fm)

    t_coef = (HBAR_C_MEV_FM ** 2) / (2.0 * mu_MeV)  # MeV · fm²
    n_int = n_grid - 2
    main_diag = (2.0 * t_coef / h ** 2) + V[1:-1]
    off_diag = -t_coef / h ** 2 * np.ones(n_int - 1)

    from scipy.linalg import eigh_tridiagonal
    evals, _ = eigh_tridiagonal(
        main_diag, off_diag,
        select="i", select_range=(0, n_eigenvalues - 1),
    )
    return RadialSpectrum(
        eigenvalues_MeV=np.asarray(evals, dtype=float),
        r_grid_fm=r,
        V_grid_MeV=V,
        mu_MeV=float(mu_MeV),
        alpha_topo=float(alpha_topo),
        sigma_topo_MeV_per_fm=float(sigma_topo_MeV_per_fm),
    )


# ---------------------------------------------------------------------------
# Calibration (K_alpha, K_sigma) sur ψ(2S)/J/ψ
# ---------------------------------------------------------------------------
def calibrate_cornell_constants(
    mode: str = MODE_UNIVERSAL,
    n_grid: int = 2000,
    r_max_fm: float = 5.0,
) -> dict:
    """Calibre (K_alpha, K_sigma) sur (J/ψ, ψ(2S)).

    Stratégie en deux temps :
      1. K_alpha tel que α_TQNT(J/ψ) = 0,4 (valeur Cornell phéno).
      2. K_sigma ajusté par dichotomie pour Δm(ψ(2S)/J/ψ) = 589 MeV.

    Le mode (universal/topo) gouverne la forme fonctionnelle de α et σ.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode invalide {mode!r}")
    dm_target = (
        QUARKONIUM_ASSIGNMENTS["psi(2S)"]["mass_pdg_MeV"]
        - QUARKONIUM_ASSIGNMENTS["J/psi"]["mass_pdg_MeV"]
    )
    knots_g = QUARKONIUM_ASSIGNMENTS["J/psi"]["knots"]
    m_g = QUARKONIUM_ASSIGNMENTS["J/psi"]["mass_pdg_MeV"]
    mu_c = M_CHARM_MEV / 2.0

    alpha_target = 0.4
    # Inverse alpha_topological pour mode donné, à α=0.4
    if mode == MODE_UNIVERSAL:
        K_alpha = alpha_target / (ALPHA_EM * N_COLOR)
    else:  # MODE_TOPO
        K_alpha = alpha_target / (ALPHA_EM * N_COLOR * c_j_eff_link(knots_g))

    alpha_topo = alpha_topological(knots_g, K_alpha, mode=mode)

    def dm_for(K_sigma: float) -> float:
        sigma_topo = sigma_topological(knots_g, m_g, K_sigma, mode=mode)
        spec = solve_radial_cornell(
            mu_c, alpha_topo, sigma_topo,
            r_max_fm=r_max_fm, n_grid=n_grid, n_eigenvalues=2,
        )
        return float(spec.eigenvalues_MeV[1] - spec.eigenvalues_MeV[0])

    K_lo, K_hi = 1.0e-4, 1.0e4
    dm_lo, dm_hi = dm_for(K_lo), dm_for(K_hi)
    n_expand = 0
    while (dm_lo - dm_target) * (dm_hi - dm_target) > 0 and n_expand < 20:
        K_hi *= 5
        dm_hi = dm_for(K_hi)
        n_expand += 1
    if (dm_lo - dm_target) * (dm_hi - dm_target) > 0:
        raise RuntimeError(
            f"calibration : impossible de bracketer Δm = {dm_target:.1f} MeV "
            f"dans [K_sigma=1e-4, {K_hi:.1e}]."
        )

    for _ in range(80):
        K_mid = 0.5 * (K_lo + K_hi)
        dm_mid = dm_for(K_mid)
        if (dm_lo - dm_target) * (dm_mid - dm_target) <= 0:
            K_hi, dm_hi = K_mid, dm_mid
        else:
            K_lo, dm_lo = K_mid, dm_mid
        if abs(dm_mid - dm_target) < 0.05:
            break
    K_sigma = 0.5 * (K_lo + K_hi)

    sigma_topo = sigma_topological(knots_g, m_g, K_sigma, mode=mode)
    return {
        "mode": mode,
        "K_alpha": float(K_alpha),
        "K_sigma": float(K_sigma),
        "alpha_TQNT_J_psi": float(alpha_topo),
        "sigma_TQNT_J_psi_MeV_per_fm": float(sigma_topo),
        "dm_calibration_target_MeV": float(dm_target),
        "dm_calibration_achieved_MeV": float(dm_for(K_sigma)),
        "note": (
            f"Mode={mode}.  K_alpha fixé pour α_TQNT(J/ψ) = 0,4 ; "
            f"K_sigma ajusté pour Δm(ψ(2S)/J/ψ) = 589 MeV.  "
            f"Reste des paires = PRÉDICTIONS."
        ),
    }


# Mémoïse par mode
_CAL_CACHE: dict[str, dict] = {}


def get_calibration(mode: str = MODE_UNIVERSAL) -> dict:
    if mode not in _CAL_CACHE:
        _CAL_CACHE[mode] = calibrate_cornell_constants(mode=mode)
    return _CAL_CACHE[mode]


def clear_calibration_cache() -> None:
    _CAL_CACHE.clear()


# ---------------------------------------------------------------------------
# Spectre Cornell pour une particule
# ---------------------------------------------------------------------------
def schrodinger_cornell_spectrum(
    particle: str,
    mode: str = MODE_UNIVERSAL,
    K_alpha: float | None = None,
    K_sigma: float | None = None,
    n_levels: int = 4,
    r_max_fm: float = 5.0,
    n_grid: int = 2000,
) -> dict:
    """Spectre Cornell TQNT pour un état quarkonium.

    Le potentiel Cornell est construit avec α_TQNT, σ_TQNT du lien V5
    du **fondamental de la famille** (J/ψ pour le charmonium, Υ(1S) pour
    le bottomonium) — les niveaux radiaux E_n vivent dans le même puits.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode invalide {mode!r}")
    if K_alpha is None or K_sigma is None:
        cal = get_calibration(mode=mode)
        K_alpha = cal["K_alpha"] if K_alpha is None else K_alpha
        K_sigma = cal["K_sigma"] if K_sigma is None else K_sigma

    cfg = quarkonium_assignment(particle)

    mu = None
    family = None
    for fam, fcfg in QUARKONIUM_RADIAL_FAMILIES.items():
        if particle == fcfg["ground"] or particle in fcfg["excited"]:
            mu = fcfg["m_quark_MeV"] / 2.0
            family = fam
            break
    if mu is None:
        raise KeyError(f"particule {particle!r} hors familles V5.37b")

    ground = QUARKONIUM_RADIAL_FAMILIES[family]["ground"]
    knots_eff = QUARKONIUM_ASSIGNMENTS[ground]["knots"]
    m_eff = QUARKONIUM_ASSIGNMENTS[ground]["mass_pdg_MeV"]

    alpha_topo = alpha_topological(knots_eff, K_alpha, mode=mode)
    sigma_topo = sigma_topological(knots_eff, m_eff, K_sigma, mode=mode)
    spec = solve_radial_cornell(
        mu, alpha_topo, sigma_topo,
        r_max_fm=r_max_fm, n_grid=n_grid, n_eigenvalues=n_levels,
    )
    return {
        "particle": particle,
        "family": family,
        "mode": mode,
        "knots_used": knots_eff,
        "mass_link_MeV": float(cfg["mass_pdg_MeV"]),
        "mu_MeV": float(mu),
        "alpha_topo": float(alpha_topo),
        "sigma_topo_MeV_per_fm": float(sigma_topo),
        "eigenvalues_MeV": spec.eigenvalues_MeV.tolist(),
        "K_alpha": float(K_alpha),
        "K_sigma": float(K_sigma),
    }


# ---------------------------------------------------------------------------
# Δm V5.37b sur les 4 paires
# ---------------------------------------------------------------------------
# (label, fondamental, excité, n_g_index, n_e_index, ref_dm_pdg_MeV)
V5P37B_PAIRS: list[tuple[str, str, str, int, int, float]] = [
    ("psi(2S)/J/psi",            "J/psi",       "psi(2S)",     0, 1, 589.2),
    ("psi(3S)/psi(2S)",          "J/psi",       "psi(3S)",     1, 2, 352.9),
    ("Upsilon(2S)/Upsilon(1S)",  "Upsilon(1S)", "Upsilon(2S)", 0, 1, 563.0),
    ("Upsilon(3S)/Upsilon(2S)",  "Upsilon(1S)", "Upsilon(3S)", 1, 2, 331.9),
]


def delta_m_cornell(
    label: str,
    ground: str,
    excited: str,
    n_g_index: int,
    n_e_index: int,
    delta_m_pdg_MeV: float,
    mode: str = MODE_UNIVERSAL,
    K_alpha: float | None = None,
    K_sigma: float | None = None,
) -> dict:
    """Δm Cornell V5.37b entre deux niveaux radiaux du même puits.

    Δm = E_{n_e} − E_{n_g} (lus dans le spectre Cornell de la famille).
    """
    spec_g = schrodinger_cornell_spectrum(
        ground, mode=mode, K_alpha=K_alpha, K_sigma=K_sigma,
        n_levels=max(n_e_index, n_g_index) + 1,
    )
    evs = np.array(spec_g["eigenvalues_MeV"], dtype=float)
    dm_v5p37b = float(evs[n_e_index] - evs[n_g_index])
    if dm_v5p37b > 0 and abs(delta_m_pdg_MeV) > 0:
        log_ratio = float(np.log10(abs(dm_v5p37b) / abs(delta_m_pdg_MeV)))
    else:
        log_ratio = float("nan")
    return {
        "label": label,
        "ground": ground,
        "excited": excited,
        "mode": mode,
        "delta_m_pdg_MeV": float(delta_m_pdg_MeV),
        "delta_m_v5p37b_MeV": dm_v5p37b,
        "E_levels_MeV": evs.tolist(),
        "log_ratio": log_ratio,
        "n_g_index": int(n_g_index),
        "n_e_index": int(n_e_index),
        "alpha_topo": float(spec_g["alpha_topo"]),
        "sigma_topo_MeV_per_fm": float(spec_g["sigma_topo_MeV_per_fm"]),
    }


def v5p37b_cornell_table(
    mode: str = MODE_UNIVERSAL,
    K_alpha: float | None = None,
    K_sigma: float | None = None,
) -> list[dict]:
    """Table 4 paires V5.37b dans le mode demandé."""
    rows: list[dict] = []
    for label, g, e, ng, ne, dm_pdg in V5P37B_PAIRS:
        rows.append(
            delta_m_cornell(label, g, e, ng, ne, dm_pdg,
                            mode=mode, K_alpha=K_alpha, K_sigma=K_sigma)
        )
    return rows


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def verdict_v5p37b(rows: list[dict], threshold_dex: float = 0.3) -> dict:
    """Σ|log| et verdict honnête de V5.37b."""
    logs = [r["log_ratio"] for r in rows]
    finite = [x for x in logs if np.isfinite(x)]
    s_abs = float(np.sum(np.abs(finite))) if finite else float("nan")
    m_abs = float(np.mean(np.abs(finite))) if finite else float("nan")
    n_ok = sum(1 for x in finite if abs(x) < threshold_dex)
    n_tot = len(rows)

    if n_ok == n_tot:
        verdict = "REPRODUIT"
        reco = (
            f"V5.37b reproduit les {n_tot} splittings radiaux à mieux que "
            f"±{threshold_dex} dex avec 2 paramètres calibrés une fois "
            f"(K_alpha, K_sigma).  Le quarkonium est CLOS."
        )
    elif n_ok == 0:
        verdict = "ECHEC"
        reco = (
            "Cornell TQNT (ce mode) ne reproduit AUCUN splitting dans le "
            "seuil malgré calibration."
        )
    else:
        verdict = "PARTIEL"
        reco = (
            f"V5.37b reproduit {n_ok}/{n_tot} splittings dans le seuil.  "
            "Cornell TQNT améliore V5.37 mais ne FERME pas totalement le "
            "quarkonium dans ce mode."
        )
    return {
        "sum_abs_log": s_abs,
        "mean_abs_log": m_abs,
        "n_within_threshold": int(n_ok),
        "n_total": int(n_tot),
        "threshold_dex": float(threshold_dex),
        "logs": [float(x) if np.isfinite(x) else float("nan") for x in logs],
        "verdict": verdict,
        "recommendation": reco,
    }
