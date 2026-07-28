"""V5.32 — Correction de plasticite P23 sur les mesures d'oscillation Delta m.

Cadre V5.32 (Case B discipline)
================================

Les experiences B-meson (LHCb 4 T, Belle II 1,5 T, CDF/D0 1,4 T, CPLEAR
~0,7 T) operent sous champ magnetique non-trivial : la mesure releve du
**Case B** de la discipline TQNT
(cf. memory/case-a-b-bext-discipline.md).

V5 §sec:P23-plastique enonce :

* Pour n = 1 composante : la plasticite radiale est nulle par symetrie
  (rotation Topologique du tube ferme annule l'integrale radiale).
* Pour n >= 2 : la superposition des champs B_i = tau_i des autres
  composantes brise la symetrie radiale, deformant chaque composante
  selon l'attracteur s_i^* = 1 + sigma(L) * gamma(L) * (cr_i - <cr>) / k_p.
* L'attracteur conduit a une correction multiplicative de la masse
  observee :
  m_obs(L) = Omega_ts(L) * (1 + sigma(L) * gamma(L) * epsilon(n))
  ou epsilon(n) encode l'amplitude radiale moyenne (= 0 pour n = 1).
* La constante globale est C_plast = -2,92 MeV, calibree sur le residu
  mu^- (3_1 + 6_1, n = 2).

Application aux oscillations
-----------------------------

Pour les mesons mixants, Delta m est lui aussi mesure dans le champ B_ext
du detecteur. La correction P23 s'applique a Delta m :

    Delta m_PDG = Delta m_topo * eps(n) * g(B_eff)

ou g(B_eff) est la modulation par le champ effectif de l'experience.
L'**unfolding** revient :

    Delta m_topo_unfolded = Delta m_PDG / [eps(n) * g(B_eff)]

Cette valeur est ensuite comparable aux predictions topologiques pures
V5.30c et V5.31 (Case A, sans B_ext).

Choix des fonctions (honnete)
-----------------------------

* eps(n) : V5 distingue explicitement n = 1 (eps = 0) et n >= 2
  (plasticite non triviale). Pour 4_1 + 6_1 (n = 2, mu^-) le residu
  C_plast = -2,92 MeV / m_mu = -2,76 % calibre l'echelle :

      eps(2) ~ 0,028   (epreuve de validation muon)
      eps(3) ~ 1,7 * eps(2) ~ 0,047

  Le facteur ~1,7 reflete que pour n = 3, la superposition radiale
  inclut deux autres champs B_j (j != i), donc le module de la
  resultante est approximativement sqrt(2) ~ 1,4 fois plus eleve ; on
  retient 1,7 pour tenir compte aussi du gain de la combinatoire
  cr_i - <cr> (variance accrue) cf. eq. (P23-attractor).

  Honnete : eps(2) et eps(3) sont **derives** d'une seule calibration
  (C_plast = -2,92 MeV sur mu^-) plus l'echelle geometrique sqrt(n-1).
  Donc V5.32 n'ajoute PAS de parametre libre par rapport a V5.31.

* g(B_eff) : V5 §13370 indique que l'amplitude plastique est
  "proportionnelle a l'intensite du champ EM rencontre". On adopte une
  dependance lineaire :

      g(B_eff) = B_eff / B_ref      avec B_ref = 4 T (LHCb)

  Les valeurs typiques pour les Delta m :
  * K^0 : CPLEAR/fixed-target/LHCb mixte    -> B_eff ~ 2 T
  * D^0 : LHCb (4T) + Belle II (1,5T)        -> B_eff ~ 3 T
  * B^0 : LHCb (4T) + CDF (1,4T)             -> B_eff ~ 3 T
  * B_s : LHCb (4T) + CDF (1,4T) + D0 (1,4T) -> B_eff ~ 3 T

  Choix conservatif. Si g(B_eff) ~ 1 (toutes les exp. dominees par LHCb
  a 4 T), on a g ~ 0,75 pour B_eff = 3 T.

Verdict honnete attendu
-----------------------

* eps(n) est calibree (1 calibration interne C_plast), pas un nouveau
  parametre.
* g(B_eff) est lineaire en B sans parametre libre supplementaire
  (B_ref = 4 T fixe par LHCb dominant).
* La correction totale eps(n)*g(B_eff) est de l'ordre de quelques %
  (max ~5 % sur Bs).
* Donc l'unfolding ne peut deplacer un log_ratio que de < 0,03 dex.
  V5.30c a deja Σ|log| = 1,05 dex, V5.31 = 1,46 dex : la correction
  P23 est trop petite pour fermer le gap a elle seule.

Ce module documente proprement cette conclusion et fournit l'outillage
de comparaison. Le verdict final est attendu PARTIAL ou NEGATIF :
P23 NE leve PAS le plafond V5 sur les oscillations.

References
----------
* docs/TQNT_detaillee_V5_FR.tex §sec:P23-plastique (postulat)
* docs/TQNT_detaillee_V5_FR.tex §sec:plast-experimental-alteration
  (alteration experimentale)
* memory/case-a-b-bext-discipline.md (Case B B_ext discipline)
* theory/v5p30c_vibrational.py (Case A baseline)
* theory/v5p31_chiral_vibrational.py (Case A + chiral, ratio Bs/B = 35)
"""
from __future__ import annotations

import numpy as np

from theory.topological_chiral_oscillation import (
    DELTA_M_PDG,
    MESON_OSCILLATION_ASSIGNMENTS,
)


# ---------------------------------------------------------------------------
# Constantes P23 (V5 §sec:P23-plastique)
# ---------------------------------------------------------------------------
# Plasticite globale, calibree sur residu mu^- (3_1 + 6_1, n = 2).
C_PLAST_MEV: float = -2.92

# Masse du muon (PDG, MeV). Sert a normaliser C_plast en correction
# relative pour n = 2.
M_MUON_MEV: float = 105.6583755

# Amplitude radiale eps(n=2) calibree directement sur mu^- :
#   eps(2) = |C_plast| / m_mu = 2.92 / 105.66 ~ 0.02763
EPS_N2_DEFAULT: float = abs(C_PLAST_MEV) / M_MUON_MEV  # ~0.02763

# Facteur d'echelle eps(n=3) / eps(n=2). V5 (eq. P23-attractor) :
# pour n composantes la resultante radiale superposee est
# proportionnelle a (n - 1) en amplitude moyenne (n-1 sources externes
# par composante), et l'ecart |cr_i - <cr>| augmente comme sqrt(n)
# pour une distribution generique des complexites. On prend donc
# le facteur geometrique sqrt((n-1)*n) / sqrt(2) qui vaut 1 pour n = 2
# et sqrt(3) ~ 1.73 pour n = 3. Honnete : pas de nouveau parametre,
# pure deduction geometrique.
def _geom_scale_n(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return float(np.sqrt(n * (n - 1) / 2.0))


# Champ de reference (LHCb dipole) pour normaliser g(B_eff).
B_REF_TESLA: float = 4.0

# Champs effectifs par systeme. Moyennes ponderees par la dominance
# experimentale dans le PDG.
B_EFF_TYPICAL: dict[str, float] = {
    # K^0 : CPLEAR (~0.7 T), fixed-target (1 T), LHCb (4 T) ;
    # moyenne ponderee historique ~ 2 T.
    "K0": 2.0,
    # D^0 : LHCb (4 T) + Belle II (1.5 T) ~3 T.
    "D0": 3.0,
    # B^0 : LHCb (4 T) + CDF/D0 (1.4 T) + Belle II (1.5 T) ~3 T.
    "B0": 3.0,
    # B_s : LHCb (4 T) + CDF/D0 (1.4 T) ~3 T.
    "Bs0": 3.0,
}


# ---------------------------------------------------------------------------
# 1. eps(n) : amplitude radiale plastique par nombre de composantes
# ---------------------------------------------------------------------------
def epsilon_n(n_components: int, eps_n2: float = EPS_N2_DEFAULT) -> float:
    """Amplitude radiale plastique eps(n) du postulat P23.

    Definition (V5 §sec:P23-plastique, eq. P23-attractor) :

    * n = 1 : eps(1) = 0 (symetrie radiale, integrale P23-radial nulle).
    * n >= 2 : eps(n) = eps(n=2) * sqrt(n*(n-1)/2)
      derive de l'amplitude de la superposition radiale (n-1 sources
      externes par composante) et de la variance des complexites.

    Parameters
    ----------
    n_components : int
        Nombre de composantes du link L.
    eps_n2 : float
        Calibration de eps(n=2) sur le residu mu^- (defaut |C_plast|/m_mu).

    Returns
    -------
    float
        eps(n) sans dimension.
    """
    if n_components < 1:
        raise ValueError("n_components doit etre >= 1.")
    if n_components == 1:
        return 0.0
    return float(eps_n2 * _geom_scale_n(n_components))


# ---------------------------------------------------------------------------
# 2. g(B_eff) : dependance lineaire au champ effectif
# ---------------------------------------------------------------------------
def g_B_field(
    B_ext_T: float,
    B_ref_T: float = B_REF_TESLA,
) -> float:
    """Modulation par champ magnetique effectif B_eff.

    Hypothese (V5 §13370) : la plasticite est proportionnelle a
    l'intensite du champ EM rencontre. On adopte une loi lineaire
    normalisee au LHCb dipole (B_ref = 4 T).

        g(B_eff) = B_eff / B_ref

    Parameters
    ----------
    B_ext_T : float
        Champ magnetique experimental effectif (Tesla).
    B_ref_T : float
        Reference (LHCb = 4 T par defaut).

    Returns
    -------
    float
        g(B_eff), sans dimension (1 a 4 T, < 1 sinon).
    """
    if B_ext_T < 0:
        raise ValueError("B_ext_T doit etre >= 0.")
    return float(B_ext_T / B_ref_T)


# ---------------------------------------------------------------------------
# 3. Correction plastique combinee
# ---------------------------------------------------------------------------
def plasticity_correction(
    n_components: int,
    B_eff_T: float,
    eps_n2: float = EPS_N2_DEFAULT,
    B_ref_T: float = B_REF_TESLA,
) -> dict:
    """Correction multiplicative P23 totale pour la mesure de Delta m.

    delta(L) = eps(n) * g(B_eff)

    La masse observee s'ecrit alors (cf. V5 eq. P23-attractor) :

        Delta m_PDG = Delta m_topo * (1 + sigma * delta)        (sigma signed)
                    ~ Delta m_topo * delta                       (approx large)

    Pour l'unfolding nous utilisons le facteur *correctif* attendu sur
    Delta m, qui est de l'ordre de delta lui-meme. Voir
    `unfold_delta_m` pour la formule precise retenue (additive et
    multiplicative).

    Returns
    -------
    dict
        { eps_n, g_B, delta, n, B_eff_T }
    """
    e = epsilon_n(n_components, eps_n2=eps_n2)
    g = g_B_field(B_eff_T, B_ref_T=B_ref_T)
    delta = e * g
    return {
        "eps_n": float(e),
        "g_B": float(g),
        "delta": float(delta),
        "n_components": int(n_components),
        "B_eff_T": float(B_eff_T),
    }


# ---------------------------------------------------------------------------
# 4. Unfolding : Delta m_topo a partir de Delta m_PDG
# ---------------------------------------------------------------------------
def unfold_delta_m(
    delta_m_PDG: float,
    n_components: int,
    B_eff_T: float,
    sigma: float = +1.0,
    eps_n2: float = EPS_N2_DEFAULT,
    B_ref_T: float = B_REF_TESLA,
) -> dict:
    """Unfold la correction P23 d'une mesure PDG.

    Formule (V5 §sec:plast-experimental-alteration, modele
    multiplicatif) :

        Delta m_PDG = Delta m_topo * (1 + sigma * eps(n) * g(B_eff))

        => Delta m_topo = Delta m_PDG / (1 + sigma * eps(n) * g(B_eff))

    avec sigma = +1 par convention pour les mesons pseudoscalaires
    (cf. P23 selecteur de signe : sigma = +1 si J = 0).

    Parameters
    ----------
    delta_m_PDG : float
        Valeur PDG mesuree (MeV ou eV peu importe : echelle conservee).
    n_components : int
        Nombre de composantes du link.
    B_eff_T : float
        Champ effectif (Tesla).
    sigma : float
        Selecteur de signe P23 (default +1, pseudoscalaire J=0).
    eps_n2, B_ref_T : voir constantes.

    Returns
    -------
    dict
        { delta_m_topo, delta_m_PDG, factor, delta, components_data }
    """
    corr = plasticity_correction(
        n_components, B_eff_T, eps_n2=eps_n2, B_ref_T=B_ref_T
    )
    delta = corr["delta"]
    factor = 1.0 + sigma * delta
    if factor <= 0:
        raise ValueError(
            f"Facteur de correction non positif ({factor}), unfold impossible."
        )
    delta_m_topo = delta_m_PDG / factor
    return {
        "delta_m_topo_unfolded": float(delta_m_topo),
        "delta_m_PDG": float(delta_m_PDG),
        "correction_factor": float(factor),
        "delta": float(delta),
        "eps_n": corr["eps_n"],
        "g_B": corr["g_B"],
        "n_components": int(n_components),
        "B_eff_T": float(B_eff_T),
        "sigma": float(sigma),
    }


# ---------------------------------------------------------------------------
# 5. Application aux 4 mesons K, D, B, Bs
# ---------------------------------------------------------------------------
def delta_m_topo_pdg_corrected(
    particle: str,
    eps_n2: float = EPS_N2_DEFAULT,
    B_ref_T: float = B_REF_TESLA,
    B_eff_T: float | None = None,
) -> dict:
    """Delta m topologique attendu apres unfolding P23 pour une particule.

    Utilise les assignations V5 (MESON_OSCILLATION_ASSIGNMENTS) pour
    determiner n_components, le PDG observe (DELTA_M_PDG), et un
    B_eff par defaut (B_EFF_TYPICAL).

    Parameters
    ----------
    particle : str
        "K0", "D0", "B0" ou "Bs0".
    eps_n2 : float
    B_ref_T : float
    B_eff_T : float, optional
        Si fourni, override la valeur typique pour la particule.

    Returns
    -------
    dict
        Resultat de unfold_delta_m enrichi du nom de particule.
    """
    if particle not in MESON_OSCILLATION_ASSIGNMENTS:
        raise KeyError(
            f"Particule '{particle}' inconnue. Choix : "
            f"{list(MESON_OSCILLATION_ASSIGNMENTS)}."
        )
    cfg = MESON_OSCILLATION_ASSIGNMENTS[particle]
    n = len(cfg["knots"])
    B = B_eff_T if B_eff_T is not None else B_EFF_TYPICAL[particle]
    dm_pdg = DELTA_M_PDG[particle]
    res = unfold_delta_m(
        delta_m_PDG=dm_pdg,
        n_components=n,
        B_eff_T=B,
        sigma=+1.0,
        eps_n2=eps_n2,
        B_ref_T=B_ref_T,
    )
    res["particle"] = particle
    res["knots"] = list(cfg["knots"])
    return res


# ---------------------------------------------------------------------------
# 6. Table P23-unfolded pour les 4 systemes
# ---------------------------------------------------------------------------
def unfolded_table(
    eps_n2: float = EPS_N2_DEFAULT,
    B_ref_T: float = B_REF_TESLA,
    B_eff_override: dict[str, float] | None = None,
) -> list[dict]:
    """Table Delta m_PDG -> Delta m_topo_unfolded pour K, D, B, Bs."""
    rows = []
    for name in ("K0", "D0", "B0", "Bs0"):
        B = (B_eff_override or {}).get(name)
        row = delta_m_topo_pdg_corrected(
            name, eps_n2=eps_n2, B_ref_T=B_ref_T, B_eff_T=B
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 7. Comparaison V5.30c / V5.31 vs PDG_unfolded
# ---------------------------------------------------------------------------
def compare_models_to_unfolded(
    v5p30c_predictions: dict[str, float],
    v5p31_predictions: dict[str, float],
    eps_n2: float = EPS_N2_DEFAULT,
    B_ref_T: float = B_REF_TESLA,
    B_eff_override: dict[str, float] | None = None,
) -> dict:
    """Compare V5.30c et V5.31 (Case A) a Delta m_PDG_unfolded (Case B
    -> A).

    Parameters
    ----------
    v5p30c_predictions : dict
        Mapping particle -> Delta m predit V5.30c en MeV.
    v5p31_predictions : dict
        Mapping particle -> Delta m predit V5.31 en MeV.

    Returns
    -------
    dict
        { rows, sum_abs_log_v5p30c_vs_PDG,
          sum_abs_log_v5p30c_vs_unfolded,
          sum_abs_log_v5p31_vs_PDG,
          sum_abs_log_v5p31_vs_unfolded,
          ratios }
    """
    rows = []
    s30c_pdg = 0.0
    s30c_unf = 0.0
    s31_pdg = 0.0
    s31_unf = 0.0
    for name in ("K0", "D0", "B0", "Bs0"):
        unf = delta_m_topo_pdg_corrected(
            name,
            eps_n2=eps_n2,
            B_ref_T=B_ref_T,
            B_eff_T=(B_eff_override or {}).get(name),
        )
        dm_pdg = unf["delta_m_PDG"]
        dm_unf = unf["delta_m_topo_unfolded"]
        dm_30c = v5p30c_predictions[name]
        dm_31 = v5p31_predictions[name]
        l30c_pdg = float(np.log10(dm_30c / dm_pdg))
        l30c_unf = float(np.log10(dm_30c / dm_unf))
        l31_pdg = float(np.log10(dm_31 / dm_pdg))
        l31_unf = float(np.log10(dm_31 / dm_unf))
        s30c_pdg += abs(l30c_pdg)
        s30c_unf += abs(l30c_unf)
        s31_pdg += abs(l31_pdg)
        s31_unf += abs(l31_unf)
        rows.append(
            {
                "name": name,
                "n_components": unf["n_components"],
                "B_eff_T": unf["B_eff_T"],
                "eps_n": unf["eps_n"],
                "g_B": unf["g_B"],
                "delta": unf["delta"],
                "correction_factor": unf["correction_factor"],
                "delta_m_PDG": dm_pdg,
                "delta_m_topo_unfolded": dm_unf,
                "delta_m_v5p30c": dm_30c,
                "delta_m_v5p31": dm_31,
                "log_ratio_v5p30c_vs_PDG": l30c_pdg,
                "log_ratio_v5p30c_vs_unfolded": l30c_unf,
                "log_ratio_v5p31_vs_PDG": l31_pdg,
                "log_ratio_v5p31_vs_unfolded": l31_unf,
            }
        )
    ratio_pdg_B = v5p31_predictions["Bs0"] / v5p31_predictions["B0"]
    pdg_ratio = DELTA_M_PDG["Bs0"] / DELTA_M_PDG["B0"]
    # PDG_unfolded ratio :
    unf_B = next(r for r in rows if r["name"] == "B0")["delta_m_topo_unfolded"]
    unf_Bs = next(r for r in rows if r["name"] == "Bs0")[
        "delta_m_topo_unfolded"
    ]
    unfolded_ratio = unf_Bs / unf_B
    return {
        "rows": rows,
        "sum_abs_log_v5p30c_vs_PDG": float(s30c_pdg),
        "sum_abs_log_v5p30c_vs_unfolded": float(s30c_unf),
        "sum_abs_log_v5p31_vs_PDG": float(s31_pdg),
        "sum_abs_log_v5p31_vs_unfolded": float(s31_unf),
        "ratio_v5p31_Bs_B": float(ratio_pdg_B),
        "ratio_pdg_Bs_B": float(pdg_ratio),
        "ratio_pdg_unfolded_Bs_B": float(unfolded_ratio),
        "eps_n2": float(eps_n2),
        "B_ref_T": float(B_ref_T),
    }


__all__ = [
    "C_PLAST_MEV",
    "M_MUON_MEV",
    "EPS_N2_DEFAULT",
    "B_REF_TESLA",
    "B_EFF_TYPICAL",
    "epsilon_n",
    "g_B_field",
    "plasticity_correction",
    "unfold_delta_m",
    "delta_m_topo_pdg_corrected",
    "unfolded_table",
    "compare_models_to_unfolded",
]
