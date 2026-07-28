"""V5.35e — Extension de V5.35d à K_S/K_L, quarkonium et états exotiques XYZ.

Contexte
========

V5.35d (commit fee4d23) ferme les Δm oscillations chirales K⁰/D⁰/B⁰/B_s⁰
à Σ|log| ≈ 0.005 dex avec la forme seq_asym et δ = -2.78 (paramètre
V5.35b : β = -1.30, forme « rel »).  La formule canonique est :

    ω_χ(L) = √( F_torque(L) / I_χ(L) )
    I_χ(L) = I_χ^V5.35b(L) × Π_{i<j<k} [ 1 + δ × f_{seq_asym}(c_i, c_j, c_k) ]
    Δm(L) = Δm^V5.30c(L) × ω_χ(L) / ω_χ(K⁰)

V5.35e teste si cette formule **chirale** s'applique à d'autres systèmes :

1. **K_S - K_L mass splitting (3.484 × 10⁻⁶ eV)** : K_S et K_L sont des
   états propres CP, pas des configurations topologiques distinctes.
   Tous deux correspondent à la même topologie K⁰ = 6_1+6_1+6_1.  Par
   définition, Δm(K_S, K_L) = Δm(K⁰, K̄⁰).  V5.35d s'applique
   TRIVIALEMENT — c'est la **même grandeur** que celle déjà calibrée.

2. **Quarkonium (J/ψ, ψ(2S), Υ(1S), Υ(2S))** : excitations radiales
   apparentes en SM.  En V5 ce sont des **topologies différentes** :
   - J/ψ      = 3_1 + 4_1 + 6_3
   - ψ(2S)    = 6_1 + 8_2     (n_comp = 2, pas de 3-corps)
   - Υ(1S)    = 6_2 + 7_1 + 7_3
   - Υ(2S)    = 6_2 + 6_2 + 8_2
   Le « splitting » entre quarkonium states n'est PAS une oscillation
   chirale L ↔ L*.  V5.35d ω_χ peut être calculé pour chaque, mais la
   différence ω_χ(ψ(2S)) - ω_χ(J/ψ) n'a pas la même signification
   physique que Δm(B⁰, B̄⁰).  On rapporte la prédiction et on identifie
   honnêtement la limite : V5.35d est une formule d'oscillation
   mirror-image, pas un splitting de niveau radial.

3. **Exotiques XYZ (X(3872), Z_c(3900), Z_b(10610))** :
   X(3872) et Z_c(3900) **partagent** la même topologie dans le catalogue
   V5 (3_1+7_2+7_7) ; ils diffèrent par leurs nombres quantiques
   internes (J^PC), pas par leur configuration topologique.  Aucune
   oscillation Δm n'est mesurée pour ces états (lifetime trop courte,
   pas d'antipartie mesurable).  V5.35d **ne s'applique pas** : pas de
   prédiction Δm extrapolable au PDG.

Conclusion attendue
===================

V5.35d est une formule **spécifique aux oscillations chirales meson↔
antimeson** (K⁰-K̄⁰, D⁰-D̄⁰, B⁰-B̄⁰, B_s⁰-B̄_s⁰).  Elle :

* s'applique trivialement à K_S/K_L (même grandeur),
* ne s'applique pas aux splittings radiales (J/ψ ↔ ψ(2S)),
* ne s'applique pas aux états exotiques sans oscillation mesurée.

Cette extension confirme V5.35d comme **formule honnêtement
spécialisée**, pas une formule universelle de mass splitting.

Référence : theory/v5p35d_3body_multiplicative.py (formule V5.35d)
            theory/topological_chiral_oscillation.py (assignations méson)
            docs/TQNT_catalog_V5_diffusion.xlsx (assignations PDG V5)
"""
from __future__ import annotations

import numpy as np

from theory.knot_catalog import crossing_number
from theory.jones import braid_index
from theory.v5p35d_3body_multiplicative import (
    delta_m_v5p35d,
    omega_chirality_v5p35d,
    chiral_inertia_v5p35d,
    _clear_cache,
)


# ---------------------------------------------------------------------------
# Topologies V5 (source : docs/TQNT_catalog_V5_diffusion.xlsx, onglet 3)
# ---------------------------------------------------------------------------
# Hors mésons d'oscillation V5.35d (K0, D0, B0, Bs0) déjà dans
# theory/topological_chiral_oscillation.py.
EXTENDED_TOPOLOGY: dict[str, dict] = {
    # K_S, K_L : même topologie que K⁰ (états propres CP).
    "K_S":      {"knots": ["6_1", "6_1", "6_1"], "mass_pdg": 497.611,
                 "note": "Etat propre CP+ de K0 (meme topologie 6_1^3)."},
    "K_L":      {"knots": ["6_1", "6_1", "6_1"], "mass_pdg": 497.611,
                 "note": "Etat propre CP- de K0 (meme topologie 6_1^3)."},

    # Charmonium 1S/2S
    "J/psi":    {"knots": ["3_1", "4_1", "6_3"], "mass_pdg": 3096.9,
                 "note": "J/psi(1S) charmonium 1--."},
    "psi(2S)":  {"knots": ["6_1", "8_2"],         "mass_pdg": 3686.10,
                 "note": "psi(2S) charmonium 1-- excitation radiale."},

    # Bottomonium 1S/2S
    "Upsilon(1S)": {"knots": ["6_2", "7_1", "7_3"], "mass_pdg": 9460.30,
                    "note": "Y(1S) bottomonium 1--."},
    "Upsilon(2S)": {"knots": ["6_2", "6_2", "8_2"], "mass_pdg": 10023.26,
                    "note": "Y(2S) bottomonium 1-- excitation radiale."},

    # Exotiques (catalogue V5)
    "X(3872)":  {"knots": ["3_1", "7_2", "7_7"], "mass_pdg": 3871.65,
                 "note": "chi_c1(3872) exotique JPC=1++."},
    "Z_c(3900)": {"knots": ["3_1", "7_2", "7_7"], "mass_pdg": 3887.1,
                  "note": "Etat charge JP=1+ (memes nœuds que X(3872))."},
    # Z_b(10610) : assignation non listée dans catalogue V5 (onglet 3).
    # On extrapole par analogie c<->b : remplace 3_1 par 4_1 (b=3 plus
    # complexe).  Honnête : prédiction approchee, marquee LOWCONF.
    "Z_b(10610)": {"knots": ["4_1", "8_2", "8_3"], "mass_pdg": 10607.2,
                   "note": "LOWCONF: assignation extrapolee, hors catalogue.",
                   "lowconf": True},
}


# ---------------------------------------------------------------------------
# Donnees PDG pour les Delta m / splittings teste
# ---------------------------------------------------------------------------
# Δm experimentaux a comparer.  Pour les oscillations chirales, Δm est
# en MeV (parité avec la convention V5.35d).  Pour les splittings de
# masse de niveaux radial, on convertit en MeV.
DELTA_M_EXTENDED_PDG_MEV: dict[str, float] = {
    # Oscillation chirale meson-antimeson :
    #   Δm(K_S, K_L) = 3.484e-6 eV = 3.484e-12 MeV   (= Δm(K0, K̄0))
    "K_S-K_L":               3.484e-12,
    # Splittings radial quarkonium :
    "psi(2S)-J/psi":         3686.10 - 3096.9,
    "Upsilon(2S)-Upsilon(1S)": 10023.26 - 9460.30,
    # Exotique : on rapporte la difference m(Z_c) - m(X) qui partagent
    # la meme topologie en V5 mais sont distincts en SM.
    "Z_c(3900)-X(3872)":     3887.1 - 3871.65,
    # Pour Z_b(10610), pas de Δm de meme nature dans PDG.
    # On reporte simplement la masse PDG dans le rapport, sans Δm cible.
}


# ---------------------------------------------------------------------------
# Parametres V5.35d calibres
# ---------------------------------------------------------------------------
V5p35D_DELTA: float = -2.78          # forme seq_asym
V5p35D_FORM: str = "seq_asym"
V5p35D_BETA: float = -1.30           # forme V5.35b "rel"
V5p35D_FORM_B: str = "rel"


# ---------------------------------------------------------------------------
# assign_topology
# ---------------------------------------------------------------------------
def assign_topology(particle_name: str) -> dict:
    """Retourne l'assignation topologique V5 du particule.

    Cherche d'abord dans EXTENDED_TOPOLOGY (V5.35e) puis dans
    MESON_OSCILLATION_ASSIGNMENTS (V5.35d mesons).

    Parameters
    ----------
    particle_name : str
        Nom (e.g. "K_S", "J/psi", "X(3872)", "K0").

    Returns
    -------
    dict
        { knots, mass_pdg, note, lowconf? }
    """
    if particle_name in EXTENDED_TOPOLOGY:
        cfg = dict(EXTENDED_TOPOLOGY[particle_name])
        cfg["source"] = "V5.35e"
        return cfg
    # Fallback : mesons V5.35d
    from theory.topological_chiral_oscillation import (
        MESON_OSCILLATION_ASSIGNMENTS,
    )
    if particle_name in MESON_OSCILLATION_ASSIGNMENTS:
        c = MESON_OSCILLATION_ASSIGNMENTS[particle_name]
        return {
            "knots": list(c["knots"]),
            "mass_pdg": float(c["mass_pdg"]),
            "note": "Meson oscillation V5.35d.",
            "source": "V5.35d",
        }
    raise KeyError(f"Particle inconnue : {particle_name!r}")


# ---------------------------------------------------------------------------
# omega_chi_v5p35d : frequence chirale pour une particule (3-corps si n>=3)
# ---------------------------------------------------------------------------
def omega_chi_v5p35d(particle_name: str) -> dict:
    """ω_χ V5.35d pour une particule (utilise δ, β calibres V5.35d).

    Avertissement : pour n_comp < 3, le facteur 3-corps multiplicatif est
    1.0 (produit vide), donc ω_χ V5.35d = ω_χ V5.35b.
    """
    cfg = assign_topology(particle_name)
    res = omega_chirality_v5p35d(
        cfg["knots"], V5p35D_DELTA, V5p35D_FORM,
        beta_v5p35b=V5p35D_BETA, form_v5p35b=V5p35D_FORM_B,
    )
    return {
        "particle": particle_name,
        "knots": cfg["knots"],
        "omega_chi": float(res["omega_chi"]),
        "F_torque": float(res["F_torque"]),
        "I_chi": float(res["I_chi"]),
        "factor_3body": float(res["factor_3body"]),
        "n_comp": int(res["n_comp"]),
        "source": cfg.get("source", "unknown"),
        "lowconf": bool(cfg.get("lowconf", False)),
    }


# ---------------------------------------------------------------------------
# delta_m_v5p35d_generalized : Δm V5.35d entre deux configurations
# ---------------------------------------------------------------------------
def delta_m_v5p35d_generalized(
    particle_1: str, particle_2: str | None = None,
) -> dict:
    """Δm V5.35d generalisee.

    Trois regimes :

    1. **particle_2 is None** ou particle_1 == particle_2 :
        Δm chirale meson↔antimeson (cas standard V5.35d).
        S'applique uniquement aux 4 mesons {K0, D0, B0, Bs0} + leurs
        equivalents CP {K_S, K_L} (meme topologie).

    2. **particle_1 != particle_2, memes nœuds** :
        Pas de splitting topologique → Δm = 0 en V5.35d.
        Exemples : K_S vs K_L (etats CP) ; X(3872) vs Z_c(3900).

    3. **particle_1 != particle_2, topologies distinctes** :
        Splitting de niveaux radiaux (J/ψ vs ψ(2S), Y(1S) vs Y(2S)).
        V5.35d **n'est pas conçue pour ça** : on rapporte la difference
        ω_χ(P_2) - ω_χ(P_1) comme **proxy** topologique, mais cette
        quantite n'a PAS la dimension d'un Δm physique (c'est l'ecart
        entre deux frequences chirales independantes).  Champ
        ``applicable = False`` dans la sortie.

    Returns
    -------
    dict
        { mode, applicable, delta_m_v5p35d_MeV, omega_chi_1, omega_chi_2,
          knots_1, knots_2, note }
    """
    if particle_2 is None or particle_2 == particle_1:
        # Cas 1 : oscillation meson-antimeson sur la meme topologie.
        cfg = assign_topology(particle_1)
        # Si on a une mass_pdg (cas K_S, K_L, K0, ...), calcule Δm V5.35d.
        # La calibration K0 doit toujours etre presente : reset cache.
        _clear_cache()
        res = delta_m_v5p35d(
            cfg["knots"], cfg["mass_pdg"], V5p35D_DELTA, V5p35D_FORM,
            beta_v5p35b=V5p35D_BETA, form_v5p35b=V5p35D_FORM_B,
        )
        return {
            "mode": "chiral_oscillation",
            "applicable": True,
            "delta_m_v5p35d_MeV": float(res["delta_m_MeV"]),
            "particle_1": particle_1,
            "particle_2": particle_1 + "*",
            "knots_1": cfg["knots"],
            "knots_2": cfg["knots"],
            "omega_chi_1": float(res["omega_chi"]),
            "omega_chi_2": float(res["omega_chi"]),
            "R_chi": float(res["R_chi"]),
            "factor_3body": float(res["factor_3body"]),
            "note": "Oscillation chirale L<->L* : formule V5.35d.",
        }

    cfg1 = assign_topology(particle_1)
    cfg2 = assign_topology(particle_2)
    knots1 = tuple(sorted(cfg1["knots"]))
    knots2 = tuple(sorted(cfg2["knots"]))

    if knots1 == knots2:
        # Cas 2 : memes nœuds, pas de splitting topologique.
        return {
            "mode": "same_topology",
            "applicable": False,
            "delta_m_v5p35d_MeV": 0.0,
            "particle_1": particle_1,
            "particle_2": particle_2,
            "knots_1": cfg1["knots"],
            "knots_2": cfg2["knots"],
            "omega_chi_1": float(omega_chi_v5p35d(particle_1)["omega_chi"]),
            "omega_chi_2": float(omega_chi_v5p35d(particle_2)["omega_chi"]),
            "R_chi": 1.0,
            "factor_3body": 1.0,
            "note": (
                "Memes nœuds → V5.35d predit Δm = 0 (pas de splitting "
                "topologique).  L'ecart PDG est d'origine non topologique "
                "(etats propres CP, J^PC distincts)."
            ),
        }

    # Cas 3 : topologies distinctes → splitting radial.
    om1 = omega_chi_v5p35d(particle_1)
    om2 = omega_chi_v5p35d(particle_2)
    # Proxy Δm : on rapporte la difference de frequences chirales (sans
    # garantir une interpretation physique).
    delta_om = float(om2["omega_chi"] - om1["omega_chi"])
    return {
        "mode": "radial_splitting",
        "applicable": False,
        "delta_m_v5p35d_MeV": delta_om,
        "particle_1": particle_1,
        "particle_2": particle_2,
        "knots_1": cfg1["knots"],
        "knots_2": cfg2["knots"],
        "omega_chi_1": float(om1["omega_chi"]),
        "omega_chi_2": float(om2["omega_chi"]),
        "R_chi": float(om2["omega_chi"] / om1["omega_chi"]),
        "factor_3body": float(om2["factor_3body"] / om1["factor_3body"])
            if om1["factor_3body"] > 0 else float("nan"),
        "note": (
            "Topologies distinctes : V5.35d est une formule d'oscillation "
            "chirale, PAS un splitting radial.  La difference ω_χ donnee "
            "comme proxy non-physique."
        ),
    }


# ---------------------------------------------------------------------------
# v5p35e_extended_table : table comparative
# ---------------------------------------------------------------------------
EXTENDED_SYSTEMS: list[tuple[str, str | None, str | None]] = [
    # (label, p1, p2)
    ("K_S - K_L",                 "K_S", "K_L"),
    ("K_S oscillation (CP+)",    "K_S", None),    # appliquer V5.35d a K_S
    ("psi(2S) - J/psi",           "J/psi", "psi(2S)"),
    ("Y(2S) - Y(1S)",             "Upsilon(1S)", "Upsilon(2S)"),
    ("Z_c(3900) - X(3872)",       "X(3872)", "Z_c(3900)"),
]


def v5p35e_extended_table() -> list[dict]:
    """Calcule Δm V5.35d sur les 5 systemes etendus + leur statut PDG.

    Returns
    -------
    list[dict]
        Liste de rows : label, mode, applicable, dm_v5p35d, dm_pdg,
        log_ratio (si applicable), note.
    """
    rows: list[dict] = []
    for label, p1, p2 in EXTENDED_SYSTEMS:
        res = delta_m_v5p35d_generalized(p1, p2)
        # Match avec PDG
        key = None
        if label.startswith("K_S - K_L"):
            key = "K_S-K_L"
        elif label.startswith("psi(2S)"):
            key = "psi(2S)-J/psi"
        elif label.startswith("Y(2S)"):
            key = "Upsilon(2S)-Upsilon(1S)"
        elif label.startswith("Z_c"):
            key = "Z_c(3900)-X(3872)"
        elif label.startswith("K_S oscillation"):
            key = "K_S-K_L"   # meme Δm que K0 (CP+/-)
        dm_pdg = DELTA_M_EXTENDED_PDG_MEV.get(key) if key else None

        if res["mode"] == "chiral_oscillation" and dm_pdg is not None:
            dm_v5 = res["delta_m_v5p35d_MeV"]
            log_r = float(np.log10(dm_v5 / dm_pdg)) if (
                dm_v5 > 0 and dm_pdg > 0
            ) else float("nan")
        elif dm_pdg is not None and abs(res["delta_m_v5p35d_MeV"]) > 0:
            dm_v5 = res["delta_m_v5p35d_MeV"]
            log_r = float(np.log10(abs(dm_v5) / dm_pdg))
        else:
            log_r = float("nan")

        rows.append({
            "label": label,
            "particle_1": res["particle_1"],
            "particle_2": res["particle_2"],
            "knots_1": res["knots_1"],
            "knots_2": res["knots_2"],
            "mode": res["mode"],
            "applicable": res["applicable"],
            "delta_m_v5p35d_MeV": float(res["delta_m_v5p35d_MeV"]),
            "delta_m_pdg_MeV": float(dm_pdg) if dm_pdg is not None
                else float("nan"),
            "log_ratio": log_r,
            "omega_chi_1": float(res["omega_chi_1"]),
            "omega_chi_2": float(res["omega_chi_2"]),
            "R_chi": float(res["R_chi"]),
            "factor_3body": res["factor_3body"],
            "note": res["note"],
        })
    return rows


# ---------------------------------------------------------------------------
# Verdict V5.35e
# ---------------------------------------------------------------------------
def verdict_v5p35e(rows: list[dict]) -> dict:
    """Bilan d'universalite de V5.35d.

    Returns
    -------
    dict
        { applicable_count, total_count, mean_log_applicable,
          universality, recommendation }
    """
    applicable = [r for r in rows if r["applicable"]
                  and np.isfinite(r["log_ratio"])]
    n_app = len(applicable)
    n_tot = len(rows)
    mean_log = (
        float(np.mean([abs(r["log_ratio"]) for r in applicable]))
        if applicable else float("nan")
    )
    if n_app == n_tot:
        universality = "UNIVERSAL"
        reco = "V5.35d applicable a tous les systemes testes."
    elif n_app == 0:
        universality = "NONE"
        reco = (
            "V5.35d s'applique uniquement aux oscillations chirales "
            "meson-antimeson (K0/D0/B0/Bs0 deja calibrees, et K_S/K_L "
            "trivialement)."
        )
    else:
        universality = "SPECIALIZED"
        reco = (
            f"V5.35d s'applique a {n_app}/{n_tot} systemes etendus : les "
            "oscillations chirales L<->L*.  Elle NE s'applique PAS aux "
            "splittings de niveaux radiaux (quarkonium) ni aux memes "
            "topologies CP/JPC (X(3872)/Z_c(3900))."
        )
    return {
        "applicable_count": int(n_app),
        "total_count": int(n_tot),
        "mean_log_applicable": float(mean_log),
        "universality": universality,
        "recommendation": reco,
    }
