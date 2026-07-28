"""
V5.19 — Catalogue des objets stellaires / compacts prédits par TQNT-V5.

Ce module catalogue SIX classes d'objets compacts dérivables des principes
V5 (§13 matière noire, §16.4 P22, §17 cinétique condensation Big Bang,
chap. 17 cascade reheating), évalue leur fonction de masse primordiale,
leur abondance cosmique, et leur contribution potentielle à la fonction de
masse stellaire (SMF) observée par JWST à haut redshift.

Les six classes sont :

* **Class A — DM Pearls** (déjà documentés ``dm_pearls_paper.tex'' v2)
  Stacking gravitationnel de nœuds DM (c > 18) sous pression de Fermi
  topologique. Chandrasekhar généralisée M_Ch ~ M_Pl^3 / m_χ^2.

* **Class B — Topological Black-Hole seeds primordiaux**
  Heavy protons (entrelacs Gen 5+, ~3 GeV) accumulés par cluster
  gravitationnel via P22 enhanced nucleation au sound horizon BBN.

* **Class C — Knot-Baryon Hybrid Stars**
  Cœur DM pearl + accrétion baryonique baryonique post-recombination.

* **Class D — Plasma Heavy Proton (transitoire)**
  Brève phase 10^2-10^3 s post-bang : cascade radiative UV transitoire.

* **Class E — Strange Knot Stars**
  Compact c=20-25 amphichiral chains. Catalogue V5 : 6 STRICT
  amphichiraux (6_3+6_3 à 1.55 GeV, etc.).

* **Class F — Primordial 0_1 Unknot Defects**
  Analogues des défauts unknot survivants de la cascade V5.17.

Pour chaque classe on calcule :
  - Mass scale (M_typical, M_max)
  - Abondance primordiale (Ω/Ω_b ou Ω/Ω_DM)
  - Signature JWST (M_apparent, r_eff, luminosité, color)
  - Contribution à la SMF résiduelle (post audit -1.5 dex)

VERDICT HONNÊTE
---------------
Sur les 6 classes, seules les classes B (PBH seeds primordiaux) et
potentiellement C (hybrid stars) ont une chance de contribuer
significativement à la SMF JWST. Les autres sont :
  - Class A : trop petite (10^-12 à 10^-3 M_⊙), MACHO/halo seulement.
  - Class D : transitoire, ne contribue pas à z>10 cumulative.
  - Class E : abondance ≤ 6 STRICT << Ω_DM.
  - Class F : abondance unknown, signature spéculative.

Voir docs/V5p19_compact_objects_catalog(_FR).tex.

References
----------
- V5 §13 (matière noire, 117 candidats)
- V5 §16.4 (P22 nucléation)
- V5 chap. 17 (cascade reheating 2-étages)
- V5 chap. 22 (cinétique condensation Big Bang, Ω_DM/Ω_b = 5.45)
- dm_pearls_paper.tex v2 (commit 5efc4e4)
- V5p18_P22_boost_rigorous.tex (boost rigoureux = 1)
- V5_JWST_impossible_audit.tex (-1.5 dex Krishnan-Abazajian 2026)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# =========================================================================
# Constantes physiques (SI + cosmologie Planck 2018)
# =========================================================================
HBAR_J_S = 1.054571817e-34
C_M_S = 299792458.0
G_NEWTON = 6.67430e-11
KB_J_K = 1.380649e-23
EV_J = 1.602176634e-19
MEV_KG = 1.78266192e-30
GEV_KG = 1.78266192e-27
MSUN_KG = 1.98892e30

# Planck mass (kg)
MPL_KG = math.sqrt(HBAR_J_S * C_M_S / G_NEWTON)  # ~ 2.176e-8 kg

# Compton scale conversion: lambdabar_C = hbar / (m c)
def lambdabar_C(m_kg: float) -> float:
    return HBAR_J_S / (m_kg * C_M_S)

# Cosmology Planck 2018
H0_KM_S_MPC = 67.4
H0_SI = H0_KM_S_MPC * 1000.0 / 3.0857e22
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
OMEGA_B = 0.0493
OMEGA_DM = OMEGA_M - OMEGA_B  # ~ 0.2657
RHO_CRIT_SI = 3 * H0_SI**2 / (8 * math.pi * G_NEWTON)  # kg/m^3
RHO_CRIT_MSUN_MPC3 = 2.775e11 * (H0_KM_S_MPC / 100.0) ** 2

# V5 specific (CLAUDE.md)
ALPHA_EM = 1.0 / 137.036
K_CS_LEVEL = 135                     # V5 §10.4
T_C_MEV = 67.0                       # condensation Big Bang
M_E_MEV = 0.51099895
M_P_GEV = 0.9382720813

# DM knots (V5 § 13.6 ; representative)
# Six STRICT amphichirals (mass GeV) from CLAUDE.md
DM_STRICT_AMPHI_GEV = [1.55, 4.70, 16.01, 17.24, 26.48, 51.78]

# Heavy proton Gen 5+ mass scale (V5 §13.6 + heavy protons paper)
HEAVY_PROTON_GEV_LOW = 2.9
HEAVY_PROTON_GEV_HIGH = 3.3

# Abundance ratio Ω_DM / Ω_b (V5 §22 prediction)
OMEGA_DM_OVER_OMEGA_B = 5.45


# =========================================================================
# 0. Dataclass: compact object class descriptor
# =========================================================================

@dataclass
class CompactObjectClass:
    """Generic descriptor for a TQNT-V5 compact object class.

    Each class wraps the four pieces of information requested in the V5.19
    spec:

      * formation_mechanism  -- 1-sentence physics
      * mass_function        -- M_typical, M_max in kg (and M_⊙ via Msun_*)
      * cosmic_abundance     -- Ω_class / Ω_b or Ω_class / Ω_DM
      * jwst_signature       -- dict with M_apparent, r_eff, color, etc.
    """
    name: str
    short_id: str  # "A".."F"
    formation_mechanism: str
    z_formation: float
    m_typical_kg: float
    m_max_kg: float
    omega_over_omega_b: float = 0.0          # mass fraction relative to Ω_b
    omega_over_omega_dm: float = 0.0         # or relative to Ω_DM (use one)
    jwst_observable: bool = False            # produces a JWST signature?
    m_apparent_Msun: float = 0.0              # M observed by JWST (effective)
    r_eff_pc: float = 0.0                    # effective radius (pc)
    surface_density_Msun_pc2: float = 0.0    # apparent surface density
    transient: bool = False
    spectral_signature: str = ""             # "UV", "no light", etc.

    @property
    def m_typical_Msun(self) -> float:
        return self.m_typical_kg / MSUN_KG

    @property
    def m_max_Msun(self) -> float:
        return self.m_max_kg / MSUN_KG


# =========================================================================
# 1. Class A — DM Pearls (existing dm_pearls_paper v2)
# =========================================================================

def chandrasekhar_mass_topological(m_chi_GeV: float) -> float:
    """Topological Chandrasekhar mass M_Ch ~ M_Pl^3 / m_chi^2 (kg).

    Equation (1) of dm_pearls_paper.tex, with the standard Landau prefactor
    absorbed into the ``~'' (we report M_Pl^3/m_chi^2 directly).
    """
    m_chi_kg = m_chi_GeV * GEV_KG
    return MPL_KG ** 3 / m_chi_kg ** 2


def class_A_dm_pearls() -> CompactObjectClass:
    """Class A: DM pearls.

    Mass spans 10^-12 .. 10^-3 M_sun for the V5 DM knot mass range
    1.55 .. 51.78 GeV (6 STRICT amphichirals).
    """
    # Mass of the lightest DM knot (sets the largest pearl Chandrasekhar)
    M_Ch_lightest = chandrasekhar_mass_topological(min(DM_STRICT_AMPHI_GEV))
    M_Ch_heaviest = chandrasekhar_mass_topological(max(DM_STRICT_AMPHI_GEV))
    # Typical = geometric mean of the range; we report M_Ch/10 (cf. paper)
    m_typical = math.sqrt(M_Ch_lightest * M_Ch_heaviest) / 10.0
    return CompactObjectClass(
        name="DM Pearls (topological Chandrasekhar)",
        short_id="A",
        formation_mechanism=(
            "Self-gravitating Fermi-degenerate aggregates of DM knots "
            "stabilized by topological hard-core exclusion."
        ),
        z_formation=200.0,  # forms throughout, but bulk freezes by recomb.
        m_typical_kg=m_typical,
        m_max_kg=M_Ch_lightest,
        omega_over_omega_dm=0.05,
        # JWST signature: NONE directly (EM-invisible).
        # Indirect contribution via microlensing, NOT in SMF -> 0 here.
        jwst_observable=False,
        m_apparent_Msun=0.0,
        r_eff_pc=0.0,
        transient=False,
        spectral_signature="dark (no EM emission)",
    )


# =========================================================================
# 2. Class B — Topological Primordial BH seeds
# =========================================================================

def heavy_proton_chandrasekhar(m_proton_GeV: float = 3.1) -> float:
    """Chandrasekhar mass for a self-gravitating gas of heavy protons.

    For m_p^(heavy) ~ 3 GeV (Gen 5+), M_Ch ~ M_Pl^3 / m_p^2 ~ 10^29 kg
    ~ 0.05 M_⊙. This is the maximum mass of a self-gravitating
    degenerate heavy-proton cluster BEFORE collapse to a BH.
    """
    m_kg = m_proton_GeV * GEV_KG
    return MPL_KG ** 3 / m_kg ** 2


def pbh_seed_mass_at_sound_horizon(T_MeV: float = 100.0) -> float:
    """PBH seed mass from sound-horizon collapse at temperature T.

    M_H(T) ~ (c^3 / G) * (1/H(T)) ~ 4 pi rho_R c^3 / (3 H(T)^3)
    For T ~ 100 MeV (QCD epoch in standard cosmology, BBN start in V5),
    M_H ~ 1 M_⊙. For T ~ 1 MeV, M_H ~ 10^5 M_⊙.
    """
    # Radiation-dominated era: H ~ 1.66 sqrt(g_*) T^2 / M_pl (units c=hbar=1).
    # Convert: T in MeV. Use simplified analytical fit from Carr+2009.
    # M_H ~ 2.7e5 * (g_*/10.75)^-1/2 * (T/MeV)^-2 M_⊙ (rad-dom)
    g_star = 61.75  # plasma effective dof at T ~ 100 MeV
    M_H_Msun = 2.7e5 * (g_star / 10.75) ** -0.5 * (T_MeV / 1.0) ** -2
    return M_H_Msun * MSUN_KG


def class_B_topological_pbh_seeds() -> CompactObjectClass:
    """Class B: Primordial BH seeds from clustered heavy protons.

    Mechanism: heavy protons (~3 GeV) decay into knot fragments at
    T ~ 100 MeV (V5 §13.7 lifetime). At sound-horizon scales, the
    accumulated cluster of heavy-proton fragments + P22 boundary
    nucleation could collapse into PBHs of mass ~ M_H(T_BBN) ~
    10^5--10^8 M_⊙.

    HONEST CAVEAT: V5.18 showed the rigorous P22 boost is ≈ 1; the
    naive clustering at sound horizon is only marginally efficient.
    Reasonable abundance estimate: Ω_PBH ≲ 10^-3 * Ω_DM (saturating
    the JWST AGN/SMBH high-z population, not the bulk SMF).
    """
    # Use the formed PBH seed at T = 100 MeV (M_H ~ 1 M_⊙)
    m_typical = pbh_seed_mass_at_sound_horizon(T_MeV=100.0)
    # Maximal seed from earlier collapse (T ~ 1 MeV epoch)
    m_max = pbh_seed_mass_at_sound_horizon(T_MeV=1.0)
    return CompactObjectClass(
        name="Topological PBH seeds (heavy-proton clustering)",
        short_id="B",
        formation_mechanism=(
            "Heavy-proton (Gen 5+, ~3 GeV) cluster gravitational collapse "
            "at the sound horizon of the V5 BBN epoch, enhanced by P22 "
            "boundary nucleation. Direct PBH seeding."
        ),
        z_formation=1.0e10,    # forms at T ~ 100 MeV (z ~ 4 x 10^9)
        m_typical_kg=m_typical,
        m_max_kg=m_max,
        # PBH abundance bound: f_PBH = Omega_PBH/Omega_DM <~ 1e-3
        # in the 10..1e9 M_⊙ window (OGLE+EROS+SMBH constraints)
        omega_over_omega_dm=1e-3,
        # PBHs of mass 10^5-10^8 M_⊙ accrete and become SMBH/AGN at z~10
        # -> visible to JWST as point-like AGN, NOT as stellar mass.
        # We report M_apparent ~ 10^5 M_⊙ for the typical seed once
        # accretion has spun up an L_AGN.
        jwst_observable=True,
        m_apparent_Msun=1e6,         # typical luminous AGN seed
        r_eff_pc=1e-3,               # point-like (sub-pc)
        surface_density_Msun_pc2=1e12,
        transient=False,
        spectral_signature="AGN-like (UV + IR power-law continuum, broad Hα)",
    )


# =========================================================================
# 3. Class C — Knot-Baryon Hybrid Stars
# =========================================================================

def class_C_hybrid_stars() -> CompactObjectClass:
    """Class C: pearl core + baryonic envelope accreted post-recombination.

    A DM pearl (Class A) acts as a gravitational seed onto which baryons
    accrete during the matter-dominated era. The resulting hybrid star
    has:
      - core: M_pearl ~ 10^-4 M_⊙ (typical)
      - envelope: M_baryon ~ 10^-3 .. 10^2 M_⊙ depending on epoch & seed

    At z ~ 10, the maximum accreted envelope is the Bondi-accretion
    bound over t_Hubble(z=10): M_acc ~ 10^-2 M_⊙ for a 10^-4 M_⊙ seed.

    HONEST: this class is just baryonic mini-stars triggered earlier
    than vanilla LCDM allows. Apparent mass to JWST = baryonic envelope
    mass only (DM core is invisible). Contribution to high-z SMF
    is potentially nontrivial but bounded by Ω_b at most.
    """
    # Typical hybrid star: pearl seed + 1 M_⊙ baryonic envelope at z ~ 15
    m_typical = 1.0 * MSUN_KG
    # Maximum: massive hybrid with 100 M_⊙ envelope
    m_max = 100.0 * MSUN_KG
    return CompactObjectClass(
        name="Knot-Baryon Hybrid Stars",
        short_id="C",
        formation_mechanism=(
            "DM pearl (Class A) gravitational seeding + Bondi-Hoyle "
            "baryonic accretion in the matter-dominated era (z ~ 50-15)."
        ),
        z_formation=20.0,
        m_typical_kg=m_typical,
        m_max_kg=m_max,
        # Fraction of baryons captured by pearl seeds: small (~10^-3 of Ω_b)
        omega_over_omega_b=1e-3,
        jwst_observable=True,
        m_apparent_Msun=1.0,
        r_eff_pc=1e-7,    # stellar scale, unresolved
        surface_density_Msun_pc2=1e13,
        transient=False,
        spectral_signature="Stellar (T_eff ~ 10^4 K, possible He-only)",
    )


# =========================================================================
# 4. Class D — Plasma Heavy Proton (transient)
# =========================================================================

def class_D_heavy_proton_plasma() -> CompactObjectClass:
    """Class D: transient heavy-proton plasma at BBN-recombination.

    During the V5 BBN epoch (10^2 .. 10^3 s post-bang, T ~ 100 keV..1 MeV),
    heavy protons (Gen 5+) live briefly before decaying into SM nucleons
    via the V5 lifetime formula. The brief radiative cascade liberates
    UV photons that contribute to the CMB-anisotropic foreground.

    HONEST: this class is TRANSIENT (τ ~ 10^3 s), so it CANNOT be
    detected by JWST at z ~ 10 (it has long since decayed). Its only
    cosmological imprint is on the CMB optical depth / spectral
    distortions (μ-distortion, COBE/FIRAS bound).
    """
    # No coherent compact body: it's a plasma. Report typical clump mass
    # corresponding to one Hubble horizon at T = 100 keV.
    m_typical = 1e3 * MSUN_KG  # rough horizon mass at z ~ 10^9
    m_max = m_typical
    return CompactObjectClass(
        name="Heavy-proton plasma (transient BBN epoch)",
        short_id="D",
        formation_mechanism=(
            "Brief population of Gen 5+ heavy protons (~3 GeV) decaying "
            "via knot fragmentation over τ ~ 10^3 s at z ~ 10^9."
        ),
        z_formation=1e9,
        m_typical_kg=m_typical,
        m_max_kg=m_max,
        omega_over_omega_b=0.0,    # vanishes at z << 10^9
        # NOT JWST-observable: decays at z >> 10
        jwst_observable=False,
        transient=True,
        spectral_signature="UV cascade at BBN; CMB μ-distortion",
    )


# =========================================================================
# 5. Class E — Strange Knot Stars (6 STRICT amphichirals)
# =========================================================================

def class_E_strange_knot_stars() -> CompactObjectClass:
    """Class E: stable compact bodies of pure amphichiral knots.

    The 6 STRICT amphichirals (4_1+8_3, 6_3+6_3, 8_3+8_3, etc.) form
    a stable sub-population (Q = 0 exactly, τ > 10^347 s via CS k=135
    barrier). Their Chandrasekhar mass span:

      m_χ = 1.55 GeV  ->  M_Ch ~ 3 x 10^31 kg ~ 0.015 M_⊙
      m_χ = 51.78 GeV ->  M_Ch ~ 2.8 x 10^28 kg ~ 1.4 x 10^-5 M_⊙

    HONEST: these are a SECONDARY DM sub-population (V5 §13.7). The
    catalogue assigns at most ~10% of Ω_DM to STRICT amphichirals
    (the bulk 90% being the 111 REALIZABLE candidates + the n=1 prime
    knot reservoir). So Ω_E / Ω_DM ≲ 0.1.

    Same JWST signature as Class A (EM-invisible) -> no SMF contribution.
    """
    # Use lightest STRICT amphichiral for maximum mass
    M_Ch_lightest = chandrasekhar_mass_topological(min(DM_STRICT_AMPHI_GEV))
    M_Ch_heaviest = chandrasekhar_mass_topological(max(DM_STRICT_AMPHI_GEV))
    m_typical = math.sqrt(M_Ch_lightest * M_Ch_heaviest) / 10.0
    return CompactObjectClass(
        name="Strange Knot Stars (STRICT amphichirals)",
        short_id="E",
        formation_mechanism=(
            "Self-gravitating Fermi aggregates of pure amphichiral knots "
            "(4_1, 6_3, 8_3 chains). Stable via Q=0 + CS k=135 barrier."
        ),
        z_formation=200.0,
        m_typical_kg=m_typical,
        m_max_kg=M_Ch_lightest,
        omega_over_omega_dm=0.10,    # at most 10% of Ω_DM in STRICT
        jwst_observable=False,
        spectral_signature="dark (Q=0 exactly; no EM coupling)",
    )


# =========================================================================
# 6. Class F — Primordial 0_1 Unknot Defects
# =========================================================================

def class_F_unknot_defects() -> CompactObjectClass:
    """Class F: 0_1 unknot defects surviving cascade reheating.

    The V5.17 two-stage reheating cascade (commit 31cb2ba) generates
    transient unknot (0_1) configurations that mostly recombine before
    BBN. A small relic fraction f_relic ~ 10^-6 could survive in
    cold/under-condensed regions and persist as primordial defects.

    Their topological hard core is set by the unknot Compton scale at
    energy E_chi (V5 §17). If E_chi ~ 1 MeV, m_unknot_eff ~ 1 MeV
    -> M_Ch ~ 10^39 kg ~ 10^9 M_⊙ (intermediate-mass black-hole scale).

    SPECULATIVE: this class is essentially Class B with a different
    progenitor (unknot rather than heavy proton). Same JWST signature
    (AGN-like seed), MORE SPECULATIVE abundance bound.
    """
    # Use m_unknot ~ 1 MeV
    m_unknot_GeV = 1e-3
    M_Ch = chandrasekhar_mass_topological(m_unknot_GeV)
    return CompactObjectClass(
        name="Primordial unknot defects (post-cascade relics)",
        short_id="F",
        formation_mechanism=(
            "0_1 unknot configurations surviving the V5.17 two-stage "
            "reheating cascade; gravitationally bound as intermediate-mass "
            "BH seeds in under-condensed primordial regions."
        ),
        z_formation=1e8,
        m_typical_kg=M_Ch / 100.0,
        m_max_kg=M_Ch,
        omega_over_omega_dm=1e-4,    # speculative upper bound
        jwst_observable=True,
        m_apparent_Msun=1e5,
        r_eff_pc=1e-2,
        surface_density_Msun_pc2=1e9,
        transient=False,
        spectral_signature="IMBH-like (AGN-faint, low-mass X-ray)",
    )


# =========================================================================
# 7. Full catalog assembly
# =========================================================================

def build_catalog() -> dict[str, CompactObjectClass]:
    """Return the dictionary of all 6 V5.19 compact-object classes."""
    return {
        "A": class_A_dm_pearls(),
        "B": class_B_topological_pbh_seeds(),
        "C": class_C_hybrid_stars(),
        "D": class_D_heavy_proton_plasma(),
        "E": class_E_strange_knot_stars(),
        "F": class_F_unknot_defects(),
    }


# =========================================================================
# 8. JWST contribution to apparent SMF
# =========================================================================

def jwst_smf_contribution(
    cls: CompactObjectClass,
    z: float = 10.0,
    omega_b_today_Msun_Mpc3: Optional[float] = None,
) -> dict:
    """Apparent SMF contribution of a compact-object class at redshift z.

    For JWST-visible classes (B, C, F), we estimate the comoving
    number density:

      n_class(z) = Omega_class * rho_crit / M_typical(z)

    and the contribution to dn/dlogM_apparent assumed log-normal of
    width 0.3 dex around M_apparent (typical observational scatter).

    Returns a dict with:
      - n_comoving_Mpc3       : comoving number density
      - phi_apparent_dex      : SMF contribution at log(M_apparent)
      - log_M_apparent        : log10(M_apparent / M_⊙)
    """
    if not cls.jwst_observable or cls.transient:
        return {
            "n_comoving_Mpc3": 0.0,
            "phi_apparent_dex": 0.0,
            "log_M_apparent": float("nan"),
            "jwst_relevant": False,
        }

    # Compute Omega_class in units of critical density today
    if cls.omega_over_omega_b > 0:
        Omega_class = cls.omega_over_omega_b * OMEGA_B
    elif cls.omega_over_omega_dm > 0:
        Omega_class = cls.omega_over_omega_dm * OMEGA_DM
    else:
        Omega_class = 0.0

    if Omega_class <= 0 or cls.m_apparent_Msun <= 0:
        return {
            "n_comoving_Mpc3": 0.0,
            "phi_apparent_dex": 0.0,
            "log_M_apparent": math.log10(max(cls.m_apparent_Msun, 1e-30)),
            "jwst_relevant": False,
        }

    rho_class_Msun_Mpc3 = Omega_class * RHO_CRIT_MSUN_MPC3
    n_comoving = rho_class_Msun_Mpc3 / cls.m_apparent_Msun  # Mpc^-3
    # log-normal SMF contribution with sigma_logM = 0.3 dex (typical)
    sigma_logM = 0.3
    # Peak of dn/dlogM at log(M_apparent) is n / (sigma sqrt(2pi))
    phi_peak = n_comoving / (sigma_logM * math.sqrt(2 * math.pi))
    return {
        "n_comoving_Mpc3": n_comoving,
        "phi_apparent_dex": phi_peak,
        "log_M_apparent": math.log10(cls.m_apparent_Msun),
        "Omega_class": Omega_class,
        "jwst_relevant": True,
    }


def total_jwst_contribution(
    z: float = 10.0,
    log_M_target: float = 10.0,
) -> dict:
    """Sum the SMF contribution of all V5 compact-object classes at
    log10(M_*/M_⊙) = log_M_target.

    Uses log-normal kernels centered at each class's M_apparent.

    Returns dict with:
      - phi_total            : total V5-compact contribution at log_M_target
      - per_class            : dict short_id -> phi contribution
      - log_phi_total        : log10
    """
    cat = build_catalog()
    total = 0.0
    per = {}
    sigma_logM = 0.3
    for sid, cls in cat.items():
        contrib = jwst_smf_contribution(cls, z=z)
        if not contrib["jwst_relevant"]:
            per[sid] = 0.0
            continue
        log_M_app = contrib["log_M_apparent"]
        # Gaussian kernel in log_M
        kernel = math.exp(-0.5 * ((log_M_target - log_M_app) / sigma_logM) ** 2)
        contribution = contrib["phi_apparent_dex"] * kernel / (
            math.exp(0) / 1.0  # already normalised in phi_apparent_dex
        )
        per[sid] = contribution
        total += contribution
    return {
        "phi_total": total,
        "per_class": per,
        "log_phi_total": math.log10(max(total, 1e-30)),
        "log_M_target": log_M_target,
        "z": z,
    }


# =========================================================================
# 9. Residual gap analysis (after audit -1.5 dex)
# =========================================================================

# Reference targets at z = 8, log_M_star = 10:
#   Raw Labbé+2023 SMF (Wang+2024 high-z): log Phi ~ -3.8 dex^-1 Mpc^-3
#   After -1.5 dex correction (Krishnan-Abazajian 2026 median):
#     log Phi ~ -5.3 (already much closer to LambdaCDM)
#   LambdaCDM nominal: log Phi ~ -10.3 (Sheth-Tormen)
#   Residual gap LambdaCDM->corrected = +5.0 dex
#   Already absorbed by FFB (Dekel+ 2023) within ~1 dex
LOG_PHI_OBS_CORRECTED = -5.3   # dex^-1 Mpc^-3 at z=8, log M_star=10
LOG_PHI_LCDM_NOMINAL = -10.3   # dex^-1 Mpc^-3 at z=8, log M_star=10
RESIDUAL_GAP_DEX = LOG_PHI_OBS_CORRECTED - LOG_PHI_LCDM_NOMINAL  # +5.0 dex


def residual_gap_analysis(z: float = 8.0, log_M_star: float = 10.0) -> dict:
    """Compute whether V5 compact objects can close the residual JWST gap.

    The residual gap is +5.0 dex at z=8 after the Krishnan-Abazajian 2026
    median correction. We compute the V5 compact-object SMF contribution
    at the same (z, log_M) and compare.
    """
    contrib = total_jwst_contribution(z=z, log_M_target=log_M_star)
    log_phi_v5_compact = contrib["log_phi_total"]
    # Gap remaining = log Phi_obs - log Phi_predicted
    log_phi_total_pred = math.log10(
        10 ** LOG_PHI_LCDM_NOMINAL + contrib["phi_total"]
    )
    gap_remaining = LOG_PHI_OBS_CORRECTED - log_phi_total_pred
    return {
        "z": z,
        "log_M_star": log_M_star,
        "log_phi_obs_corrected": LOG_PHI_OBS_CORRECTED,
        "log_phi_LCDM": LOG_PHI_LCDM_NOMINAL,
        "log_phi_V5_compact": log_phi_v5_compact,
        "log_phi_LCDM_plus_compact": log_phi_total_pred,
        "residual_gap_dex": gap_remaining,
        "V5_compact_closes_gap": gap_remaining < 1.0,
        "per_class": contrib["per_class"],
    }


# =========================================================================
# 10. CLI summary
# =========================================================================

def summary() -> dict:
    """Compact diagnostic summary suitable for CLI / tests."""
    cat = build_catalog()
    out = {"classes": {}}
    for sid, cls in cat.items():
        out["classes"][sid] = {
            "name": cls.name,
            "z_formation": cls.z_formation,
            "M_typical_Msun": cls.m_typical_Msun,
            "M_max_Msun": cls.m_max_Msun,
            "Omega_over_Omega_b": cls.omega_over_omega_b,
            "Omega_over_Omega_DM": cls.omega_over_omega_dm,
            "jwst_observable": cls.jwst_observable,
            "transient": cls.transient,
            "M_apparent_Msun": cls.m_apparent_Msun,
            "spectral_signature": cls.spectral_signature,
        }
    out["residual_gap_z8"] = residual_gap_analysis(z=8.0, log_M_star=10.0)
    return out


if __name__ == "__main__":
    s = summary()
    print("=" * 78)
    print("V5.19 — Catalogue des objets compacts TQNT-V5")
    print("=" * 78)
    for sid, cls in s["classes"].items():
        print(f"\nClass {sid}: {cls['name']}")
        print(f"  z_formation    = {cls['z_formation']:.2e}")
        print(f"  M_typical      = {cls['M_typical_Msun']:.3e} M_sun")
        print(f"  M_max          = {cls['M_max_Msun']:.3e} M_sun")
        print(f"  Omega/Omega_b  = {cls['Omega_over_Omega_b']:.3e}")
        print(f"  Omega/Omega_DM = {cls['Omega_over_Omega_DM']:.3e}")
        print(f"  JWST visible   = {cls['jwst_observable']}, "
              f"transient = {cls['transient']}")
        print(f"  Signature      = {cls['spectral_signature']}")
    rg = s["residual_gap_z8"]
    print(f"\nResidual gap @ z=8, log M*=10:")
    print(f"  log Phi obs (corrected)        = {rg['log_phi_obs_corrected']:.2f}")
    print(f"  log Phi LambdaCDM              = {rg['log_phi_LCDM']:.2f}")
    print(f"  log Phi V5 compact-objects     = {rg['log_phi_V5_compact']:.2f}")
    print(f"  log Phi LCDM + V5 compact      = {rg['log_phi_LCDM_plus_compact']:.2f}")
    print(f"  residual gap                   = {rg['residual_gap_dex']:.2f} dex")
    print(f"  V5 compact closes gap?         = {rg['V5_compact_closes_gap']}")
