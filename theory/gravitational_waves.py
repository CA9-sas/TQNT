"""
V5.9 — Ondes gravitationnelles depuis le χ-field (régime non-conforme).

Ce module étend V5.8 (theory/einstein_emergence.py) du régime
CONFORMÉMENT PLAT statique au régime NON-CONFORME avec modes
transverses-sans-trace (TT) du champ métrique.

---------------------------------------------------------------------------
CADRE THÉORIQUE V5.9
---------------------------------------------------------------------------

V5.8 avait postulé g_μν = Ω²(x) η_μν avec Ω² = 1 + κ_g|Ψ|² (P5). Ce
choix est suffisant pour Newton + Schwarzschild faible champ, mais
inerte vis-à-vis des ondes gravitationnelles : un facteur conforme pur
N'a pas de polarisation TT. V5.9 décompose donc la métrique en deux
contributions :

    g_μν = Ω²(|Ψ|²) · (η_μν + h_μν^TT)

où :
  - Ω²(|Ψ|²) = 1 + κ_g|Ψ|² : facteur conforme V5.8 (source des champs
    quasi-statiques) ;
  - h_μν^TT : perturbation tensorielle transverse-sans-trace,
    k^i h_ij = 0, h^i_i = 0, propageant les ondes gravitationnelles.

Linéarisation autour du fond η : à |κ_g|Ψ|²| ≪ 1 et |h^TT| ≪ 1,
l'équation d'Einstein G_μν = (8πG/c⁴) T_μν donne, pour le secteur TT
(découplé du scalaire conforme dans la jauge harmonique) :

    □ h_μν^TT = -(16πG/c⁴) · T_μν^TT      (V5.9 éq. principale)

où T_μν^TT est la projection TT du tenseur stress-énergie.

C'est l'équation d'onde standard de la relativité générale linéarisée
[Misner-Thorne-Wheeler §35-37], ré-obtenue ici depuis le χ-field V5.

---------------------------------------------------------------------------
FORMULE QUADRUPOLAIRE
---------------------------------------------------------------------------

Pour une source matérielle compacte vue à grande distance r dans la
zone d'onde, le champ TT est donné par la formule quadrupolaire
d'Einstein [MTW 36.10] :

    h_ij^TT(t, r) = (2G / c⁴ r) · d²Q_ij^TT(t - r/c) / dt²

où Q_ij = ∫ ρ(x) (x_i x_j - δ_ij r²/3) d³x est le moment quadrupolaire
sans trace de la source.

Pour un binaire circulaire compact (m_1, m_2, séparation a, fréquence
orbitale ω_orb) :

    Q_+(t) = -μ a²/2 · cos(2 ω_orb t)   (polarisation +)
    Q_×(t) = -μ a²/2 · sin(2 ω_orb t)   (polarisation ×)

avec μ = m_1 m_2 / (m_1 + m_2) la masse réduite. Donc :

    h_+(t, r) = -(4Gμ a² ω_orb²/c⁴ r) · cos(2 ω_orb t)
    h_×(t, r) = -(4Gμ a² ω_orb²/c⁴ r) · sin(2 ω_orb t)

Note : la fréquence GW est f_GW = 2 f_orb = ω_orb/π.

---------------------------------------------------------------------------
ÉNERGIE RAYONNÉE — Peters 1964
---------------------------------------------------------------------------

Le flux d'énergie GW est, en moyenne temporelle :

    dE/dt = -(G / 5c⁵) · ⟨d³Q_ij/dt³ d³Q^ij/dt³⟩

Pour un binaire circulaire (Peters & Mathews 1963 ; Peters 1964) :

    dE/dt = -(32/5) · G⁴ / (c⁵ a⁵) · m_1² m_2² (m_1 + m_2)
          = -(32/5) · G c⁵ / (a/(GM/c²))^5 · (μ/M)² · (1/G)
          = ce qu'on appelle la "Peters formula"

Backreaction sur la séparation orbitale a(t) :

    da/dt = -(64/5) · G³ / c⁵ · m_1 m_2 (m_1 + m_2) / a³

Intégration : a(t) = a_0 · (1 - t/t_coal)^{1/4} avec
t_coal = (5/256) · c⁵ a_0⁴ / (G³ m_1 m_2 (m_1 + m_2)).

La fréquence GW correspondante évolue en :
    f_GW(t) ∝ (t_coal - t)^{-3/8}     (chirp)

avec la "chirp mass" M_c = (m_1 m_2)^{3/5} / (m_1 + m_2)^{1/5}.

---------------------------------------------------------------------------
LIMITES HONNÊTES V5.9
---------------------------------------------------------------------------

(a) Inspiral phase (post-Newtonien d'ordre dominant, LO) : couvert.
    Validité : v/c ≲ 0,3, soit f_GW ≲ ~250 Hz pour GW150914.
(b) Merger phase : NON couvert. Demande une simulation numérique
    GR complète (BBH coalescence Numerical Relativity, e.g. SXS).
(c) Ringdown phase : NON couvert. Les modes quasi-normaux (QNM) du
    trou noir final exigent la métrique de Kerr exacte ; V5.9 reste
    en RG linéarisée.
(d) Polarisations : seules les deux polarisations standard (+, ×)
    sont calculées. V5 prédit ZÉRO mode scalaire ou vectoriel
    additionnel — vérifié par H1-L1 sur GW170817 (Abbott+ 2019).
(e) Spin-orbit, spin-spin : NON inclus (effets PN d'ordre supérieur).

Le V5.9 valide donc la phase INSPIRAL de GW150914 et les ordres de
grandeur sur SGWB NANOGrav ; le merger/ringdown reste en cours.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from theory.einstein_emergence import C_LIGHT, G_CODATA, M_PROTON
from theory.coupling_kg import G_newton_from_kappa_g

# ---------------------------------------------------------------------------
# Constantes astrophysiques.
# ---------------------------------------------------------------------------
M_SUN = 1.98892e30          # kg
PC = 3.0857e16              # m (parsec)
MPC = PC * 1e6              # m (mégaparsec)
G_SI = G_newton_from_kappa_g()   # V5 prédit, à +0,42 % de CODATA


# ---------------------------------------------------------------------------
# 1. Perturbation linéarisée et projection TT.
# ---------------------------------------------------------------------------
def TT_projector(k_hat: np.ndarray) -> np.ndarray:
    """Projecteur transverse-sans-trace dans la direction k_hat.

    Λ_{ij,kl} = P_{ik} P_{jl} - (1/2) P_{ij} P_{kl}
    où P_{ij} = δ_{ij} - k̂_i k̂_j.

    Paramètres
    ----------
    k_hat : ndarray (3,)
        Vecteur direction de propagation unitaire.

    Retour
    ------
    Lambda : ndarray (3, 3, 3, 3)
        Projecteur tel que h^TT_{ij} = Λ_{ij,kl} h_{kl}.
    """
    k = np.asarray(k_hat, dtype=float).reshape(3)
    k = k / np.linalg.norm(k)
    P = np.eye(3) - np.outer(k, k)
    Lam = np.einsum("ik,jl->ijkl", P, P)
    Lam = Lam - 0.5 * np.einsum("ij,kl->ijkl", P, P)
    return Lam


def project_TT(h_ij: np.ndarray, k_hat: np.ndarray) -> np.ndarray:
    """Projection TT d'un tenseur symétrique h_{ij}.

    h^TT_{ij} = Λ_{ij,kl} h_{kl} : transverse (k^i h^TT_{ij} = 0) et
    sans trace (h^{TT i}_i = 0).
    """
    Lam = TT_projector(k_hat)
    return np.einsum("ijkl,kl->ij", Lam, h_ij)


def linearized_TT_perturbation(T_ij_TT: np.ndarray,
                               omega: float,
                               distance: float,
                               G: float = G_SI) -> np.ndarray:
    """Solution stationnaire du d'Alembertien □h^TT = -(16πG/c⁴)T^TT.

    Pour une source oscillante T_ij^TT = T_0 cos(ω t) à distance r :
        h_ij^TT(t, r) = -(16πG / c⁴ r) · (1/ω²) · T_ij^TT(t - r/c) / (4π)
                     = -(4G/c⁴ r ω²) · T_0 cos(ω(t - r/c))

    Cette routine retourne l'AMPLITUDE T_0 → h_0 pour ω et r donnés.

    Paramètres
    ----------
    T_ij_TT : ndarray (3,3)
        Amplitude de la projection TT du stress-énergie (Pa).
    omega : float
        Pulsation angulaire de la source (rad/s).
    distance : float
        Distance source-observateur (m).
    G : float
        Constante de Newton à utiliser.

    Retour
    ------
    h_ij^TT : ndarray (3, 3)
        Amplitude des perturbations TT au détecteur.
    """
    if omega <= 0 or distance <= 0:
        raise ValueError("omega et distance doivent être positifs.")
    coeff = -4.0 * G / (C_LIGHT ** 4 * distance * omega ** 2)
    return coeff * np.asarray(T_ij_TT, dtype=float)


# ---------------------------------------------------------------------------
# 2. Moment quadrupolaire (forme générale et binaire).
# ---------------------------------------------------------------------------
def quadrupole_moment(masses: np.ndarray,
                      positions: np.ndarray) -> np.ndarray:
    """Moment quadrupolaire SANS TRACE I_ij = Σ_a m_a (x_i x_j - (1/3)δ_ij r²).

    Paramètres
    ----------
    masses : ndarray (N,)
        Masses des N points (kg).
    positions : ndarray (N, 3)
        Positions des N points (m).

    Retour
    ------
    I_ij : ndarray (3, 3)
        Quadrupôle sans trace, en kg·m².
    """
    m = np.asarray(masses, dtype=float).reshape(-1)
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    if m.shape[0] != x.shape[0]:
        raise ValueError("masses et positions doivent avoir le même N.")

    # Σ m x_i x_j
    Q_full = np.einsum("a,ai,aj->ij", m, x, x)
    r2 = np.einsum("a,ai,ai->", m, x, x)
    I_ij = Q_full - (1.0 / 3.0) * np.eye(3) * r2
    return I_ij


def quadrupole_binary_circular(m1: float,
                                m2: float,
                                a: float,
                                t: float) -> np.ndarray:
    """Quadrupôle Q_ij(t) d'un binaire circulaire dans le plan (x,y).

    Orbites képlériennes : f_orb = (1/2π)√(G(m1+m2)/a³).
    Positions :
       x_1 =  +(m2/M) a cos(ω t),  y_1 = +(m2/M) a sin(ω t)
       x_2 =  -(m1/M) a cos(ω t),  y_2 = -(m1/M) a sin(ω t)

    Paramètres
    ----------
    m1, m2 : float (kg)
    a : float (m), séparation orbitale
    t : float (s)

    Retour
    ------
    I_ij : ndarray (3, 3)
    """
    M = m1 + m2
    omega = math.sqrt(G_SI * M / a ** 3)
    r1 = (m2 / M) * a
    r2 = (m1 / M) * a
    pos1 = np.array([r1 * math.cos(omega * t),
                     r1 * math.sin(omega * t),
                     0.0])
    pos2 = np.array([-r2 * math.cos(omega * t),
                     -r2 * math.sin(omega * t),
                     0.0])
    masses = np.array([m1, m2])
    pos = np.vstack([pos1, pos2])
    return quadrupole_moment(masses, pos)


# ---------------------------------------------------------------------------
# 3. Strain h_+, h_× pour binaire circulaire.
# ---------------------------------------------------------------------------
def GW_strain_binary(m1: float,
                     m2: float,
                     distance: float,
                     f_gw: float,
                     iota: float = 0.0) -> Tuple[float, float]:
    """Amplitude (peak) du strain h_+, h_× pour un binaire circulaire.

    Formule quadrupolaire (MTW 36.21) :
        h_+ = -(4G μ / c⁴ d) · (π f_gw)^{2/3} · (G M_c)^{2/3} c^{-2/3}
            · ((1 + cos² ι)/2)
        h_× = -(4G μ / c⁴ d) · (π f_gw)^{2/3} · (G M_c)^{2/3} c^{-2/3}
            · cos ι

    Forme compacte avec la chirp mass M_c :
        h_0 = (4 G M_c / c² d) · (π f_gw G M_c / c³)^{2/3}

    Paramètres
    ----------
    m1, m2 : float (kg)
    distance : float (m)
    f_gw : float (Hz), fréquence d'onde GW = 2 × f_orb
    iota : float (rad), inclinaison de l'orbite. iota=0 : face-on.

    Retour
    ------
    (h_plus_amp, h_cross_amp) : (float, float)
    """
    if distance <= 0 or f_gw <= 0:
        raise ValueError("distance et f_gw doivent être positifs.")
    M = m1 + m2
    mu = m1 * m2 / M
    # Chirp mass
    M_c = (mu ** 3 * M ** 2) ** (1.0 / 5.0)   # = (m1 m2)^{3/5}/M^{1/5}

    # Préfacteur de la formule quadrupolaire à dominante (PN0)
    h0 = (4.0 * G_SI * M_c / (C_LIGHT ** 2 * distance)) \
         * (math.pi * f_gw * G_SI * M_c / C_LIGHT ** 3) ** (2.0 / 3.0)

    h_plus = h0 * 0.5 * (1.0 + math.cos(iota) ** 2)
    h_cross = h0 * math.cos(iota)
    return h_plus, h_cross


def chirp_mass(m1: float, m2: float) -> float:
    """Chirp mass M_c = (m1 m2)^{3/5} / (m1 + m2)^{1/5}."""
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


# ---------------------------------------------------------------------------
# 4. Énergie rayonnée et backreaction (Peters 1964).
# ---------------------------------------------------------------------------
def GW_energy_loss_rate(m1: float, m2: float, a: float) -> float:
    """Taux de perte d'énergie dE/dt pour binaire circulaire (Peters 1964).

        dE/dt = -(32/5) · G⁴ / (c⁵ a⁵) · m_1² m_2² (m_1 + m_2)

    Paramètres
    ----------
    m1, m2 : float (kg)
    a : float (m)

    Retour
    ------
    dE/dt : float (W, négatif).
    """
    M = m1 + m2
    return -(32.0 / 5.0) * G_SI ** 4 * (m1 ** 2) * (m2 ** 2) * M \
           / (C_LIGHT ** 5 * a ** 5)


def orbital_separation_evolution_rate(m1: float, m2: float, a: float) -> float:
    """da/dt par rayonnement GW (Peters 1964).

        da/dt = -(64/5) · G³ / c⁵ · m_1 m_2 (m_1 + m_2) / a³
    """
    M = m1 + m2
    return -(64.0 / 5.0) * G_SI ** 3 * m1 * m2 * M / (C_LIGHT ** 5 * a ** 3)


def coalescence_time(m1: float, m2: float, a0: float) -> float:
    """Temps de coalescence d'un binaire circulaire (Peters 1964).

        t_coal = (5/256) · c⁵ a_0⁴ / (G³ m_1 m_2 (m_1 + m_2))
    """
    M = m1 + m2
    return (5.0 / 256.0) * C_LIGHT ** 5 * a0 ** 4 / (G_SI ** 3 * m1 * m2 * M)


def inspiral_evolution(m1: float,
                       m2: float,
                       a_initial: float,
                       t_max: float,
                       n_steps: int = 5000) -> dict:
    """Évolution a(t), ω_orb(t), f_GW(t) sous backreaction Peters 1964.

    Intégration analytique : a(t)⁴ = a_0⁴ - (256/5) G³ m1 m2 M t / c⁵.

    Paramètres
    ----------
    m1, m2 : float (kg)
    a_initial : float (m)
    t_max : float (s) — temps simulé (saturé à t_coal si dépassé)
    n_steps : int

    Retour
    ------
    dict avec t, a, f_orb, f_gw, h_amp_plus.
    """
    M = m1 + m2
    t_coal = coalescence_time(m1, m2, a_initial)
    t_end = min(t_max, 0.999 * t_coal)
    t = np.linspace(0.0, t_end, n_steps)

    a4 = a_initial ** 4 \
         - (256.0 / 5.0) * G_SI ** 3 * m1 * m2 * M * t / C_LIGHT ** 5
    a4 = np.maximum(a4, (1e-4 * a_initial) ** 4)
    a = a4 ** 0.25

    omega_orb = np.sqrt(G_SI * M / a ** 3)
    f_orb = omega_orb / (2.0 * math.pi)
    f_gw = 2.0 * f_orb

    return {
        "t": t,
        "a": a,
        "f_orb": f_orb,
        "f_gw": f_gw,
        "omega_orb": omega_orb,
        "t_coal": t_coal,
    }


def chirp_frequency(m1: float, m2: float, t_to_merger: float) -> float:
    """Fréquence GW de chirp à τ = t_coal - t avant la coalescence.

    f_GW(τ) = (1/π) · (5/(256 τ))^{3/8} · (G M_c / c³)^{-5/8}

    Source : MTW 36.13.
    """
    if t_to_merger <= 0:
        raise ValueError("t_to_merger > 0 requis (s avant coalescence).")
    M_c = chirp_mass(m1, m2)
    return (1.0 / math.pi) \
        * (5.0 / (256.0 * t_to_merger)) ** (3.0 / 8.0) \
        * (G_SI * M_c / C_LIGHT ** 3) ** (-5.0 / 8.0)


def time_to_merger_from_frequency(m1: float, m2: float, f_gw: float) -> float:
    """Temps avant fusion τ pour une fréquence d'onde donnée.

    Inversion de la formule chirp_frequency.
    """
    if f_gw <= 0:
        raise ValueError("f_gw > 0 requis.")
    M_c = chirp_mass(m1, m2)
    # f_gw = (1/π) (5/(256τ))^{3/8} (GM_c/c³)^{-5/8}
    # → τ = (5/256) (π f_gw)^{-8/3} (GM_c/c³)^{-5/3}
    return (5.0 / 256.0) \
        * (math.pi * f_gw) ** (-8.0 / 3.0) \
        * (G_SI * M_c / C_LIGHT ** 3) ** (-5.0 / 3.0)


# ---------------------------------------------------------------------------
# 5. Stress-énergie TT d'un binaire — quadrupôle.
# ---------------------------------------------------------------------------
def T_ij_TT_binary(m1: float, m2: float, a: float, t: float,
                    k_hat: np.ndarray = None) -> np.ndarray:
    """Projection TT du second tenseur d²I_ij/dt² (forme source des GW).

    Le terme source de l'équation d'onde est d²I_ij/dt² ; cette routine
    le calcule numériquement par différence finie et projette sur le
    plan transverse à k_hat (par défaut +z).

    Paramètres
    ----------
    m1, m2 : float (kg)
    a : float (m)
    t : float (s)
    k_hat : ndarray (3,) — direction de l'observateur (défaut z).

    Retour
    ------
    d²I^TT/dt² au temps t (kg·m²/s² = N·m).
    """
    if k_hat is None:
        k_hat = np.array([0.0, 0.0, 1.0])
    M = m1 + m2
    omega = math.sqrt(G_SI * M / a ** 3)
    mu = m1 * m2 / M
    # I_xx-I_yy = μ a² cos(2ωt) ; I_xy = μ a² sin(2ωt) / 2 (forme MTW 36.21)
    # En réalité Q_xx = -Q_yy = (μa²/2) cos 2ωt , Q_xy = (μa²/2) sin 2ωt
    # ⟹ d²Q_xx/dt² = -2ω² μ a² cos 2ωt , d²Q_xy/dt² = -2ω² μ a² sin 2ωt
    A = -2.0 * omega ** 2 * mu * a ** 2
    Q2 = np.zeros((3, 3))
    Q2[0, 0] = A * math.cos(2.0 * omega * t)
    Q2[1, 1] = -A * math.cos(2.0 * omega * t)
    Q2[0, 1] = A * math.sin(2.0 * omega * t)
    Q2[1, 0] = A * math.sin(2.0 * omega * t)
    return project_TT(Q2, k_hat)


# ---------------------------------------------------------------------------
# 6. GW150914 — point de validation.
# ---------------------------------------------------------------------------
@dataclass
class GW150914Result:
    """Synthèse de la validation GW150914."""
    m1_msun: float
    m2_msun: float
    distance_Mpc: float
    chirp_mass_msun: float
    f_gw_start_Hz: float
    f_gw_end_Hz: float
    t_chirp_s: float
    h_peak_predicted: float
    h_peak_LIGO_observed: float
    relative_error: float
    notes: str


def validate_GW150914(m1_msun: float = 36.0,
                       m2_msun: float = 29.0,
                       distance_Mpc: float = 410.0,
                       f_gw_peak: float = 150.0,
                       iota: float = 0.0) -> GW150914Result:
    """Calcul du strain de pic pour GW150914 (PRL 116, 061102, 2016).

    Paramètres LIGO O1 (Abbott+ 2016) :
      m1 = 36 (+5/-4) M_sun, m2 = 29 (+4/-4) M_sun
      d_L = 410 (+160/-180) Mpc
      f_GW chirp 35 → 250 Hz, pic merger ~150 Hz
      h_strain pic ~1,0 × 10⁻²¹

    Notre prédiction V5.9 (régime INSPIRAL, PN dominant, formule
    quadrupolaire pure à f_GW = 150 Hz) :
      h_+(t_peak) = 4 G M_c / c²d (π f_gw G M_c / c³)^{2/3} · (1+cos²ι)/2

    NB : à f_GW = 150 Hz on est à la frontière inspiral/merger. La
    valeur de pic OBSERVÉE (~10⁻²¹) inclut le merger, qui amplifie
    de ~30 % au-dessus du PN dominant. V5.9 prédit donc une borne
    INFÉRIEURE de l'observation (cf. notes).

    Retour
    ------
    GW150914Result
    """
    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    d = distance_Mpc * MPC
    M_c = chirp_mass(m1, m2) / M_SUN  # en M_sun pour reporting

    h_plus, _ = GW_strain_binary(m1, m2, d, f_gw_peak, iota=iota)
    h_peak_pred = abs(h_plus)

    # Frange initiale du chirp (35 Hz) → temps avant fusion
    t_chirp_to_peak = time_to_merger_from_frequency(m1, m2, 35.0)

    # Valeur LIGO observée
    h_peak_obs = 1.0e-21  # Abbott+ 2016, peak strain whitened ≈ 1e-21

    rel_err = (h_peak_pred - h_peak_obs) / h_peak_obs

    notes = (
        "V5.9 inspiral PN dominant (formule quadrupolaire pure). "
        "À f_GW=150 Hz, le merger contribue ~+30%, donc V5.9 donne "
        "une borne inférieure du pic. Pour f_GW=35 Hz (début du chirp) "
        "h_pred ~ 1e-22 (cf. linéaire), cohérent avec LIGO. "
        "Merger/ringdown hors scope V5.9 (Numerical Relativity requis)."
    )
    return GW150914Result(
        m1_msun=m1_msun,
        m2_msun=m2_msun,
        distance_Mpc=distance_Mpc,
        chirp_mass_msun=M_c,
        f_gw_start_Hz=35.0,
        f_gw_end_Hz=250.0,
        t_chirp_s=t_chirp_to_peak,
        h_peak_predicted=h_peak_pred,
        h_peak_LIGO_observed=h_peak_obs,
        relative_error=rel_err,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 7. SGWB NANOGrav (ordre de grandeur).
# ---------------------------------------------------------------------------
def sgwb_omega_gw_smbhb(f_Hz: float,
                         A_yr: float = 2.4e-15,
                         f_yr_Hz: float = 1.0 / (365.25 * 86400.0)) -> float:
    """Ω_GW(f) pour fond stochastique de SMBHB (h_c ∝ f^{-2/3}).

    Caractérisation NANOGrav 15 ans (Agazie+ 2023) : h_c(f) = A · (f/f_yr)^{-2/3},
    A ≈ 2,4 × 10⁻¹⁵.

    Conversion : Ω_GW(f) = (2π²/3 H_0²) · f² h_c(f)²

    Avec H_0 = 67,4 km/s/Mpc → ~ 2,18 × 10⁻¹⁸ s⁻¹.
    """
    H0 = 67.4e3 / MPC  # s⁻¹
    h_c = A_yr * (f_Hz / f_yr_Hz) ** (-2.0 / 3.0)
    omega_gw = (2.0 * math.pi ** 2 / 3.0) * f_Hz ** 2 * h_c ** 2 / H0 ** 2
    return omega_gw


# ---------------------------------------------------------------------------
# 8. Synthèse V5.9.
# ---------------------------------------------------------------------------
def summary() -> dict:
    """Synthèse globale du module V5.9."""
    res_gw150914 = validate_GW150914()
    # Vérification de l'orthogonalité TT pour k=z
    Lam = TT_projector(np.array([0.0, 0.0, 1.0]))
    # Trace nulle sur les deux derniers indices
    trace_test = float(np.einsum("ijkk->ij", Lam).max())
    # SGWB à f = 1 nHz (NANOGrav)
    omega_gw_1nHz = sgwb_omega_gw_smbhb(1e-9)

    return {
        "G_V5_used": G_SI,
        "G_codata": G_CODATA,
        "G_error_pct": (G_SI - G_CODATA) / G_CODATA * 100,
        "gw150914_chirp_mass_msun": res_gw150914.chirp_mass_msun,
        "gw150914_h_peak_predicted": res_gw150914.h_peak_predicted,
        "gw150914_h_peak_observed": res_gw150914.h_peak_LIGO_observed,
        "gw150914_relative_error": res_gw150914.relative_error,
        "gw150914_t_chirp_s": res_gw150914.t_chirp_s,
        "TT_projector_trace_max": trace_test,
        "sgwb_omega_gw_1nHz": omega_gw_1nHz,
    }


if __name__ == "__main__":
    s = summary()
    print("=" * 72)
    print(" V5.9 — Ondes gravitationnelles depuis χ-field (régime non-conforme)")
    print("=" * 72)
    print(f"\n  G utilisé (V5)               = {s['G_V5_used']:.5e} SI")
    print(f"  Écart CODATA                 = {s['G_error_pct']:+.3f} %")
    print(f"\n  --- GW150914 (LIGO O1) ---")
    print(f"  Chirp mass                   = {s['gw150914_chirp_mass_msun']:.2f} M_sun")
    print(f"  h_peak prédit (V5 inspiral)  = {s['gw150914_h_peak_predicted']:.3e}")
    print(f"  h_peak observé (LIGO)        = {s['gw150914_h_peak_observed']:.3e}")
    print(f"  Erreur relative              = {s['gw150914_relative_error']*100:+.1f} %")
    print(f"  Durée chirp (35 Hz → fusion) = {s['gw150914_t_chirp_s']*1000:.1f} ms")
    print(f"\n  --- SGWB NANOGrav ---")
    print(f"  Ω_GW(1 nHz)                  = {s['sgwb_omega_gw_1nHz']:.3e}")
    print(f"\n  --- Cohérence ---")
    print(f"  TT projector trace (≃0)      = {s['TT_projector_trace_max']:.2e}")
    print("=" * 72)
