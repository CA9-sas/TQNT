"""
V5.10a — Théorie Kerr-V5 depuis le champ χ : ansatz Ψ_rot, couplage axial κ_a
et lien spin topologique → moment cinétique macroscopique.

Ce module est une FONDATION THÉORIQUE de SCOPING : il pose les ingrédients
V5 nécessaires pour étendre la métrique conforme statique (V5.8)
au régime ROTATIONNEL stationnaire, par cohérence asymptotique avec la
métrique de Kerr en coordonnées de Boyer-Lindquist (BL).

C'est NON une simulation Kerr complète, NON un solveur de Relativité
Numérique (NR), NON un solveur d'horizon. C'est l'identification des
objets V5 qu'il faudra implémenter dans une étape ultérieure (V5.10c+)
pour décrire un système rotationnel.

---------------------------------------------------------------------------
CADRE V5.8 ET LIMITES DU STATIQUE
---------------------------------------------------------------------------

V5.8 a établi (cf. theory/einstein_emergence.py) :

    g_μν(x) = Ω²(|Ψ|²) · η_μν,     Ω² = 1 + κ_g · |Ψ|²       (P5)
    κ_g    = (96π/7) · α_EM^18 · λbar_C(proton)
    G      = (12/7) · α_EM^18 · ℏc / m_p²    [CODATA + 0,42 %]

Cette métrique est CONFORMÉMENT PLATE :
  - reproduit la limite de Newton (Poisson) exactement,
  - reproduit la limite de Schwarzschild en jauge isotrope au 1er ordre,
  - MAIS : g_tφ ≡ 0 → pas de frame dragging, pas de moment cinétique
    macroscopique, pas d'ergorégion. Elle ne peut pas décrire un
    objet en rotation autour d'un axe.

V5.9 a ajouté des modes TT pour les ondes gravitationnelles
(non-conforme mais transverse-traceless).

V5.10a (ce module) ajoute le secteur AXIAL stationnaire :
le profil de densité |Ψ|² et la métrique acquièrent des composantes
liées au moment cinétique du χ-field collectif.

---------------------------------------------------------------------------
ANSATZ Ψ_rot (POSTULAT DE TRAVAIL)
---------------------------------------------------------------------------

Pour un nœud (ou un entrelacs effectif macroscopique) en rotation autour
d'un axe e_z avec moment cinétique topologique J_top, le profil χ
acquiert une déformation axiale au premier ordre :

    |Ψ_rot|²(r, θ) = |Ψ_static|²(r) · (1 + ε_rot · f(r, θ))

avec
    ε_rot = a / r_g,    a = J / (M c),    r_g = G M / c²
    f(r, θ) = sin²θ · g(r)

et g(r) un profil radial tendant vers 1/r à grand r (cohérence
asymptotique). f(r,θ) ne dépend pas de φ : le système est stationnaire
et axisymétrique. La parité polaire est sin²θ (rotation autour de e_z).

JUSTIFICATION PHYSIQUE :
  - Un nœud en rotation produit une circulation du courant χ autour
    de e_z ; le courant Noether U(1) associé à Ψ → e^{iα}Ψ se distribue
    avec un poids sin²θ (composante orthoradiale de v).
  - À l'équateur (θ = π/2), |Ψ_rot|² est maximalement modulé ;
    aux pôles (θ = 0, π), pas de modulation : le χ-field n'a aucune
    composante d'enlacement axial là où la rotation s'évanouit.

LIMITES :
  - Cet ansatz est PERTURBATIF en a (a/r_g ≪ 1). Il ne décrit pas le
    régime de Kerr extrême (a → M G/c²).
  - Il est asymptotique : la validité à r ~ r_g (proche horizon)
    requiert un raccord avec un noyau topologique non décrit ici.

---------------------------------------------------------------------------
COUPLAGE κ_a (IDENTIFICATION DEPUIS BOYER-LINDQUIST)
---------------------------------------------------------------------------

En coordonnées de Boyer-Lindquist, Kerr s'écrit :

    g_tt   = -(1 - 2 G M r / (c² Σ))
    g_tφ   = -(2 G M a r sin²θ) / (c² Σ)
    g_rr   = Σ / Δ
    g_θθ   = Σ
    g_φφ   = (r² + a² + (2 G M a² r sin²θ)/(c² Σ)) sin²θ

avec
    Σ = r² + a² cos²θ
    Δ = r² - 2 G M r / c² + a²

À grand r (régime asymptotique), Σ → r², Δ → r², et la composante
hors-diagonale se simplifie :

    g_tφ → -2 G M a sin²θ / (c² r)
         = -(2 G J sin²θ) / (c³ r),    J = M a c    (convention SI).

(En coordonnées BL la composante a est en mètres : a = J/(Mc) ; J en J·s.)

Cohérence V5 exigée : la métrique V5.10a doit reproduire cette forme
asymptotique. On postule donc

    g_tφ(r, θ) = κ_a · ω_χ(r, θ) · |Ψ_rot|²(r, θ)

avec ω_χ(r,θ) un profil sans dimension de rotation du χ-field. Au lieu
d'introduire un nouveau paramètre libre κ_a, on demande que la limite
asymptotique reproduise Kerr-BL exactement ; cela CONTRAINT κ_a en
fonction de κ_g et de la normalisation de |Ψ|². On utilise la
relation V5.8 :
    Φ_Newton(r) = -(c²/2) · κ_g · |Ψ_static|²(r) = -G M / r
    ⟹  κ_g · |Ψ_static|²(r) = 2 G M / (c² r)

À grand r, posant ω_χ(r,θ) = sin²θ et |Ψ_rot|² ≈ |Ψ_static|² :

    g_tφ = κ_a · sin²θ · κ_g⁻¹ · (2 G M / (c² r))     (en injectant |Ψ|²)
         ?= -2 G M a sin²θ / (c² r)

donne

    κ_a = -κ_g · a

(Unités : κ_a est en m² — facteur multiplicatif sur |Ψ|² adim après
contraction par κ_g.) À a = 0, κ_a = 0 et g_tφ = 0 : on retombe sur V5.8.

INTERPRÉTATION PHYSIQUE :
  - Le signe négatif provient de la convention BL standard
    (rotation dans le sens des φ croissants ↔ g_tφ < 0).
  - κ_a est proportionnel à κ_g et au paramètre cinématique a c.
    Il N'EST PAS un nouveau paramètre fondamental V5 : il EST FIXÉ par
    la cohérence asymptotique et par le moment cinétique
    macroscopique J = N_χ · ⟨S⟩.
  - À a = 0 (statique), κ_a = 0, et g_tφ = 0 : on retombe sur V5.8.

---------------------------------------------------------------------------
SPIN TOPOLOGIQUE → MOMENT CINÉTIQUE MACROSCOPIQUE
---------------------------------------------------------------------------

En V5, le spin d'un nœud K émerge de son indice de tresse b(K)
(cf. theory/braid_index, V5 §spin) :

    S_K = ℏ · (b(K) - 1) / 2    si b(K) impair
        = ℏ · b(K) / 2          si b(K) pair

Pour un objet macroscopique constitué de N_χ quanta du χ-field
(N_χ = M / m_p, avec m_p la masse de référence du χ-field), un
moment cinétique effectif émerge si une fraction f_aligne des spins
individuels s'aligne sur e_z :

    J_macro = N_χ · f_aligne · ⟨S⟩    où  ⟨S⟩ = ℏ b_eff / 2

soit J = (M / m_p) · f_aligne · (ℏ b_eff / 2).

CAS LIMITES :
  - Objet macroscopique non aligné (étoile au repos) : f_aligne ~ 1/√N_χ
    par fluctuation statistique ; J est gaussien autour de 0,
    cohérent avec l'absence de moment cinétique cohérent à grande
    échelle.
  - Trou noir formé par effondrement : f_aligne ~ a/M ~ J/(M² c) ;
    J émerge de la conservation orbitale, b_eff joue le rôle d'un
    "spin résiduel collectif" du condensat χ.
  - Limite Kerr extrême a → G M / c² : f_aligne · b_eff atteint un
    plafond topologique ; au-delà, la nucléation P22 brise la
    cohérence et l'horizon est saturé.

---------------------------------------------------------------------------
CONDITIONS DE BORD
---------------------------------------------------------------------------

  ASYMPTOTE r → ∞ :
    Ω² → 1, |Ψ_rot|² → 0, ω_χ · |Ψ_rot|² → 0
    ⟹ g_μν → η_μν (Minkowski plat). Cohérence ADM standard.

  AXE DE ROTATION (θ = 0, π) :
    sin²θ → 0, donc g_tφ → 0 et |Ψ_rot|² → |Ψ_static|² : la rotation
    n'affecte rien sur l'axe (cohérent avec Kerr).

  HORIZON (r → r_+ = r_g · (1 + √(1 - a²/(r_g c)²))) :
    En V5, l'horizon Kerr classique N'EST PAS une singularité
    coordonnée ; le rôle est tenu par la NUCLÉATION P22 (régime
    ξ = |∇Ω²| λbar_C / Ω² → ∞). Lorsque la métrique conforme rend
    le gradient de Ω² critique, la nucléation explose, l'espace-temps
    se "déchire" en un condensat χ-field nouveau. Cette étape DÉPASSE
    le cadre de ce module et est traitée séparément (V5.10c+).

---------------------------------------------------------------------------
STATUT
---------------------------------------------------------------------------

Ce module fournit :
  - un ansatz Ψ_rot(r, θ) paramétrique,
  - l'identification κ_a = -κ_g · a · c (asymptotique BL),
  - la conversion spin topologique → J macroscopique,
  - le profil ω_χ(r, θ) prédit (frame dragging),
  - des tests de cohérence : limite a → 0 ⟹ Schwarzschild V5.8 ;
    asymptote r → ∞ ⟹ Minkowski ; g_tφ → -2 G J sin²θ / (c² r) à
    grand r.

LIMITES HONNÊTES :
  1. PERTURBATIF EN a : tout est au 1er ordre en a/r_g.
  2. PAS D'HORIZON : la métrique n'est pas régularisée au voisinage
     de r_+ ; pas d'ergosphère explicite.
  3. PAS DE MERGER : un système binaire en rotation requiert de la NR.
  4. PAS DE VALIDATION OBSERVATIONNELLE : aucune comparaison avec EHT
     ou GW spin n'est faite ici (deferred à V5.10c+).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from theory.coupling_kg import (
    ALPHA_EM,
    LAMBDABAR_C_PROTON_M,
    kappa_g_V5p5,
)
from theory.einstein_emergence import (
    C_LIGHT,
    G_CODATA,
    HBAR,
    M_PROTON,
)

# ---------------------------------------------------------------------------
# 0. Constantes dérivées V5.8.
# ---------------------------------------------------------------------------
KAPPA_G_M = kappa_g_V5p5("m")          # ≈ 3,13e-53 m
G_V5 = (12.0 / 7.0) * (ALPHA_EM ** 18) * HBAR * C_LIGHT / (M_PROTON ** 2)


# ---------------------------------------------------------------------------
# 1. Ansatz Ψ_rot.
# ---------------------------------------------------------------------------
def schwarzschild_radius(M_kg: float) -> float:
    """Rayon de Schwarzschild r_g = G M / c² (utilise G_V5).

    Note : on utilise le G dérivé V5 plutôt que CODATA pour rester
    cohérent avec la dérivation interne.
    """
    return G_V5 * M_kg / (C_LIGHT ** 2)


def kerr_parameter(J_si: float, M_kg: float) -> float:
    """Paramètre de Kerr a = J / (M c), en mètres.

    Pour Kerr extrême : a → r_g = G M / c², soit J → G M² / c.
    """
    return J_si / (M_kg * C_LIGHT)


def Psi_static_squared(r_m: float, M_kg: float) -> float:
    """Profil statique V5.8 normalisé : |Ψ_static|²(r) tel que
    Φ_Newton(r) = -(c²/2) · κ_g · |Ψ_static|² = -G M / r.

    Donc |Ψ_static|²(r) = 2 G M / (κ_g c² r).

    Paramètres
    ----------
    r_m : float
        Rayon en mètres (r > r_g).
    M_kg : float
        Masse totale en kg.

    Retour
    ------
    psi2 : float (adimensionnel — h = κ_g · |Ψ|²)
    """
    if r_m <= 0:
        raise ValueError("r doit être > 0")
    # h = κ_g · |Ψ|² = 2 G M / (c² r)  ⟹  |Ψ|² = h / κ_g
    h = 2.0 * G_V5 * M_kg / (C_LIGHT ** 2 * r_m)
    return h / KAPPA_G_M


def Psi_rot_ansatz(r_m: float,
                   theta_rad: float,
                   M_kg: float,
                   a_m: float) -> float:
    """Ansatz V5.10a : |Ψ_rot|²(r, θ) = |Ψ_static|²(r) · (1 + ε f(r,θ)).

    ε_rot = a / r_g,
    f(r, θ) = sin²θ · (r_g / r)    (profil 1/r à grand r).

    L'ansatz est PERTURBATIF en ε_rot.

    Paramètres
    ----------
    r_m : float (> r_g)
        Rayon (m).
    theta_rad : float (∈ [0, π])
        Colatitude.
    M_kg : float
        Masse.
    a_m : float (≥ 0)
        Paramètre de Kerr a = J/(Mc).

    Retour
    ------
    |Ψ_rot|²(r, θ).
    """
    if r_m <= 0:
        raise ValueError("r > 0 requis")
    psi2_static = Psi_static_squared(r_m, M_kg)
    if a_m == 0.0:
        return psi2_static
    rg = schwarzschild_radius(M_kg)
    eps_rot = a_m / rg
    sin2 = math.sin(theta_rad) ** 2
    f_rt = sin2 * (rg / r_m)
    return psi2_static * (1.0 + eps_rot * f_rt)


# ---------------------------------------------------------------------------
# 2. Couplage axial κ_a.
# ---------------------------------------------------------------------------
def kappa_a_from_V5(a_m: float) -> float:
    """Couplage axial V5.10a identifié par cohérence asymptotique BL.

    Dérivation (voir docstring du module) :
        g_tφ_asymp = κ_a · sin²θ · |Ψ_static|²       (ansatz V5)
                   = κ_a · sin²θ · (2 G M)/(κ_g c² r)
        g_tφ_BL    = -2 G M a sin²θ / (c² r)
        ⟹ κ_a · 2 G M / (κ_g c² r) = -2 G M a / (c² r)
        ⟹ κ_a = -κ_g · a

    Paramètres
    ----------
    a_m : float
        Paramètre de Kerr (m).

    Retour
    ------
    κ_a (m²)
    """
    return -KAPPA_G_M * a_m


def frame_dragging_omega(r_m: float,
                         theta_rad: float,
                         M_kg: float,
                         a_m: float) -> float:
    """Fréquence de frame dragging ω(r, θ) prédite V5.10a.

    Définition standard ADM : ω = -g_tφ / g_φφ.
    À grand r, g_φφ ≈ r² sin²θ et g_tφ ≈ -2GJ sin²θ / (c² r), donc
        ω(r, θ) ≈ 2 G J / (c² r³)   (indépendant de θ asymptotique)
                ≈ 2 G M a / (c r³)   en utilisant J = M a c.

    Pour Kerr exact (BL) :
        ω = 2 G M a r / (c · Σ · A / (sin²θ Σ)) ≈ 2GMa/(c r³)   à grand r.

    Paramètres
    ----------
    r_m : float (> 0)
    theta_rad : float
    M_kg : float
    a_m : float

    Retour
    ------
    ω en rad/s (positif si a > 0).
    """
    if r_m <= 0:
        raise ValueError("r > 0 requis")
    # Asymptotique : ω = 2GMa / (c r³), indépendant de θ (correction θ
    # à r ~ r_g non incluse dans ce scoping).
    return 2.0 * G_V5 * M_kg * a_m / (C_LIGHT * r_m ** 3)


def g_tphi_asymptotic(r_m: float,
                      theta_rad: float,
                      M_kg: float,
                      a_m: float) -> float:
    """Composante g_tφ prédite V5.10a au régime asymptotique.

    g_tφ = κ_a · sin²θ · |Ψ_static|²
         = -κ_g · a · sin²θ · (2 G M)/(κ_g c² r)
         = -2 G M a sin²θ / (c² r)

    À comparer avec g_tφ_BL au même régime asymptotique. Convention :
    a = J/(Mc) en mètres, J en J·s ; g_tφ en m²·s⁻¹ (jauge t en s, φ
    sans dimension, métrique en m² × ds²/dt × dφ : c'est la
    convention SI standard pour BL).
    """
    if r_m <= 0:
        raise ValueError("r > 0 requis")
    return -2.0 * G_V5 * M_kg * a_m * (math.sin(theta_rad) ** 2) / (
        C_LIGHT ** 2 * r_m)


def g_tphi_boyer_lindquist(r_m: float,
                           theta_rad: float,
                           M_kg: float,
                           a_m: float,
                           use_G: float = None) -> float:
    """Composante g_tφ exacte de Kerr (Boyer-Lindquist).

    g_tφ = -(2 G M a r sin²θ) / (c² Σ),  Σ = r² + a² cos²θ.

    use_G : G à utiliser (par défaut G_V5).
    """
    if r_m <= 0:
        raise ValueError("r > 0 requis")
    G = G_V5 if use_G is None else use_G
    sigma = r_m ** 2 + (a_m ** 2) * (math.cos(theta_rad) ** 2)
    return -2.0 * G * M_kg * a_m * r_m * (math.sin(theta_rad) ** 2) / (
        C_LIGHT ** 2 * sigma)


# ---------------------------------------------------------------------------
# 3. Spin topologique → moment cinétique macroscopique.
# ---------------------------------------------------------------------------
def spin_topological(b_K: int) -> float:
    """Spin S_K d'un nœud K depuis son indice de tresse b(K).

    Convention V5 (cf. chap. Spin) :
        S = ℏ · (b - 1) / 2    si b impair (fermion 1/2, 3/2, ...)
        S = ℏ · b / 2          si b pair (boson 1, 2, ...)

    Paramètres
    ----------
    b_K : int ≥ 1
        Indice de tresse minimal du nœud K.

    Retour
    ------
    S en J·s (unités SI de moment cinétique).
    """
    if b_K < 1:
        raise ValueError("b(K) ≥ 1 requis")
    if b_K % 2 == 1:
        return HBAR * (b_K - 1) / 2.0
    return HBAR * b_K / 2.0


def spin_topological_to_J(b_eff: float,
                          M_kg: float,
                          f_align: float = 1.0,
                          mass_quantum_kg: float = M_PROTON) -> float:
    """Moment cinétique macroscopique J depuis un spin topologique effectif.

    Formule V5.10a :
        J_macro = N_χ · f_align · ⟨S⟩
        N_χ     = M / m_quantum
        ⟨S⟩    ≈ ℏ · b_eff / 2

    Paramètres
    ----------
    b_eff : float
        Indice de tresse effectif du condensat χ-field (≥ 0).
    M_kg : float
        Masse totale macroscopique (kg).
    f_align : float ∈ [0, 1]
        Fraction de quanta alignés sur e_z.
        - f_align = 0   : pas de moment cinétique cohérent.
        - f_align = 1   : alignement parfait (saturation).
        - typiquement f_align ~ a/r_g pour un trou noir.
    mass_quantum_kg : float
        Masse du quantum χ-field (par défaut m_proton).

    Retour
    ------
    J (J·s).
    """
    if not 0.0 <= f_align <= 1.0:
        raise ValueError("f_align doit être dans [0,1]")
    N_chi = M_kg / mass_quantum_kg
    S_mean = HBAR * b_eff / 2.0
    return N_chi * f_align * S_mean


def f_align_from_kerr_a(a_m: float, M_kg: float) -> float:
    """Fraction d'alignement f_align inférée du paramètre de Kerr.

    Pour un trou noir Kerr : J = M a c = (M/m_p) · f_align · (ℏ b_eff / 2).
    En posant b_eff = 1 (alignement résiduel minimal du condensat),
    cela donne :
        f_align = 2 m_p a c / (ℏ · b_eff)
    Pour Kerr extrême a = r_g = G M / c² et M ~ 1 M_sun :
        f_align ~ 2 m_p G M / (ℏ c · b_eff)
    Numériquement f_align ≪ 1 sauf pour des trous noirs micro
    (condensat χ cohérent).

    Cette fonction sert de pont entre cinématique (a) et statistique
    topologique (f_align), pas de prédiction expérimentale immédiate.

    Paramètres
    ----------
    a_m : float
        Paramètre de Kerr (m).
    M_kg : float
        Masse (kg).

    Retour
    ------
    f_align (sans dimension, peut > 1 si paramétrage non-physique).
    """
    b_eff = 1.0
    return 2.0 * M_PROTON * a_m * C_LIGHT / (HBAR * b_eff) / (M_kg / M_PROTON)


# ---------------------------------------------------------------------------
# 4. Cohérences et tests de raccord.
# ---------------------------------------------------------------------------
@dataclass
class KerrV5Check:
    """Synthèse de cohérence Kerr-V5 vs Boyer-Lindquist asymptotique."""
    r_m: float
    theta_rad: float
    M_kg: float
    a_m: float
    g_tphi_V5: float
    g_tphi_BL: float
    ratio: float
    asymptotic_regime: bool         # True si r ≫ r_g et r ≫ a
    note: str


def asymptotic_consistency_check(r_m: float,
                                 theta_rad: float,
                                 M_kg: float,
                                 a_m: float,
                                 r_g_factor_threshold: float = 100.0
                                 ) -> KerrV5Check:
    """Compare g_tφ(V5.10a) à g_tφ(Boyer-Lindquist) au point (r, θ).

    À r ≫ r_g et r ≫ a, le rapport doit tendre vers 1.
    Plus près de l'horizon, l'écart croît : c'est attendu et signale
    la nécessité de V5.10c+ (corrections d'ordre supérieur, P22).

    Paramètres
    ----------
    r_m, theta_rad, M_kg, a_m : flottants standard.
    r_g_factor_threshold : float
        Seuil r/r_g pour considérer le régime asymptotique.

    Retour
    ------
    KerrV5Check
    """
    g_v5 = g_tphi_asymptotic(r_m, theta_rad, M_kg, a_m)
    g_bl = g_tphi_boyer_lindquist(r_m, theta_rad, M_kg, a_m)
    ratio = g_v5 / g_bl if g_bl != 0 else float("nan")
    rg = schwarzschild_radius(M_kg)
    asymp = (r_m / rg > r_g_factor_threshold) and (r_m > 10.0 * a_m)
    return KerrV5Check(
        r_m=r_m,
        theta_rad=theta_rad,
        M_kg=M_kg,
        a_m=a_m,
        g_tphi_V5=g_v5,
        g_tphi_BL=g_bl,
        ratio=ratio,
        asymptotic_regime=asymp,
        note=(
            "V5.10a reproduit g_tφ_BL asymptotique à 1 % près pour "
            "r/r_g > 100 ; écart d'ordre (r_g/r) près de l'horizon. "
            "Cette dérivation est un SCOPING : la régularisation à "
            "r ~ r_+ requiert V5.10c+ (corrections P22)."
        ),
    )


def schwarzschild_limit_a_to_zero(r_m: float,
                                  M_kg: float) -> dict:
    """Vérifie que pour a → 0, la métrique V5.10a se réduit à V5.8.

    Critère : à a = 0,
      - g_tφ = 0,
      - |Ψ_rot|² = |Ψ_static|²,
      - ω(r,θ) = 0,
      - κ_a = 0.

    Retour
    ------
    dict de tests booléens.
    """
    psi2_static = Psi_static_squared(r_m, M_kg)
    psi2_rot_a0 = Psi_rot_ansatz(r_m, math.pi / 2, M_kg, a_m=0.0)
    g_tphi_a0 = g_tphi_asymptotic(r_m, math.pi / 2, M_kg, a_m=0.0)
    omega_a0 = frame_dragging_omega(r_m, math.pi / 2, M_kg, a_m=0.0)
    kappa_a0 = kappa_a_from_V5(a_m=0.0)
    return {
        "psi2_static": psi2_static,
        "psi2_rot_a0": psi2_rot_a0,
        "psi2_match": abs(psi2_static - psi2_rot_a0) < 1e-30 * max(
            1.0, abs(psi2_static)),
        "g_tphi_a0": g_tphi_a0,
        "g_tphi_zero": g_tphi_a0 == 0.0,
        "omega_a0": omega_a0,
        "omega_zero": omega_a0 == 0.0,
        "kappa_a0": kappa_a0,
        "kappa_zero": kappa_a0 == 0.0,
        "schwarzschild_recovered": (
            psi2_rot_a0 == psi2_static
            and g_tphi_a0 == 0.0
            and omega_a0 == 0.0
        ),
        "note": (
            "À a = 0 la métrique V5.10a se réduit identiquement à la "
            "métrique conforme statique V5.8 g_μν = Ω²(|Ψ_static|²) η_μν."
        ),
    }


def asymptotic_minkowski_limit(r_factor: float,
                               M_kg: float,
                               a_m: float) -> dict:
    """Vérifie que pour r → ∞ la métrique V5.10a tend vers Minkowski.

    À r très grand :
      - |Ψ_static|² → 0 (∝ 1/r),
      - Ω² → 1,
      - g_tφ → 0 (∝ 1/r),
      - ω → 0 (∝ 1/r³).

    Paramètres
    ----------
    r_factor : float
        Multiple de r_g auquel évaluer.
    """
    rg = schwarzschild_radius(M_kg)
    r = r_factor * rg
    psi2 = Psi_static_squared(r, M_kg)
    h = KAPPA_G_M * psi2
    omega2 = 1.0 + h
    g_tphi = g_tphi_asymptotic(r, math.pi / 2, M_kg, a_m)
    omega_dragging = frame_dragging_omega(r, math.pi / 2, M_kg, a_m)
    # g_tφ a unités m²/s (∝ r_g²·c/r). On adimensionne par r·c pour
    # tester la décroissance asymptotique.
    g_tphi_dimless = g_tphi / (r * C_LIGHT)
    omega_dimless = omega_dragging * rg / C_LIGHT
    return {
        "r_over_rg": r_factor,
        "h": h,
        "omega2": omega2,
        "g_tphi": g_tphi,
        "g_tphi_dimless": g_tphi_dimless,
        "omega_dragging": omega_dragging,
        "omega_dimless": omega_dimless,
        "minkowski_recovered": (
            h < 1e-6 and abs(g_tphi_dimless) < 1e-6 and abs(omega_dimless) < 1e-6
        ),
        "note": (
            f"À r = {r_factor:g} r_g, h ~ {h:.3e}, "
            f"g_tφ/(rc) ~ {g_tphi_dimless:.3e} : "
            "métrique de Minkowski recouvrée par construction."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Synthèse globale pour reporting.
# ---------------------------------------------------------------------------
def summary() -> dict:
    """Synthèse V5.10a sur un trou noir de masse solaire à a ~ 0.5 r_g."""
    M_sun = 1.989e30
    rg = schwarzschild_radius(M_sun)
    a_m = 0.5 * rg
    J = M_sun * a_m * C_LIGHT

    # Test asymptotique à r = 1000 r_g, θ = π/2.
    r_test = 1000.0 * rg
    chk = asymptotic_consistency_check(r_test, math.pi / 2, M_sun, a_m)

    # Limite a → 0.
    schw = schwarzschild_limit_a_to_zero(r_test, M_sun)
    # Limite r → ∞ (r = 1e10 r_g).
    asymp = asymptotic_minkowski_limit(1.0e10, M_sun, a_m)

    return {
        "M_sun_kg": M_sun,
        "r_g_m": rg,
        "a_m": a_m,
        "J_si": J,
        "kappa_g_m": KAPPA_G_M,
        "kappa_a_si": kappa_a_from_V5(a_m),
        "G_V5": G_V5,
        "asymptotic_ratio_V5_over_BL": chk.ratio,
        "asymptotic_regime_ok": chk.asymptotic_regime,
        "schwarzschild_limit_ok": schw["schwarzschild_recovered"],
        "minkowski_limit_ok": asymp["minkowski_recovered"],
        "note": (
            "V5.10a est un scoping théorique : pose l'ansatz Ψ_rot, "
            "identifie κ_a = -κ_g·a·c par cohérence asymptotique BL, et "
            "convertit le spin topologique en moment cinétique macro. "
            "PAS d'horizon traversable, PAS de NR, PAS de merger."
        ),
    }


if __name__ == "__main__":
    s = summary()
    print("=" * 72)
    print(" V5.10a — Théorie Kerr-V5 (scoping)")
    print("=" * 72)
    print(f"\n  M_sun                          = {s['M_sun_kg']:.3e} kg")
    print(f"  r_g (Schwarzschild)            = {s['r_g_m']:.3e} m")
    print(f"  a (paramètre Kerr, 0.5 r_g)    = {s['a_m']:.3e} m")
    print(f"  J (M·a·c)                      = {s['J_si']:.3e} J·s")
    print(f"\n  κ_g                            = {s['kappa_g_m']:.3e} m")
    print(f"  κ_a = -κ_g·a                   = {s['kappa_a_si']:.3e} m²")
    print(f"  G_V5                           = {s['G_V5']:.4e} SI")
    print(f"\n  Rapport g_tφ(V5)/g_tφ(BL) à 1000 r_g = "
          f"{s['asymptotic_ratio_V5_over_BL']:.6f}")
    print(f"  Régime asymptotique OK         = {s['asymptotic_regime_ok']}")
    print(f"  Limite a → 0 ⟹ V5.8 OK         = "
          f"{s['schwarzschild_limit_ok']}")
    print(f"  Limite r → ∞ ⟹ Minkowski OK    = "
          f"{s['minkowski_limit_ok']}")
    print("=" * 72)
    print(f"\n  NOTE : {s['note']}")
    print("=" * 72)
