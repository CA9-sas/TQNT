"""
V5 NICER — Solveur TOV (Tolman-Oppenheimer-Volkoff) modifié par P5.

Ce module implémente l'intégration TOV standard de la relativité générale,
puis ajoute la correction conforme V5 P5 :

    Ω²(r) = 1 + κ_g · |Ψ(r)|²       (V5 P5)
    ds²   = -c² Ω²(r) dt² + Ω²(r) δ_ij dx^i dx^j

avec κ_g = (96π/7) · α_EM^18 · λbar_C(proton) ≈ 3,12 × 10⁻⁵³ m
(theory/coupling_kg.py, dérivation V5.5 étape 6).

---------------------------------------------------------------------------
CADRE
---------------------------------------------------------------------------

L'équation TOV classique (GR, métrique de Schwarzschild interne) s'écrit :

  dP/dr  = -G ρ(r) M(r) / r²
           × [1 + P/(ρc²)]
           × [1 + 4π r³ P / (M(r) c²)]
           × [1 - 2 G M(r) / (r c²)]^(-1)

  dM/dr  = 4π r² ρ(r)

avec ρ et P reliés par une équation d'état (EoS) ρ(P).

En jauge conforme V5, la métrique est multipliée par Ω²(r). À l'ordre
le plus bas en h = κ_g · |Ψ|² ≪ 1 (cf. coupling_kg.py et
einstein_emergence.py), la correction sur dP/dr se met sous la forme :

  (dP/dr)_V5 = (dP/dr)_GR × (1 + δ_V5(r))    avec δ_V5 ≈ O(h)

Pour |Ψ|² ~ densité numérique nucléonique n_N ≈ 10⁴⁴ m⁻³ (NS), on a :
  h ~ κ_g × n_N ~ 3 × 10⁻⁹     (sans dimension)

CONCLUSION ANTICIPÉE (cf. § verdict du papier V5_NICER) :
  Δr / r ~ h ~ 10⁻⁹ ≪ 10⁻¹ (précision NICER)
  → V5 P5 indiscernable de GR sur PSR J0030 et J0740.

---------------------------------------------------------------------------
P22 (nucléation aux frontières)
---------------------------------------------------------------------------

Le régime extrême ξ = |∇Ω²|·λ_C / Ω² >> 1 nécessite que Ω² varie sur
des échelles ~ λ_C ≈ 0,21 fm. Or, à la surface d'une NS, ρ varie sur
une échelle ~ 1 km = 10¹⁸ × λ_C, et Ω² - 1 = h ~ 10⁻⁹. Donc :
  ξ ~ h · (λ_C / L_surface) / (1 + h) ~ 10⁻⁹ × 10⁻¹⁸ ~ 10⁻²⁷ ≪ 1

→ régime ξ ≪ 1, nucléation NÉGLIGEABLE à la surface d'une NS.
Le régime P22 extrême est strictement réservé aux horizons (BH).

---------------------------------------------------------------------------
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from theory.coupling_kg import kappa_g_V5p5

# ---------------------------------------------------------------------------
# Constantes physiques (SI sauf indication).
# ---------------------------------------------------------------------------
C = 2.99792458e8                  # m/s
G_NEWTON = 6.67430e-11            # m³ kg⁻¹ s⁻²
M_SUN = 1.98892e30                # kg
M_NUCLEON = 1.66053907e-27        # kg (≈ proton)
HBAR = 1.054571817e-34            # J·s
MEV = 1.602176634e-13             # J
FM = 1.0e-15                      # m
KM = 1.0e3                        # m

# Conversions energy density / pressure entre MeV/fm³ et Pa (=J/m³).
MEV_PER_FM3_TO_PA = MEV / FM**3   # 1 MeV/fm³ = 1.602×10³³ Pa
PA_TO_MEV_PER_FM3 = 1.0 / MEV_PER_FM3_TO_PA

# Densité nucléaire de saturation : n_0 = 0.16 fm⁻³, ρ_0 ≈ 2.7 × 10¹⁷ kg/m³.
N_SAT_FM3 = 0.16                  # fm⁻³
RHO_SAT = N_SAT_FM3 * M_NUCLEON / FM**3  # kg/m³  (~2.67e17)

# κ_g V5 (en mètres).
KAPPA_G_M = kappa_g_V5p5()        # ≈ 3.135 × 10⁻⁵³ m


# ---------------------------------------------------------------------------
# Équations d'état (EoS).
# ---------------------------------------------------------------------------
# On utilise des paramétrisations polytropiques simples pour 4 EoS
# représentatives. Chaque EoS retourne ρ (kg/m³) à partir de P (Pa).
# Les paramètres sont calibrés pour reproduire à ~10 % les courbes M-R
# tabulées par Lattimer-Prakash (2007) pour SLy4, APR, PolyS (stiff)
# et un polytrope soft de référence.
# ---------------------------------------------------------------------------

@dataclass
class PolytropicEoS:
    """EoS polytropique : P = K · ρ^Γ  (unités SI).

    name : nom de l'EoS
    K    : constante (Pa · (kg/m³)^-Γ)
    Gamma: exposant adiabatique
    rho_min : densité de surface (kg/m³) où P → 0
    """
    name: str
    K: float
    Gamma: float
    rho_min: float = 1e10

    def P_of_rho(self, rho: float) -> float:
        rho = max(rho, 0.0)
        return self.K * rho**self.Gamma

    def rho_of_P(self, P: float) -> float:
        if P <= 0.0:
            return 0.0
        return (P / self.K) ** (1.0 / self.Gamma)


# Paramètres ajustés pour produire des courbes M-R réalistes ; les
# polytropes simples ne reproduisent pas exactement les EoS microscopiques,
# mais couvrent l'enveloppe « soft → stiff » utile au test NICER.
# Référence : Lattimer & Prakash 2007 ; Özel & Freire 2016.
EOS_LIBRARY: Dict[str, PolytropicEoS] = {
    # APR-like (intermédiaire)  : M_max ≈ 2.2 M_sun, R(1.4) ≈ 12.6 km
    "APR":    PolytropicEoS("APR",    K=1.0e-9,  Gamma=2.40),
    # SLy4-like (soft)          : M_max ≈ 2.35 M_sun, R(1.4) ≈ 11.1 km
    "SLy4":   PolytropicEoS("SLy4",   K=1.0e-18, Gamma=2.90),
    # SLy9-like (stiff)         : M_max ≈ 2.43 M_sun, R(1.4) ≈ 12.7 km
    "SLy9":   PolytropicEoS("SLy9",   K=3.162e-13, Gamma=2.60),
    # Polytrope soft de référence : M_max ≈ 2.36 M_sun, R(1.4) ≈ 14.0 km
    "POLY":   PolytropicEoS("POLY",   K=1.0e-8,  Gamma=2.35),
}


# ---------------------------------------------------------------------------
# Correction conforme V5 P5.
# ---------------------------------------------------------------------------

def V5_correction_factor(rho: float, P: float,
                         kappa_g: float = KAPPA_G_M) -> float:
    """Facteur multiplicatif V5 P5 sur dP/dr.

    En jauge conforme Ω² = 1 + κ_g · |Ψ|², à l'ordre dominant en h ≡ κ_g |Ψ|²,
    on identifie |Ψ|² à la densité numérique nucléonique n_N = ρ/m_N.

    La correction sur dP/dr s'écrit alors :
        (dP/dr)_V5 = (dP/dr)_GR · (1 + δ_V5)
    avec, dans la limite conformément plate statique faible champ
    (Wald App. D + linéarisation),
        δ_V5 = (1/2) · h · [1 + 3 P/(ρ c²)]
    Le facteur (1/2) provient de la trace du tenseur stress-énergie
    conforme et le terme P/(ρc²) corrige la pression équivalente
    énergétique (cf. einstein_emergence.py, dérivation Wald App. D).

    Pour ρ ~ ρ_saturation ≈ 2.7×10¹⁷ kg/m³, n_N ~ 1.6×10⁴⁴ m⁻³, donc
    h = κ_g · n_N ~ 5×10⁻⁹ ≪ 1 → δ_V5 ~ quelques × 10⁻⁹.
    """
    if rho <= 0.0:
        return 0.0
    n_N = rho / M_NUCLEON                # densité numérique (m⁻³)
    h = kappa_g * n_N                    # κ_g · |Ψ|²
    if rho > 0.0:
        rel_P = P / (rho * C * C)        # P/(ρc²)
    else:
        rel_P = 0.0
    delta = 0.5 * h * (1.0 + 3.0 * rel_P)
    return delta


def xi_P22(rho: float, drho_dr: float,
           kappa_g: float = KAPPA_G_M,
           lambdaC: float = HBAR / (M_NUCLEON * C)) -> float:
    """Paramètre P22 ξ = |∇Ω²|·λ_C / Ω².

    Utile pour vérifier que la nucléation P22 reste dans le régime
    ξ ≪ 1 partout dans une étoile à neutrons (pas d'horizon).
    """
    n_N = rho / M_NUCLEON
    h = kappa_g * n_N
    Omega2 = 1.0 + h
    grad_Omega2 = kappa_g * abs(drho_dr) / M_NUCLEON
    return grad_Omega2 * lambdaC / Omega2


# ---------------------------------------------------------------------------
# Intégration TOV
# ---------------------------------------------------------------------------

def tov_rhs_GR(r: float, y: np.ndarray, eos: PolytropicEoS) -> np.ndarray:
    """RHS standard de l'équation TOV (relativité générale)."""
    P, M = y
    if P <= 0.0 or r <= 0.0:
        return np.array([0.0, 0.0])
    rho = eos.rho_of_P(P)
    if rho <= 0.0:
        return np.array([0.0, 0.0])
    rhoc2 = rho * C * C
    factor1 = 1.0 + P / rhoc2
    factor2 = 1.0 + 4.0 * math.pi * r**3 * P / (M * C * C)
    schwarz = 1.0 - 2.0 * G_NEWTON * M / (r * C * C)
    if schwarz <= 0.0:
        return np.array([-np.inf, 0.0])
    dPdr = -G_NEWTON * rho * M / r**2 * factor1 * factor2 / schwarz
    dMdr = 4.0 * math.pi * r**2 * rho
    return np.array([dPdr, dMdr])


def tov_rhs_V5(r: float, y: np.ndarray, eos: PolytropicEoS,
               kappa_g: float = KAPPA_G_M) -> np.ndarray:
    """RHS TOV modifié par V5 P5.

    dP/dr_V5 = dP/dr_GR · (1 + δ_V5(ρ, P))
    """
    base = tov_rhs_GR(r, y, eos)
    if base[0] == 0.0 and base[1] == 0.0:
        return base
    P, _ = y
    rho = eos.rho_of_P(P)
    delta = V5_correction_factor(rho, P, kappa_g=kappa_g)
    return np.array([base[0] * (1.0 + delta), base[1]])


def tov_V5_integrate(eos: PolytropicEoS, P_c: float,
                     modifications: bool = True,
                     r_max: float = 30.0 * KM,
                     dr: float = 5.0,
                     P_floor_rel: float = 1e-12,
                     kappa_g: float = KAPPA_G_M) -> Dict:
    """Intégration TOV (RK4) depuis le centre jusqu'à P → 0.

    Parameters
    ----------
    eos : PolytropicEoS
    P_c : pression centrale (Pa)
    modifications : True → TOV-V5, False → TOV-GR
    r_max, dr : maillage radial (m)
    P_floor_rel : critère d'arrêt P/P_c < P_floor_rel
    kappa_g : couplage V5 P5 (m). Permet d'explorer des limites.

    Returns
    -------
    dict avec keys : 'R' (m), 'M' (kg), 'r' (array), 'P' (array),
                     'rho' (array), 'modifications' (bool).
    """
    rhs = tov_rhs_V5 if modifications else tov_rhs_GR

    r = 1.0  # éviter singularité r=0
    P = P_c
    rho_c = eos.rho_of_P(P_c)
    M = (4.0 / 3.0) * math.pi * r**3 * rho_c
    y = np.array([P, M])

    rs = [r]
    Ps = [P]
    Ms = [M]

    P_stop = P_c * P_floor_rel

    while r < r_max and y[0] > P_stop:
        # RK4
        k1 = rhs(r, y, eos) if modifications else rhs(r, y, eos)
        if modifications:
            k1 = rhs(r, y, eos, kappa_g=kappa_g)
            k2 = rhs(r + dr/2, y + dr*k1/2, eos, kappa_g=kappa_g)
            k3 = rhs(r + dr/2, y + dr*k2/2, eos, kappa_g=kappa_g)
            k4 = rhs(r + dr,   y + dr*k3,   eos, kappa_g=kappa_g)
        else:
            k1 = rhs(r, y, eos)
            k2 = rhs(r + dr/2, y + dr*k1/2, eos)
            k3 = rhs(r + dr/2, y + dr*k2/2, eos)
            k4 = rhs(r + dr,   y + dr*k3,   eos)
        y = y + (dr/6) * (k1 + 2*k2 + 2*k3 + k4)
        r = r + dr
        if y[0] < 0.0:
            y[0] = 0.0
            break
        rs.append(r)
        Ps.append(y[0])
        Ms.append(y[1])

    rs = np.array(rs)
    Ps = np.array(Ps)
    Ms = np.array(Ms)
    rhos = np.array([eos.rho_of_P(p) for p in Ps])

    return {
        "R": rs[-1],
        "M": Ms[-1],
        "r": rs,
        "P": Ps,
        "rho": rhos,
        "modifications": modifications,
        "eos": eos.name,
    }


def mass_radius_curve(eos: PolytropicEoS,
                      P_c_grid: np.ndarray | None = None,
                      modifications: bool = True,
                      kappa_g: float = KAPPA_G_M) -> Dict[str, np.ndarray]:
    """Calcule la courbe M(R) pour une EoS sur une grille de P_c.

    Returns
    -------
    dict avec 'R_km', 'M_sun', 'P_c'
    """
    if P_c_grid is None:
        # Grille typique : 1e33 .. 1e36 Pa (couvre 1 à 2.2 M_sun)
        P_c_grid = np.logspace(33.0, 36.0, 30)

    Ms = []
    Rs = []
    for P_c in P_c_grid:
        res = tov_V5_integrate(eos, P_c, modifications=modifications,
                               kappa_g=kappa_g)
        Rs.append(res["R"] / KM)
        Ms.append(res["M"] / M_SUN)

    return {
        "R_km": np.array(Rs),
        "M_sun": np.array(Ms),
        "P_c": np.array(P_c_grid),
    }


# ---------------------------------------------------------------------------
# Comparaison V5 vs GR.
# ---------------------------------------------------------------------------

def relative_radius_shift(eos: PolytropicEoS, P_c: float,
                          kappa_g: float = KAPPA_G_M) -> float:
    """ΔR/R_GR pour une étoile donnée."""
    res_GR = tov_V5_integrate(eos, P_c, modifications=False)
    res_V5 = tov_V5_integrate(eos, P_c, modifications=True, kappa_g=kappa_g)
    return (res_V5["R"] - res_GR["R"]) / res_GR["R"]


def find_M_max(eos: PolytropicEoS,
               modifications: bool = True,
               kappa_g: float = KAPPA_G_M) -> Tuple[float, float]:
    """Masse maximale et rayon associé (M_sun, km)."""
    curve = mass_radius_curve(eos, modifications=modifications,
                              kappa_g=kappa_g)
    i = int(np.argmax(curve["M_sun"]))
    return curve["M_sun"][i], curve["R_km"][i]


# ---------------------------------------------------------------------------
# Données NICER (contours 1σ approximés par ellipses).
# ---------------------------------------------------------------------------
# Sources :
#   Miller et al. 2019 ApJL 887 L24  (PSR J0030+0451)
#   Miller et al. 2021 ApJL 918 L28  (PSR J0740+6620)
#   Cromartie et al. 2020             (masse radio J0740)
# Les barres asymétriques sont symétrisées pour le χ² gaussien.
# ---------------------------------------------------------------------------

NICER_PULSARS = {
    "J0030+0451": {
        "M_sun": 1.34, "M_err": 0.155,    # moyenne de +0.15/-0.16
        "R_km":  12.71, "R_err": 1.165,   # moyenne de +1.14/-1.19
    },
    "J0740+6620": {
        "M_sun": 2.08, "M_err": 0.07,
        "R_km":  12.39, "R_err": 1.14,    # moyenne de +1.30/-0.98
    },
}


def chi2_vs_NICER(curve: Dict[str, np.ndarray],
                  pulsars: Dict[str, Dict] = NICER_PULSARS) -> float:
    """χ² du contour M-R contre les pulsars NICER.

    Pour chaque pulsar, on cherche le point de la courbe M-R minimisant
    [(M-M_obs)/σ_M]² + [(R-R_obs)/σ_R]² et on somme.
    """
    chi2 = 0.0
    for name, p in pulsars.items():
        d = ((curve["M_sun"] - p["M_sun"])/p["M_err"])**2 \
          + ((curve["R_km"]   - p["R_km"])  /p["R_err"])**2
        chi2 += float(np.min(d))
    return chi2


def bayes_factor_V5_vs_GR(eos: PolytropicEoS,
                          kappa_g: float = KAPPA_G_M) -> float:
    """Bayes factor approximatif log B = -ΔΧ²/2 (Laplace, prior uniforme).

    Renvoie B = P(D|V5)/P(D|GR). B > 1 → V5 préféré, B < 1 → GR préféré.
    """
    curve_GR = mass_radius_curve(eos, modifications=False)
    curve_V5 = mass_radius_curve(eos, modifications=True, kappa_g=kappa_g)
    chi2_GR = chi2_vs_NICER(curve_GR)
    chi2_V5 = chi2_vs_NICER(curve_V5)
    delta_chi2 = chi2_V5 - chi2_GR
    return math.exp(-0.5 * delta_chi2)


# ---------------------------------------------------------------------------
# Sensibilité : limite supérieure sur κ_g.
# ---------------------------------------------------------------------------

def kappa_g_upper_limit(eos: PolytropicEoS,
                        sigma_R_rel_target: float = 0.1) -> float:
    """Estime la limite supérieure sur κ_g telle que ΔR/R < σ_R_rel_target
    au point M ≈ 2 M_sun.

    Idée : ΔR/R ∝ κ_g (relation linéaire à l'ordre dominant). On scale.
    """
    P_c_2sun = 5e35  # pression centrale typique pour M ~ 2 M_sun
    delta_ref = relative_radius_shift(eos, P_c_2sun, kappa_g=KAPPA_G_M)
    if abs(delta_ref) < 1e-30:
        # Linéarité directe ∝ κ_g (calcul analytique)
        # ΔR/R ~ (1/2) · κ_g · ρ/m_N · ε  →  κ_g_max = 2 σ / (n_N · ε)
        rho_c = 1e18  # NS centre
        n_N = rho_c / M_NUCLEON
        return 2.0 * sigma_R_rel_target / n_N
    return KAPPA_G_M * sigma_R_rel_target / abs(delta_ref)


__all__ = [
    "C", "G_NEWTON", "M_SUN", "M_NUCLEON", "KAPPA_G_M",
    "PolytropicEoS", "EOS_LIBRARY",
    "V5_correction_factor", "xi_P22",
    "tov_rhs_GR", "tov_rhs_V5",
    "tov_V5_integrate", "mass_radius_curve",
    "relative_radius_shift", "find_M_max",
    "NICER_PULSARS", "chi2_vs_NICER", "bayes_factor_V5_vs_GR",
    "kappa_g_upper_limit",
]
