"""
V5.20 -- Physique interne d'une perle de matière noire (Class A) et taux
d'enchevêtrement topologique spontané.

Cadre
-----
V5.19 Class A (commit 86cc097 + 5efc4e4 v2 + b0528ce v3) prédit des
agrégats compacts de matière noire formés par empilement de nœuds isolés
(c > 18, n = 1) sous pression de Fermi topologique. La masse de
Chandrasekhar topologique est

    M_Ch_topo(m_chi) ~ M_Pl^3 / m_chi^2

avec m_chi la masse d'un nœud DM (V5 §13.6, candidats catalogue).

Pour la sous-population des 6 STRICT amphichiraux (V5 §13.7 ; CLAUDE.md),
les masses des nœuds individuels valent 1,55 ; 4,70 ; 16,01 ; 17,24 ;
26,48 ; 51,78 GeV. La masse typique d'une perle Class A est environ
2 x 10^-3 M_sun (V5.19 catalogue), et le nœud le plus léger
(6_3 + 6_3 = 1,55 GeV) fournit la limite supérieure M_max = 6,79 x 10^-1
M_sun (Chandrasekhar topologique).

Ce module V5.20 répond à DEUX questions complémentaires :

Partie A -- Structure interne :
  * Nombre de nœuds N = M_pearl / m_knot
  * Équation d'état (EoS) Fermi topologique relativiste/non-relativiste
  * Profil TOV (réutilise theory/tov_V5.py) adapté à l'EoS pearl
  * Rayon de la perle r_pearl, densité centrale rho_c
  * Comparaison Chandrasekhar pearl vs naine blanche (m_chi vs m_e)

Partie B -- Enchevêtrement spontané :
  * Cross-section de reconnexion topologique sigma ~ sigma_QED * alpha^{Delta c}
  * Taux par paire Gamma_pair = sigma * v_rel * n_local
  * Taux total Gamma_pearl ~ N^2 / V * sigma * v / 2
  * Événements sur l'âge de l'Univers N_events = Gamma * t_Hubble
  * Verdict : enchevêtrement spontané possible / exclu

Le module est volontairement minimaliste -- toutes les constantes sont
explicites et le code est entièrement dérivable des formules indiquées
dans les docstrings.

Références
----------
* V5 §13 (matière noire, 117 candidats)
* V5 §13.6.4 (suppression Vassiliev)
* V5 §17 (cinétique de condensation Big Bang)
* dm_pearls_paper.tex v3 (commit b0528ce)
* V5p19_compact_objects_catalog (commit 22f606a)
* CLAUDE.md (formules canoniques V5)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constantes physiques (SI sauf indication)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34       # J s
C_LIGHT = 2.99792458e8       # m/s
G_NEWTON = 6.67430e-11       # m^3 kg^-1 s^-2
KB = 1.380649e-23            # J/K
EV_J = 1.602176634e-19       # J/eV
MEV_KG = 1.78266192e-30      # kg per MeV/c^2
GEV_KG = 1.78266192e-27      # kg per GeV/c^2
M_SUN_KG = 1.98892e30        # kg
M_E_KG = 9.1093837015e-31    # kg
M_E_MEV = 0.51099895         # MeV/c^2
M_P_GEV = 0.93827208816      # GeV/c^2

# Planck mass
MPL_KG = math.sqrt(HBAR * C_LIGHT / G_NEWTON)   # ~ 2.176e-8 kg

# Age of universe (Planck 2018)
T_HUBBLE_S = 4.35e17         # ~ 13.8 Gyr

# Volume du proche univers observable (Hubble volume)
R_HUBBLE_M = C_LIGHT / (67.4e3 / 3.0857e22)
V_HUBBLE_M3 = (4.0/3.0) * math.pi * R_HUBBLE_M ** 3

# Classical electron radius (sigma_QED scale)
R_E_M = 2.8179403262e-15     # m
SIGMA_QED = math.pi * R_E_M ** 2   # ~ 2.5e-29 m^2

# Fine structure constant (V5: 1/(k+2) avec k=135)
ALPHA_EM = 1.0 / 137.036

# V5 DM strict amphichiral masses (CLAUDE.md, GeV)
DM_STRICT_AMPHI_GEV = (1.55, 4.70, 16.01, 17.24, 26.48, 51.78)

# Number of DM knots per pearl (typical V5.19)
M_PEARL_TYPICAL_MSUN = 2.0e-3


# ---------------------------------------------------------------------------
# 1. Comptage des nœuds dans une perle
# ---------------------------------------------------------------------------

def n_knots(M_pearl_Msun: float, m_knot_GeV: float) -> float:
    """Nombre total de nœuds dans une perle de masse M_pearl.

    Formule simple :
        N = M_pearl / m_knot
    en convertissant en unités SI.

    Pour M_pearl = 2e-3 M_sun et m_knot = 1,55 GeV :
        N ~ 1.4 x 10^60.
    """
    M_pearl_kg = M_pearl_Msun * M_SUN_KG
    m_knot_kg = m_knot_GeV * GEV_KG
    return M_pearl_kg / m_knot_kg


def n_knots_table() -> Dict[float, float]:
    """Renvoie N(m_knot) pour les 6 STRICT amphichiraux.

    À M_pearl = 2e-3 M_sun fixée.
    """
    return {
        m: n_knots(M_PEARL_TYPICAL_MSUN, m) for m in DM_STRICT_AMPHI_GEV
    }


# ---------------------------------------------------------------------------
# 2. Équation d'état (EoS) topologique
# ---------------------------------------------------------------------------
#
# On modélise les nœuds DM comme un gaz de Fermi dégénéré. La justification
# physique du "Fermi topologique" est que les nœuds isolés (n=1) ont un
# cœur dur topologique (exclusion de leurs cœurs c > 18 -> non-coalescence),
# ce qui mimique la statistique de Fermi-Dirac : un seul nœud par cellule
# élémentaire de volume ~ lambda_chi^3 où lambda_chi = hbar/(m_chi c).
#
# On dispose de deux régimes :
#  - Non-relativiste : p_F << m_chi c -> P ~ K_NR * n^{5/3}
#  - Ultra-relativiste : p_F >> m_chi c -> P ~ K_UR * n^{4/3}
#
# Le passage de l'un à l'autre se produit à n_crit ~ (m_chi c / hbar)^3
# (la densité où p_F ~ m_chi c).
# ---------------------------------------------------------------------------

def fermi_momentum(n: float) -> float:
    """Impulsion de Fermi pour densité numérique n (m^-3).

    p_F = hbar * (3 pi^2 n)^{1/3}
    """
    return HBAR * (3.0 * math.pi ** 2 * max(n, 0.0)) ** (1.0/3.0)


def n_crit_relativistic(m_knot_GeV: float) -> float:
    """Densité critique à laquelle p_F ~ m_knot c.

    n_crit = (m c / hbar)^3 / (3 pi^2)
    """
    m_kg = m_knot_GeV * GEV_KG
    return (m_kg * C_LIGHT / HBAR) ** 3 / (3.0 * math.pi ** 2)


def pressure_NR(n: float, m_knot_GeV: float) -> float:
    """Pression non-relativiste d'un gaz de Fermi de nœuds (Pa).

    P_NR = (hbar^2 / (5 m)) * (3 pi^2)^{2/3} * n^{5/3}
    """
    m_kg = m_knot_GeV * GEV_KG
    if n <= 0:
        return 0.0
    return (HBAR ** 2 / (5.0 * m_kg)) * (3.0 * math.pi ** 2) ** (2.0/3.0) * n ** (5.0/3.0)


def pressure_UR(n: float) -> float:
    """Pression ultra-relativiste d'un gaz de Fermi (Pa).

    P_UR = (hbar c / 4) * (3 pi^2)^{1/3} * n^{4/3}

    Coefficient standard du gaz de Fermi UR (Landau-Lifshitz, vol.5 §61).
    """
    if n <= 0:
        return 0.0
    return (HBAR * C_LIGHT / 4.0) * (3.0 * math.pi ** 2) ** (1.0/3.0) * n ** (4.0/3.0)


def eos_topological(n: float, m_knot_GeV: float) -> Dict[str, float]:
    """EoS topologique : pression et densité de masse-énergie.

    Renvoie {"P", "rho", "n_crit", "regime"}.

    rho est exprimée en kg/m^3 et inclut la contribution d'énergie de
    Fermi (pas seulement n * m). Pour les besoins de TOV on retient :
        rho c^2 = n * m c^2 + epsilon_F        (NR)
        rho c^2 = (3/4) p_F c * n              (UR, fluide UR à pression P,
                                                rho c^2 = 3 P)
    """
    m_kg = m_knot_GeV * GEV_KG
    n_c = n_crit_relativistic(m_knot_GeV)
    if n <= 0:
        return {"P": 0.0, "rho": 0.0, "n_crit": n_c, "regime": "vacuum"}

    p_F = fermi_momentum(n)
    p_F_norm = p_F / (m_kg * C_LIGHT)   # = p_F / (m c)

    if p_F_norm < 0.5:
        # Non-relativiste
        P = pressure_NR(n, m_knot_GeV)
        # rho c^2 = n m c^2 (1 + 3/5 p_F^2/(m^2 c^2) + ...)
        eps_F_per_n = (3.0/5.0) * (p_F ** 2) / (2.0 * m_kg)   # énergie cinétique moyenne
        rho_c2 = n * m_kg * C_LIGHT ** 2 + n * eps_F_per_n
        regime = "NR"
    elif p_F_norm > 2.0:
        # Ultra-relativiste
        P = pressure_UR(n)
        # rho c^2 = 3 P (gaz UR)
        rho_c2 = 3.0 * P + n * m_kg * C_LIGHT ** 2   # garde la masse au repos
        regime = "UR"
    else:
        # Transition : on interpole linéairement entre les deux pour la
        # stabilité numérique (pas d'analyticité requise ici)
        P_NR = pressure_NR(n, m_knot_GeV)
        P_UR = pressure_UR(n)
        w = (p_F_norm - 0.5) / 1.5
        P = (1 - w) * P_NR + w * P_UR
        rho_c2 = n * m_kg * C_LIGHT ** 2 + 3.0 * P
        regime = "transition"

    rho = rho_c2 / C_LIGHT ** 2
    return {"P": P, "rho": rho, "n_crit": n_c, "regime": regime}


def n_of_rho_NR(rho: float, m_knot_GeV: float) -> float:
    """Inversion : densité numérique pour densité de masse rho (NR).

    rho ~ n m -> n = rho / m
    """
    m_kg = m_knot_GeV * GEV_KG
    return max(rho / m_kg, 0.0)


def P_of_rho_NR_polytrope(rho: float, m_knot_GeV: float) -> float:
    """EoS polytropique non-relativiste : P = K_NR rho^{5/3}.

    Convertit la forme P(n) en P(rho) en posant n = rho/m.
    """
    n = n_of_rho_NR(rho, m_knot_GeV)
    return pressure_NR(n, m_knot_GeV)


# ---------------------------------------------------------------------------
# 3. Profil TOV adapté à l'EoS topologique
# ---------------------------------------------------------------------------
#
# On utilise un intégrateur TOV simple en RK4, qui n'a PAS besoin
# d'importer tov_V5.py car l'EoS ici est polytropique pure (n_p = 3 UR,
# n_p = 3/2 NR). La correction conforme V5 P5 (kappa_g) est ignorée car
# elle est negligible (~10^-9, cf. tov_V5.py docstring) -- pour une perle,
# rho_c << rho_NS donc l'effet est encore plus petit.
# ---------------------------------------------------------------------------

@dataclass
class PearlPolytropicEoS:
    """EoS polytropique non-relativiste pour une perle.

    P(rho) = K * rho^Gamma   avec Gamma = 5/3 (Fermi NR).
    """
    m_knot_GeV: float
    Gamma: float = 5.0/3.0

    @property
    def K(self) -> float:
        m_kg = self.m_knot_GeV * GEV_KG
        # P = (hbar^2 / (5 m)) (3 pi^2)^(2/3) n^(5/3) avec n = rho/m
        # -> P = (hbar^2 / (5 m^(8/3))) (3 pi^2)^(2/3) rho^(5/3)
        return (HBAR ** 2 / (5.0 * m_kg ** (8.0/3.0))) \
               * (3.0 * math.pi ** 2) ** (2.0/3.0)

    def P_of_rho(self, rho: float) -> float:
        if rho <= 0.0:
            return 0.0
        return self.K * rho ** self.Gamma

    def rho_of_P(self, P: float) -> float:
        if P <= 0.0:
            return 0.0
        return (P / self.K) ** (1.0 / self.Gamma)


def tov_rhs_pearl(r: float, y: np.ndarray, eos: PearlPolytropicEoS) -> np.ndarray:
    """RHS TOV standard pour une perle (NR, négligeons les corrections GR
    aux ordres dominants car la perle est très loin du régime relativiste
    de Schwarzschild)."""
    P, M = y
    if P <= 0.0 or r <= 0.0 or M <= 0.0:
        return np.array([0.0, 0.0])
    rho = eos.rho_of_P(P)
    if rho <= 0.0:
        return np.array([0.0, 0.0])
    # Forme Newtonienne (compactness M G / (r c^2) << 1 pour une perle)
    dPdr = -G_NEWTON * rho * M / r ** 2
    dMdr = 4.0 * math.pi * r ** 2 * rho
    return np.array([dPdr, dMdr])


def tov_pearl(M_pearl_Msun: float, m_knot_GeV: float,
              rho_c_guess: Optional[float] = None,
              max_iter: int = 25,
              tol: float = 1.0e-3) -> Dict:
    """Intègre l'équation TOV pour une perle de masse cible M_pearl.

    Algorithme :
      1) Estime rho_c initial via Chandrasekhar Newtonien (polytrope n=3/2)
      2) Intègre dP/dr, dM/dr depuis r=eps jusqu'à P->0
      3) Ajuste rho_c par bissection pour atteindre M(R) ~ M_pearl

    Retourne dict :
      'R'         : rayon (m)
      'M'         : masse totale (kg)
      'rho_c'     : densité centrale (kg/m^3)
      'r','P','rho','n' : profils
      'eos'       : nom EoS
    """
    eos = PearlPolytropicEoS(m_knot_GeV=m_knot_GeV)

    if rho_c_guess is None:
        # Échelle de densité Newtonienne :
        # M = 4pi R^3 rho / 3 et R ~ (K/(G rho^{2/3}))^{1/2}
        # On démarre à l'échelle Chandrasekhar topologique
        m_kg = m_knot_GeV * GEV_KG
        rho_c_guess = (m_kg / HBAR ** 3) ** 3 * 1e-90 + 1e6  # placeholder
        # Plus simple : utiliser densité critique relativiste / 100
        rho_c_guess = 0.01 * m_kg * n_crit_relativistic(m_knot_GeV)

    def integrate(rho_c: float) -> Dict:
        P_c = eos.P_of_rho(rho_c)
        # Auto-tune dr : on prend dr = 1e-4 * R_estimate
        # R_estimate Newtonien polytrope n=3/2 : R ~ ((5K/(4 pi G)) * rho_c^(-1/3))^(1/2)
        K = eos.K
        R_est = math.sqrt(5.0 * K * rho_c ** (-1.0/3.0) / (4.0 * math.pi * G_NEWTON))
        dr = max(R_est * 5.0e-3, 1.0)
        r_max = R_est * 5.0
        r = max(dr, 1.0e-3)
        M = (4.0/3.0) * math.pi * r ** 3 * rho_c
        y = np.array([P_c, M])
        rs = [r]
        Ps = [P_c]
        Ms = [M]
        P_stop = P_c * 1.0e-10
        n_steps = 0
        max_steps = 200000
        while r < r_max and y[0] > P_stop and n_steps < max_steps:
            k1 = tov_rhs_pearl(r, y, eos)
            k2 = tov_rhs_pearl(r + dr/2, y + dr*k1/2, eos)
            k3 = tov_rhs_pearl(r + dr/2, y + dr*k2/2, eos)
            k4 = tov_rhs_pearl(r + dr, y + dr*k3, eos)
            y = y + (dr/6) * (k1 + 2*k2 + 2*k3 + k4)
            r = r + dr
            n_steps += 1
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
        return {"R": rs[-1], "M": Ms[-1], "r": rs, "P": Ps,
                "rho": rhos, "rho_c": rho_c}

    # Bissection sur rho_c pour viser M_target
    M_target_kg = M_pearl_Msun * M_SUN_KG
    rho_c_lo = rho_c_guess * 1.0e-4
    rho_c_hi = rho_c_guess * 1.0e4
    res = None
    for _ in range(max_iter):
        rho_c_mid = math.sqrt(rho_c_lo * rho_c_hi)
        res = integrate(rho_c_mid)
        M_curr = res["M"]
        if M_curr <= 0:
            rho_c_lo = rho_c_mid
            continue
        rel = (M_curr - M_target_kg) / M_target_kg
        if abs(rel) < tol:
            break
        if M_curr < M_target_kg:
            rho_c_lo = rho_c_mid
        else:
            rho_c_hi = rho_c_mid

    n_profile = np.array([
        n_of_rho_NR(r_, m_knot_GeV) for r_ in res["rho"]
    ])
    res["n"] = n_profile
    res["eos"] = "Fermi NR topological"
    res["m_knot_GeV"] = m_knot_GeV
    return res


# ---------------------------------------------------------------------------
# 4. Relation Masse-Rayon de la perle
# ---------------------------------------------------------------------------

def mass_radius_pearl(m_knot_GeV: float,
                      M_grid_Msun: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Renvoie la courbe M-R pour une perle dans le régime NR.

    Pour le polytrope NR (Gamma = 5/3, indice n=3/2), la relation
    Newtonienne est :
        R = R_0 * (M/M_0)^(-1/3)
    où R_0, M_0 dépendent de K et G.

    On échantillonne via TOV pour vérifier.
    """
    if M_grid_Msun is None:
        M_grid_Msun = np.logspace(-6, 0, 12)
    Rs = []
    Ms = []
    rho_cs = []
    for Mt in M_grid_Msun:
        res = tov_pearl(Mt, m_knot_GeV)
        Rs.append(res["R"])
        Ms.append(res["M"] / M_SUN_KG)
        rho_cs.append(res["rho_c"])
    return {"M_Msun": np.array(Ms),
            "R_m": np.array(Rs),
            "rho_c": np.array(rho_cs)}


# ---------------------------------------------------------------------------
# 5. Chandrasekhar topologique vs naine blanche
# ---------------------------------------------------------------------------

def chandrasekhar_topological_pearl(m_knot_GeV: float) -> float:
    """Masse de Chandrasekhar topologique : M_Ch ~ M_Pl^3 / m_knot^2 (kg).

    Cette formule est la forme dimensionnelle générique du gaz de Fermi
    relativiste dégénéré (cf. Landau-Lifshitz vol.5 §61) :

        M_Ch = (3/2)^{1/2} (4 pi)^{-1/2} (hbar c / G)^{3/2} / m^2

    Le coefficient numérique exact (~3.1) est inclus pour la naine blanche
    standard ; on adopte ici la forme dimensionnelle dépouillée pour la
    perle (cohérent avec dm_pearls_paper).
    """
    m_kg = m_knot_GeV * GEV_KG
    return MPL_KG ** 3 / m_kg ** 2


def chandrasekhar_white_dwarf() -> float:
    """Masse de Chandrasekhar de la naine blanche (kg).

    M_Ch_WD ~ 5.76 * mu_e^-2 * M_sun (Chandrasekhar 1931)
    Pour mu_e = 2 (matière baryonique 12C/16O) : 1.44 M_sun.
    """
    return 1.4 * M_SUN_KG


def chandrasekhar_ratio(m_knot_GeV: float) -> float:
    """Rapport M_Ch_pearl / M_Ch_WD pour un nœud DM de masse m_knot."""
    return chandrasekhar_topological_pearl(m_knot_GeV) / chandrasekhar_white_dwarf()


# ---------------------------------------------------------------------------
# 6. Enchevêtrement spontané : cross-section + taux
# ---------------------------------------------------------------------------
#
# La suppression Vassiliev (V5 §13.6.4) donne :
#     sigma(c+1) ~ sigma(c) * alpha_EM
# soit pour passer de deux nœuds c chacun à un link c_link :
#     sigma_entangle = sigma_QED * alpha_EM^(Delta c)
#   où Delta c = c_link - 2*c_node (crossings additionnels créés par
#   la reconnexion topologique).
#
# Pour deux nœuds DM avec c=10 chacun (n=1 isolé, c~10 typique pour les
# 6 STRICT amphichiraux) -> link n=2 avec c_link ~ 22 -> Delta c = 2,
# MAIS le nœud isolé "intrinsèque" c > 18 (V5 §13.6.4) implique
# c_link ~ 30-40 -> Delta c ~ 10-20.
#
# C'est extrêmement supprimé : alpha^10 ~ 1e-22, alpha^20 ~ 1e-44.

def sigma_entanglement(delta_c: int = 10) -> float:
    """Cross-section de reconnexion topologique (m^2).

    sigma = sigma_QED * alpha_EM^{delta_c}

    Pour delta_c = 10 : sigma ~ 2.5e-29 * (1/137)^10 ~ 1.2e-50 m^2
                              = 1.2e-46 cm^2

    Pour delta_c = 20 : sigma ~ 5e-72 m^2
                              = 5e-68 cm^2

    Pour delta_c =  5 : sigma ~ 7e-40 m^2
                              = 7e-36 cm^2
    """
    return SIGMA_QED * ALPHA_EM ** delta_c


def fermi_velocity(n: float, m_knot_GeV: float) -> float:
    """Vitesse de Fermi : v_F = p_F / m (NR) ou ~ c (UR).

    Renvoie min(v_F_NR, c) pour rester causale.
    """
    m_kg = m_knot_GeV * GEV_KG
    p_F = fermi_momentum(n)
    v_F = p_F / m_kg
    return min(v_F, C_LIGHT)


def gamma_pair_local(n: float, m_knot_GeV: float, delta_c: int = 10) -> float:
    """Taux de reconnexion par paire et par seconde (Hz).

    Gamma_pair = sigma * v_rel * n

    où v_rel = sqrt(2) v_F est la vitesse relative typique dans un gaz
    de Fermi NR.
    """
    sigma = sigma_entanglement(delta_c)
    v_F = fermi_velocity(n, m_knot_GeV)
    v_rel = math.sqrt(2.0) * v_F
    return sigma * v_rel * n


_TOV_CACHE: dict = {}


def _tov_cached(M_pearl_Msun: float, m_knot_GeV: float) -> Dict:
    key = (round(M_pearl_Msun, 12), round(m_knot_GeV, 6))
    if key not in _TOV_CACHE:
        _TOV_CACHE[key] = tov_pearl(M_pearl_Msun, m_knot_GeV)
    return _TOV_CACHE[key]


def entanglement_rate(M_pearl_Msun: float, m_knot_GeV: float,
                      delta_c: int = 10) -> Dict[str, float]:
    """Taux total d'enchevêtrement intégré sur le volume de la perle.

    Gamma_total = ∫ (1/2) n^2 sigma v_rel dV ~ (1/2) <n> N_knots sigma v_F

    Pour une perle uniformément dense de volume V_pearl et N nœuds :
        Gamma_total = (N(N-1)/2) * sigma * v_F / V_pearl
                    ≈ (N^2 / 2) * sigma * v_F / V_pearl    (N grand)

    Renvoie un dict :
      'N_knots'             : nombre de nœuds dans la perle
      'r_pearl_m'           : rayon (m)
      'rho_central'         : densité centrale (kg/m^3)
      'n_central'           : densité numérique centrale (m^-3)
      'sigma_cm2'           : sigma_entanglement (cm^2)
      'v_F_central'         : vitesse de Fermi (m/s)
      'Gamma_pair_central'  : taux par paire au centre (Hz)
      'Gamma_total_pearl'   : taux total intégré (Hz)
    """
    # Profil TOV (caché)
    res = _tov_cached(M_pearl_Msun, m_knot_GeV)
    r = res["r"]
    n_arr = res["n"]
    V_pearl = (4.0/3.0) * math.pi * res["R"] ** 3
    sigma = sigma_entanglement(delta_c)
    v_F_c = fermi_velocity(n_arr[0], m_knot_GeV)
    v_rel = math.sqrt(2.0) * v_F_c

    # Intégrale ∫ n^2 r^2 dr (volume sphérique)
    # Gamma_total = 2 pi sigma v_rel * ∫ n^2 r^2 dr
    integrand = n_arr ** 2 * r ** 2
    # Use trapezoid (np.trapz is deprecated in numpy 2.x)
    try:
        integral = np.trapezoid(integrand, r)
    except AttributeError:
        integral = np.trapz(integrand, r)
    Gamma_total = 2.0 * math.pi * sigma * v_rel * integral

    Gamma_pair_c = sigma * v_rel * n_arr[0]

    N = n_knots(M_pearl_Msun, m_knot_GeV)
    return {
        "N_knots": N,
        "r_pearl_m": float(res["R"]),
        "rho_central": float(res["rho_c"]),
        "n_central": float(n_arr[0]),
        "sigma_cm2": sigma * 1e4,
        "v_F_central": v_F_c,
        "Gamma_pair_central_Hz": Gamma_pair_c,
        "Gamma_total_pearl_Hz": Gamma_total,
        "V_pearl_m3": V_pearl,
        "delta_c": delta_c,
    }


def n_events_over_universe_age(M_pearl_Msun: float, m_knot_GeV: float,
                               delta_c: int = 10) -> Dict[str, float]:
    """Nombre d'événements d'enchevêtrement spontané dans 1 perle sur t_Hubble.

    N_events = Gamma_total_pearl * t_Hubble

    Si N_events < 1 : aucune perle n'a probablement enchevêtré
    spontanément depuis le Big Bang.

    On renvoie aussi N_events_per_Hubble_volume = N_events * N_pearls
    avec N_pearls = (Omega_A * rho_crit * V_Hubble) / M_pearl.
    """
    rate = entanglement_rate(M_pearl_Msun, m_knot_GeV, delta_c)
    N_events_per_pearl = rate["Gamma_total_pearl_Hz"] * T_HUBBLE_S

    # Abondance des perles Class A : 5% de Omega_DM (V5.19 catalogue)
    # Omega_DM ~ 0.265
    OMEGA_A = 0.05 * 0.265
    H0_SI = 67.4e3 / 3.0857e22
    RHO_CRIT = 3 * H0_SI ** 2 / (8 * math.pi * G_NEWTON)
    M_DM_per_volume = OMEGA_A * RHO_CRIT  # kg/m^3 comoving
    M_pearl_kg = M_pearl_Msun * M_SUN_KG
    n_pearls_per_m3 = M_DM_per_volume / M_pearl_kg
    N_pearls_in_Hubble = n_pearls_per_m3 * V_HUBBLE_M3

    return {
        "N_events_per_pearl": N_events_per_pearl,
        "N_pearls_in_Hubble_volume": N_pearls_in_Hubble,
        "N_events_total_universe": N_events_per_pearl * N_pearls_in_Hubble,
        "Gamma_total_pearl_Hz": rate["Gamma_total_pearl_Hz"],
        "t_Hubble_s": T_HUBBLE_S,
        "sigma_cm2": rate["sigma_cm2"],
        "delta_c": delta_c,
        "N_knots_per_pearl": rate["N_knots"],
    }


# ---------------------------------------------------------------------------
# 7. Verdict V5.20
# ---------------------------------------------------------------------------

def verdict_v5p20(M_pearl_Msun: float = M_PEARL_TYPICAL_MSUN,
                  m_knot_GeV: float = 1.55,
                  delta_c: int = 10) -> Dict:
    """Synthèse globale : enchevêtrement possible ou exclu ?"""
    res = n_events_over_universe_age(M_pearl_Msun, m_knot_GeV, delta_c)
    N_universe = res["N_events_total_universe"]
    N_per_pearl = res["N_events_per_pearl"]

    if N_universe < 1.0e-3:
        status = "STRICTLY EXCLUDED"
        comment = (
            f"Suppression Vassiliev domine : N_events_total < 0.001 sur "
            f"l'univers observable. Aucune perle DM n'a jamais entrelace "
            f"spontanement (alpha^{delta_c} ~ {ALPHA_EM**delta_c:.2e})."
        )
    elif N_universe < 1.0:
        status = "MARGINAL"
        comment = (
            "Au plus quelques perles ont entrelacé : événement "
            "essentiellement non observable."
        )
    elif N_universe < 1.0e10:
        status = "RARE BUT POSSIBLE"
        comment = (
            f"~{N_universe:.2e} perles ont au moins 1 link n=2 formé "
            "sur l'univers observable. Recherche de signature de "
            "désintégration cascade pertinente."
        )
    else:
        status = "FREQUENT"
        comment = (
            "Enchevêtrement spontané fréquent : la suppression Vassiliev "
            "ne suffit pas, V5.20 a besoin d'une révision."
        )

    return {
        "M_pearl_Msun": M_pearl_Msun,
        "m_knot_GeV": m_knot_GeV,
        "delta_c": delta_c,
        "N_knots_per_pearl": res["N_knots_per_pearl"],
        "N_events_per_pearl_over_t_Hubble": N_per_pearl,
        "N_pearls_in_Hubble_volume": res["N_pearls_in_Hubble_volume"],
        "N_events_total_universe": N_universe,
        "sigma_cm2": res["sigma_cm2"],
        "status": status,
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("V5.20 -- Physique interne d'une perle DM (Class A)")
    print("=" * 78)
    print(f"\nM_pearl typique = {M_PEARL_TYPICAL_MSUN:.2e} M_sun")
    print(f"\nN(m_knot) pour les 6 STRICT amphichiraux :")
    for m, N in n_knots_table().items():
        print(f"  m_knot = {m:6.2f} GeV  ->  N = {N:.3e}")
    print()
    for m_knot in [1.55, 17.24, 51.78]:
        print(f"\n--- Profil TOV : m_knot = {m_knot} GeV ---")
        res = tov_pearl(M_PEARL_TYPICAL_MSUN, m_knot)
        print(f"  R_pearl = {res['R']:.3e} m  ({res['R']/1000:.2f} km)")
        print(f"  rho_c   = {res['rho_c']:.3e} kg/m^3")
        print(f"  n_c     = {res['n'][0]:.3e} m^-3")
        M_Ch = chandrasekhar_topological_pearl(m_knot)
        print(f"  M_Ch    = {M_Ch:.3e} kg ({M_Ch/M_SUN_KG:.3e} M_sun)")
        print(f"  M_Ch/M_Ch_WD = {chandrasekhar_ratio(m_knot):.3e}")
    print()
    for delta_c in [5, 10, 15, 20]:
        v = verdict_v5p20(delta_c=delta_c)
        print(f"\nDelta_c = {delta_c} : {v['status']}")
        print(f"  sigma = {v['sigma_cm2']:.3e} cm^2")
        print(f"  N_events_total_universe = {v['N_events_total_universe']:.3e}")


__all__ = [
    "n_knots", "n_knots_table",
    "fermi_momentum", "n_crit_relativistic",
    "pressure_NR", "pressure_UR", "eos_topological",
    "PearlPolytropicEoS",
    "tov_rhs_pearl", "tov_pearl", "mass_radius_pearl",
    "chandrasekhar_topological_pearl", "chandrasekhar_white_dwarf",
    "chandrasekhar_ratio",
    "sigma_entanglement", "fermi_velocity", "gamma_pair_local",
    "entanglement_rate", "n_events_over_universe_age",
    "verdict_v5p20",
    "M_PEARL_TYPICAL_MSUN", "DM_STRICT_AMPHI_GEV",
    "ALPHA_EM", "SIGMA_QED", "MPL_KG", "M_SUN_KG", "T_HUBBLE_S",
]
