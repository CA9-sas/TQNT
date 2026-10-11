"""Modèle stellaire complet — naine brune avec et sans noyau de perle.

Suite de naine_brune_tqnt.py (2026-08-05) : remplace le modèle de cuspide
« ordre déclaré » par une VRAIE structure hydrostatique :

  * EoS complète : électrons de Fermi-Dirac non relativistes (dégénérescence
    partielle exacte via F_{1/2}, F_{3/2}) + ions parfaits (X=0,75, Y=0,25) ;
    table (ρ, T) → (P, s) précalculée, adiabates par inversion de s.
  * Intérieur ISENTROPE (naines brunes entièrement convectives — l'hypothèse
    est étendue à la cuspide, déclaré) intégré par RK adaptatif depuis
    r₀ = 100 km avec la condition interne m(r₀) = M_p (perle) ou m≈0
    (contrôle). La cuspide, l'auto-écrantage et le plafonnement par la
    dégénérescence sortent DU MODÈLE (plus de prescription à la main).
  * Taux pp NACRE + écran de Salpeter (faible, plafonné ×10 — déclaré).
  * Critère d'allumage standard de la littérature MMSM : le long de la
    séquence de contraction d'un objet de masse M (balayage des états
    centraux), l'hydrogène s'allume si max L_nuc ≥ L_crit(M), avec
    L_crit(M) = A·(M/0,075 Msun)^2,2 CALIBRÉ pour que le CONTRÔLE sans
    perle donne M_min = 0,075 Msun — le décalage AVEC perle est alors une
    prédiction DIFFÉRENTIELLE, robuste au modèle de photosphère.

Sorties : M_min(contrôle) [calibré], M_min(perle), L et structure des
objets sous-seuil allumés, garde-fous (R(M) dégénéré ~ R_Jup, ρ_c, T_c).
"""
from __future__ import annotations

import math
import numpy as np

G = 6.674e-11
KB = 1.380649e-23
HBAR = 1.054571817e-34
H_PL = 2 * math.pi * HBAR
ME = 9.1093837e-31
MH = 1.67262192e-27
MSUN = 1.98892e30
LSUN = 3.828e26
RJUP = 6.9911e7

X_H, Y_HE = 0.75, 0.25
MU_E = 2.0 / (1 + X_H)                  # 1,143
MU_I = 1.0 / (X_H + Y_HE / 4.0)         # 1,231 (ions seuls)
M_P = 2.03e-3 * MSUN
R0_PEARL = 1.0e5                        # 100 km : bord interne (>> 21 km)

print(__doc__)

# ================================================== EoS Fermi-Dirac
def fd_integral(k, eta):
    """F_k(η) = ∫ x^k/(1+exp(x-η)) dx, quadrature découpée (vectorisé sur η)."""
    eta = np.atleast_1d(eta).astype(float)
    out = np.zeros_like(eta)
    # domaine : [0, max(60, η+40)]
    for i, e in enumerate(eta):
        top = max(60.0, e + 40.0)
        x = np.linspace(1e-8, top, 3000)
        out[i] = np.trapezoid(x**k / (1.0 + np.exp(np.clip(x - e, -700, 700))), x)
    return out if out.size > 1 else float(out[0])

# table η(ρ,T) : n_e = C_n T^{3/2} F_{1/2}(η), C_n = 4π(2 m_e k)^{3/2}/h³
C_N = 4 * math.pi * (2 * ME * KB) ** 1.5 / H_PL**3
C_P = (2.0 / 3.0) * C_N * KB            # P_e = C_P T^{5/2} F_{3/2}(η)

ETA_GRID = np.linspace(-25, 400, 500)
F12 = np.array([fd_integral(0.5, e) for e in ETA_GRID])
F32 = np.array([fd_integral(1.5, e) for e in ETA_GRID])
LOGF12 = np.log(F12)

def eos(rho, T):
    """P (Pa), s (J/K/kg) pour (ρ, T) scalaires ; branche dégénérée
    analytique au-delà de la table (F_1/2→(2/3)η^{3/2}, F_3/2→(2/5)η^{5/2})."""
    n_e = rho / (MU_E * MH)
    # η par inversion de F_{1/2} = n_e/(C_N T^{3/2})
    val = math.log(n_e / (C_N * T**1.5))
    if val > LOGF12[-1] - 0.5:
        eta = (1.5 * math.exp(val)) ** (2.0 / 3.0)
        f32 = 0.4 * eta ** 2.5
    else:
        eta = float(np.interp(val, LOGF12, ETA_GRID))
        f32 = float(np.exp(np.interp(eta, ETA_GRID, np.log(F32))))
    P_e = C_P * T**2.5 * f32
    n_i = rho / (MU_I * MH)
    P_i = n_i * KB * T
    # entropies : ions Sackur-Tetrode ; électrons s_e = (u+P-μn)/(ρT), u = (3/2)P_e
    lam_i = H_PL / math.sqrt(2 * math.pi * (MU_I * MH) * KB * T)
    s_i = (n_i * KB / rho) * (2.5 - math.log(max(n_i * lam_i**3, 1e-300)))
    s_e = ((2.5 * P_e) - eta * KB * T * n_e) / (rho * T)
    return P_e + P_i, s_i + s_e

# table (logρ, logT) → (logP, s) puis adiabates T(ρ; s)
LR = np.linspace(-1, 9.0, 201)          # kg/m³
LT = np.linspace(4.4, 8.3, 196)
P_TAB = np.zeros((len(LR), len(LT)))
S_TAB = np.zeros_like(P_TAB)
for i, lr in enumerate(LR):
    for j, lt in enumerate(LT):
        P, s = eos(10**lr, 10**lt)
        P_TAB[i, j] = math.log10(P)
        S_TAB[i, j] = s
print(f"table EoS : {P_TAB.shape}, s ∈ [{S_TAB.min():.1e}, {S_TAB.max():.1e}] J/K/kg")

def t_adiabat(lr, s):
    """logT sur l'adiabate s à logρ=lr (interpolation dans la table)."""
    i = np.clip(np.searchsorted(LR, lr) - 1, 0, len(LR) - 2)
    w = (lr - LR[i]) / (LR[i + 1] - LR[i])
    scol = (1 - w) * S_TAB[i] + w * S_TAB[i + 1]     # s(logT) croissant
    return float(np.interp(s, scol, LT))

def p_of(lr, lt):
    i = np.clip(np.searchsorted(LR, lr) - 1, 0, len(LR) - 2)
    j = np.clip(np.searchsorted(LT, lt) - 1, 0, len(LT) - 2)
    wi = (lr - LR[i]) / (LR[i + 1] - LR[i])
    wj = (lt - LT[j]) / (LT[j + 1] - LT[j])
    return ((1 - wi) * (1 - wj) * P_TAB[i, j] + wi * (1 - wj) * P_TAB[i + 1, j]
            + (1 - wi) * wj * P_TAB[i, j + 1] + wi * wj * P_TAB[i + 1, j + 1])

def rho_of_P(lp, s, lr_guess):
    """inverse ρ(P) sur l'adiabate (Newton discret sur la table)."""
    lo, hi = -1.0, 9.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if p_of(mid, t_adiabat(mid, s)) < lp:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ================================================== taux pp écranté
def eps_pp(rho, T):
    T9 = T / 1e9
    if T9 < 1e-4:
        return 0.0
    base = 2.4e-3 * rho * X_H**2 * T9**(-2/3) * math.exp(-3.381 / T9**(1/3))
    # écran de Salpeter faible : H = 0,188 ζ √(ρ_ccg) T6^{-3/2}, ζ(H/He)~1,3 ; plafond ×10
    T6 = T / 1e6
    Hs = 0.188 * 1.3 * math.sqrt(rho * 1e-3) * T6 ** (-1.5)
    return base * min(math.exp(min(Hs, 2.302585)), 10.0)

# ================================================== intégrateur de structure
def structure(rho0, T0, pearl):
    """Intègre depuis r0 ; renvoie (M_tot, R, L_nuc, rho_c=rho0, T0)."""
    s = eos(rho0, T0)[1]
    r = R0_PEARL if pearl else 1.0e4
    m = M_P if pearl else (4/3) * math.pi * r**3 * rho0
    lr = math.log10(rho0)
    lt = math.log10(T0)
    lp = p_of(lr, lt)
    L = 0.0
    for _ in range(400000):
        rho = 10**lr
        T = 10**(t_adiabat(lr, s))
        P = 10**lp
        dPdr = -G * m * rho / r**2
        # pas adaptatif : 1 % en pression ou 2 % en rayon
        dr = min(0.02 * r, 0.01 * P / abs(dPdr))
        L += eps_pp(rho, T) * rho * 4 * math.pi * r**2 * dr
        m += 4 * math.pi * r**2 * rho * dr
        lp += (dPdr * dr) / (P * math.log(10))
        r += dr
        if lp < 4.0 or lr < -0.5:        # P < 1e4 Pa : surface atteinte
            break
        lr = rho_of_P(lp, s, lr)
    return m, r, L

# ================================================== séquences et seuil
def sweep(pearl, label):
    """Contrôle : grille (ρ_c, T_c). Perle : la condition interne doit être
    SUR l'adiabate de l'étoile — grille (entropie du bulk s) × (densité du
    bord interne lr0), T0 = T_adiabate(lr0, s) : la cuspide, sa compression
    et son plafonnement dégénéré sortent de l'EoS, pas d'une prescription."""
    print(f"\n--- séquences de contraction ({label})")
    rows = []
    if not pearl:
        for lrho in np.linspace(4.0, 6.8, 8):
            for lT in np.linspace(5.6, 6.9, 8):
                M, R, L = structure(10**lrho, 10**lT, pearl)
                if 0.005 * MSUN < M < 0.12 * MSUN:
                    rows.append((M, R, L, 10**lrho, 10**lT))
        return rows
    # entropies représentatives (états bulk du contrôle)
    s_set = sorted({round(eos(10**lr, 10**lt)[1], -2)
                    for lr in np.linspace(4.2, 6.6, 6)
                    for lt in np.linspace(5.7, 6.8, 6)})
    for s in s_set:
        for lr0 in np.linspace(6.0, 8.6, 9):
            lt0 = t_adiabat(lr0, s)
            if lt0 >= LT[-1] - 0.02:
                continue
            M, R, L = structure(10**lr0, 10**lt0, True)
            if 0.005 * MSUN < M < 0.12 * MSUN:
                rows.append((M, R, L, 10**lr0, 10**lt0))
    return rows

def L_max_at(rows, M_target, tol=0.12):
    sel = [r for r in rows if abs(r[0] - M_target) / M_target < tol]
    if not sel:
        return None
    return max(sel, key=lambda r: r[2])

def main():
    global A_CAL
    CTRL = sweep(False, "contrôle sans perle")
    PEARL = sweep(True, f"avec noyau de perle {M_P/MSUN:.2e} Msun")

    # garde-fous du contrôle
    best075 = L_max_at(CTRL, 0.075 * MSUN)
    print(f"\ncontrôle 0,075 Msun : R = {best075[1]/RJUP:.2f} R_Jup, "
          f"ρ_c = {best075[3]:.1e} kg/m³, T_c = {best075[4]:.2e} K, "
          f"L_nuc,max = {best075[2]/LSUN:.2e} Lsun")

    # calibration : L_crit(M) = A (M/0,075)^2,2 avec A = L_nuc,max(contrôle à 0,075)
    A_CAL = best075[2]
    def L_crit(M):
        return A_CAL * (M / (0.075 * MSUN)) ** 2.2

    print(f"calibration : L_crit(0,075) := L_nuc,max(contrôle) = {A_CAL/LSUN:.2e} Lsun"
          f" → M_min(contrôle) = 0,075 Msun PAR CONSTRUCTION")

    print("\n===== SEUIL D'ALLUMAGE : contrôle vs perle =====")
    print("  M[Msun]   L_max/L_crit (ctrl)   L_max/L_crit (perle)")
    for Mt in (0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09):
        c = L_max_at(CTRL, Mt * MSUN)
        p = L_max_at(PEARL, Mt * MSUN)
        rc = c[2] / L_crit(Mt * MSUN) if c else float('nan')
        rp = p[2] / L_crit(Mt * MSUN) if p else float('nan')
        tag = ""
        if p and rp >= 1 and rc < 1:
            tag = "  ← ALLUMÉ PAR LA PERLE SEULE"
        print(f"  {Mt:6.3f}   {rc:12.3f}        {rp:12.3f}{tag}")


if __name__ == '__main__':
    main()