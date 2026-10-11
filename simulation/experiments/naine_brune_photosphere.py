"""Photosphère réaliste T_eff(g, s) — clôture du critère de stabilité absolu.

Le maillon manquant nommé au §3quater : la réponse de refroidissement
T_eff(s). Implémentation auto-cohérente, sans fit externe :

  * MINI-EoS D'ENVELOPPE (ρ < ~1 kg/m³ … 1e3 kg/m³, T 800 K … 1e6 K) :
    mélange H/He (X=0,75) avec DISSOCIATION H₂⇌2H (D₀=4,478 eV, rotation
    z_rot=T/2θ_r) et SAHA H⇌p+e (χ=13,6 eV) ; entropie de Sackur-Tetrode
    PAR ESPÈCE (H₂ + rotation, H, p, e, He) — l'entropie du mélange à
    l'équilibre est continue à travers les zones de recombinaison :
    l'adiabate d'enveloppe est physique.
  * RACCORD : l'entropie intérieure (EoS FD du modèle complet) et celle
    de l'enveloppe vivent sur des échelles différentes (espèce « ion »
    moyenne vs mélange) → on raccorde AU POINT D'ANCRAGE ρ_m = 300 kg/m³
    (entièrement ionisé, non dégénéré des deux côtés) : l'intérieur y
    fixe T_m via SON adiabate, l'enveloppe continue depuis (ρ_m, T_m)
    avec la SIENNE — aucun décalage d'échelle ne subsiste.
  * PHOTOSPHÈRE : P_ph = (2/3) g/κ à τ=2/3, κ = κ_H⁻ (2,5e-31 ρ_cgs^½ T⁹
    cm²/g, régime 2500-8000 K) + plancher moléculaire 1e-2 cm²/g
    (déclaré) ; T_eff = racine de s_env(ρ_ph, T_eff) = s_env(ρ_m, T_m).
  * VERDICTS : refait la table gravothermique avec L_surf = 4πR²σT_eff⁴
    VRAI — VALIDATION : le contrôle 0,09 Msun doit sortir STABLE ;
    alors les verdicts perle deviennent ABSOLUS, et les (L*, T_eff*)
    d'équilibre des étoiles-perle sont des PRÉDICTIONS.

Réserves déclarées : couches superadiabatiques et fenêtres radiatives
des atmosphères BD ignorées ; κ_H⁻ hors domaine sous 2500 K (plancher) ;
poussières/molécules non résolues — la RÉPONSE dT_eff/ds est l'objet,
pas le zéro absolu de T_eff.
"""
from __future__ import annotations

import math
import numpy as np

from simulation.experiments.naine_brune_modele_complet import (
    structure, eos, t_adiabat, LT, MSUN, LSUN, M_P, G)

KB = 1.380649e-23
H_PL = 6.62607015e-34
ME = 9.1093837e-31
MH = 1.67262192e-27
SIGMA_SB = 5.670374e-8
EV = 1.602176634e-19
X_H, Y_HE = 0.75, 0.25
D0, CHI = 4.478 * EV, 13.598 * EV
THETA_ROT = 87.5

print(__doc__)

# =========================================================== EoS d'enveloppe
def _lam3(m, T):
    return (H_PL / math.sqrt(2 * math.pi * m * KB * T)) ** 3

def envelope(rho, T):
    """Composition d'équilibre + (P, s) du mélange H2/H/p/e/He."""
    a = X_H * rho / MH                    # noyaux H par m³
    n_he = Y_HE * rho / (4 * MH)
    zrot = max(T / (2 * THETA_ROT), 1.0)
    Kd = (math.pi * MH * KB * T) ** 1.5 / H_PL**3 * (4.0 / zrot) * math.exp(-D0 / (KB * T))
    Ks = 1.0 / _lam3(ME, T) * math.exp(-CHI / (KB * T))
    # y : fraction de noyaux H libres (atomes+ions) ; z : fraction ionisée des libres
    y, z = 0.5, 0.0
    for _ in range(80):
        # dissociation : [y a (1-z)]² = Kd (1-y) a / 2  → résoudre y à z fixé
        lo, hi = 1e-12, 1.0 - 1e-12
        for _ in range(60):
            ym = 0.5 * (lo + hi)
            lhs = (ym * a * max(1 - z, 1e-12)) ** 2
            rhs = Kd * (1 - ym) * a / 2
            if lhs < rhs:
                lo = ym
            else:
                hi = ym
        y = 0.5 * (lo + hi)
        # Saha : (y a z)² / (y a (1-z)) = Ks → z à y fixé
        lo, hi = 1e-15, 1.0 - 1e-15
        for _ in range(60):
            zm = 0.5 * (lo + hi)
            lhs = (y * a * zm) ** 2
            rhs = Ks * y * a * max(1 - zm, 1e-15)
            if lhs < rhs:
                lo = zm
            else:
                hi = zm
        z_new = 0.5 * (lo + hi)
        if abs(z_new - z) < 1e-10:
            z = z_new
            break
        z = z_new
    n_h2 = (1 - y) * a / 2
    n_h0 = y * a * (1 - z)
    n_p = y * a * z
    n_e = n_p
    P = (n_h2 + n_h0 + n_p + n_e + n_he) * KB * T
    def st(n, m, g, extra=0.0):
        if n <= 0:
            return 0.0
        return n * KB * (2.5 + math.log(g / (_lam3(m, T) * n)) + extra)
    s = (st(n_h2, 2 * MH, 1.0, extra=1.0 + math.log(zrot))
         + st(n_h0, MH, 2.0) + st(n_p, MH, 1.0)
         + st(n_e, ME, 2.0) + st(n_he, 4 * MH, 1.0)) / rho
    return P, s, y, z

def kappa(rho, T):
    """κ [m²/kg] : H⁻ (fit standard, ρ en g/cm³) + plancher moléculaire."""
    k_hm = 2.5e-31 * math.sqrt(rho * 1e-3) * T**9 * 0.1   # cm²/g → m²/kg
    return min(k_hm, 1e3) + 1e-3

def t_eff(g, s_int):
    """Photosphère : T_eff(g, s_int) via raccord à ρ_m et adiabate d'enveloppe."""
    RHO_M = 300.0
    T_m = 10 ** t_adiabat(math.log10(RHO_M), s_int)
    s_star = envelope(RHO_M, T_m)[1]
    def mismatch(Te):
        # ρ_ph : P_env(ρ, Te) = (2/3) g / κ(ρ, Te)
        lo, hi = -9.0, 1.5
        for _ in range(60):
            lr = 0.5 * (lo + hi)
            rho = 10**lr
            if envelope(rho, Te)[0] < (2.0 / 3.0) * g / kappa(rho, Te):
                lo = lr
            else:
                hi = lr
        rho_ph = 10 ** (0.5 * (lo + hi))
        return envelope(rho_ph, Te)[1] - s_star
    lo, hi = 800.0, 9000.0
    if mismatch(lo) * mismatch(hi) > 0:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mismatch(mid) * mismatch(lo) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ---------- contrôles de la brique enveloppe
print("--- contrôles enveloppe")
for rho, T in [(1e-3, 1500), (1e-3, 3000), (1e-3, 6000), (1e-2, 2e4), (300, 3e5)]:
    P, s, y, z = envelope(rho, T)
    print(f"  ρ={rho:8.0e} T={T:7.0f} : libres y={y:.3f}, ionisés z={z:.3e},"
          f" P={P:.2e} Pa, s={s:.3e}")
print("  (attendu : moléculaire à 1500 K, atomique à 6000 K, ionisé à 3e5 K)")

# =========================================================== structures fixes
_MEMO = {}
def run(lr0, s, pearl):
    lt0 = t_adiabat(lr0, s)
    if lt0 >= LT[-1] - 0.02 or lt0 <= LT[0] + 0.02:
        return None
    key = (round(lr0, 4), round(s, 1), pearl)
    if key not in _MEMO:
        _MEMO[key] = structure(10**lr0, 10**lt0, pearl)
    return _MEMO[key]

def M_exact(s, M_target, pearl, tol=0.01):
    lo_hi = (5.0, 8.6) if pearl else (3.8, 7.0)
    grid = []
    for lr0 in np.linspace(*lo_hi, 11):
        out = run(lr0, s, pearl)
        if out:
            grid.append((lr0, out[0]))
    for k in range(len(grid) - 1):
        m1, m2 = grid[k][1], grid[k + 1][1]
        if (m1 - M_target) * (m2 - M_target) <= 0:
            lo, hi = grid[k][0], grid[k + 1][0]
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                out = run(mid, s, pearl)
                if out is None:
                    return None
                if (out[0] - M_target) * (m1 - M_target) > 0:
                    lo = mid
                else:
                    hi = mid
            out = run(0.5 * (lo + hi), s, pearl)
            if out and abs(out[0] - M_target) / M_target < tol:
                return out
    return None

def verdict(Mt_msun, pearl, label):
    Mt = Mt_msun * MSUN
    _s = sorted({round(eos(10**lr, 10**lt)[1], -2)
                 for lr in np.linspace(4.0, 6.8, 7)
                 for lt in np.linspace(5.2, 6.8, 7)})
    s_all = sorted({_s[int(q * (len(_s) - 1))] for q in np.linspace(0, 1, 10)})
    pts = []
    for s in s_all:
        out = M_exact(s, Mt, pearl)
        if out is None:
            continue
        M, R, Lnuc = out
        g = G * M / R**2
        Te = t_eff(g, s)
        if Te is None:
            continue
        Lsurf = 4 * math.pi * R**2 * SIGMA_SB * Te**4
        pts.append((s, Lnuc, Lsurf, Te, R))
    pts.sort()
    print(f"\n  [{label}] M = {Mt_msun} Msun : {len(pts)} structures")
    if len(pts) < 4:
        print("    couverture insuffisante")
        return None
    s_a = np.array([p[0] for p in pts])
    Ln = np.array([max(p[1], 1e-30) for p in pts])
    Ls = np.array([p[2] for p in pts])
    Te_a = np.array([p[3] for p in pts])
    print(f"    T_eff : {Te_a.min():.0f} → {Te_a.max():.0f} K sur la gamme d'entropies"
          f" ; dlnL_surf/ds = {np.polyfit(s_a, np.log(Ls), 1)[0]:.2e}"
          f" vs dlnL_nuc/ds = {np.polyfit(s_a, np.log(Ln), 1)[0]:.2e}")
    diff = np.log(Ln) - np.log(Ls)
    for k in range(len(pts) - 1):
        if diff[k] * diff[k + 1] <= 0:
            dLn = (math.log(Ln[k+1]) - math.log(Ln[k])) / (s_a[k+1] - s_a[k])
            dLs = (math.log(Ls[k+1]) - math.log(Ls[k])) / (s_a[k+1] - s_a[k])
            stable = dLn < dLs
            Lstar = math.sqrt(Ls[k] * Ls[k+1])
            Testar = math.sqrt(Te_a[k] * Te_a[k+1])
            print(f"    ÉQUILIBRE : L* = {Lstar/LSUN:.2e} Lsun, T_eff* = {Testar:.0f} K ;"
                  f" dlnL_nuc/ds={dLn:.2e} vs dlnL_surf/ds={dLs:.2e} → "
                  f"{'THERMOSTAT STABLE' if stable else 'INSTABLE (flash)'}")
            return stable
    side = "L_nuc>L_surf partout" if diff.min() > 0 else "L_nuc<L_surf partout (se contracte encore)"
    print(f"    pas de croisement dans la gamme : {side}")
    return None

print("\n===== GRAVOTHERMIQUE AVEC PHOTOSPHÈRE RÉALISTE =====")
ok = verdict(0.090, False, "CONTRÔLE 0,090 sans perle — VALIDATION : doit être STABLE")
p5 = verdict(0.050, True, "perle 0,050")
p3 = verdict(0.030, True, "perle 0,030")
print(f"\n  BILAN : contrôle={'STABLE ✓ critère VALIDÉ' if ok else ('non validé' if ok is not None else 'sans croisement')} ;"
      f" perle 0,05={'stable' if p5 else ('INSTABLE' if p5 is not None else 'n/a')} ;"
      f" perle 0,03={'stable' if p3 else ('INSTABLE' if p3 is not None else 'n/a')}")
