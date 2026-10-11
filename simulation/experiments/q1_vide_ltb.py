# -*- coding: utf-8 -*-
"""Q-1 — Le profil de vide au-delà du linéaire (03/09/2026).

LTB à temps de bang synchrone : chaque coquille évolue comme un univers FRW
indépendant (top-hat = la coquille intérieure) ; on intègre l'équation de
coquille exacte avec Λ,
    R̈ = −(H0²Ω_m/2)(1+δ_i)a_i³ r³/R² + H0²Ω_Λ R,
conditions initiales en mode croissant à a_i=1e-3 (z≈1000) :
    R_i = a_i(1−δ_i/3),  Ṙ_i = H(a_i)·a_i·(1−2δ_i/3).
Sorties : ΔH/H exact vs linéaire (−fδ/3) pour un contraste d'aujourd'hui
donné ; profondeur requise pour la tension pleine ; couverture KBC raffinée
avec pondération du volume d'échantillonnage SH0ES.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

Om, OL = 0.315, 0.685
H0 = 1.0                       # unités H0=1
a_i = 1e-3
f_growth = Om**0.55

def H_bg(a):
    return H0 * np.sqrt(Om / a**3 + OL)

# temps cosmique du fond
t_of = lambda a: quad(lambda x: 1.0 / (x * H_bg(x)), 1e-8, a)[0]
t_i, t_0 = t_of(a_i), t_of(1.0)

def shell(delta_i):
    """Intègre la coquille ; renvoie (delta_aujourd'hui, ΔH/H exact)."""
    Ri = a_i * (1.0 - delta_i / 3.0)
    Vi = H_bg(a_i) * a_i * (1.0 - 2.0 * delta_i / 3.0)
    GM = 0.5 * H0**2 * Om * (1.0 + delta_i) * a_i**3 / a_i**3 * 1.0  # r=1
    GM = 0.5 * H0**2 * Om * (1.0 + delta_i)                          # a_i³r³ absorbé
    def rhs(t, y):
        R, V = y
        return [V, -GM / R**2 + H0**2 * OL * R]
    sol = solve_ivp(rhs, (t_i, t_0), [Ri, Vi], rtol=1e-10, atol=1e-12)
    R, V = sol.y[0][-1], sol.y[1][-1]
    delta_now = (1.0 / R)**3 * (1.0 + delta_i) - 1.0
    dH = V / R / H_bg(1.0) - 1.0
    return delta_now, dH

def dH_of_delta_now(target):
    di = brentq(lambda d: shell(d)[0] - target, -0.5, 0.5, xtol=1e-10)
    return shell(di)[1]

print("δ_aujourd'hui   ΔH/H exact   ΔH/H linéaire (−fδ/3)   rapport")
for d in (-0.1, -0.2, -0.3, -0.4, -0.5):
    ex = dH_of_delta_now(d)
    lin = -f_growth * d / 3.0
    print(f"{d:12.2f} {ex:+12.4f} {lin:+12.4f}            {ex/lin:8.3f}")

# tension pleine : profondeur requise exacte
dH_req = 73.04 / 67.4 - 1.0
d_req = brentq(lambda d: dH_of_delta_now(d) - dH_req, -0.9, -0.05, xtol=1e-8)
print(f"\ntension pleine (+{dH_req*100:.1f}%) : δ requis EXACT = {d_req:+.3f} "
      f"(linéaire : {-3*dH_req/f_growth:+.3f})")

# KBC exact + pondération d'échantillon
d_kbc = -0.20
dH_kbc = dH_of_delta_now(d_kbc)
print(f"vide KBC δ=-0,20 : ΔH/H exact = {dH_kbc*100:+.2f}% "
      f"(linéaire +3,53%) → couverture max {dH_kbc/dH_req*100:.0f}%")
# pondération : SN SH0ES 0,023<z<0,15 ; vide de 300 Mpc → z_v≈0,070
z_lo, z_hi, z_v = 0.023, 0.15, 0.070
# poids ~ volume comobile dans la coquille des SN (dN/dz ∝ z²)
w = (z_v**3 - z_lo**3) / (z_hi**3 - z_lo**3)
print(f"pondération échantillon SH0ES (z_v=0,07 sur [0,023, 0,15], dN∝z²) : "
      f"w = {w:.2f}")
print(f"→ couverture RÉALISTE : {w*dH_kbc/dH_req*100:.0f}% de la tension "
      f"(+{w*dH_kbc*67.4:.1f} km/s/Mpc)")
print("\ncontrôle externe : le diagramme de Hubble SN lui-même borne une")
print("marche locale à |ΔH/H| ≲ 1,8% (Kenworthy+ 2019) — compatible avec")
print("la couverture réaliste, exclut le vide plein.")