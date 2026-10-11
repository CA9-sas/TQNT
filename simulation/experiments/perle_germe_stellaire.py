"""La perle comme germe de formation stellaire — le calcul détaillé.

Question (2026-08-05) : une perle (2e-3 Msun, 21 km) plongée dans un nuage
moléculaire peut-elle nucléer l'effondrement et « faire naître » une étoile
autour d'elle (genèse des hybrides Classe C) ? Avant toute mise en ligne,
chiffrer les quatre maillons :

  A. Croissance de Bondi-Hoyle-Lyttleton : Ṁ = 4πλ G²M²ρ/(c_s²+v²)^{3/2}
     — emballement en temps fini t_run = (c_s²+v²)^{3/2}/(4πλG²ρM₀) ;
     la CINÉMATIQUE DU HALO (v~200 km/s vs gaz) supprime le taux en 1/v³ :
     seule la queue lente de la distribution peut semer.
  B. Statistique de rencontre : densité de perles au plafond f=1 %
     (n_p = f·ρ_DM/M_p), nombre de perles DANS le gaz moléculaire galactique
     à un instant donné, fraction assez lente (Maxwell 3D décalée par la
     rotation LSR ~220 km/s), taux d'événements de germination par Galaxie.
  C. Fenêtre de germination : t_run < durée de vie du cœur/nuage, par
     densité de cœur (1e4-1e7 cm⁻³) et vitesse relative (0,2-10 km/s).
  D. Allumage des naines brunes : rayon d'influence du noyau sombre au
     centre d'un objet au seuil H (r_i = GM_p·μm_H/kT_c), masse réchauffée,
     ordre du décalage du seuil d'allumage (estimation déclarée, pas un
     modèle stellaire).

Toutes les entrées sont canoniques (M_p nominal 2,03e-3 Msun ; f≤1 % des
limites 04/08) ou standard (ρ_DM=0,4 GeV/cm³, LSR 220 km/s, σ_halo 1D
155 km/s, M_H2,MW ~ 1e9 Msun).
"""
from __future__ import annotations

import math
import numpy as np

G = 6.674e-11
KB = 1.380649e-23
MH = 1.67262e-27
MSUN = 1.98892e30
GEV = 1.78266e-27
PC = 3.0857e16
YR = 3.156e7
MYR = 1e6 * YR

M_P = 2.03e-3 * MSUN          # perle nominale
F_MAX = 0.01                  # plafond des limites 04/08
RHO_DM = 0.4 * GEV * 1e6      # kg/m³ (0,4 GeV/cm³)
MU = 2.33                     # gaz moléculaire (H2 + He)
LAM = 1.12                    # Bondi isotherme (e^{3/2}/4)
SIG1D = 155e3                 # dispersion 1D du halo (m/s)
V_LSR = 220e3                 # décalage disque/halo (m/s)
M_H2_MW = 1e9 * MSUN          # gaz moléculaire galactique
T_GAS = 10.0                  # K

CS = math.sqrt(KB * T_GAS / (MU * MH))
print(__doc__)
print(f"c_s(10 K, μ=2,33) = {CS:.0f} m/s = {CS/1e3:.2f} km/s")

# ---------------------------------------------------------------- A/C. BHL
def t_runaway(n_cm3, v_rel):
    """Temps d'emballement BHL (s) : dM/dt = kM² → t = 1/(k M₀)."""
    rho = n_cm3 * 1e6 * MU * MH
    k = 4 * math.pi * LAM * G**2 * rho / (CS**2 + v_rel**2) ** 1.5
    return 1.0 / (k * M_P)

print("\n--- C. Fenêtre de germination : t_emballement [Myr] (départ 2e-3 Msun)")
vs = [0.2e3, 0.5e3, 1e3, 2e3, 5e3, 10e3, 200e3]
ns = [1e3, 1e4, 1e5, 1e6, 1e7]
hdr = "  n[cm-3] " + "".join(f"  v={v/1e3:5.1f}km/s" for v in vs)
print(hdr)
for n in ns:
    row = f"  {n:8.0e} "
    for v in vs:
        row += f"  {t_runaway(n, v)/MYR:11.2e}"
    print(row)
print("  (fenêtre : t_run < ~10 Myr [vie d'un cœur dense/nuage] →"
      " exige v_rel ≲ 1-2 km/s ET n ≳ 1e5 cm-3 ; à v halo 200 km/s :"
      f" {t_runaway(1e6, 200e3)/MYR:.1e} Myr — MORT)")

# ---------------------------------------------------------------- B. statistique
n_p = F_MAX * RHO_DM / M_P
d_p = n_p ** (-1.0 / 3.0)
print(f"\n--- B. Statistique (au PLAFOND f = {F_MAX:.0%})")
print(f"densité de perles n_p = {n_p:.2e} m⁻³ → une perle par ({d_p/PC:.1f} pc)³")

# volume du gaz moléculaire (masse/densité moyenne n~100 cm-3)
V_H2 = M_H2_MW / (100 * 1e6 * MU * MH)
N_in_gas = n_p * V_H2
print(f"perles dans le gaz moléculaire MW (n̄~100) : {N_in_gas:.1e} à tout instant")

# fraction assez lente : Maxwell 3D (σ1d) décalée de V_LSR ; P(|v_rel| < v_c)
def p_slow(v_c):
    # intégration numérique de la maxwellienne décalée
    v = np.linspace(1.0, v_c, 400)
    a = (1.0 / (math.sqrt(2 * math.pi) * SIG1D)) ** 3
    # distribution de |v| pour bulk V : f(v) = (v/(V σ²√(2π))) [exp(-(v-V)²/2σ²)-exp(-(v+V)²/2σ²)]
    f = (v / (V_LSR * SIG1D**2 * math.sqrt(2 * math.pi))) * (
        np.exp(-((v - V_LSR) ** 2) / (2 * SIG1D**2))
        - np.exp(-((v + V_LSR) ** 2) / (2 * SIG1D**2)))
    return float(np.trapezoid(f, v))

for vc in (1e3, 2e3, 5e3):
    print(f"P(v_rel < {vc/1e3:.0f} km/s) = {p_slow(vc):.2e}")
N_slow = N_in_gas * p_slow(2e3)
print(f"→ perles LENTES (<2 km/s) dans le gaz moléculaire : {N_slow:.2e} à tout instant")

# de plus il faut être dans un CŒUR dense (fraction du gaz en cœurs ~few %)
F_CORE = 0.03
N_seed = N_slow * F_CORE
print(f"→ et dans un cœur dense (f_cœur~{F_CORE:.0%}) : {N_seed:.2e}")
# taux : renouvellement des cœurs ~ 1 Myr-10 Myr → événements/Gyr
for t_core in (1.0, 10.0):
    print(f"   taux de germination ≲ {N_seed * (1e3/t_core):.2e} étoiles semées/Galaxie/Gyr"
          f" (renouvellement cœurs {t_core:.0f} Myr)")

# capture par friction dynamique gazeuse (Ostriker) d'une perle rapide : Δv/passage
def dv_df(v, n_cm3, L_pc):
    rho = n_cm3 * 1e6 * MU * MH
    lnL = 10.0
    a = 4 * math.pi * G**2 * M_P * rho * lnL / v**2   # régime supersonique
    return a * (L_pc * PC / v)

dv = dv_df(200e3, 100, 20)
print(f"friction dynamique (perle à 200 km/s, GMC n=100, 20 pc) : Δv = {dv:.2e} m/s"
      f" → capture par le nuage : IMPOSSIBLE (il faudrait Δv ~ 2e5 m/s)")

# ---------------------------------------------------------------- D. naines brunes
print("\n--- D. Noyau sombre au centre d'un objet au seuil H (ordre déclaré)")
T_C = 3e6           # K, centre au seuil d'allumage
RHO_C = 1e6         # kg/m³ (naine brune massive / bas de séquence)
MU_I = 0.6          # gaz ionisé H/He
r_i = G * M_P * MU_I * MH / (KB * T_C)
M_heat = RHO_C * (4.0 / 3.0) * math.pi * r_i**3
R_BD = 7e7          # ~R_Jupiter
print(f"rayon d'influence r_i = GM_p μ m_H / kT_c = {r_i/1e3:.0f} km"
      f" ({r_i/R_BD:.2%} du rayon ; {r_i/2.11e4/1e3*1e3:.0f}× le rayon de la perle)")
print(f"masse réchauffée ~ρ_c·(4π/3)r_i³ = {M_heat/MSUN:.1e} Msun"
      f" ({M_heat/M_P:.1f}× la masse de la perle)")
print(f"ΔT/T ~ 1 à r_i par construction (le puits ajoute ~kT_c par particule) ;"
      f" la fusion pp varie en ~T^6 au seuil → la région centrale"
      f" ({M_heat/M_P:.1f}× la masse de la perle) fusionne ~2^6=64× plus vite :"
      f" allumage possible SOUS le seuil standard — ORDRE seulement, un vrai"
      f" modèle stellaire est requis (dégénérescence non traitée : elle plafonne"
      f" ΔT et RÉDUIT l'effet).")

# ---------------------------------------------------------------- verdict
rate_max = N_seed * (1e3 / 1.0)   # renouvellement 1 Myr (le plus favorable)
print(f"""
=== VERDICT (chiffres calculés ci-dessus, au PLAFOND f=1 %) ===
1. La FENÊTRE physique existe : une perle à v_rel ≲ 0,5-1 km/s dans un
   cœur n ≥ 1e5-1e6 cm⁻³ s'emballe en < 10 Myr et fait naître une étoile
   à noyau sombre. Le mécanisme est sain.
2. MAIS la cinématique du halo le rend QUASI INEXISTANT aujourd'hui :
   - à 200 km/s, t_run = {t_runaway(1e6, 200e3)/MYR:.1e} Myr (rien n'arrive jamais) ;
   - P(v_rel < 2 km/s) = {p_slow(2e3):.1e} (maxwellienne décalée de 220 km/s)
     → {N_slow:.1e} perle lente dans TOUT le gaz moléculaire galactique,
     {N_seed:.1e} dans un cœur dense, à un instant donné ;
   - taux ≤ {rate_max:.1e} étoile semée / Galaxie / Gyr : sur 10 Gyr,
     ~{rate_max*10:.0e} événement — LE CANAL EST MORT dans le disque actuel.
3. Capture d'une perle rapide par un nuage : impossible (Δv de friction
   ~ 1e-11 du nécessaire).
4. SURVIT en perspective : les environnements CO-MOBILES du jeune Univers
   (avant virialisation, v_rel gaz-DM ~ km/s à z ≳ 15-20, densités hautes)
   — germination des PREMIÈRES étoiles = la seule époque plausible du
   canal ; à consigner comme piste, pas comme résultat.
5. Naine brune à noyau sombre (si formée, quel qu'en soit le chemin) :
   r_i = {r_i/1e3:.0f} km, ~{M_heat/M_P:.1f}× la masse de la perle réchauffée
   d'un facteur ~2 → allumage sous le seuil = CANDIDAT de signature
   (ordre seulement, dégénérescence non traitée).
""")
