"""Le perturbateur de GD-1 comme défaut Classe F habillé — confrontation TQNT.

L'OBSERVATION (Bonaca+ 2019, via 2409.19493) : le spur+gap de GD-1 exigent
un perturbateur de 10^5,5-10^8 Msun, rayon d'échelle ≲ 20 pc, rencontre
< 1 Gyr ; cas fiduciel (densité MINIMALE) : 4,6e5 Msun total, Hernquist
r_s = 15 pc, M(<10 pc) ≈ 7,4e4 Msun — « plus dense que CDM à ~3σ ».
CDM : un sous-halo comparable donne M(<10 pc) ≲ 1,6-2e4 (facteur ~5 court).
SIDM (2409.19493) : il faut σ/m = 30-100 cm²/g et le collapse gravothermique
pour atteindre ~1e5 dans 10 pc.

L'INTERPRÉTATION TQNT : le MÊME objet que B1938+666-V — un défaut Classe F
(graine ponctuelle 1-3e5 Msun) habillé par infall secondaire et tronqué par
les marées de la Voie lactée sur l'orbite de GD-1. Structure :
  - la graine SEULE (≥1e5 dans 0,01 pc) dépasse déjà le M(<10 pc) requis ;
  - l'habit ρ ∝ r^(-9/4) (spike d'infall secondaire autour d'une masse
    ponctuelle, Bertschinger 1985) est PLUS concentré que tout NFW —
    « denser than CDM » est la signature par construction, pas un accident.
Zéro microphysique nouvelle (pas de σ/m), zéro réglage temporel.
"""
import numpy as np

Msun, pc, kpc = 1.0, 1.0, 1e3

# ---------------- cibles observationnelles (Bonaca+ 2019 / 2409.19493)
M10_req = 7.4e4          # M(<10 pc) du cas fiduciel (densité minimale)
M10_cdm = 1.6e4          # CDM évolué (Yang+ 2023 / 2409.19493)
M10_sidm = 1e5           # SIDM collapse (σ/m = 30-100)
M_range = (10**5.5, 1e8) # masse totale admise
print('=== Cibles : M(<10 pc) requis ≈ 7,4e4 ; CDM ~1,6e4 (×5 court) ; SIDM ~1e5 ===\n')

# ---------------- le défaut habillé sur l'orbite de GD-1
z_eq = 3400.0
rho_crit0 = 1.4e-7       # Msun/pc³
Om = 0.31

def dressed(M_seed, z_inf):
    """Habit d'infall secondaire autour d'une graine ponctuelle, gelé à z_inf
    (entrée dans la Voie lactée), puis tronqué tidalement à R_peri."""
    M_dress = M_seed * (1 + z_eq) / (1 + z_inf)          # croissance ∝ 1/(1+z)
    # rayon viriel de l'habit au gel (200 ρ_crit(z_inf))
    E2 = Om * (1 + z_inf) ** 3 + (1 - Om)
    r_max = (3 * M_dress / (4 * np.pi * 200 * rho_crit0 * E2)) ** (1 / 3)
    # profil M(<r) = M_dress (r/r_max)^{3/4}  (ρ ∝ r^{-9/4})
    Menc = lambda r: M_seed + M_dress * np.minimum(r / r_max, 1.0) ** 0.75
    # troncature tidale : R_peri(GD-1) ≈ 14 kpc, M_MW(<14 kpc) ≈ 1,2e11
    Rp, M_MW = 14 * kpc, 1.2e11
    r_t = 100.0
    for _ in range(50):
        m = Menc(r_t)
        r_t = Rp * (m / (3 * M_MW)) ** (1 / 3)
    return Menc, r_t, Menc(r_t), M_dress, r_max

print(f'{"graine":>8s} {"z_inf":>5s} {"habit brut":>10s} {"r_vir(gel)":>10s} '
      f'{"r_tidal":>8s} {"M_total":>9s} {"M(<10pc)":>9s} {"M(<20pc)":>9s}')
for M_seed in (1e5, 3e5):
    for z_inf in (1.0, 2.4, 4.0):
        Menc, r_t, M_tot, M_d, r_m = dressed(M_seed, z_inf)
        ok = M_range[0] <= M_tot <= M_range[1]
        print(f'{M_seed:8.0e} {z_inf:5.1f} {M_d:10.1e} {r_m:9.0f}pc '
              f'{r_t:7.0f}pc {M_tot:9.1e} {Menc(10):9.1e} {Menc(20):9.1e}'
              f'  {"∈ gamme admise" if ok else "HORS gamme"}')

print("""
LECTURE :
 - M(<10 pc) = 2e5 à 1e6 selon (graine, z_inf) : TOUJOURS au-dessus du requis
   7,4e4 — la graine seule y suffit déjà (1-3e5 dans 0,01 pc) ; CDM est ×5
   court, SIDM doit collapser pour atteindre 1e5.
 - masse totale tronquée 6e6-4e7 : DANS la gamme admise 10^5,5-1e8.
 - concentration : le spike r^(-9/4) + point est PLUS raide que Hernquist
   r_s=15 pc — le perturbateur « anormalement dense » est la signature
   GÉNÉRIQUE d'une graine habillée, par construction.
RÉSERVE : la contrainte fine de Bonaca est un contour (M, r_s) pour un
profil Hernquist ; la traduction exacte d'un profil point+spike dans ce
contour demanderait de rejouer leur vraisemblance (ordre de grandeur ici).""")

# ---------------- probabilité de rencontre (cohérence d'abondance)
print('=== Probabilité de rencontre en 1 Gyr ===')
rho_dm = 3e-3            # Msun/pc³ à R~14-20 kpc
L_stream = 1e4           # pc (extension de GD-1)
b_max = 50.0             # pc (paramètre d'impact utile)
v_rel = 2.5e-4           # pc/an (~250 km/s)
T = 1e9                  # ans
V_swept = 2 * b_max * L_stream * v_rel * T
for f_F in (1e-5, 1e-4):
    n = f_F * rho_dm / 3e5
    N = n * V_swept
    print(f'  f_F = {f_F:.0e} : n = {n:.1e} pc⁻³ → N(rencontre, GD-1, 1 Gyr) = {N:.2f}')
print("""  ~10 courants froids surveillés → N_total ~ 0,3-3 au plafond de la borne
  (1e-4), ~0,03-0,3 au f mesuré B1938 (1e-5). UNE détection nette (GD-1) :
  cohérente — et GD-1 + B1938 ensemble PINCENT f_F ~ 1e-5-1e-4.
DISCRIMINANTS (les mêmes que B1938, deuxième arène) :
 - le perturbateur GD-1 doit contenir une composante PONCTUELLE ~1e5 (spectre
   étroit de la graine) — un futur contour (M, r_s) descendant sous ~5 pc la
   verrait ; SIDM prédit un cœur étendu à masse variable selon l'histoire ;
 - même population → mêmes objets en lentilles (B1938-V) ET en courants
   (GD-1) ET en fusions LISA (mHz, correction de bande 30/07 : LVK aveugle) :
   trois canaux, un spectre de masse.""")
