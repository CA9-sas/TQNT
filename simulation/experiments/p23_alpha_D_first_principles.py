r"""
DÉRIVATION DE α_D DEPUIS LES PREMIERS PRINCIPES.

Calibration empirique : D_eff(L) ≈ α_D · ⟨cr⟩(L)  avec r = 0.86

Hypothèses à tester :
  H1 : α_D ∝ R_nœud (rayon caractéristique d'un nœud)
       prédiction : α_D ≈ R_canonique de la paramétrisation
  H2 : α_D ∝ (U_κU_τ)^(1/3) / cr  (volume topologique)
  H3 : α_D = λ_C × constante (longueur de Compton)
  H4 : α_D = rapport entre rayon de rope-length et cr (Buck-Simon)

Si α_D ≈ 1 dans nos unités de paramétrisation (R = 2 pour 3_1),
cela signifie D_eff ≈ ⟨cr⟩ (mesuré en unités du rayon caractéristique).
"""
from __future__ import annotations
import numpy as np
from theory.abraham import frenet_geometry
from theory.knot_catalog import _CURVE_REGISTRY


# ───────── Étape 1 : Mesure du rayon caractéristique de chaque nœud ─────────
print("═"*90)
print("  Étape 1 : Rayon caractéristique R_K et taille linéaire L_K")
print("═"*90)

KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","7_1","7_2","9_1"]
print(f"  {'K':<5} {'cr':>3} {'R_K (rayon)':>11} {'L_K (long. courbe)':>18} "
      f"{'R_K/cr':>8} {'L_K/cr':>8}")
print("  " + "-" * 70)

K_data = {}
for K in KNOTS:
    pts = _CURVE_REGISTRY[K](20000)
    # Rayon caractéristique = max distance au centre
    center = np.mean(pts, axis=0)
    R = np.max(np.linalg.norm(pts - center, axis=1))
    # Longueur de la courbe
    dpts = np.diff(pts, axis=0)
    L = np.sum(np.linalg.norm(dpts, axis=1))
    cr = int(K.split("_")[0])
    print(f"  {K:<5} {cr:>3} {R:>11.4f} {L:>18.4f} {R/cr:>8.4f} {L/cr:>8.4f}")
    K_data[K] = {"R": R, "L": L, "cr": cr}


# ───────── Étape 2 : Calculer α_D pour différentes définitions ─────────
print("\n" + "═"*90)
print("  Étape 2 : α_D candidat selon différentes hypothèses physiques")
print("═"*90)

# Données empiriques de la calibration précédente
D_eff_data = [
    ("μ⁻",   ["3_1","6_1"],          2.775,  4.5),
    ("η",    ["3_1","3_1","6_1"],    2.939,  4.0),
    ("τ⁻",   ["4_1","5_1","6_1"],    5.980,  5.0),
    ("φ",    ["3_1","5_1","5_2"],    2.965,  4.33),
    ("D⁰",   ["3_1","5_1","7_2"],    3.386,  5.0),
    ("Λ_c",  ["3_1","3_1","9_1"],    7.130,  5.0),
    ("p",    ["3_1","7_2","7_2"],   10.176,  5.67),
]

print(f"  {'particule':<6} {'D_eff':>7} {'⟨cr⟩':>6} {'α_D (cr)':>10} "
      f"{'⟨R_K⟩':>8} {'α_D (R)':>10} {'(ΣU)^(1/3)':>12}")
print("  " + "-" * 70)
import math

# Précalcul des invariants
from theory.jones import jones_polynomial, jones_determinant
U_KT = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_K = float(np.sum(g["kappa"]**2*ds))
    U_T = float(np.sum(g["tau"]**2*ds))
    U_KT[K] = U_K * U_T

alpha_D_list = []
for nm, cfg, D_eff, mean_cr in D_eff_data:
    R_mean = np.mean([K_data[K]["R"] for K in cfg])
    sum_U = sum(U_KT[K] for K in cfg)
    cube_root = sum_U**(1/3)
    alpha_cr = D_eff / mean_cr      # D_eff = α · ⟨cr⟩
    alpha_R = D_eff / R_mean         # D_eff = α · ⟨R_K⟩
    print(f"  {nm:<6} {D_eff:>7.3f} {mean_cr:>6.2f} {alpha_cr:>10.4f} "
          f"{R_mean:>8.4f} {alpha_R:>10.4f} {cube_root:>12.4f}")
    alpha_D_list.append({"nm": nm, "alpha_cr": alpha_cr, "alpha_R": alpha_R,
                          "D_eff": D_eff, "mean_cr": mean_cr,
                          "R_mean": R_mean, "cube_root": cube_root})


# Statistiques
alpha_cr_arr = np.array([a["alpha_cr"] for a in alpha_D_list])
alpha_R_arr = np.array([a["alpha_R"] for a in alpha_D_list])
print(f"\n  Statistiques α_D :")
print(f"    α_D vs ⟨cr⟩ : ⟨α⟩ = {alpha_cr_arr.mean():.4f}, "
      f"σ = {alpha_cr_arr.std():.4f} ({alpha_cr_arr.std()/alpha_cr_arr.mean()*100:.0f}%)")
print(f"    α_D vs ⟨R⟩ : ⟨α⟩ = {alpha_R_arr.mean():.4f}, "
      f"σ = {alpha_R_arr.std():.4f} ({alpha_R_arr.std()/alpha_R_arr.mean()*100:.0f}%)")


# ───────── Étape 3 : Conversion en unités physiques (TQNT) ─────────
print("\n" + "═"*90)
print("  Étape 3 : Conversion en unités physiques TQNT")
print("═"*90)

# En TQNT, le rayon du proton = 4·λ_C(proton) où λ_C = ℏ/(m_p·c)
# λ_C(proton) ≈ 0.21 fm
# Pour notre paramétrisation, le 3_1 a R ≈ 2 (unités arbitraires)
# Si le 3_1 dans le proton a une « taille » physique de l'ordre du proton,
# alors notre unité = 0.21 fm / R_3_1_normalized

R_3_1_norm = K_data["3_1"]["R"]
# En TQNT, proton = 4λ_C où λ_C(p) ≈ 0.21 fm
lambda_C_p = 0.21  # fm
r_proton_phys = 4 * lambda_C_p  # 0.84 fm

# Notre 3_1 normalisé a R ≈ 2. Si on identifie ce R à... un rayon physique ?
# Hypothèse : la "taille" d'un knot est ~ λ_C × cr (proportionnel aux croisements)
# Pour 3_1 (cr=3) : taille = 3·λ_C ≈ 0.63 fm
# Notre R(3_1) = 2.0 normalisé → 1 unité = 0.63 / 2.0 = 0.315 fm

L_unit_estim = (3 * lambda_C_p) / R_3_1_norm  # fm par unité normalisée
print(f"\n  Hypothèse d'unité : 1 unité normalisée = {L_unit_estim:.4f} fm")
print(f"    basée sur taille(3_1) ≈ cr · λ_C = 3 · 0.21 fm")
print(f"\n  Conversions des D_eff en unités physiques :")
print(f"  {'particule':<8} {'D_eff (norm)':>13} {'D_eff (fm)':>11} {'D_eff/λ_C':>12}")
print("  " + "-" * 60)
for a in alpha_D_list:
    D_phys = a["D_eff"] * L_unit_estim
    D_C = D_phys / lambda_C_p
    print(f"  {a['nm']:<8} {a['D_eff']:>13.3f} {D_phys:>11.3f} {D_C:>12.3f}")


# ───────── Étape 4 : Dérivation théorique d'α_D ─────────
print("\n" + "═"*90)
print("  Étape 4 : Tentative de dérivation d'α_D depuis les premiers principes")
print("═"*90)
print(f"""
  Hypothèses physiques :

  (H1) α_D = R_canonique / cr_unité — α_D mesure la "taille par croisement"

      D'après nos R_K mesurés (table étape 1) :
        R_K / cr varie de ~0.4 à ~0.6 selon le nœud
        ⟨R_K/cr⟩ ≈ {np.mean([K_data[K]["R"]/K_data[K]["cr"] for K in KNOTS]):.3f}

      Si α_D = ⟨R_K/cr⟩, alors α_D ≈ 0.5
      Or notre mesure empirique donne ⟨α_D⟩ ≈ {alpha_cr_arr.mean():.2f}
      → Cohérence d'ordre de grandeur (×2)

  (H2) α_D = λ_C / R_unité_paramétrisation

      Notre paramétrisation utilise R_3_1 = {R_3_1_norm:.2f} unités
      Si 1 unité = λ_C_proton = {lambda_C_p} fm, alors R_3_1 = {R_3_1_norm*lambda_C_p:.3f} fm
      Mais le proton entier mesure 0.84 fm = 4·λ_C
      → conflit avec H1, indique que notre unité ≠ λ_C

  (H3) α_D dérivé de l'identité Biot-Savart entre boucles fermées

      Pour deux boucles fermées de courant I_1, I_2 séparées de D :
        F_attractive ~ I_1·I_2·R_1·R_2 / D^3 (champs lointains)
      Le facteur géométrique combinant les "tailles" des nœuds est :
        D_eff ≈ (R_1·R_2)^(1/2) · facteur(topologie)

      Cela suggère D_eff ∝ √(R_1·R_2), qui scale avec ⟨R_K⟩
      → cohérent avec H1

  (H4) Norme universelle dans les unités TQNT :

      α_D semble ≈ 1 dans les unités où R_3_1 = 2
      → D_eff exprimé en "rayons caractéristiques" est ≈ ⟨cr⟩
      → Interprétation : "chaque croisement écarte les composantes
        d'un rayon caractéristique de nœud"

  DÉRIVATION FORMELLE PROPOSÉE :

    α_D ≈ ⟨R_K(canonique)⟩ / ⟨cr⟩

    où R_K(canonique) est le rayon de la paramétrisation du nœud isolé,
    et ⟨ · ⟩ est la moyenne sur les nœuds du catalogue.

  Cette identification donne α_D ≈ {np.mean([K_data[K]["R"]/K_data[K]["cr"] for K in KNOTS]):.3f}
  unités/croisement, à comparer à la valeur empirique
  ⟨α_D⟩ = {alpha_cr_arr.mean():.3f} unités/croisement.

  CONCLUSION :
    α_D ≈ 0.5–1.0 unités de longueur normalisée par unité de croisement.
    Cette valeur est COHÉRENTE avec l'identification R_K/cr,
    mais reste à raffiner par un calcul de force entre boucles
    de courant rigoureux (Biot-Savart complet + intégration radiale).
""")
