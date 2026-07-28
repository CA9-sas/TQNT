r"""
SIMPLIFICATION ALGÉBRIQUE de l'équation de masse Ω_ts.

Forme originelle V5 :
  m = Φ_n · Φ_p · skel · G^Q_EXP

avec :
  Φ_n = ((n+1)/2)^(-x),   x = -√6
  Φ_p = Π_{K, ct≥2} (ρ_0 · cr_K^(-β))^(ct-1)
  skel = (Σ_K n_K · w_K^a)^(1/a),   w_K = M_naive(K) · (b·det/T_31)^p
  G = Σ_K n_K · U_K / P_31
  M_naive(K) = cr_K · M_C + n_lobes_K · M_L
  M_C = m_e·F_κ/3,  M_L = m_e·(1-F_κ)/3

Paramètres analytiques fermés :
  a = √(7/(8π)),  p = q = 3/4,  x = -√6,  β = 33/20,  ρ_0 = 10 - 3/(8π)
  T_31 = b(3_1)·det(3_1) = 6,  P_31 = U_κU_τ(3_1) ≈ 9.34,  F_κ = 0.8682

FORME SIMPLIFIÉE PROPOSÉE :

Soit pour chaque nœud K :
  μ_K = M_naive(K) / m_e        (masse naïve normalisée)
  τ_K = b(K)·det(K) / 6          (facteur Jones-tresse normalisé)
  υ_K = U_κU_τ(K) / 9{,}34       (énergie EM normalisée)

Alors :
  m_L / m_e = Φ_n · Φ_p · ‖μ · τ^{3/4}‖_a · ‖υ‖_1^{3/4}

où ‖·‖_a est la norme L^a au sens de Minkowski.

VERIFICATION numérique sur quelques particules : la forme simplifiée
doit donner les mêmes valeurs que la formule originelle.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_FREN = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)),
               float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


# ───────── Formule ORIGINELLE ─────────
def omega_original(components):
    n = len(components); ws=[]; Us=[]
    for K in components:
        Us.append(U_KT[K][0]*U_KT[K][1])
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── Formule SIMPLIFIÉE (dimensionnée) ─────────
def omega_simplified(components):
    """Forme simplifiée en quantités normalisées."""
    n = len(components)
    counter = Counter(components)
    # Quantités normalisées par knot
    mu = {K: M_NAIVE[K] / M_E for K in counter}
    tau = {K: TOPO[K]["b"] * TOPO[K]["det"] / T_31 for K in counter}
    upsilon = {K: U_KT[K][0] * U_KT[K][1] / P_31 for K in counter}
    # L^a norm de μ·τ^{3/4}
    a = A_FREN
    norm_a = sum(counter[K] * (mu[K] * tau[K]**P_EXP)**a
                  for K in counter)**(1/a)
    # L^1 norm de υ
    norm_1 = sum(counter[K] * upsilon[K] for K in counter)
    # Phi_n
    Phi_n = ((n+1)/2)**(-X_EXP)
    # Phi_p
    Phi_p = 1.0
    for K, ct in counter.items():
        if ct >= 2:
            Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return M_E * Phi_n * Phi_p * norm_a * norm_1**Q_EXP


# ───────── Test de cohérence ─────────
test_particles = [
    ("e⁻", ["3_1"]),
    ("μ⁻", ["3_1", "6_1"]),
    ("π", ["3_1", "3_1", "3_1"]),
    ("p", ["3_1", "7_2", "7_2"]),
    ("K", ["6_1", "6_1", "6_1"]),
    ("η", ["3_1", "4_1", "5_1"]),
    ("J/ψ", ["3_1", "4_1", "6_3"]),
]

print("\n" + "═"*100)
print("  Test : la forme simplifiée donne-t-elle les mêmes masses ?")
print("═"*100)
print(f"\n  {'Particule':<10} {'Originelle':>13} {'Simplifiée':>13} {'Écart':>10}")
print("  " + "-" * 50)
for nm, cfg in test_particles:
    m_orig = omega_original(cfg)
    m_simp = omega_simplified(cfg)
    diff = (m_simp - m_orig) / m_orig * 100
    flag = "✓" if abs(diff) < 1e-9 else "✗"
    print(f"  {nm:<10} {m_orig:>13.6f} {m_simp:>13.6f} {diff:>+9.2e} {flag}")


# ───────── Présentation de la forme simplifiée ─────────
print("\n" + "═"*100)
print("  FORME SIMPLIFIÉE FINALE")
print("═"*100)

print(f"""
  Définitions par nœud K :

    μ_K = M_naive(K) / m_e        (masse naïve dim.less)
    τ_K = b(K)·det(K) / T_31      (Jones-tresse dim.less, T_31 = 6)
    υ_K = U_κU_τ(K) / U_31         (énergie EM dim.less, U_31 ≈ 9.34)

  Formule :

    m_L / m_e  =  Φ_n  ·  Φ_p  ·  ‖μ τ^{{p}}‖_a  ·  ‖υ‖_1^{{q}}

  Paramètres analytiques fermés (origine SU(2)_8) :

    p = q = 3/4                     (exposants quantum dim)
    a = √(7/(8π)) ≈ 0.529           (norme L^a)
    x = -√6 ≈ -2.449                (exposant Φ_n)
    β = 33/20                        (exposant Φ_p)
    ρ₀ = 10 - 3/(8π) ≈ 9.881         (préfacteur Φ_p)

  Facteurs combinatoires :

    Φ_n(n)        = ((n+1)/2)^(-x)
    Φ_p({{n_K}})    = ∏_{{K, n_K≥2}} (ρ₀ cr_K^(-β))^(n_K - 1)

  Quantités élémentaires :

    M_naive(K)    = cr_K · M_C + n_lobes_K · M_L
    M_C, M_L      = m_e · F_κ / 3 ,  m_e · (1-F_κ) / 3   (F_κ = 0.8682)
    T_31          = b(3_1) · det(3_1) = 2 · 3 = 6
    U_31          = U_κU_τ(3_1) ≈ 9.34

  ┌──────────────────────────────────────────────────────────────────┐
  │ INTERPRÉTATION :                                                 │
  │                                                                  │
  │ • ‖μ·τ^(3/4)‖_a  est une norme L^a (Minkowski) des poids         │
  │   topologiques pondérés par la masse naïve.                       │
  │ • ‖υ‖_1^(3/4)   est la norme L^1 (somme) des énergies EM        │
  │   topologiques, élevée à la puissance 3/4.                        │
  │ • Φ_n   est un facteur de combinatoire des composantes.         │
  │ • Φ_p   est un facteur de répétition (composantes identiques).  │
  │                                                                  │
  │ La masse émerge comme PRODUIT de trois fonctions séparées :     │
  │  (taille_combinatoire) × (poids_topologique) × (énergie^3/4)   │
  │                                                                  │
  │ Toutes les constantes (a, p, q, x, β, ρ₀) sont ANALYTIQUEMENT  │
  │ FERMÉES — dérivées des dimensions quantiques de SU(2)_8.        │
  └──────────────────────────────────────────────────────────────────┘

  Notation LaTeX compacte :

    \\boxed{{\\;
      \\frac{{m_L}}{{m_e}} = \\Phi_n(n) \\cdot \\Phi_p\\!\\left(\\{{n_K\\}}\\right)
      \\cdot \\!\\left\\|\\mu\\cdot\\tau^{{3/4}}\\right\\|_a
      \\cdot \\!\\left\\|\\upsilon\\right\\|_1^{{3/4}}
      \\;}}

  Cette forme rend explicite :
    • LE RÔLE de chaque facteur,
    • LA NATURE topologique de chaque terme,
    • L'ORIGINE analytique des constantes (SU(2)_8 representation theory),
    • L'ABSENCE de paramètre par particule (modèle universel).
""")


# ───────── Identités et limites utiles ─────────
print("\n" + "═"*100)
print("  Identités et cas particuliers")
print("═"*100)
print(f"""
  Cas 1 — Nœud unique (n = 1, pas de répétition) :

    Φ_n(1) = 1^√6 = 1
    Φ_p = 1
    ‖μ τ^p‖_a = μ_K · τ_K^(3/4)
    ‖υ‖_1 = υ_K

    Donc :  m_K = m_e · μ_K · τ_K^(3/4) · υ_K^(3/4)
                = m_e · μ_K · (τ_K · υ_K)^(3/4)
                = M_naive(K) · (b·det·U_κU_τ / (T_31·U_31))^(3/4)

  Cas 2 — Tous les nœuds différents (n_K = 1 ∀K) :

    Φ_p = 1
    ‖μ τ^p‖_a = (Σ_K (μ_K τ_K^p)^a)^(1/a)
    ‖υ‖_1 = Σ_K υ_K

  Cas 3 — Tous identiques (un seul K avec multiplicité n) :

    ‖μ τ^p‖_a = n^(1/a) · μ_K · τ_K^(3/4)
    ‖υ‖_1 = n · υ_K
    Φ_p = (ρ_0 · cr_K^(-β))^(n-1)

    m_L = m_e · Φ_n · (ρ_0 cr_K^{{-β}})^{{n-1}} · n^{{1/a + 3/4}} · μ_K · (τ_K υ_K)^{{3/4}}

  Cas 4 — Pi (3_1×3) : n=3, Phi_n = 2^√6 ≈ 5.85
    cr = 3, Phi_p = (ρ_0 · 3^(-β))^2 ≈ (9.881 · 0.218)^2 ≈ 4.64
    3^(1/a + 3/4) ≈ 3^2.64 ≈ 17.7
    μ τ^(3/4) υ^(3/4) ≈ 0.511/0.511 · 1^(3/4) · 1^(3/4) = 1

    m_π ≈ 0.511 · 5.85 · 4.64 · 17.7 = ~245 MeV (estimation rapide)
    Note : approximation rapide, valeur exacte par formule complète.
""")
