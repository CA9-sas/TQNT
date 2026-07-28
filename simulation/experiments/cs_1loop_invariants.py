r"""
CALCUL 1-LOOP SYSTÉMATIQUE via INVARIANTS TOPOLOGIQUES.

Formulation Lawrence-Rozansky du 1-loop pour SU(2)_k :
  WRT_k(K) ~ (k+2)^(-d/2) · exp(2πi(k+2)·CS(α)) · τ_R(α)^(1/2)

où :
  CS(α) : invariant de Chern-Simons de la connexion plate α
  τ_R(α) : torsion de Reidemeister
  d : dimension du moduli space

Pour les torus knots T(s,t), il y a (s-1)(t-1)/2 connexions plates
irréductibles. Le calcul exact existe (Rozansky 1996, Lawrence 1995).

Le coefficient c_1(K) du développement en boucle prend la forme :
  c_1(K) = A·σ(K) + B·λ(K) + C·log(τ_R(K)) + ...

où :
  σ(K) : signature
  λ(K) : invariant de Casson
  τ_R(K) : torsion de Reidemeister

Ces invariants sont CALCULABLES depuis les paramètres topologiques.

Pour les torus knots T(s,t) avec s ≤ t coprimes :
  σ(T(s,t)) = -(s-1)(t-1)  (avec convention TQNT)
  λ(T(s,t)) = (s²-1)(t²-1)/24

Pour les non-torus, on utilise des tables ou calculs explicites.
"""
from __future__ import annotations
import numpy as np
from theory.jones import jones_polynomial, jones_determinant, braid_index


# ───────── Tables des invariants topologiques ─────────
# Signature σ(K) — convention positive : σ ≥ 0 pour 3_1 droit
# (varies in literature; ici on prend |σ|)
SIGNATURE = {
    "3_1": -2,  "4_1": 0,   "5_1": -4,  "5_2": -2,
    "6_1": 0,   "6_2": -2,  "6_3": 0,
    "7_1": -6,  "7_2": -2,  "7_3": -4,  "7_4": -2,
    "7_5": -4,  "7_6": -2,  "7_7": 0,
    "8_1": 0,   "8_2": -4,  "8_3": 0,
    "9_1": -8,  "10_1": 0,
}

# Casson invariant λ(K) (calculé pour torus knots, tabulé pour hyperbolic)
# Pour T(2, 2n+1) : λ = ((2n+1)² - 1)/8 · 3/3 = (4n² + 4n)/8 = n(n+1)/2
CASSON = {
    "3_1": 1,    # T(2,3) : (3²-1)·(2²-1)/24 = 8·3/24 = 1
    "4_1": -1,   # figure-eight : -1
    "5_1": 3,    # T(2,5) : (5²-1)·3/24 = 24·3/24 = 3
    "5_2": 1,    # 5_2 : 1
    "6_1": -1,   # 6_1 : -1
    "6_2": 1,    # 6_2 : 1
    "6_3": 1,    # 6_3 : 1
    "7_1": 6,    # T(2,7) : (49-1)·3/24 = 48·3/24 = 6
    "7_2": 1,    # 7_2 : 1
    "7_3": 5,    "7_4": 4,   "7_5": 4,   "7_6": 1,   "7_7": -1,
    "8_1": -2,   "8_2": 4,   "8_3": -4,
    "9_1": 10,   # T(2,9) : (81-1)·3/24 = 80·3/24 = 10
    "10_1": -3,
}


def n_crossings(K):
    return int(K.split("_")[0])


def genus(K):
    """Genus de Seifert (estimé via le braid index pour torus knots).
    Pour T(s,t) : g = (s-1)(t-1)/2 ; pour autres, tabulé.
    """
    table = {
        "3_1": 1, "4_1": 1, "5_1": 2, "5_2": 1,
        "6_1": 1, "6_2": 2, "6_3": 2,
        "7_1": 3, "7_2": 1, "7_3": 2, "7_4": 1,
        "7_5": 2, "7_6": 2, "7_7": 1,
        "8_1": 1, "8_2": 3, "8_3": 1,
        "9_1": 4, "10_1": 1,
    }
    return table.get(K, 1)


def det_knot(K):
    """Déterminant via Jones."""
    return jones_determinant(jones_polynomial(K))


def is_amphichiral(K):
    """1 si amphichiral (σ = 0 et structure miroir)."""
    table = {"4_1": True, "6_3": True, "8_3": True,
             "8_9": True, "8_12": True, "8_17": True, "8_18": True,
             "10_17": True, "10_33": True}
    return 1 if table.get(K, False) else 0


# ───────── Calcul de la « torsion de Reidemeister » approximée ─────────
def reidemeister_proxy(K):
    """Proxy de la torsion de Reidemeister (heuristique).
    Pour SU(2)_k au connexion triviale : τ_R(0) ~ det(K) / (some factor)
    """
    return abs(det_knot(K))


# ───────── Construction de c_1(K) selon le modèle Lawrence-Rozansky ─────────
def c1_LR_model(K, A_sig=0.05, B_cass=0.005, C_tor=0.005):
    """Modèle c_1 = A·σ + B·λ + C·log(τ_R)."""
    sig = SIGNATURE.get(K, 0)
    cass = CASSON.get(K, 0)
    tor = reidemeister_proxy(K)
    return A_sig * sig + B_cass * cass + C_tor * np.log(tor)


# ───────── Données empiriques de l'ansatz A5 (k=8) ─────────
C1_EMP = {
    "3_1": +0.12917,
    "4_1": -0.00684,
    "5_1": +0.02309,
    "5_2": +0.23891,
    "6_1": -0.02906,
    "6_2": -0.00911,
    "6_3": -0.11126,
    "7_1": -0.06822,
    "7_2": -0.05940,
}


# ───────── Fit du modèle Lawrence-Rozansky aux données empiriques ─────────
from scipy.optimize import minimize


def fit_LR(c1_data):
    """Fit A_sig, B_cass, C_tor pour reproduire c_1 empirique."""
    knots = list(c1_data.keys())
    target = np.array([c1_data[K] for K in knots])

    def chi2(params):
        A, B, C = params
        pred = np.array([
            A * SIGNATURE.get(K, 0)
            + B * CASSON.get(K, 0)
            + C * np.log(reidemeister_proxy(K))
            for K in knots
        ])
        return np.sum((pred - target)**2)
    res = minimize(chi2, [0.05, 0.005, 0.005], method='Nelder-Mead')
    return res.x, np.sqrt(res.fun / len(knots))


print("═"*90)
print("  Calcul 1-loop systématique — invariants topologiques")
print("═"*90)

print(f"\n  {'K':<5} {'σ':>4} {'λ Casson':>10} {'log(det)':>10} {'c_1 emp':>10}")
print("  " + "-" * 55)
for K, c in C1_EMP.items():
    print(f"  {K:<5} {SIGNATURE.get(K, 0):>+4} {CASSON.get(K, 0):>+10} "
          f"{np.log(reidemeister_proxy(K)):>10.3f} {c:>+10.4f}")


(A, B, C), rms_fit = fit_LR(C1_EMP)
print(f"\n  Fit Lawrence-Rozansky : c_1 ≈ A·σ + B·λ + C·log(det)")
print(f"    A_sig = {A:+.5f}")
print(f"    B_cass = {B:+.5f}")
print(f"    C_tor = {C:+.5f}")
print(f"    RMS fit = {rms_fit:.5f}")


print(f"\n  Prédictions LR vs empirique :")
print(f"  {'K':<5} {'c_1 empirique':>14} {'c_1 LR prédit':>15} {'résidu':>10}")
print("  " + "-" * 55)
for K, c_emp in C1_EMP.items():
    c_pred = A * SIGNATURE.get(K, 0) + B * CASSON.get(K, 0) \
              + C * np.log(reidemeister_proxy(K))
    residu = c_pred - c_emp
    flag = "✓" if abs(residu) < 0.05 else ("·" if abs(residu) < 0.1 else "✗")
    print(f"  {K:<5} {c_emp:>+14.4f} {c_pred:>+15.4f} {residu:>+9.4f} {flag}")


# ───────── Test avec une caractéristique additionnelle : amphichiralité ─────────
print("\n" + "═"*90)
print("  Test étendu : ajout d'amphichiralité (σ = 0)")
print("═"*90)


def fit_LR_extended(c1_data):
    knots = list(c1_data.keys())
    target = np.array([c1_data[K] for K in knots])

    def chi2(params):
        A, B, C, D = params
        pred = np.array([
            A * SIGNATURE.get(K, 0)
            + B * CASSON.get(K, 0)
            + C * np.log(reidemeister_proxy(K))
            + D * is_amphichiral(K)
            for K in knots
        ])
        return np.sum((pred - target)**2)
    res = minimize(chi2, [0.05, 0.005, 0.005, 0], method='Nelder-Mead')
    return res.x, np.sqrt(res.fun / len(knots))


(A2, B2, C2, D2), rms_ext = fit_LR_extended(C1_EMP)
print(f"\n  Fit étendu : c_1 ≈ A·σ + B·λ + C·log(det) + D·amphichiral")
print(f"    A_sig = {A2:+.5f}")
print(f"    B_cass = {B2:+.5f}")
print(f"    C_tor = {C2:+.5f}")
print(f"    D_amph = {D2:+.5f}")
print(f"    RMS étendu = {rms_ext:.5f}")


# ───────── Prédiction pour les nœuds NON inclus dans le fit ─────────
print("\n" + "═"*90)
print("  Prédictions LR pour les nœuds NON utilisés dans le fit")
print("═"*90)
print(f"\n  {'K':<5} {'σ':>4} {'λ':>5} {'log(det)':>10} {'c_1 LR prédit':>15}")
print("  " + "-" * 55)
not_fit = [K for K in SIGNATURE if K not in C1_EMP]
for K in not_fit:
    c_LR = A * SIGNATURE.get(K, 0) + B * CASSON.get(K, 0) \
            + C * np.log(reidemeister_proxy(K))
    print(f"  {K:<5} {SIGNATURE.get(K, 0):>+4} {CASSON.get(K, 0):>+5} "
          f"{np.log(reidemeister_proxy(K)):>10.3f} {c_LR:>+15.4f}")


# ───────── Synthèse ─────────
print("\n" + "═"*90)
print("  SYNTHÈSE")
print("═"*90)
print(f"""
  RMS du modèle Lawrence-Rozansky : {rms_fit:.4f}
  RMS du modèle étendu (avec amphichiralité) : {rms_ext:.4f}

  Si RMS < 0.05 : modèle LR essentiellement validé sur ces 9 nœuds
  Si RMS ∈ [0.05, 0.15] : direction correcte, raffinement nécessaire
  Si RMS > 0.15 : invariants topologiques choisis insuffisants

  Coefficients identifiés :
    A_sig (signature)  : {A:+.5f}  — interprétation : pondération de σ
    B_cass (Casson)    : {B:+.5f}  — pondération de λ
    C_tor (Reidemeister) : {C:+.5f}  — pondération de log(τ_R)

  Si A, B, C correspondent à des constantes Lawrence-Rozansky connues
  (impliquant 1/(k+2) = 1/10 pour k=8), on a une dérivation.

  Étapes restantes pour un vrai 1-loop :
    1. Calculer σ(K), λ(K), τ_R(K) RIGOUREUSEMENT pour les 19 nœuds
    2. Comparer A, B, C aux formules de Lawrence-Rozansky exactes
    3. Étendre aux liens multi-composantes (entrelacs)
""")
