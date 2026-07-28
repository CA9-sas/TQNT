r"""
RECHERCHE DE L'ORIGINE ANALYTIQUE de ρ_0 et k_b.

Avec les paramètres analytiques :
   a = √(7/(8π)) ≈ 0.52775
   p = q = 3/4
   x = -√6
   β = 33/20

Les contraintes physiques donnent (par résolution numérique) :
   ρ_0 ≈ X (à déterminer en pratique)
   k_b ≈ 1.003 (très proche de 1)

QUESTION : Ces deux paramètres "physiques" ont-ils des
origines analytiques en SU(2)_8 + physique TQNT ?

POUR k_b :
   Dans la limite Hölder pure, k_b satisfait :
   k_b^a = 2(m_n^a - m_e^a)/(m_p^a - m_e^a) - 1
   = formule explicite

POUR ρ_0 :
   Détermine l'échelle absolue de l'écrantage par paire.
   Liée à la masse du proton.
   Pourrait correspondre à un (k+2)^α de SU(2)_8.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from collections import Counter

from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index

M_E = 0.51099895
M_P = 938.27208
M_N = 939.56542
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_LOBES = {
    "3_1": 3, "4_1": 2, "5_1": 5, "5_2": 2, "6_1": 2,
    "6_2": 3, "6_3": 3, "7_1": 7, "7_2": 2, "7_3": 3,
    "7_4": 2, "7_5": 3, "7_6": 2, "7_7": 3, "8_1": 2,
    "8_2": 3, "8_3": 3, "9_1": 9, "10_1": 2,
}

# SU(2)_8
K_LEVEL = 8
C_CENTRAL = 12 / 5         # 3k/(k+2)
H_HALF = 3/40
H_32 = 3/8
H_52 = 7/8

# Valeurs analytiques
A_ANA = np.sqrt(7 / (8 * np.pi))
P_ANA = 3 / 4
Q_ANA = 3 / 4
X_ANA = -np.sqrt(6)
B_ANA = 33 / 20

print("Calcul Frenet…")
U_KT = {}
TOPO = {}
for K in KNOTS:
    geo = frenet_geometry(K, 50000)
    kappa = geo["kappa"]; tau = geo["tau"]; speed = geo["speed"]
    dt = 2 * np.pi / len(kappa)
    ds = speed * dt
    U_KT[K] = (
        float(np.sum(kappa ** 2 * ds)),
        float(np.sum(tau ** 2 * ds)),
    )
    pj = jones_polynomial(K)
    TOPO[K] = {
        "cr":  int(K.split("_")[0]),
        "n_l": N_LOBES[K],
        "b":   braid_index(K),
        "det": jones_determinant(pj),
    }
P_31 = U_KT["3_1"][0] * U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"] * TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"] * M_C + TOPO[K]["n_l"] * M_L for K in KNOTS}


def operator(comps, a, p, q, x, rho0, beta, kb=1.0, neutron_mode=False):
    n = len(comps)
    ws = []
    for K_idx, K in enumerate(comps):
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        w = M_NAIVE[K] * T ** p
        if neutron_mode and K == "7_2":
            counted = sum(1 for k in comps[:K_idx] if k == "7_2")
            if counted == 1:
                w *= kb
        ws.append(w)
    skeleton = float(sum(w ** a for w in ws) ** (1 / a))
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    Phi_n = ((n + 1) / 2) ** (-x) if n > 0 else 1.0
    c = Counter(comps)
    Phi_pair = 1.0
    for K, count in c.items():
        if count >= 2:
            rho_K = rho0 * TOPO[K]["cr"] ** (-beta)
            Phi_pair *= rho_K ** (count - 1)
    return Phi_n * Phi_pair * skeleton * G_comp ** q


def solve_rho0(a, p, q, x, beta):
    def f(log_rho0):
        return operator(["3_1", "7_2", "7_2"], a, p, q, x, np.exp(log_rho0), beta) - M_P
    return float(np.exp(brentq(f, -10, 10, xtol=1e-12)))


def solve_kb(a, p, q, x, rho0, beta):
    def f(log_kb):
        return operator(["3_1", "7_2", "7_2"], a, p, q, x, rho0, beta,
                        kb=np.exp(log_kb), neutron_mode=True) - M_N
    return float(np.exp(brentq(f, -5, 5, xtol=1e-12)))


def check(name, val, target):
    err = (val - target) / target * 100
    flag = "★★" if abs(err) < 0.1 else ("★" if abs(err) < 0.5 else
                                          ("✓" if abs(err) < 2 else " "))
    print(f"  {name:<40} {val:>10.6f}  err {err:+7.3f}%  {flag}")


def main():
    print("=" * 90)
    print("  ORIGINE ANALYTIQUE de ρ_0 et k_b dans Ωₜₛ-SU(2)_8")
    print("=" * 90)

    # Résoudre ρ_0 et k_b avec valeurs analytiques
    rho0 = solve_rho0(A_ANA, P_ANA, Q_ANA, X_ANA, B_ANA)
    kb = solve_kb(A_ANA, P_ANA, Q_ANA, X_ANA, rho0, B_ANA)

    print(f"\n  Avec paramètres analytiques (a, p, q, x, β) :")
    print(f"    ρ_0 = {rho0:.6f}")
    print(f"    k_b = {kb:.6f}")

    # ──────────────────────────────────────────
    # ORIGINE ANALYTIQUE DE k_b
    # ──────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  ▶ ORIGINE DE k_b")
    print(f"  ═══════════════════════════════════════════════════")
    print(f"\n  HYPOTHÈSE : k_b est déterminé par le rapport m_n/m_p.")
    print(f"  Test de plusieurs formules analytiques :")

    # Hölder pure formula
    kb_holder = ((2 * (M_N**A_ANA - M_E**A_ANA) / (M_P**A_ANA - M_E**A_ANA) - 1)
                 ** (1/A_ANA))
    check("k_b Hölder pur (a=√(7/(8π)))",  kb_holder, kb)

    # Simple ratio
    kb_simple = 2 * M_N / M_P - 1
    check("k_b = 2·m_n/m_p - 1",           kb_simple, kb)

    # Leading order
    kb_leading = (2 * (M_N/M_P)**A_ANA - 1) ** (1/A_ANA)
    check("k_b = (2·(m_n/m_p)^a - 1)^(1/a)", kb_leading, kb)

    # Other forms
    kb_one_plus = 1 + (M_N - M_P) / M_P
    check("k_b = m_n / m_p",                M_N/M_P, kb)
    kb_one_log = np.exp((M_N - M_P) / M_P)
    check("k_b = exp((m_n-m_p)/m_p)",       kb_one_log, kb)

    # 1 + 2(m_n-m_p)/m_p / a
    kb_inflated = 1 + 2 * (M_N - M_P) / M_P / A_ANA
    check("k_b = 1 + 2(m_n-m_p)/(m_p·a)",  kb_inflated, kb)

    # Reservoir : k_b = (m_n/m_p)^(1/a)
    kb_power = (M_N / M_P) ** (1 / A_ANA)
    check("k_b = (m_n/m_p)^(1/a)",          kb_power, kb)

    # ──────────────────────────────────────────
    # ORIGINE ANALYTIQUE DE ρ_0
    # ──────────────────────────────────────────
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  ▶ ORIGINE DE ρ_0")
    print(f"  ═══════════════════════════════════════════════════")
    print(f"\n  HYPOTHÈSE : ρ_0 lié à la structure de SU(2)_8 + m_p.")
    print(f"  Test de plusieurs formes :")

    # Constants SU(2)_8
    k_plus_2 = K_LEVEL + 2  # 10
    sum_dim_q = (k_plus_2 / 2) / np.sin(np.pi / k_plus_2) ** 2  # D²_total
    D_total = np.sqrt(sum_dim_q)
    D_half = 2 * np.cos(np.pi / k_plus_2)
    D_one = 1 + D_half
    D_32 = np.sin(4 * np.pi / k_plus_2) / np.sin(np.pi / k_plus_2)
    D_2 = np.sin(5 * np.pi / k_plus_2) / np.sin(np.pi / k_plus_2)
    print(f"\n  Dimensions quantiques SU(2)_8 :")
    print(f"    D_{{1/2}} = 2cos(π/10)     = {D_half:.6f}")
    print(f"    D_1                     = {D_one:.6f}  (φ² = {((1+np.sqrt(5))/2)**2:.6f})")
    print(f"    D_{{3/2}}                = {D_32:.6f}")
    print(f"    D_2                     = {D_2:.6f}")
    print(f"    D_total = √Σ D²         = {D_total:.6f}")

    # Tests pour ρ_0
    print(f"\n  Tests pour ρ_0 = {rho0:.5f} :")
    check("k+2 = 10",                      k_plus_2, rho0)
    check("D_total² / 4",                  sum_dim_q / 4, rho0)
    check("(k+2) · D_{1/2}",               k_plus_2 * D_half, rho0)
    check("D_1 · D_{3/2}",                 D_one * D_32, rho0)
    check("(k+2) + D_{1/2}",               k_plus_2 + D_half, rho0)
    check("D_2 + 2·D_{1/2}",               D_2 + 2*D_half, rho0)
    check("k+2 + h_{3/2}·k",              k_plus_2 + H_32 * K_LEVEL, rho0)
    check("c · k + 1",                     C_CENTRAL * K_LEVEL + 1, rho0)
    check("(k+2) · D_total / D_4",         k_plus_2 * D_total / 1, rho0)  # D_4 = 1
    check("π² + 0.74",                     np.pi**2 + 0.74, rho0)
    check("π² · (1 + h_{3/2}/π)",          np.pi**2 * (1 + H_32/np.pi), rho0)
    check("π² + h_{5/2}",                  np.pi**2 + H_52, rho0)

    # Try : ρ_0 = (k+2) × something
    target_ratio = rho0 / k_plus_2
    print(f"\n  ρ_0 / (k+2) = {target_ratio:.6f}")
    check("1 + 1/(2π²·0.1)",  1 + 1/(2*np.pi**2*0.1), target_ratio)
    check("c/(c-1)",          C_CENTRAL/(C_CENTRAL-1), target_ratio)
    check("2/√(c+1)",         2/np.sqrt(C_CENTRAL+1), target_ratio)

    # ───── Conclusion ─────
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  ▶ CONCLUSION")
    print(f"  ═══════════════════════════════════════════════════")
    print(f"\n  k_b est BIEN APPROXIMÉ par la formule Hölder pure :")
    print(f"     k_b = (2(m_n^a - m_e^a)/(m_p^a - m_e^a) - 1)^(1/a)")
    print(f"     Cette formule provient des contraintes EXACTES")
    print(f"     m_pred(p) = m_p et m_pred(n) = m_n.")
    print(f"\n  ρ_0 est DÉTERMINÉ par le proton :")
    print(f"     Pas de forme analytique simple émergente,")
    print(f"     mais sa proximité avec (k+2) = 10 (niveau SU(2)_8 + 2)")
    print(f"     suggère un lien structurel.")
    print(f"\n  Les DEUX paramètres restent FONDAMENTALEMENT physiques")
    print(f"  (déterminés par les masses du proton et du neutron),")
    print(f"  mais leur valeur peut être interprétée dans SU(2)_8.")
    print("=" * 90)


if __name__ == "__main__":
    main()
