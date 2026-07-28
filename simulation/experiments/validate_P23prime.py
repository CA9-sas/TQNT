r"""
VALIDATION P23' (plasticité de confinement) sur m_u, m_d, m_s, m_c, m_b.

Postulat à valider :
  m_eff(K_i | L) = Ω_ts(K_i) · Π_C(K_i | L)
  Π_C(K_i | L) = exp(α_C · ξ_C^(i)(L))
  ξ_C^(i)(L) = Σ_{j≠i} (b_i b_j √(U_i U_j)) / (Σ_k U_k)²

Identification quark = nœud confiné :
  u, d : 3_1 confinés dans le proton (3_1 + 7_2 + 7_2)
  s    : 5_1 confiné dans un hadron étrange à déterminer
  c    : 7_1 confiné dans charmonium (η_c ou J/ψ)
  b    : 9_1 confiné dans bottomonium (η_b ou Υ)

Programme :
1. Précalcul des invariants topologiques U_K, b_K pour chaque nœud
2. Choix des entrelacs hôtes L (avec justification basée sur catalogue V5)
3. Calcul de ξ_C^(i)(L) pour chaque assignation quark
4. Calibration de α_C sur une donnée (typiquement m_d, le plus précis)
5. Prédiction des 4 autres masses
6. Comparaison aux valeurs PDG MS-bar à μ=2 GeV
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


# ─── Constantes globales TQNT-V5 ───
M_E = 0.51099895  # MeV
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

# ─── Nœuds du catalogue ───
KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}

print("Précalcul des invariants topologiques (Frenet + Jones)…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_K = float(np.sum(g["kappa"]**2 * ds))
    U_tau = float(np.sum(g["tau"]**2 * ds))
    U_KT[K] = (U_K, U_tau)
    pj = jones_polynomial(K)
    TOPO[K] = {
        "cr": int(K.split("_")[0]),
        "n_l": N_LOBES[K],
        "b": braid_index(K),
        "det": jones_determinant(pj),
        "U": U_K * U_tau,
    }

P_31 = U_KT["3_1"][0] * U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"] * TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"] * M_C + TOPO[K]["n_l"] * M_L for K in KNOTS}


def omega_ts(comps):
    """Ω_ts canonique : masse d'un entrelacs L = {K_1,...,K_n}."""
    n = len(comps)
    ws = [M_NAIVE[K] * (TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2:
            Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


def xi_C_component(K_i, L):
    """ξ_C^(i)(L) = Σ_{j≠i} (b_i b_j √(U_i U_j)) / (Σ_k U_k)²

    K_i désigne la composante focale ; L est la liste des composantes (multiset)
    où K_i apparaît au moins une fois. La somme est faite sur les autres
    composantes (donc on retire UNE occurrence de K_i de L).
    """
    L_others = list(L)
    if K_i not in L_others:
        raise ValueError(f"{K_i} pas dans L = {L}")
    L_others.remove(K_i)

    b_i = TOPO[K_i]["b"]
    U_i = TOPO[K_i]["U"]
    sum_U = sum(TOPO[K]["U"] for K in L)
    sum_U_sq = sum_U ** 2

    xi = 0.0
    for K_j in L_others:
        b_j = TOPO[K_j]["b"]
        U_j = TOPO[K_j]["U"]
        xi += (b_i * b_j * np.sqrt(U_i * U_j)) / sum_U_sq
    return xi


def m_eff(K_i, L, alpha_C):
    """Masse effective de K_i confiné dans L via P23'."""
    m_can = omega_ts([K_i])  # canonique du nœud isolé
    xi = xi_C_component(K_i, L)
    Pi_C = np.exp(alpha_C * xi)
    return m_can * Pi_C, xi, Pi_C, m_can


# ─── Hôtes pour chaque quark ───
# Choix basés sur catalogue V5 et structure baryonique/mésonique
# - Nucléon : 3_1 + 7_2 + 7_2 (canonique V5)
# - Lambda  : 3_1 + 4_1 + 7_2 (plast V5, contient un s-quark)
# - η_c(1S) : 2 charmonia ; doit contenir le nœud-c
# - η_b(1S) : 2 bottomonia ; doit contenir le nœud-b

HOSTS = {
    "u": {
        "knot": "3_1",
        "host": ["3_1", "7_2", "7_2"],  # proton
        "host_label": "proton",
        "m_PDG": 2.16,  # MeV, MS-bar à μ=2 GeV
        "m_PDG_low": 1.67, "m_PDG_high": 2.81,
    },
    "d": {
        "knot": "3_1",
        "host": ["3_1", "7_2", "7_2"],  # proton (même hôte que u par défaut)
        "host_label": "proton",
        "m_PDG": 4.67,
        "m_PDG_low": 4.37, "m_PDG_high": 5.21,
    },
    "s": {
        "knot": "5_1",
        "host": ["3_1", "5_1", "7_2"],  # Λ baryon plausible
        "host_label": "Λ-like (3_1+5_1+7_2)",
        "m_PDG": 93.4,
        "m_PDG_low": 84, "m_PDG_high": 104,
    },
    "c": {
        "knot": "7_1",
        "host": ["7_1", "7_1"],  # charmonium η_c
        "host_label": "η_c-like (7_1+7_1)",
        "m_PDG": 1270.0,
        "m_PDG_low": 1230, "m_PDG_high": 1310,
    },
    "b": {
        "knot": "9_1",
        "host": ["9_1", "9_1"],  # bottomonium η_b
        "host_label": "η_b-like (9_1+9_1)",
        "m_PDG": 4180.0,
        "m_PDG_low": 4090, "m_PDG_high": 4280,
    },
}


# ═══════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" CALCULS DES INVARIANTS DE CONFINEMENT  ξ_C^(i)(L)")
print("═"*78)
print(f"{'quark':>5} {'nœud':>5} {'hôte':<24} {'U_K':>7} {'b_K':>3} "
      f"{'ξ_C':>9} {'m_can':>8}")
print("─"*78)

for q, data in HOSTS.items():
    K = data["knot"]
    L = data["host"]
    m_can = omega_ts([K])
    xi = xi_C_component(K, L)
    print(f"{q:>5} {K:>5} {data['host_label']:<24} {TOPO[K]['U']:>7.2f} "
          f"{TOPO[K]['b']:>3} {xi:>9.5f} {m_can:>8.3f}")


# ═══════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" CALIBRATION : test de plusieurs valeurs de α_C")
print("═"*78)

# Calibrer sur m_d (mieux mesuré que m_u, et 3_1 confiné dans proton)
data_d = HOSTS["d"]
K_d = data_d["knot"]
L_d = data_d["host"]
m_can_d = omega_ts([K_d])
xi_d = xi_C_component(K_d, L_d)
target_d = data_d["m_PDG"]

alpha_C_fit = np.log(target_d / m_can_d) / xi_d
print(f"\n  Calibration sur m_d = {target_d} MeV (3_1 dans {data_d['host_label']})")
print(f"  ξ_C^(31)(proton) = {xi_d:.5f}")
print(f"  m_can(3_1) = {m_can_d:.3f} MeV")
print(f"  → α_C = ln({target_d}/{m_can_d:.3f}) / {xi_d:.5f} = {alpha_C_fit:.3f}")


# ═══════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(f" PRÉDICTIONS avec α_C = {alpha_C_fit:.3f} (calibré sur m_d)")
print("═"*78)
print(f"{'quark':>5} {'nœud':>5} {'hôte':<26} {'m_PDG':>10} {'m_eff':>10} "
      f"{'Δ %':>8} {'Π_C':>7}")
print("─"*78)

results = []
for q, data in HOSTS.items():
    K = data["knot"]; L = data["host"]
    m, xi, PiC, m_can = m_eff(K, L, alpha_C_fit)
    delta = (m - data["m_PDG"]) / data["m_PDG"] * 100
    results.append((q, K, data["host_label"], data["m_PDG"], m, delta, PiC))
    flag = "✓" if abs(delta) < 30 else ("~" if abs(delta) < 100 else "✗")
    print(f"{q:>5} {K:>5} {data['host_label']:<26} {data['m_PDG']:>10.2f} "
          f"{m:>10.2f} {delta:>+7.1f}% {PiC:>7.2f} {flag}")


# ═══════════════════════════════════════════════════════════
# Calibration alternative : minimiser le RMS log sur 4 saveurs (sans u)
# pour évaluer la qualité multi-saveurs
print("\n" + "═"*78)
print(" CALIBRATION GLOBALE : α_C minimisant le RMS log sur d, s, c, b")
print("═"*78)


def rms_log_residual(alpha_C, quarks=("d", "s", "c", "b")):
    log_resid = []
    for q in quarks:
        data = HOSTS[q]
        m, _, _, _ = m_eff(data["knot"], data["host"], alpha_C)
        log_resid.append(np.log10(m / data["m_PDG"]))
    return np.sqrt(np.mean(np.array(log_resid)**2))


# Scan α_C ∈ [1, 30]
from scipy.optimize import minimize_scalar
res = minimize_scalar(rms_log_residual, bounds=(1, 50), method="bounded")
alpha_C_global = res.x
rms_global = res.fun

print(f"\n  Optimum global : α_C = {alpha_C_global:.3f}")
print(f"  RMS log_10 sur d, s, c, b : {rms_global:.3f} (soit ×{10**rms_global:.2f}-divergence)")
print()

print(f"{'quark':>5} {'nœud':>5} {'hôte':<26} {'m_PDG':>10} {'m_eff':>10} "
      f"{'Δ %':>8} {'log Δ':>7}")
print("─"*78)
for q, data in HOSTS.items():
    K = data["knot"]; L = data["host"]
    m, xi, PiC, m_can = m_eff(K, L, alpha_C_global)
    delta = (m - data["m_PDG"]) / data["m_PDG"] * 100
    log_delta = np.log10(m / data["m_PDG"])
    flag = "✓" if abs(delta) < 30 else ("~" if abs(delta) < 100 else "✗")
    print(f"{q:>5} {K:>5} {data['host_label']:<26} {data['m_PDG']:>10.2f} "
          f"{m:>10.2f} {delta:>+7.1f}% {log_delta:>+7.3f} {flag}")


# ═══════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" VERDICT")
print("═"*78)
n_pass = sum(1 for r in results if abs(r[5]) < 30)
n_partial = sum(1 for r in results if 30 <= abs(r[5]) < 100)
n_fail = sum(1 for r in results if abs(r[5]) >= 100)
print(f"\n  Avec α_C calibré sur m_d :")
print(f"    ✓ écart < 30%   : {n_pass}/5")
print(f"    ~ 30-100%       : {n_partial}/5")
print(f"    ✗ écart ≥ 100% : {n_fail}/5")
print(f"\n  RMS log_10 optimal multi-saveurs : {rms_global:.3f}")
if rms_global < 0.3:
    print("  ★ Validation partielle de P23' (RMS < 0.3 décade ≈ ×2 d'écart).")
elif rms_global < 0.5:
    print("  ⊙ Validation faible de P23' (RMS < 0.5 décade ≈ ×3 d'écart).")
elif rms_global < 1.0:
    print("  ⚠ Validation négative : Ansatz ξ_C insuffisant à seul paramètre.")
else:
    print("  ✗ ÉCHEC : la forme exponentielle simple ne capture pas la hiérarchie.")


# ═══════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" ANSATZ ALTERNATIF : Π_C en loi de puissance (multi-décade)")
print("═"*78)

# Π_C = (Σ U_j / U_i)^p  (share of geometric energy)
# C'est plus naturel pour les hiérarchies > 3 décades.

def m_eff_power(K_i, L, p):
    m_can = omega_ts([K_i])
    U_i = TOPO[K_i]["U"]
    sum_U = sum(TOPO[K]["U"] for K in L)
    ratio = sum_U / U_i
    return m_can * (ratio ** p)


def rms_log_power(p, quarks=("d", "s", "c", "b")):
    log_resid = []
    for q in quarks:
        data = HOSTS[q]
        m = m_eff_power(data["knot"], data["host"], p)
        log_resid.append(np.log10(m / data["m_PDG"]))
    return np.sqrt(np.mean(np.array(log_resid)**2))


res = minimize_scalar(rms_log_power, bounds=(0.1, 10), method="bounded")
p_opt = res.x
print(f"\n  Optimum global : p = {p_opt:.3f}")
print(f"  RMS log_10 sur d, s, c, b : {res.fun:.3f}")
print()

print(f"{'quark':>5} {'nœud':>5} {'hôte':<26} {'m_PDG':>10} {'m_eff':>10} "
      f"{'Δ %':>8} {'log Δ':>7}")
print("─"*78)
for q, data in HOSTS.items():
    K = data["knot"]; L = data["host"]
    m = m_eff_power(K, L, p_opt)
    delta = (m - data["m_PDG"]) / data["m_PDG"] * 100
    log_delta = np.log10(m / data["m_PDG"])
    flag = "✓" if abs(delta) < 30 else ("~" if abs(delta) < 100 else "✗")
    print(f"{q:>5} {K:>5} {data['host_label']:<26} {data['m_PDG']:>10.2f} "
          f"{m:>10.2f} {delta:>+7.1f}% {log_delta:>+7.3f} {flag}")


# Maintenant essayons un Ansatz inclluant le nombre de croisements
# Π_C = (cr_total / cr_i)^p · (b_total / b_i)^q

def m_eff_struct(K_i, L, p, q):
    m_can = omega_ts([K_i])
    cr_i = TOPO[K_i]["cr"]; b_i = TOPO[K_i]["b"]
    cr_tot = sum(TOPO[K]["cr"] for K in L)
    b_tot = sum(TOPO[K]["b"] for K in L)
    return m_can * ((cr_tot/cr_i)**p) * ((b_tot/b_i)**q)


from scipy.optimize import minimize


def rms_log_struct(params, quarks=("d", "s", "c", "b")):
    p, q = params
    log_resid = []
    for qk in quarks:
        data = HOSTS[qk]
        m = m_eff_struct(data["knot"], data["host"], p, q)
        log_resid.append(np.log10(m / data["m_PDG"]))
    return np.sqrt(np.mean(np.array(log_resid)**2))


print("\n" + "═"*78)
print(" ANSATZ TRIPARAMÉTRIQUE : (cr_total/cr_i)^p · (b_total/b_i)^q")
print("═"*78)

res = minimize(rms_log_struct, [1.0, 0.5], method="Nelder-Mead")
p_opt, q_opt = res.x
print(f"\n  Optimum : p = {p_opt:.3f}, q = {q_opt:.3f}")
print(f"  RMS log_10 sur d, s, c, b : {res.fun:.3f}")
print()
print(f"{'quark':>5} {'nœud':>5} {'hôte':<26} {'m_PDG':>10} {'m_eff':>10} "
      f"{'Δ %':>8} {'log Δ':>7}")
print("─"*78)
for qk, data in HOSTS.items():
    K = data["knot"]; L = data["host"]
    m = m_eff_struct(K, L, p_opt, q_opt)
    delta = (m - data["m_PDG"]) / data["m_PDG"] * 100
    log_delta = np.log10(m / data["m_PDG"])
    flag = "✓" if abs(delta) < 30 else ("~" if abs(delta) < 100 else "✗")
    print(f"{qk:>5} {K:>5} {data['host_label']:<26} {data['m_PDG']:>10.2f} "
          f"{m:>10.2f} {delta:>+7.1f}% {log_delta:>+7.3f} {flag}")
