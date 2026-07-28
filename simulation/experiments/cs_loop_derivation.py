r"""
PROGRAMME ANALYTIQUE CS — étape 1 : extraction du terme 1-loop pour chaque nœud.

Pour chaque nœud K du catalogue où le polynôme de Jones coloré est connu
(0_1, 3_1, 4_1, 5_1, 7_1, 9_1), on calcule J_n(K; q) à q = exp(2πi/(k+2))
pour plusieurs valeurs de k, puis on EXTRAIT analytiquement la
décomposition :

  m_K(k) = m_K^(0) + c_1(K)/(k+2) + c_2(K)/(k+2)^2 + ...

où m_K(k) est défini comme -log|J_n(K; q)| / log(k+2) (à normalisation
près).

L'extraction de c_1(K) se fait par régression sur la série en 1/(k+2),
puis comparaison à c_1 empirique de l'ansatz A5 précédent.

Si la concordance est numérique, c_1 est analytiquement dérivable
depuis Chern-Simons. Sinon, on identifie le décalage.
"""
from __future__ import annotations
import numpy as np
from theory.chern_simons import colored_jones, cs_q, COLORED_JONES_KNOTS


print("═"*90)
print("  Programme analytique CS : extraction de c_1(K) pour chaque nœud")
print("═"*90)

# Nœuds avec colored Jones implémenté
KNOTS_AVAIL = ["3_1", "4_1", "5_1", "7_1", "9_1"]
print(f"\n  Nœuds disponibles (Jones coloré): {KNOTS_AVAIL}")

# Calculer m_K(k) pour plusieurs k pour chaque nœud
def m_CS(knot, n, k):
    """Masse Chern-Simons effective : -log|J_n(K; q=e^{2πi/(k+2)})|."""
    q = cs_q(k)
    J = colored_jones(knot, n, q)
    if abs(J) < 1e-15:
        return float('nan')
    return -np.log(abs(J))


# Niveau k variable
K_grid = [5, 8, 10, 15, 20, 30, 50, 100, 200]
print(f"\n  Niveaux testés : {K_grid}")
print(f"  Représentation : n = 2 (fundamentale SU(2), fermions)")


print(f"\n  m_CS(K, k) pour n = 2 :")
print(f"  {'K':<5} " + " ".join(f"k={k:<5}" for k in K_grid))
print("  " + "-" * 90)

m_data = {}
for K in KNOTS_AVAIL:
    m_K = []
    for k in K_grid:
        try:
            m = m_CS(K, 2, k)
        except Exception as e:
            m = float("nan")
        m_K.append(m)
    m_data[K] = m_K
    print(f"  {K:<5} " + " ".join(f"{m:>6.3f}" for m in m_K))


# ───────── Extraction du développement en boucle ─────────
print("\n" + "═"*90)
print("  Extraction du développement : m(k) = m_0 + c_1/(k+2) + c_2/(k+2)^2 + ...")
print("═"*90)


def fit_loop_expansion(k_arr, m_arr, n_terms=3):
    """Fit m = a_0 + a_1·x + a_2·x^2 + ... avec x = 1/(k+2)."""
    x = 1.0 / (np.array(k_arr) + 2)
    valid = ~np.isnan(m_arr)
    if valid.sum() < n_terms + 1:
        return None
    # Polyfit en x
    coeffs = np.polyfit(x[valid], np.array(m_arr)[valid], n_terms)
    # coeffs en ordre décroissant : [a_n, ..., a_1, a_0]
    coeffs_asc = coeffs[::-1]
    return coeffs_asc  # [a_0, a_1, a_2, ...]


print(f"\n  Fit polynomial en 1/(k+2) à 3 termes :")
print(f"  {'K':<5} {'m_0 (LO)':>10} {'c_1 (NLO)':>12} {'c_2 (NNLO)':>12}")
print("  " + "-" * 50)

c1_CS = {}
for K in KNOTS_AVAIL:
    m_arr = np.array(m_data[K])
    coeffs = fit_loop_expansion(K_grid, m_arr, n_terms=3)
    if coeffs is None:
        print(f"  {K:<5} (fit échoué)")
        continue
    m_0, c_1, c_2, c_3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    c1_CS[K] = c_1
    print(f"  {K:<5} {m_0:>+10.4f} {c_1:>+12.4f} {c_2:>+12.4f}")


# ───────── Comparaison aux c_1 empiriques (ansatz A5) ─────────
print("\n" + "═"*90)
print("  Comparaison aux c_1 EMPIRIQUES (ansatz A5 à k=8)")
print("═"*90)

# Valeurs c_1 empiriques de l'expérience précédente
c1_emp = {
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

print(f"\n  {'K':<5} {'c_1 (CS calculé)':>17} {'c_1 (empirique A5)':>20} "
      f"{'ratio':>10}")
print("  " + "-" * 65)
for K in KNOTS_AVAIL:
    if K not in c1_CS or K not in c1_emp:
        continue
    c_calc = c1_CS[K]
    c_emp = c1_emp[K]
    ratio = c_calc / c_emp if abs(c_emp) > 1e-10 else float('inf')
    print(f"  {K:<5} {c_calc:>+17.4f} {c_emp:>+20.4f} {ratio:>+9.3f}")


# ───────── Diagnostic ─────────
print("\n" + "═"*90)
print("  DIAGNOSTIC")
print("═"*90)

print(f"""
  Le développement m_CS(k) = m_0 + c_1/(k+2) + c_2/(k+2)^2 + ...
  donne pour chaque nœud des coefficients c_1 numériquement dérivés.

  Si c_1_CS et c_1_emp sont du MEME SIGNE et proportionnels par une
  constante universelle (rapport ~ constant), alors :
    → La structure CS 1-loop EST l'origine du résidu Ω_ts
    → La normalisation (rapport) donne la conversion entre m_CS et m_phys

  Si les rapports varient sauvagement, c_1 n'est pas une simple correction
  CS au sens où nous l'avons modélisé.

  Note importante : m_CS = -log|J_n| n'est pas exactement la masse
  physique TQNT. La conversion implique des facteurs normaux (α_0,
  conventions de signe, etc.). Une vraie validation nécessiterait
  d'ajuster ces facteurs pour TOUTE la formule Ω_ts, pas seulement
  le résidu.

  Étape suivante :
    Pour chaque nœud disponible (3_1, 4_1, 5_1, 7_1, 9_1), comparer
    m_CS(k=8) à m_pred Ω_ts du même nœud isolé (m_e pour 3_1=e^-,
    etc.). Si correspondance, on a un dictionnaire CS ↔ TQNT.
""")


# ───────── Test additionnel : m_CS vs U_EM ─────────
print("\n" + "═"*90)
print("  Test additionnel : m_CS(K, k=8) vs U_EM(K)")
print("═"*90)

# Computer U_EM = U_kappa + U_tau pour comparaison
from theory.abraham import frenet_geometry

print(f"\n  Comparaison m_CS et invariants géométriques de TQNT :")
print(f"  {'K':<5} {'m_CS(k=8)':>11} {'U_EM = U_κ+U_τ':>16} {'ratio':>10}")
print("  " + "-" * 50)

for K in KNOTS_AVAIL:
    if K == "0_1":
        continue
    try:
        m8 = m_CS(K, 2, 8)
        g = frenet_geometry(K, 30000)
        ds = g["speed"] * 2*np.pi/len(g["kappa"])
        U_kappa = float(np.sum(g["kappa"]**2*ds))
        U_tau = float(np.sum(g["tau"]**2*ds))
        U_EM = U_kappa + U_tau
        ratio = m8 / U_EM if abs(U_EM) > 1e-10 else float('inf')
        print(f"  {K:<5} {m8:>+11.4f} {U_EM:>16.3f} {ratio:>+10.5f}")
    except Exception as e:
        print(f"  {K:<5} (erreur: {e})")

print(f"""
  Si le rapport m_CS / U_EM est constant à travers les nœuds, alors :
    m_CS = (constante) · U_EM
  Cela confirmerait que U_EM (intégrale géométrique TQNT) est le
  proxy classique de la masse CS au leading order.
""")
