r"""
EXTENSION SYSTÉMATIQUE — déformation d'embedding pour particules à gros résidu.

Suite au succès sur μ⁻ (3_1 allongé λ_z=1.58 → m_μ exact), test sur
toutes les particules à |Δ V5| > 0.5%. Pour chacune, essayer :
  • Déformer chaque composante UNE PAR UNE
  • Trouver le λ_z qui donne m_exp exact
  • Vérifier que la déformation est PHYSIQUEMENT MODÉRÉE (λ_z ∈ [0.3, 3.0])

Si une déformation modérée existe pour la plupart des particules,
on a un principe TQNT-V6 unifié : embedding contextuel des composantes.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import brentq
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin
from theory.knot_catalog import knot_curve

# ───────── précalcul standard ─────────
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul standard…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]),
               "n_l": N_LOBES[K], "b": braid_index(K),
               "det": jones_determinant(pj), "s": topological_spin(K)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def frenet_curvature_torsion(pts):
    """κ et τ par différences finies sur courbe discrétisée."""
    n = len(pts)
    dt = 2*np.pi / n
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    r3 = np.gradient(r2, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1)
    cross = np.cross(r1, r2)
    kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
    triple = np.einsum('ij,ij->i', cross, r3)
    tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
    return kappa, tau, speed


def U_KT_deformed(knot_name, lam_z, n_points=8000):
    """U_κ et U_τ pour nœud déformé en z par facteur λ_z."""
    pts = knot_curve(knot_name, n_points)
    # Déformer en z
    pts_def = pts.copy()
    pts_def[:, 2] *= lam_z
    kappa, tau, speed = frenet_curvature_torsion(pts_def)
    ds = speed * 2*np.pi / n_points
    return np.sum(kappa**2 * ds), np.sum(tau**2 * ds)


# Normalisation : facteur correctif pour matcher la référence V5
def normalize_factors(knot_name):
    """Renvoie (norm_K, norm_T) pour que λ_z=1 donne la valeur V5."""
    Uk, Ut = U_KT_deformed(knot_name, 1.0)
    norm_K = U_KT[knot_name][0] / Uk if Uk > 0 else 1
    norm_T = U_KT[knot_name][1] / Ut if Ut > 0 else 1
    return norm_K, norm_T

print("\nNormalisation des déformations…")
NORM = {K: normalize_factors(K) for K in KNOTS}


def U_KT_def_normalized(knot_name, lam_z):
    """U_κ, U_τ déformés et normalisés (λ_z=1 ≡ V5)."""
    Uk, Ut = U_KT_deformed(knot_name, lam_z)
    nK, nT = NORM[knot_name]
    return Uk * nK, Ut * nT


def omega_def(comps, deform_idx, lam_z):
    """Ω_ts avec composante index deform_idx déformée par λ_z.

    Si plusieurs occurrences du même nœud, seule la composante à
    l'index spécifié est déformée.
    """
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))

    # G : somme des U_κ × U_τ par composante
    G_sum = 0
    for i, K in enumerate(comps):
        if i == deform_idx:
            Uk, Ut = U_KT_def_normalized(K, lam_z)
        else:
            Uk, Ut = U_KT[K]
        G_sum += Uk * Ut
    G = G_sum / P_31

    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


def find_lambda(comps, deform_idx, m_target, lam_range=(0.1, 4.0)):
    """Trouve λ_z qui donne m_target en déformant la composante idx."""
    def f(lz): return omega_def(comps, deform_idx, lz) - m_target
    # Sampling pour trouver changement de signe
    test_lz = np.linspace(lam_range[0], lam_range[1], 30)
    vals = [f(lz) for lz in test_lz]
    for i in range(len(vals)-1):
        if vals[i] * vals[i+1] < 0:
            try:
                return brentq(f, test_lz[i], test_lz[i+1])
            except ValueError:
                continue
    return None  # pas de solution


# ───────── PDG des particules à gros résidu ─────────
PDG_TEST = [
    # (nom, config, m_exp)
    ("μ⁻",        ["3_1","6_1"],         105.66),
    ("π⁰",        ["3_1","3_1","5_2"],   134.98),
    ("π±",        ["3_1","3_1","5_2"],   139.57),
    ("K±",        ["4_1","4_1","4_1"],   493.68),
    ("η",         ["3_1","3_1","6_1"],   547.86),
    ("η'",        ["7_7","7_7"],         957.78),
    ("K*(892)",   ["6_1","6_2"],         891.66),
    ("φ",         ["3_1","5_1","5_2"],   1019.46),
    ("Λ",         ["5_1","8_1"],         1115.68),
    ("Σ⁺",        ["4_1","4_1","5_1"],   1189.37),
    ("Ξ⁰",        ["4_1","5_1","5_2"],   1314.86),
    ("D⁰",        ["3_1","5_1","7_2"],   1864.84),
    ("D±",        ["3_1","5_1","7_2"],   1869.66),
    ("ψ(2S)",     ["6_1","8_2"],         3686.1),
    ("η_c(2S)",   ["6_1","8_2"],         3637.5),
    ("χ_c2",      ["6_1","8_2"],         3556.17),
    ("Λ_c",       ["3_1","3_1","9_1"],   2286.46),
]

print("\n" + "═"*82)
print("  TEST SYSTÉMATIQUE — déformation d'embedding pour particules à gros résidu")
print("═"*82)

# Pour chaque particule, essayer de déformer chaque composante
print(f"\n  Notation : {'particule':<10} | {'config':<22} | Δ V5 | meilleur λ_z + composante")
print("  " + "-" * 90)
results = []
for nm, cfg, m_exp in PDG_TEST:
    n = len(cfg)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in cfg]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in cfg)/P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(cfg); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    m_v5 = Phi_n * Phi_p * skel * G**Q_EXP
    d_v5 = (m_v5 - m_exp) / m_exp * 100

    # Essayer chaque composante (indice)
    best = None
    seen_K = set()
    for i, K in enumerate(cfg):
        if K in seen_K: continue  # ne pas répéter le même nœud
        seen_K.add(K)
        lam = find_lambda(cfg, i, m_exp)
        if lam is not None and 0.2 <= lam <= 4.0:
            # Score : préférer λ proche de 1 (déformation modérée)
            score = abs(np.log(lam))  # 0 si λ=1
            if best is None or score < best[2]:
                best = (K, lam, score)

    if best:
        K_best, lam_best, _ = best
        cls = "modérée" if 0.7 <= lam_best <= 1.5 else (
              "forte"   if 0.4 <= lam_best <= 2.5 else "extrême")
        deform_pct = (lam_best - 1) * 100
        flag = "★" if cls == "modérée" else ("⚠" if cls == "forte" else "✗")
        line = f"  {nm:<10} | {'+'.join(cfg):<22} | {d_v5:>+6.2f}% | λ_z({K_best})={lam_best:.3f} ({deform_pct:+.0f}%) {cls} {flag}"
        results.append((nm, cfg, m_exp, d_v5, K_best, lam_best, cls))
    else:
        line = f"  {nm:<10} | {'+'.join(cfg):<22} | {d_v5:>+6.2f}% | AUCUNE déformation valide trouvée  ✗"
        results.append((nm, cfg, m_exp, d_v5, None, None, None))
    print(line)

# Statistiques
print(f"\n  ─── BILAN ───")
n_modere = sum(1 for r in results if r[6] == "modérée")
n_forte = sum(1 for r in results if r[6] == "forte")
n_extreme = sum(1 for r in results if r[6] == "extrême")
n_none = sum(1 for r in results if r[6] is None)
total = len(results)
print(f"    Particules avec déformation MODÉRÉE (λ_z ∈ [0.7, 1.5]) : {n_modere}/{total}")
print(f"    Particules avec déformation FORTE (λ_z ∈ [0.4, 2.5])   : {n_forte}/{total}")
print(f"    Particules avec déformation EXTRÊME                    : {n_extreme}/{total}")
print(f"    Particules SANS solution                                : {n_none}/{total}")

# Analyse : quels nœuds sont le plus souvent déformés ?
deformed_count = Counter()
for r in results:
    if r[4]:
        deformed_count[r[4]] += 1
print(f"\n  ─── NŒUDS LES PLUS DÉFORMÉS ───")
for K, ct in deformed_count.most_common():
    print(f"    {K}  : {ct} particules")

# Lambda moyens par nœud
print(f"\n  ─── LAMBDA MOYEN PAR NŒUD DÉFORMÉ ───")
by_knot = {}
for r in results:
    if r[4]:
        by_knot.setdefault(r[4], []).append(r[5])
for K, lams in sorted(by_knot.items()):
    if lams:
        mean_lam = np.mean(lams)
        std_lam = np.std(lams)
        print(f"    {K:<5} : ⟨λ_z⟩ = {mean_lam:.3f} ± {std_lam:.3f}  ({len(lams)} particules)")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Sur {total} particules à gros résidu :
    • Modérée   : {n_modere} ({n_modere/total*100:.0f}%)
    • Forte     : {n_forte} ({n_forte/total*100:.0f}%)
    • Extrême   : {n_extreme} ({n_extreme/total*100:.0f}%)
    • Sans sol  : {n_none}

  Si la majorité des particules nécessite une déformation MODÉRÉE
  (λ_z entre 0.7 et 1.5), l'hypothèse d'embedding contextuel est
  CONFIRMÉE : les résidus V5 viennent de l'usage d'embeddings standard
  au lieu des embeddings d'équilibre contextuel.

  Si certaines particules nécessitent une déformation EXTRÊME
  (λ_z < 0.4 ou > 2.5), c'est probablement la mauvaise composante
  à déformer ou un effet QM standard (mixing, fine structure).
""")
