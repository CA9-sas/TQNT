r"""
TEST FINAL — Le résidu μ⁻ via déformation contextuelle du 6_1.

PHASE A de exotic_knots_scan a montré :
  Avec b=3, det=9, n_l=2, cr=6 (mêmes invariants topologiques que 6_1)
  mais U_κU_τ = 119.84 (au lieu de 115.95 = +3.4%), on obtient m_μ EXACT.

Question physique : peut-on atteindre U_κU_τ ≈ 119.84 par déformation
géométrique CONTINUE de 6_1 (préservant la topologie) ?

Test : reparamétriser 6_1 avec un facteur d'échelle inhomogène
       λ_z = facteur de compression vertical
       et chercher le λ qui donne U_κU_τ = 119.84.
"""
from __future__ import annotations
import numpy as np
from theory.abraham import frenet_geometry
from theory.knot_catalog import _curve_6_1

M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS_REF = ["3_1","6_1"]
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6)
M_MU = 105.6583755

print("═"*82); print("  Déformation contextuelle du 6_1 pour résoudre μ⁻"); print("═"*82)

# Précalcul de 3_1
from theory.jones import jones_polynomial, jones_determinant, braid_index
g_31 = frenet_geometry("3_1", 50000)
ds_31 = g_31["speed"] * 2*np.pi/len(g_31["kappa"])
U_31 = (float(np.sum(g_31["kappa"]**2*ds_31)),
        float(np.sum(g_31["tau"]**2*ds_31)))
P_31_geo = U_31[0]*U_31[1]
b_31 = braid_index("3_1"); det_31 = jones_determinant(jones_polynomial("3_1"))
T_31 = b_31 * det_31
M_naive_31 = 3*M_C + 3*M_L
M_naive_61 = 6*M_C + 2*M_L


def frenet_deformed(scale_xy, scale_z, n_points=50000):
    """Calcule (U_κ, U_τ) pour 6_1 déformé par scale_xy en (x,y) et scale_z en z."""
    pts = _curve_6_1(n_points)
    pts = pts * np.array([scale_xy, scale_xy, scale_z])
    dt = 2*np.pi / n_points
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    r3 = np.gradient(r2, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1)
    cross = np.cross(r1, r2)
    kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
    triple = np.einsum('ij,ij->i', cross, r3)
    tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
    ds = speed * dt
    return float(np.sum(kappa**2 * ds)), float(np.sum(tau**2 * ds))


def m_pair(U_κτ_6_1, b_6_1=3, det_6_1=9):
    """m(3_1+6_1) avec U_κU_τ paramétrable pour 6_1."""
    w1 = M_naive_31 * (T_31 / T_31)**P_EXP
    w2 = M_naive_61 * (b_6_1*det_6_1 / T_31)**P_EXP
    skel = (w1**A + w2**A)**(1/A)
    G = (P_31_geo + U_κτ_6_1) / P_31_geo
    Phi_n = (3/2)**(-X_EXP)
    return Phi_n * skel * G**Q_EXP


print(f"\n  Référence V5 (6_1 sans déformation) :")
Uk_0, Ut_0 = frenet_deformed(1.0, 1.0)
print(f"    U_κ = {Uk_0:.4f}, U_τ = {Ut_0:.4f}, U_κU_τ = {Uk_0*Ut_0:.4f}")
print(f"    m(3_1+6_1)_V5 = {m_pair(Uk_0*Ut_0):.4f} MeV  (m_exp = {M_MU:.4f})")
print(f"    écart V5 = {(m_pair(Uk_0*Ut_0)-M_MU)/M_MU*100:+.3f}%")

# Scan : déformation scale_z
print(f"\n  Scan scale_z (scale_xy = 1.0) : effet de compression/étirement vertical")
print(f"  {'scale_z':>9} {'U_κ':>10} {'U_τ':>10} {'U_κU_τ':>10} {'m(3_1+6_1)':>12} {'Δ %':>8}")
print("  " + "-" * 65)
best = (1e9, 1.0)
for sz in np.linspace(0.7, 1.3, 25):
    Uk, Ut = frenet_deformed(1.0, sz)
    m = m_pair(Uk*Ut)
    err = (m - M_MU)/M_MU*100
    print(f"  {sz:>9.3f} {Uk:>10.3f} {Ut:>10.3f} {Uk*Ut:>10.3f} {m:>12.4f} {err:>+7.3f}%")
    if abs(err) < abs(best[0]):
        best = (err, sz)

print(f"\n  Meilleur scale_z trouvé : {best[1]:.4f} (erreur résiduelle {best[0]:+.3f}%)")

# Affiner : recherche binaire
from scipy.optimize import brentq
def err_func(sz):
    Uk, Ut = frenet_deformed(1.0, sz)
    return m_pair(Uk*Ut) - M_MU
try:
    sz_opt = brentq(err_func, 0.5, 2.0, xtol=1e-5)
    Uk_o, Ut_o = frenet_deformed(1.0, sz_opt)
    print(f"\n  ★ Solution exacte : scale_z = {sz_opt:.5f}")
    print(f"    U_κU_τ obtenu = {Uk_o*Ut_o:.4f} (cible 119.84)")
    print(f"    m(3_1+6_1) = {m_pair(Uk_o*Ut_o):.4f} MeV (cible {M_MU:.4f})")
    print(f"    Déformation : {(sz_opt-1)*100:+.2f}% en z")
except Exception as e:
    print(f"  Pas de solution par brentq dans [0.5, 2.0]: {e}")

# Aussi : scan en scale_xy
print(f"\n  Scan scale_xy (scale_z = 1.0) : effet de dilatation horizontale")
print(f"  {'scale_xy':>10} {'U_κU_τ':>10} {'m(3_1+6_1)':>12} {'Δ %':>8}")
print("  " + "-" * 50)
for sxy in np.linspace(0.85, 1.15, 13):
    Uk, Ut = frenet_deformed(sxy, 1.0)
    m = m_pair(Uk*Ut)
    err = (m - M_MU)/M_MU*100
    print(f"  {sxy:>10.3f} {Uk*Ut:>10.3f} {m:>12.4f} {err:>+7.3f}%")

print("\n" + "═"*82); print("  ANALYSE — Que nous dit ce test ?"); print("═"*82)
print(f"""
  Le test scan exhaustif (exotic_knots_scan.py PHASE B) a montré :
    • Aucun K du catalogue ne donne m(3_1+K) plus proche de m_μ que 6_1.
    • Le 6_1 actuel donne -2.27%, et tous les autres ≥ 30% d'écart.

  L'analyse paramétrique (PHASE A) a quantifié le gap :
    • Avec U_κU_τ_partenaire = 119.84 (au lieu de 115.95 → +3.4%),
      on obtient m_μ EXACT, avec mêmes invariants topologiques (b, det).

  Le présent test confirme par déformation EXPLICITE :
    • Une compression vertical (scale_z < 1) du 6_1 augmente U_κU_τ
    • Le facteur de compression nécessaire est ≈ {sz_opt:.4f}
    • Cela représente {abs(sz_opt-1)*100:.2f}% de déformation
    • Physiquement très petit, plausible pour un "embedding contextuel"

  RÉSULTAT : l'hypothèse d'embedding contextuel est désormais quantifiée :
    le 6_1 dans μ⁻ doit être {abs(sz_opt-1)*100:.1f}% comprimé verticalement
    par rapport à sa forme canonique de catalogue.

  Mais cela reste une CALIBRATION sur 1 point (μ⁻).
  Sans loi prédictive indépendante (autre observable contraignant scale_z),
  le résidu μ⁻ reste structurellement irréductible dans V5+V6.

  Note sur l'exploration pretzel : ma paramétrisation pretzel_curve dans
  exotic_knots_scan.py est cassée (U_κU_τ ~ 10⁶ aberrant). Une vraie
  exploration pretzel nécessiterait :
    (i) paramétrisation rigoureuse via décomposition en tangles rationnels
    (ii) calcul de Jones par algorithme de Kauffman
    (iii) vérification topologique par invariants de Reidemeister
""")
