r"""
RECHERCHE DE L'ORIGINE ANALYTIQUE des 5 paramètres libres de Ωₜₛ.

Paramètres trouvés par optimisation Nelder-Mead :
   a = 0.52825
   p = 0.77955
   q = 0.72177
   x = -2.45599
   β = 1.65095

On cherche des valeurs analytiques exactes qui reproduisent
l'optimum, en testant :
   - Fractions simples
   - Constantes mathématiques (π, e, √n, ratios)
   - Relations entre paramètres (sommes, produits, ratios)
   - Lien avec invariants TQNT (3/8 de R6, 4/3 de Gennes, 32/9, etc.)
"""
from __future__ import annotations
import numpy as np
from collections import Counter

from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index

# Valeurs OPTIMALES numériques
A_OPT = 0.52825
P_OPT = 0.77955
Q_OPT = 0.72177
X_OPT = -2.45599
B_OPT = 1.65095


def report(name, candidates):
    """Affiche candidats triés par erreur relative."""
    print(f"\n  ── {name} ──")
    print(f"  {'expression':<30} {'valeur':>10} {'erreur %':>10}")
    print("  " + "-" * 50)
    candidates.sort(key=lambda c: abs(c[2]))
    for expr, val, err in candidates[:8]:
        flag = " ★" if abs(err) < 0.3 else ("  ✓" if abs(err) < 1.0 else "")
        print(f"  {expr:<30} {val:>10.5f} {err:>+9.3f}%{flag}")


def main():
    print("=" * 80)
    print("  RECHERCHE DE L'ORIGINE ANALYTIQUE DES 5 PARAMÈTRES")
    print("=" * 80)
    print(f"\n  Valeurs numériques optimisées :")
    print(f"    a = {A_OPT}")
    print(f"    p = {P_OPT}")
    print(f"    q = {Q_OPT}")
    print(f"    x = {X_OPT}")
    print(f"    β = {B_OPT}")

    # ===== a =====
    candidates_a = [
        ("1/2",              0.5,                (0.5 - A_OPT) / A_OPT * 100),
        ("π/6",              np.pi / 6,          (np.pi/6 - A_OPT) / A_OPT * 100),
        ("17/32",            17 / 32,            (17/32 - A_OPT) / A_OPT * 100),
        ("8/15",             8 / 15,             (8/15 - A_OPT) / A_OPT * 100),
        ("11/21",            11 / 21,            (11/21 - A_OPT) / A_OPT * 100),
        ("ln(e^(1/2))",      0.5,                (0.5 - A_OPT) / A_OPT * 100),
        ("√(2/(π+1/2))",     np.sqrt(2/(np.pi+0.5)), (np.sqrt(2/(np.pi+0.5)) - A_OPT) / A_OPT * 100),
        ("1/2 + 1/32",       0.5 + 1/32,         (0.5+1/32 - A_OPT) / A_OPT * 100),
        ("1/√(2·e^(-1/3))",  1/np.sqrt(2*np.exp(-1/3)), (1/np.sqrt(2*np.exp(-1/3)) - A_OPT) / A_OPT * 100),
        ("(√e - 1) / √2",    (np.sqrt(np.e) - 1) / np.sqrt(2), ((np.sqrt(np.e)-1)/np.sqrt(2) - A_OPT) / A_OPT * 100),
    ]
    report("a", candidates_a)

    # ===== p =====
    candidates_p = [
        ("π/4",              np.pi / 4,          (np.pi/4 - P_OPT) / P_OPT * 100),
        ("3/4",              3 / 4,              (3/4 - P_OPT) / P_OPT * 100),
        ("25/32",            25 / 32,            (25/32 - P_OPT) / P_OPT * 100),
        ("4/5",              4 / 5,              (4/5 - P_OPT) / P_OPT * 100),
        ("3/4 + 1/32",       3/4 + 1/32,         (3/4 + 1/32 - P_OPT) / P_OPT * 100),
        ("ln(e·π/(π+1))",    np.log(np.e * np.pi / (np.pi+1)), (np.log(np.e*np.pi/(np.pi+1)) - P_OPT) / P_OPT * 100),
        ("√(3/5)·1.005",     np.sqrt(3/5)*1.005, (np.sqrt(3/5)*1.005 - P_OPT) / P_OPT * 100),
    ]
    report("p", candidates_p)

    # ===== q =====
    candidates_q = [
        ("3/4",              3 / 4,              (3/4 - Q_OPT) / Q_OPT * 100),
        ("5/7",              5 / 7,              (5/7 - Q_OPT) / Q_OPT * 100),
        ("23/32",            23 / 32,            (23/32 - Q_OPT) / Q_OPT * 100),
        ("3/4 - 1/32",       3/4 - 1/32,         (3/4 - 1/32 - Q_OPT) / Q_OPT * 100),
        ("π/4 - 1/16",       np.pi/4 - 1/16,     (np.pi/4 - 1/16 - Q_OPT) / Q_OPT * 100),
        ("ln(2.0625)",       np.log(2.0625),     (np.log(2.0625) - Q_OPT) / Q_OPT * 100),
        ("3·π / (4π+1/2)",   3*np.pi/(4*np.pi + 0.5), (3*np.pi/(4*np.pi+0.5) - Q_OPT) / Q_OPT * 100),
    ]
    report("q", candidates_q)

    # ===== x (négatif) =====
    candidates_x = [
        ("-√6",              -np.sqrt(6),        (-np.sqrt(6) - X_OPT) / X_OPT * 100),
        ("-5/2",             -2.5,               (-2.5 - X_OPT) / X_OPT * 100),
        ("-π/√(8/π²)",       -np.pi*np.sqrt(8)/np.pi/np.pi**0.5, ((-np.pi*np.sqrt(8)/np.pi/np.pi**0.5) - X_OPT) / X_OPT * 100),
        ("-e + 1/4",         -np.e + 0.25,       (-np.e + 0.25 - X_OPT) / X_OPT * 100),
        ("-79/32",           -79/32,             (-79/32 - X_OPT) / X_OPT * 100),
        ("-√(2π+1/2)",       -np.sqrt(2*np.pi+0.5), (-np.sqrt(2*np.pi+0.5) - X_OPT) / X_OPT * 100),
        ("-2·5/(3+π/16)",    -2*5/(3+np.pi/16),  ((-2*5/(3+np.pi/16)) - X_OPT) / X_OPT * 100),
    ]
    report("x", candidates_x)

    # ===== β =====
    candidates_b = [
        ("√e",               np.sqrt(np.e),      (np.sqrt(np.e) - B_OPT) / B_OPT * 100),
        ("5/3",              5/3,                (5/3 - B_OPT) / B_OPT * 100),
        ("φ (or)",           (1+np.sqrt(5))/2,   ((1+np.sqrt(5))/2 - B_OPT) / B_OPT * 100),
        ("53/32",            53/32,              (53/32 - B_OPT) / B_OPT * 100),
        ("ln(5+1/4)",        np.log(5.25),       (np.log(5.25) - B_OPT) / B_OPT * 100),
        ("π/(2-1/π)",        np.pi/(2-1/np.pi),  (np.pi/(2-1/np.pi) - B_OPT) / B_OPT * 100),
        ("e^(1/2)",          np.exp(0.5),        (np.exp(0.5) - B_OPT) / B_OPT * 100),
    ]
    report("β", candidates_b)

    # ===== RELATIONS ENTRE PARAMÈTRES =====
    print(f"\n  ─── RELATIONS ENTRE PARAMÈTRES ───")
    print(f"  Sommes, produits, ratios :")
    relations = [
        ("p + q",            P_OPT + Q_OPT,         3/2),
        ("p × q",            P_OPT * Q_OPT,         9/16),
        ("a × p × q",        A_OPT * P_OPT * Q_OPT, 1/4),  # test
        ("a + p + q",        A_OPT + P_OPT + Q_OPT, 2),
        ("p² + q²",          P_OPT**2 + Q_OPT**2,   9/8),
        ("a × β",            A_OPT * B_OPT,         8/9),  # test
        ("a × |x|",          A_OPT * abs(X_OPT),    4/3),
        ("β² (β = √e ?)",    B_OPT**2,              np.e),
        ("x² (x = -√6 ?)",   X_OPT**2,              6),
        ("β / a",            B_OPT / A_OPT,         π_test := np.pi),
        ("|x| / β",          abs(X_OPT) / B_OPT,    3/2),
        ("β·a / √x²",        B_OPT*A_OPT/abs(X_OPT), 3/8),  # test
    ]
    print(f"  {'expression':<25} {'valeur':>10} {'cible':>12} {'erreur %':>10}")
    print("  " + "-" * 60)
    for expr, val, target in relations:
        err = (val - target) / target * 100
        flag = " ★" if abs(err) < 0.3 else ("  ✓" if abs(err) < 1.0 else "")
        print(f"  {expr:<25} {val:>10.5f} {target:>12.5f} {err:>+9.3f}%{flag}")

    # ===== SYNTHÈSE =====
    print(f"\n  ═══ MEILLEURES HYPOTHÈSES ANALYTIQUES ═══")
    print(f"  Les candidats à <0.5 % d'erreur :")
    print(f"    a  ≈ 17/32 = 0.53125         (erreur 0.6%)")
    print(f"    p  ≈ π/4 = 0.78540           (erreur 0.7%)")
    print(f"    q  ≈ 23/32 = 0.71875         (erreur 0.4%)")
    print(f"    |x| ≈ √6 = 2.44949           (erreur 0.27%)")
    print(f"    β  ≈ √e = 1.64872            (erreur 0.14%)")
    print(f"  ")
    print(f"  Relations contraintes :")
    print(f"    p + q  ≈ 3/2  (erreur 0.09%)")
    print(f"    p · q  ≈ 9/16  (erreur 0.05%)")
    print(f"    x²     ≈ 6     (erreur 0.5%)")
    print(f"    β²     ≈ e     (erreur 0.28%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
