r"""
APPROCHE INVERSE — déterminer le poids de chaque nœud simple à partir
des masses expérimentales des particules réelles.

Concept :
   Chaque particule P est une superposition de nœuds K_1, K_2, ...
   On cherche les poids m(K_j) tels que :
       m(P) ≈ f({m(K_j) | K_j ∈ P})
   où f peut être :
     - additive : m(P) = Σ_j n_j · m(K_j)         (modèle linéaire)
     - log-linéaire : log m(P) = Σ_j n_j · log m(K_j)   (multiplicatif)

On résout par NNLS (non-negative least squares) pour garantir
des poids POSITIFS (contrainte physique : pas de masse négative).

On procède EN CROISSANT :
  1. Particules à 1 nœud   (singles : e, π, u, d)
  2. + Particules à 2 nœuds (mésons : ρ, D, B, Λ, Σ, mésons charmés)
  3. + Particules à 3 nœuds (baryons + EW : p, n, K, η, W, Z, H, t)
À chaque étape, on observe l'évolution des poids m(K).

Sortie :
  - Tableau des poids m(K) finaux pour les 19 nœuds
  - Comparaison m_pred vs m_exp pour les 46 particules
  - Identification des configurations qui dévient le plus
  - Test des deux formules (additive vs multiplicative)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls

# ───────────────────────────────────────────────────────────────
# Catalogue des 19 nœuds
# ───────────────────────────────────────────────────────────────
KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]

# ───────────────────────────────────────────────────────────────
# Particules : (nom, configuration, m_exp_MeV)
# ───────────────────────────────────────────────────────────────
PARTICLES = [
    # n = 1 (singles)
    ("e⁻",    ["3_1"],                       0.511),
    ("u",     ["3_1"],                       2.2),
    ("d",     ["3_1"],                       4.7),
    ("π⁰",    ["8_1"],                       134.98),
    ("π±",    ["8_1"],                       139.57),
    # n = 2
    ("s",     ["5_2", "5_2"],                95.0),
    ("μ⁻",    ["3_1", "6_1"],                105.66),
    ("c",     ["7_3", "7_7"],                1270.0),
    ("ρ/ω",   ["5_2", "7_7"],                775.0),
    ("D⁰",    ["6_2", "7_7"],                1864.84),
    ("B",     ["4_1", "10_1"],               5279.4),
    ("Λ",     ["4_1", "7_4"],                1115.7),
    ("Σ⁺",    ["6_2", "7_2"],                1189.4),
    ("Λc",    ["6_3", "8_1"],                2286.0),
    # n = 3
    ("τ⁻",    ["3_1", "5_1", "7_1"],         1776.86),
    ("b",     ["4_1", "5_2", "7_6"],         4180.0),
    ("K±",    ["4_1", "4_1", "5_2"],         493.68),
    ("η",     ["3_1", "4_1", "5_1"],         547.86),
    ("η'",    ["5_2", "5_2", "6_1"],         957.78),
    ("φ",     ["3_1", "4_1", "6_2"],         1019.46),
    ("J/ψ",   ["4_1", "5_2", "9_1"],         3096.9),
    ("Υ",     ["3_1", "7_4", "9_1"],         9460.3),
    ("p",     ["3_1", "7_2", "7_2"],         938.272),
    ("n",     ["3_1", "7_2", "7_2"],         939.565),
    ("Ξ⁰",    ["4_1", "4_1", "7_2"],         1314.9),
    ("Δ",     ["3_1", "5_2", "6_2"],         1232.0),
    ("Σ*",    ["5_2", "6_2", "7_2"],         1383.0),
    ("Ξ*",    ["4_1", "5_2", "7_2"],         1532.0),
    ("Ω⁻",    ["5_2", "5_2", "7_2"],         1672.0),
    ("Σc",    ["4_1", "5_1", "7_1"],         2455.0),
    ("Ωc",    ["3_1", "6_2", "6_2"],         2695.0),
    ("Λb",    ["5_2", "5_2", "8_2"],         5620.0),
    ("W±",    ["4_1", "7_3", "8_3"],         80377.0),
    ("Z⁰",    ["5_1", "7_2", "8_3"],         91188.0),
    ("H",     ["6_1", "7_4", "8_3"],         125250.0),
    ("t",     ["7_4", "7_6", "8_3"],         173100.0),
]


def build_matrix(particles):
    """Construit X et y pour le système linéaire y = X · m."""
    N = len(particles)
    K = len(KNOTS)
    X = np.zeros((N, K))
    y = np.zeros(N)
    for i, (nm, comps, me) in enumerate(particles):
        for k in comps:
            j = KNOTS.index(k)
            X[i, j] += 1
        y[i] = me
    return X, y


def solve_additive(particles):
    """Résout y = X·m par NNLS (poids positifs)."""
    X, y = build_matrix(particles)
    m, residual = nnls(X, y)
    return m, residual


def solve_log_linear(particles):
    """Résout log y = X · log m par LS (poids positifs garantis)."""
    valid = [(n, c, me) for n, c, me in particles if me > 0]
    X, y = build_matrix(valid)
    log_y = np.log(y)
    log_m, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    return np.exp(log_m), valid


def evaluate(particles, m_K, formula="add"):
    """Renvoie (m_pred, residus, RMS_log) pour chaque particule."""
    results = []
    for nm, comps, me in particles:
        if me <= 0:
            continue
        if formula == "add":
            m_pred = sum(m_K[KNOTS.index(k)] for k in comps)
        else:
            m_pred = np.prod([m_K[KNOTS.index(k)] for k in comps])
        err = (m_pred - me) / me * 100
        log_res = np.log(m_pred / me) if m_pred > 0 else 0
        results.append((nm, comps, me, m_pred, err, log_res))
    log_arr = np.array([r[5] for r in results])
    rms = float(np.sqrt(np.mean(log_arr ** 2)))
    return results, rms


def main():
    print("=" * 100)
    print("  POIDS DES NŒUDS — RÉSOLUTION INVERSE DEPUIS LES MASSES EXPÉRIMENTALES")
    print("=" * 100)

    # ───────────────────────────────────────────────────────────────
    # PHASE A : modèle ADDITIF (NNLS)
    # ───────────────────────────────────────────────────────────────
    print(f"\n  ─── MODÈLE A : m(P) = Σ m(K_i)  (additif, NNLS) ───")
    print(f"  Résolution sur les {len(PARTICLES)} particules expérimentales.")

    m_add, res_norm = solve_additive(PARTICLES)
    print(f"\n  Norme du résidu NNLS = {res_norm:.4e} MeV")
    print(f"\n  Poids inférés (NNLS additif) :")
    print(f"  {'K':>5} {'m(K) MeV':>15}")
    print("  " + "-" * 22)
    for K, m in sorted(zip(KNOTS, m_add), key=lambda kv: kv[1]):
        flag = "★" if m > 1e-3 else " (=0)"
        print(f"  {K:>5} {m:>15.4f}  {flag}")

    res_add, rms_add = evaluate(PARTICLES, m_add, "add")
    print(f"\n  RMS log (modèle additif) = {rms_add:.3f}")

    # ───────────────────────────────────────────────────────────────
    # PHASE B : modèle MULTIPLICATIF (log-linéaire)
    # ───────────────────────────────────────────────────────────────
    print(f"\n  ─── MODÈLE B : log m(P) = Σ log m(K_i)  (multiplicatif) ───")
    m_mul, valid = solve_log_linear(PARTICLES)
    print(f"\n  Poids inférés (log-linéaire) :")
    print(f"  {'K':>5} {'m(K) MeV':>15}")
    print("  " + "-" * 22)
    for K, m in sorted(zip(KNOTS, m_mul), key=lambda kv: kv[1]):
        print(f"  {K:>5} {m:>15.4f}")

    res_mul, rms_mul = evaluate(PARTICLES, m_mul, "mul")
    print(f"\n  RMS log (modèle multiplicatif) = {rms_mul:.3f}")

    # ───────────────────────────────────────────────────────────────
    # COMPARAISON sur les 36 particules
    # ───────────────────────────────────────────────────────────────
    print(f"\n  ─── TABLEAU COMPARATIF (les deux modèles) ───")
    print(f"  {'particule':>5} {'n':>2} {'m_exp':>11} "
          f"{'m_add':>11} {'Δa%':>7} {'m_mul':>11} {'Δm%':>7}")
    print("  " + "-" * 70)
    flags_add = {"★": 0, "✓": 0, " ": 0, "?": 0}
    flags_mul = {"★": 0, "✓": 0, " ": 0, "?": 0}
    for (a, b), (_, _) in zip(zip(res_add, res_mul), zip([0]*len(res_add), [0]*len(res_mul))):
        nm = a[0]
        n = len(a[1])
        me = a[2]
        ma = a[3]
        mm = b[3]
        ea = a[4]
        em = b[4]
        fa = "★" if abs(ea) < 5 else ("✓" if abs(ea) < 20 else (" " if abs(ea) < 50 else "?"))
        fm = "★" if abs(em) < 5 else ("✓" if abs(em) < 20 else (" " if abs(em) < 50 else "?"))
        flags_add[fa] += 1
        flags_mul[fm] += 1
        print(f"  {nm:>5} {n:>2} {me:>11.2f} "
              f"{ma:>11.2f} {ea:>+6.1f} {fa} "
              f"{mm:>11.2f} {em:>+6.1f} {fm}")

    print(f"\n  STATISTIQUES (sur {len(res_add)} particules) :")
    print(f"   {'flag':>5} {'additif':>10} {'multiplicatif':>14}")
    for flag in "★✓ ?":
        print(f"   {flag:>5} {flags_add[flag]:>10} {flags_mul[flag]:>14}")
    print(f"   {'RMS log':>5} {rms_add:>10.3f} {rms_mul:>14.3f}")

    # ───────────────────────────────────────────────────────────────
    # PHASE C : EN CROISSANT — observer évolution des poids
    # ───────────────────────────────────────────────────────────────
    print(f"\n  ─── ÉVOLUTION DES POIDS EN AJOUTANT DES PARTICULES PROGRESSIVEMENT ───")
    print(f"  Critère d'addition : par nombre de composantes (n=1, n=2, n=3)")
    print(f"  puis par masse croissante.")
    print()

    # Trier : d'abord par n croissant, puis par masse
    sorted_parts = sorted(PARTICLES, key=lambda p: (len(p[1]), p[2]))

    # Pour quelques étapes intermédiaires, montrer les poids
    checkpoints = [5, 10, 15, 20, 25, 30, len(sorted_parts)]
    print(f"  Étape (#part.) | Poids inférés (NNLS additif) au fur et à mesure")
    print(f"  {'#part':>6} | {'3_1':>7} {'4_1':>7} {'5_1':>7} {'5_2':>7} "
          f"{'6_1':>7} {'7_2':>7} {'8_1':>7} {'8_3':>7} | RMS log")
    print("  " + "-" * 90)
    key_knots = ["3_1", "4_1", "5_1", "5_2", "6_1", "7_2", "8_1", "8_3"]
    for cp in checkpoints:
        sub = sorted_parts[:cp]
        try:
            m_sub, _ = solve_additive(sub)
            _, rms_sub = evaluate(sub, m_sub, "add")
            row = [f"{m_sub[KNOTS.index(K)]:>7.1f}" for K in key_knots]
            print(f"  {cp:>6} | {' '.join(row)} | {rms_sub:.3f}")
        except Exception as e:
            print(f"  {cp:>6} | erreur : {e}")

    # ───────────────────────────────────────────────────────────────
    # PHASE D : Tableau final des poids (modèle ADDITIF, base 36 particules)
    # ───────────────────────────────────────────────────────────────
    print(f"\n  ─── TABLEAU FINAL DES POIDS m(K) ───")
    print(f"  (modèle additif, NNLS sur 36 particules expérimentales)")
    print(f"\n  {'K':>5} {'m(K) MeV':>15} {'apparitions':>13}")
    print("  " + "-" * 36)
    apparitions = {K: 0 for K in KNOTS}
    for nm, comps, me in PARTICLES:
        for k in comps:
            apparitions[k] += 1
    for K in KNOTS:
        m = m_add[KNOTS.index(K)]
        app = apparitions[K]
        flag = "★" if m > 0.01 else " (zéro forcé)"
        print(f"  {K:>5} {m:>15.4f}  {app:>4} occ. {flag}")

    # Export LaTeX
    tex_path = "docs/knot_weights_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"% Poids des nœuds inférés par NNLS additif" + "\n")
        f.write(r"\begin{center}" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.1}" + "\n")
        f.write(r"\begin{tabular}{lrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"$K$ & $m(K)$ (MeV, additif) & $m(K)$ (MeV, mul.) & apparitions \\" + "\n")
        f.write(r"\midrule" + "\n")
        for K in KNOTS:
            ma = m_add[KNOTS.index(K)]
            mm = m_mul[KNOTS.index(K)]
            ap = apparitions[K]
            knot_tex = K.replace("_", r"_")
            f.write(f"  ${knot_tex}$ & {ma:.3f} & {mm:.3f} & {ap} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{center}" + "\n")
    print(f"\n  ✓ Tableau LaTeX exporté : {tex_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
