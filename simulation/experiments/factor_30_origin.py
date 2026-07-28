r"""
RECHERCHE DE L'ORIGINE DU FACTEUR 30,6× POUR 7_2

Le naive (cr·m_c + lobes·m_l) × k_s donne 2,16 MeV pour 7_2 alors que
la valeur Hölder cohérente avec le proton est 66,08 MeV. Le facteur 30
manquant doit avoir une origine géométrique-topologique.

On teste systématiquement toutes les combinaisons des invariants
géométriques (Frenet) et topologiques (Jones) pour identifier celle
qui produit exactement 30,6 pour 7_2 normalisé à 3_1.

Candidats :
   ratios simples : cr, n_lobes, det, c_J
   intégrales     : U_κ, U_τ, ∫|κ|ds, ∫|τ|ds, L
   produits       : U_κ·U_τ, ∫κ·∫τ, c_J·cr, etc.
   puissances     : U_κ^p, ∫τ²/L, etc.
"""
from __future__ import annotations
import numpy as np

from theory.abraham import frenet_geometry, REFERENCE_N50K
from theory.jones import (jones_polynomial, jones_span, jones_determinant,
                          braid_index)

M_E = 0.51099895
M_72_HOLDER = 66.083    # déduit du proton
K_S = 2.0               # boucle serrée

KNOTS_TEST = ["3_1", "4_1", "5_1", "7_1", "7_2"]
N_LOBES = {"3_1": 3, "4_1": 2, "5_1": 5, "7_1": 7, "7_2": 2}


def compute_invariants(K, n_points=50000):
    geo = frenet_geometry(K, n_points)
    kappa = geo["kappa"]
    tau = geo["tau"]
    speed = geo["speed"]
    N = len(kappa)
    dt = 2 * np.pi / N
    ds = speed * dt
    p = jones_polynomial(K)
    return {
        "K":        K,
        "cr":       int(K.split("_")[0]),
        "n_lobes":  N_LOBES[K],
        "L":        float(np.sum(ds)),
        "int_k":    float(np.sum(np.abs(kappa) * ds)),
        "int_t":    float(np.sum(np.abs(tau) * ds)),
        "U_k":      float(np.sum(kappa ** 2 * ds)),
        "U_t":      float(np.sum(tau ** 2 * ds)),
        "span":     float(jones_span(p)),
        "det":      float(jones_determinant(p)),
        "b":        float(braid_index(K)),
    }


def main():
    print("=" * 100)
    print("  RECHERCHE DE L'ORIGINE DU FACTEUR 30,6× pour 7_2")
    print("=" * 100)

    inv = {K: compute_invariants(K) for K in KNOTS_TEST}

    # Calcul m_naive et facteur Hölder pour chaque
    print(f"\n  m_per_crossing = {M_E*0.8682/3:.6f} MeV")
    print(f"  m_per_loop     = {M_E*0.1318/3:.6f} MeV")
    m_c = M_E * 0.8682 / 3
    m_l = M_E * 0.1318 / 3

    # Cibles Hölder
    HOLDER = {"3_1": M_E, "4_1": 15.367, "5_1": 70.165,
              "7_1": 199.934, "7_2": M_72_HOLDER}

    print(f"\n  ─── m_naive vs cible Hölder ───")
    print(f"  {'K':>5} {'cr':>3} {'lobes':>5} {'m_naive':>10} "
          f"{'cible':>10} {'facteur':>9}")
    print("  " + "-" * 50)
    factors = {}
    for K in KNOTS_TEST:
        d = inv[K]
        m_n = d["cr"] * m_c + d["n_lobes"] * m_l
        ks = K_S if K == "7_2" else 1.0
        m_naive_final = m_n * ks
        target = HOLDER[K]
        fac = target / m_naive_final
        factors[K] = fac
        print(f"  {K:>5} {d['cr']:>3} {d['n_lobes']:>5} "
              f"{m_naive_final:>10.4f} {target:>10.4f} {fac:>9.3f}×")

    # ─── Recherche de l'origine ────────────────────────────────────
    print(f"\n  ─── TEST : différentes combinaisons d'invariants ───")
    print(f"  (ratio K/3_1 normalisé pour chaque combinaison)")
    print(f"\n  {'combinaison':<26} " + " ".join(
        f"{K:>8}" for K in KNOTS_TEST))
    print("  " + "-" * 72)

    def show(name, fn):
        ref = fn(inv["3_1"])
        row = []
        for K in KNOTS_TEST:
            try:
                v = fn(inv[K]) / ref if ref != 0 else float("nan")
            except Exception:
                v = float("nan")
            row.append(v)
        # Affichage avec mise en évidence de 7_2
        cells = []
        for K, v in zip(KNOTS_TEST, row):
            if K == "7_2" and abs(v - 30.6) < 1.5:
                cells.append(f"{v:>7.2f}★")
            else:
                cells.append(f"{v:>8.2f}")
        print(f"  {name:<26} " + " ".join(cells))
        return row

    print(f"\n  Cible factor :              " + " ".join(
        f"{factors[K]:>8.2f}" for K in KNOTS_TEST))
    print("  " + "-" * 72)

    show("cr",                 lambda d: d["cr"])
    show("det",                lambda d: d["det"])
    show("cr × n_lobes",       lambda d: d["cr"] * d["n_lobes"])
    show("cr + n_lobes",       lambda d: d["cr"] + d["n_lobes"])
    show("U_κ",                lambda d: d["U_k"])
    show("U_τ",                lambda d: d["U_t"])
    show("U_κ + U_τ (=U_tot)", lambda d: d["U_k"] + d["U_t"])
    show("U_κ × U_τ ★",        lambda d: d["U_k"] * d["U_t"])
    show("U_κ²",               lambda d: d["U_k"] ** 2)
    show("U_τ²",               lambda d: d["U_t"] ** 2)
    show("∫|κ|·∫|τ|",          lambda d: d["int_k"] * d["int_t"])
    show("L · U_τ",            lambda d: d["L"] * d["U_t"])
    show("c_J = span + γ·det", lambda d: d["span"] + 0.1 * d["det"])
    show("(c_J)^3",            lambda d: (d["span"] + 0.1 * d["det"]) ** 3)
    show("U_EM · c_J",         lambda d: (d["U_k"] + d["U_t"]) * (d["span"] + 0.1 * d["det"]))
    show("√(U_κ · U_τ) · cr",  lambda d: np.sqrt(d["U_k"] * d["U_t"]) * d["cr"])
    show("U_τ · cr",           lambda d: d["U_t"] * d["cr"])

    # ─── Vérification de l'hypothèse U_κ · U_τ ───────────────────────
    print(f"\n  ─── VÉRIFICATION : facteur = U_κ · U_τ / (U_κ · U_τ)_3_1 ───")
    P_31 = inv["3_1"]["U_k"] * inv["3_1"]["U_t"]
    for K in KNOTS_TEST:
        P_K = inv[K]["U_k"] * inv[K]["U_t"]
        ratio = P_K / P_31
        d = inv[K]
        m_n = d["cr"] * m_c + d["n_lobes"] * m_l
        ks = K_S if K == "7_2" else 1.0
        m_pred = m_n * ks * ratio
        target = HOLDER[K]
        err = (m_pred - target) / target * 100
        flag = "★" if abs(err) < 5 else " "
        print(f"  {K:>5}: P_K/P_31 = {ratio:>7.3f}, m_pred = {m_pred:>9.3f}, "
              f"cible = {target:>9.3f}, Δ = {err:>+7.2f}% {flag}")

    # Décomposition pour 7_2 explicitement
    print(f"\n  ─── DÉCOMPOSITION POUR 7_2 ───")
    d72 = inv["7_2"]
    d31 = inv["3_1"]
    P_72 = d72["U_k"] * d72["U_t"]
    P_31 = d31["U_k"] * d31["U_t"]
    print(f"  U_κ(7_2)         = {d72['U_k']:.4f}")
    print(f"  U_τ(7_2)         = {d72['U_t']:.4f}")
    print(f"  U_κ · U_τ (7_2)  = {P_72:.4f}")
    print(f"  U_κ · U_τ (3_1)  = {P_31:.4f}")
    print(f"  Ratio P_72/P_31  = {P_72/P_31:.4f}  ← ce serait le facteur 30,6")
    print(f"\n  Application :")
    m_naive_72 = d72["cr"] * m_c + d72["n_lobes"] * m_l
    print(f"  m_naive(7_2)     = {m_naive_72:.4f} MeV  (cr·m_c + lobes·m_l)")
    print(f"  × k_s = {K_S}      → {m_naive_72*K_S:.4f}")
    print(f"  × (U_κ·U_τ ratio) → {m_naive_72*K_S*P_72/P_31:.4f}")
    print(f"  Cible Hölder     = {M_72_HOLDER:.4f}")
    err = (m_naive_72*K_S*P_72/P_31 - M_72_HOLDER) / M_72_HOLDER * 100
    print(f"  écart            = {err:+.3f} %")
    print("=" * 100)


if __name__ == "__main__":
    main()
