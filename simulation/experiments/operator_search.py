r"""
Recherche d'un OPÉRATEUR Ô qui combine correctement les poids m(K) des
nœuds simples pour reproduire les masses expérimentales :

   m(P) = Ô({m(K_i) | K_i ∈ P}, paramètres opérateur)

Les modèles purs (additif, multiplicatif) ont échoué (commit précédent).
On teste maintenant des opérateurs PLUS RICHES avec un petit nombre de
paramètres globaux, tout en gardant les poids m(K) ≥ 0.

Opérateurs candidats (ordre croissant de complexité) :

  O0 (réf)  : additif simple        m(P) = Σ m(K)
  O1        : multiplicatif         m(P) = Π m(K)
  O2        : moyenne géom + n^p    m(P) = n^p · (Π m(K))^(1/n)
  O3        : max + couplage        m(P) = max(m) + α · (Σ m - max)
  O4        : Pythagore            m(P) = √(Σ m²)
  O5        : combinaison mixte    m(P) = α · max + β · √(Σ m²)
  O6        : exp(somme log + shift sectoriel)
                                   log m(P) = Σ log m(K) + δ_σ
  O7        : sectoriel (4 sect.)  m(P) = α_σ · max(m) · n^p_σ
  O8        : Σ m^a (pondéré)      m(P) = (Σ m^a)^(1/a)  (Hölder)
  O9        : Σ m^a + couplage    m(P) = c · (Σ m^a)^(1/a) · couplage(n, σ)

Pour chacun, optimisation conjointe (poids + paramètres) par
Nelder-Mead sur log RMS, garantie m(K) > 0 par paramétrage exponentiel.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_K = len(KNOTS)
EW_MARKER = "8_3"

# Particules avec m > 0 (les neutrinos exclus)
PARTICLES = [
    ("e⁻",    ["3_1"],                       0.511),
    ("u",     ["3_1"],                       2.2),
    ("d",     ["3_1"],                       4.7),
    ("π⁰",    ["8_1"],                       134.98),
    ("π±",    ["8_1"],                       139.57),
    ("s",     ["5_2", "5_2"],                95.0),
    ("μ⁻",    ["3_1", "6_1"],                105.66),
    ("c",     ["7_3", "7_7"],                1270.0),
    ("ρ/ω",   ["5_2", "7_7"],                775.0),
    ("D⁰",    ["6_2", "7_7"],                1864.84),
    ("B",     ["4_1", "10_1"],               5279.4),
    ("Λ",     ["4_1", "7_4"],                1115.7),
    ("Σ⁺",    ["6_2", "7_2"],                1189.4),
    ("Λc",    ["6_3", "8_1"],                2286.0),
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


def sector_of(comps):
    n = len(comps)
    if n == 1:
        return 0
    if EW_MARKER in comps:
        return 2
    return 1   # hadron


def widx(comps, weights):
    return [weights[KNOTS.index(K)] for K in comps]


# ─── Opérateurs candidats ──────────────────────────────────────────
def O0_additive(comps, w, p):
    return sum(widx(comps, w))

def O1_multiplicative(comps, w, p):
    return float(np.prod(widx(comps, w)))

def O2_geom_mean(comps, w, p):
    ws = widx(comps, w)
    n = len(ws)
    return n ** p[0] * np.prod(ws) ** (1 / n)

def O3_max_plus(comps, w, p):
    ws = widx(comps, w)
    m_max = max(ws)
    return m_max + p[0] * (sum(ws) - m_max)

def O4_pythag(comps, w, p):
    ws = widx(comps, w)
    return float(np.sqrt(sum(x ** 2 for x in ws)))

def O5_mixed(comps, w, p):
    ws = widx(comps, w)
    return p[0] * max(ws) + p[1] * float(np.sqrt(sum(x ** 2 for x in ws)))

def O6_log_shift(comps, w, p):
    """log m(P) = Σ log m(K) + δ_σ ; δ_0 = 0 (single)"""
    ws = widx(comps, w)
    sigma = sector_of(comps)
    deltas = [0.0, p[0], p[1]]
    return float(np.exp(sum(np.log(x + 1e-30) for x in ws) + deltas[sigma]))

def O7_sect_max(comps, w, p):
    """m(P) = α_σ · max(m) · n^β_σ , 3 secteurs"""
    ws = widx(comps, w)
    n = len(ws)
    sigma = sector_of(comps)
    alpha = [p[0], p[2], p[4]][sigma]
    beta = [p[1], p[3], p[5]][sigma]
    return alpha * max(ws) * n ** beta

def O8_holder(comps, w, p):
    """Hölder mean : (Σ m^a)^(1/a)"""
    a = p[0]
    ws = widx(comps, w)
    if abs(a) < 1e-6:
        return float(np.prod(ws) ** (1 / len(ws)))
    return float((sum(x ** a for x in ws)) ** (1 / a))

def O9_sect_holder(comps, w, p):
    """Hölder mean × couplage sectoriel : c_σ · (Σ m^a_σ)^(1/a_σ)"""
    sigma = sector_of(comps)
    a = [p[0], p[2], p[4]][sigma]
    c = [p[1], p[3], p[5]][sigma]
    ws = widx(comps, w)
    if abs(a) < 1e-6:
        return c * float(np.prod(ws) ** (1 / len(ws)))
    return c * float((sum(x ** a for x in ws)) ** (1 / a))


OPERATORS = [
    ("O0  additif      m(P)=Σ m",                 O0_additive,       0),
    ("O1  multiplicatif m(P)=Π m",                O1_multiplicative, 0),
    ("O2  géom · n^p   m(P)=n^p·(Π m)^(1/n)",     O2_geom_mean,      1),
    ("O3  max + α·rest m(P)=max + α·(Σ-max)",     O3_max_plus,       1),
    ("O4  Pythagore    m(P)=√(Σ m²)",             O4_pythag,         0),
    ("O5  α·max+β·√Σm² (mixte)",                  O5_mixed,          2),
    ("O6  log-shift σ  log m=Σ log m+δ_σ",        O6_log_shift,      2),
    ("O7  α_σ·max·n^β_σ (3 secteurs)",            O7_sect_max,       6),
    ("O8  Hölder       (Σ m^a)^(1/a)",            O8_holder,         1),
    ("O9  Hölder · sect (3 secteurs, 6 params)",  O9_sect_holder,    6),
]


def fit_operator(op, n_params):
    """Optimise jointement {log m(K)} et paramètres op."""
    def loss(x):
        w = np.exp(x[:N_K])
        p = x[N_K:] if n_params > 0 else []
        residuals = []
        for nm, comps, me in PARTICLES:
            try:
                m_pred = op(comps, w, p)
            except (OverflowError, ValueError):
                return 1e6
            if m_pred <= 0 or not np.isfinite(m_pred):
                residuals.append(10)
            else:
                residuals.append(np.log(m_pred / me))
        return float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    # Initialisation : poids = 10, params = 1
    x0 = np.concatenate([np.log(np.ones(N_K) * 10), np.ones(n_params)])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": 30000, "xatol": 1e-7, "fatol": 1e-9,
                            "adaptive": True})
    w_opt = np.exp(res.x[:N_K])
    p_opt = res.x[N_K:] if n_params > 0 else []
    return w_opt, p_opt, res.fun


def main():
    print("=" * 100)
    print("  RECHERCHE D'OPÉRATEUR — COMBINAISON DES POIDS NŒUDS → MASSES")
    print("=" * 100)

    print(f"\n  Optimisation de 10 opérateurs candidats sur {len(PARTICLES)} particules…")
    print(f"  ({N_K} poids m(K) + n_params globaux, Nelder-Mead adaptive)")
    print()

    results = []
    for name, op, npar in OPERATORS:
        try:
            w, p, rms = fit_operator(op, npar)
            results.append((name, op, npar, w, p, rms))
            print(f"  {name:<50} {N_K + npar:>2}p  RMS log = {rms:.4f}  "
                  f"(={(np.exp(rms)-1)*100:>5.0f}%)")
        except Exception as e:
            print(f"  {name:<50} ÉCHEC : {e}")

    # ─── Meilleur opérateur ─────────────────────────────────────────
    results.sort(key=lambda r: r[5])
    print(f"\n  ─── TOP 3 OPÉRATEURS ───")
    for name, op, npar, w, p, rms in results[:3]:
        print(f"\n  {name}  (RMS log = {rms:.4f})")
        print(f"    Paramètres opérateur : {[f'{v:.3f}' for v in p]}")
        print(f"    Poids m(K) MeV :")
        for K, mK in sorted(zip(KNOTS, w), key=lambda kv: kv[1]):
            print(f"       m({K}) = {mK:>10.4f}")

    # ─── Prédictions du meilleur opérateur ───────────────────────────
    best_name, best_op, _, best_w, best_p, best_rms = results[0]
    print(f"\n  ─── PRÉDICTIONS DU MEILLEUR OPÉRATEUR ({best_name.split()[0]}) ───")
    print(f"  RMS log = {best_rms:.4f}  (= +{(np.exp(best_rms)-1)*100:.0f}% relatif)")
    print(f"\n  {'particule':>5} {'n':>2} {'sect':>3} {'m_exp':>11} "
          f"{'m_pred':>11} {'Δ %':>8}")
    print("  " + "-" * 55)
    flags = {"★": 0, "✓": 0, " ": 0, "?": 0}
    for nm, comps, me in PARTICLES:
        m_pred = best_op(comps, best_w, best_p)
        e = (m_pred - me) / me * 100
        sigma = sector_of(comps)
        sect_label = ["L", "H", "E"][sigma]
        if abs(e) < 5:
            flag = "★"
        elif abs(e) < 20:
            flag = "✓"
        elif abs(e) < 50:
            flag = " "
        else:
            flag = "?"
        flags[flag] += 1
        print(f"  {nm:>5} {len(comps):>2} {sect_label:>3} {me:>11.2f} "
              f"{m_pred:>11.2f} {e:>+7.1f} {flag}")

    print(f"\n  Bilan flags : ★ {flags['★']}/  ✓ {flags['✓']}/  "
          f"{flags[' ']}/  ? {flags['?']}  sur {len(PARTICLES)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
