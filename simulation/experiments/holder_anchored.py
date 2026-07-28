r"""
OPÉRATEUR DE HÖLDER ANCRÉ SUR L'ÉLECTRON

m(P) = (Σ m(K)^a)^(1/a)

avec ANCRAGE : m(3_1) = m_e = 0.51099895 MeV exactement.
(L'électron est un nœud unique 3_1, donc m(e⁻) = m(3_1) par construction.)

Les "masses des quarks" (u, d, s, c, b) sont des constructions
théoriques (MS-bar, scheme-dependent, current vs constituent),
PAS des observables directes. On les EXCLUT du fit pour ne garder
que les masses directement mesurables.

→ 31 particules observables (au lieu de 36) :
   leptons : e, μ, τ
   mésons  : π, K, η, η', ρ, φ, J/ψ, Υ, D, B
   baryons : p, n, Λ, Σ, Ξ, Δ, Σ*, Ξ*, Ω, Λc, Σc, Ωc, Λb
   EW      : W, Z, H, t

On résout :
   variables    : log m(K) pour K ≠ 3_1 (18 valeurs)  + paramètre a
   contrainte   : m(3_1) = 0.51099895 fixe
   objectif     : minimiser RMS log sur 31 particules

Le résultat donne les "vraies" masses des nœuds simples telles que
l'opérateur de Hölder reproduit le spectre expérimental observable.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_K = len(KNOTS)
M_E = 0.51099895   # masse exacte de l'électron
IDX_31 = KNOTS.index("3_1")

# 31 particules avec masses directement mesurables (quarks exclus)
PARTICLES = [
    ("e⁻",    ["3_1"],                       M_E),         # ANCRAGE
    ("μ⁻",    ["3_1", "6_1"],                105.6584),
    ("τ⁻",    ["3_1", "5_1", "7_1"],         1776.86),
    ("π⁰",    ["8_1"],                       134.9768),
    ("π±",    ["8_1"],                       139.5704),
    ("K±",    ["4_1", "4_1", "5_2"],         493.677),
    ("K⁰_S",  ["4_1", "4_1", "5_2"],         497.611),
    ("η",     ["3_1", "4_1", "5_1"],         547.862),
    ("η'",    ["5_2", "5_2", "6_1"],         957.78),
    ("ρ/ω",   ["5_2", "7_7"],                775.0),
    ("φ",     ["3_1", "4_1", "6_2"],         1019.461),
    ("J/ψ",   ["4_1", "5_2", "9_1"],         3096.900),
    ("Υ",     ["3_1", "7_4", "9_1"],         9460.30),
    ("D⁰",    ["6_2", "7_7"],                1864.84),
    ("D±",    ["6_2", "7_7"],                1869.66),
    ("B",     ["4_1", "10_1"],               5279.4),
    ("p",     ["3_1", "7_2", "7_2"],         938.27208),
    ("n",     ["3_1", "7_2", "7_2"],         939.5654),
    ("Λ",     ["4_1", "7_4"],                1115.683),
    ("Σ⁺",    ["6_2", "7_2"],                1189.37),
    ("Σ⁰",    ["6_2", "7_2"],                1192.642),
    ("Σ⁻",    ["6_2", "7_2"],                1197.449),
    ("Ξ⁰",    ["4_1", "4_1", "7_2"],         1314.86),
    ("Ξ⁻",    ["4_1", "4_1", "7_2"],         1321.71),
    ("Δ",     ["3_1", "5_2", "6_2"],         1232.0),
    ("Σ*",    ["5_2", "6_2", "7_2"],         1383.7),
    ("Ξ*",    ["4_1", "5_2", "7_2"],         1531.8),
    ("Ω⁻",    ["5_2", "5_2", "7_2"],         1672.45),
    ("Λc",    ["6_3", "8_1"],                2286.46),
    ("Σc",    ["4_1", "5_1", "7_1"],         2453.97),
    ("Ωc",    ["3_1", "6_2", "6_2"],         2695.2),
    ("Λb",    ["5_2", "5_2", "8_2"],         5619.60),
    ("W±",    ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",    ["5_1", "7_2", "8_3"],         91187.6),
    ("H",     ["6_1", "7_4", "8_3"],         125250.0),
    ("t",     ["7_4", "7_6", "8_3"],         173100.0),
]


def holder_mean(values, a):
    """Moyenne de Hölder (Σ x^a)^(1/a) avec cas limite a → 0."""
    if abs(a) < 1e-8:
        return float(np.prod(values) ** (1 / len(values)))
    s = sum(v ** a for v in values)
    return float(s ** (1 / a))


def predict_mass(comps, w, a):
    ws = [w[KNOTS.index(K)] for K in comps]
    return holder_mean(ws, a)


def fit_anchored():
    """Optimisation : 18 poids (les autres) + a, m(3_1) = M_E fixe."""
    free_indices = [i for i in range(N_K) if i != IDX_31]
    N_free = len(free_indices)

    def unpack(x):
        log_w = x[:N_free]
        a = x[-1]
        w = np.zeros(N_K)
        w[IDX_31] = M_E
        for k, idx in enumerate(free_indices):
            w[idx] = np.exp(log_w[k])
        return w, a

    def loss(x):
        w, a = unpack(x)
        residuals = []
        for nm, comps, me in PARTICLES:
            try:
                m_pred = predict_mass(comps, w, a)
            except (OverflowError, ZeroDivisionError, ValueError):
                return 1e6
            if m_pred <= 0 or not np.isfinite(m_pred):
                residuals.append(20)
            else:
                residuals.append(np.log(m_pred / me))
        return float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    # Initialisation : poids croissants avec cr, a = 0.5
    init_w_log = []
    for idx in free_indices:
        cr = int(KNOTS[idx].split("_")[0])
        init_w_log.append(np.log(2 * cr))
    x0 = np.concatenate([init_w_log, [0.5]])

    # Plusieurs restarts pour éviter minima locaux
    best = None
    for trial_a in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
        x_try = x0.copy()
        x_try[-1] = trial_a
        res = minimize(loss, x_try, method="Nelder-Mead",
                       options={"maxiter": 30000, "xatol": 1e-8,
                                "fatol": 1e-10, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
            print(f"    trial a₀={trial_a}: RMS = {res.fun:.4f}, "
                  f"a_opt = {res.x[-1]:.4f}")
    return unpack(best.x) + (best.fun,)


def main():
    print("=" * 100)
    print("  OPÉRATEUR DE HÖLDER ANCRÉ SUR L'ÉLECTRON")
    print("=" * 100)
    print(f"\n  Ancrage : m(3_1) = m_e = {M_E} MeV (exact)")
    print(f"  Particules : {len(PARTICLES)} observables directs (quarks exclus)")
    print(f"  Variables : 18 poids log m(K) + 1 paramètre a")
    print(f"\n  Optimisation (Nelder-Mead, 6 restarts) :")

    w, a, rms = fit_anchored()

    print(f"\n  ─── PARAMÈTRE OPÉRATEUR ───")
    print(f"  a optimal = {a:.5f}")
    if abs(a - 1) < 0.05:
        print(f"  ≈ additif (Σ m)")
    elif abs(a - 2) < 0.1:
        print(f"  ≈ Pythagore (√Σm²)")
    elif abs(a) < 0.1:
        print(f"  ≈ moyenne géométrique")
    elif a < 0:
        print(f"  → moyenne anti-pondérée (dominée par les petits)")
    elif 0 < a < 1:
        print(f"  → moyenne sub-additive (entre géométrique et additif)")
    elif a > 1:
        print(f"  → moyenne sur-additive (entre additif et max)")

    print(f"\n  ─── POIDS m(K) INFÉRÉS ───")
    print(f"  (croissants par valeur)")
    print(f"  {'K':>5} {'m(K) MeV':>15} {'cr':>3} {'commentaire':<20}")
    print("  " + "-" * 50)
    for K, mK in sorted(zip(KNOTS, w), key=lambda kv: kv[1]):
        cr = int(K.split("_")[0])
        anchor = " ★ ANCRÉ (e⁻)" if K == "3_1" else ""
        print(f"  {K:>5} {mK:>15.5f} {cr:>3} {anchor:<20}")

    # Prédictions complètes
    print(f"\n  ─── PRÉDICTIONS SUR LES 31 PARTICULES OBSERVABLES ───")
    print(f"  {'particule':>5} {'n':>2} {'configuration':>16} "
          f"{'m_exp':>11} {'m_pred':>11} {'Δ %':>8} {'flag':>5}")
    print("  " + "-" * 75)
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    residuals_log = []
    rows = []
    for nm, comps, me in PARTICLES:
        m_pred = predict_mass(comps, w, a)
        err = (m_pred - me) / me * 100
        residuals_log.append(np.log(m_pred / me))
        rows.append((nm, comps, me, m_pred, err))
        if abs(err) < 0.5:
            flag = "★★"
        elif abs(err) < 5:
            flag = "★"
        elif abs(err) < 20:
            flag = "✓"
        elif abs(err) < 50:
            flag = " "
        else:
            flag = "?"
        flags[flag] += 1
        cfg_str = "+".join(comps)
        print(f"  {nm:>5} {len(comps):>2} {cfg_str:>16} "
              f"{me:>11.3f} {m_pred:>11.3f} {err:>+7.2f} {flag:>5}")

    rms_log = float(np.sqrt(np.mean(np.array(residuals_log) ** 2)))
    print(f"\n  RMS log = {rms_log:.4f}  (= +{(np.exp(rms_log)-1)*100:.1f}% relatif)")

    print(f"\n  ─── BILAN ───")
    total = len(PARTICLES)
    print(f"  ★★ (<0.5%) : {flags['★★']}/{total}")
    print(f"  ★  (<5%)   : {flags['★']}/{total}")
    print(f"  ✓  (<20%)  : {flags['✓']}/{total}")
    print(f"      (<50%) : {flags[' ']}/{total}")
    print(f"  ?  (>50%)  : {flags['?']}/{total}")

    excellent = flags["★★"] + flags["★"]
    print(f"\n  Précision < 5%  : {excellent}/{total} ({excellent/total*100:.0f}%)")
    print(f"  Précision < 20% : {excellent + flags['✓']}/{total} "
          f"({(excellent + flags['✓'])/total*100:.0f}%)")

    # Export LaTeX
    tex_path = "docs/holder_anchored_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"% Opérateur de Hölder ancré sur l'électron" + "\n")
        f.write(r"% Poids m(K) :" + "\n")
        f.write(r"\begin{center}" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.1}" + "\n")
        f.write(r"\begin{tabular}{lr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(f"\\multicolumn{{2}}{{c}}{{Opérateur : $m(P) = "
                f"\\left(\\sum_K m(K)^{{{a:.4f}}}\\right)^{{1/{a:.4f}}}$}} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(r"$K$ & $m(K)$ (MeV) \\" + "\n")
        f.write(r"\midrule" + "\n")
        for K, mK in sorted(zip(KNOTS, w), key=lambda kv: kv[1]):
            anchor = r" $\star$" if K == "3_1" else ""
            f.write(f"  ${K.replace('_', '_')}$ & {mK:.4f}{anchor} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{center}" + "\n\n")
        # Prédictions
        f.write(r"\begin{center}\small" + "\n")
        f.write(r"\begin{tabular}{lrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"particule & $m_{\text{exp}}$ (MeV) & $m_{\text{pred}}$ (MeV) & $\Delta$ \% \\" + "\n")
        f.write(r"\midrule" + "\n")
        for nm, comps, me, m_pred, err in rows:
            f.write(f"  {nm} & {me:.3f} & {m_pred:.3f} & {err:+.2f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{center}" + "\n")
    print(f"\n  ✓ Tableau LaTeX exporté : {tex_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
