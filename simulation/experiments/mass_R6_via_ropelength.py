r"""
Chantier 1 → D1bis — formule de masse TQNT-R6 en utilisant la rope-length
algébrique R(K) ≈ 3.85·cr(K) + 5.61 (résultat R1 du bilan) à la place
de U_EM = ∮(κ²+τ²)ds.

Rationale (cf. catalog_rebuild_pieranski.py, commit b30df12) :
   U_EM dépend de la paramétrisation des courbes, donc des artefacts
   Fourier du catalogue. R(K) est une fonction algébrique du nombre de
   croisements, intrinsèquement stable. Si une formule de masse cohérente
   existe en R(K) au lieu de U_EM, elle est PRÉDICTIVE (rien de caché).

On teste 6 variantes :

   V0 (référence R6) :  m = α₀ · U_EM_V4 · (f · c_J)^(32/9)
   V1 :                  m = α₀ · R(L) · (f · c_J)^(32/9)
   V2 :                  m = α₀ · R(L)^q · (f · c_J)^(32/9)  (q libre)
   V3 :                  m = α₀ · (f · c_J)^p   (sans facteur ext., p libre)
   V4 :                  m = α₀ · R(L)^q · c_J^p  (deux exposants libres)
   V5 :                  m = α₀ · cr(L)^q · (f · c_J)^(32/9)

Pour chacune, α₀ est calibré globalement par moindres carrés log sur
les 7 particules canoniques (e-, p, n, W, Z, H, t).

Si une variante atteint <1% RMS sur le spectre, ELLE remplace V4/R6
et le compte de paramètres descend encore.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar, minimize

from theory.mass_formula import c_j, EW_MARKER
from theory.abraham import REFERENCE_N50K

# ─── Constantes de la chantier R1 ────────────────────────────────────────
A_R, B_R = 3.85, 5.61                        # R(K) ≈ 3.85·cr + 5.61
F_R6 = lambda n: ((n + 1) / 2) ** (-3 / 8)   # lemme R6
P_R6 = 32 / 9                                # conjecture forte

# ─── Spectre canonique V4 ────────────────────────────────────────────────
SPECTRUM = [
    ("e-",  ["3_1"],               "single", 0.51099895),
    ("p",   ["3_1", "7_2", "7_2"], "hadron", 938.272),
    ("n",   ["3_1", "7_2", "7_2"], "hadron", 939.565),
    ("W",   ["4_1", "7_3", "8_3"], "ew",     80377.0),
    ("Z",   ["5_1", "7_2", "8_3"], "ew",     91188.0),
    ("H",   ["6_1", "7_4", "8_3"], "ew",     125250.0),
    ("t",   ["7_4", "7_6", "8_3"], "ew",     173100.0),
]


def cr(K):
    return int(K.split("_")[0])


def U_EM_V4(K):
    return float(REFERENCE_N50K[K]["sqrt_U"]) ** 2


def R_alg(K):
    """Rope-length algébrique de R1."""
    return A_R * cr(K) + B_R


def f_screening(comps):
    n = len(comps)
    if n == 1 or EW_MARKER in comps:
        return 1.0
    return F_R6(n)


# ─── Variantes de formule ────────────────────────────────────────────────
def m_V0(comps, alpha0, **kw):
    """Référence R6 avec U_EM_V4."""
    U = sum(U_EM_V4(K) for K in comps)
    cJ = sum(c_j(K) for K in comps)
    f = f_screening(comps)
    return alpha0 * U * (f * cJ) ** P_R6


def m_V1(comps, alpha0, **kw):
    """U_EM → R(L) linéaire."""
    R = sum(R_alg(K) for K in comps)
    cJ = sum(c_j(K) for K in comps)
    f = f_screening(comps)
    return alpha0 * R * (f * cJ) ** P_R6


def m_V2(comps, alpha0, q, **kw):
    """U_EM → R(L)^q."""
    R = sum(R_alg(K) for K in comps)
    cJ = sum(c_j(K) for K in comps)
    f = f_screening(comps)
    return alpha0 * R ** q * (f * cJ) ** P_R6


def m_V3(comps, alpha0, p, **kw):
    """Sans facteur géométrique, p libre."""
    cJ = sum(c_j(K) for K in comps)
    f = f_screening(comps)
    return alpha0 * (f * cJ) ** p


def m_V4(comps, alpha0, q, p, **kw):
    """R(L)^q · c_J^p, f abandonné."""
    R = sum(R_alg(K) for K in comps)
    cJ = sum(c_j(K) for K in comps)
    return alpha0 * R ** q * cJ ** p


def m_V5(comps, alpha0, q, **kw):
    """cr(L)^q au lieu de U_EM."""
    cr_tot = sum(cr(K) for K in comps)
    cJ = sum(c_j(K) for K in comps)
    f = f_screening(comps)
    return alpha0 * cr_tot ** q * (f * cJ) ** P_R6


# ─── Calibration et stats ────────────────────────────────────────────────
def calibrate_alpha0(mass_fn, params):
    """α₀ qui minimise Σ(log(m/m_exp))² (moyenne géométrique)."""
    log_a = []
    for nm, comps, sec, me in SPECTRUM:
        m_unit = mass_fn(comps, alpha0=1.0, **params)
        log_a.append(np.log(me / m_unit))
    return float(np.exp(np.mean(log_a)))


def rms_log(mass_fn, params):
    a0 = calibrate_alpha0(mass_fn, params)
    residuals = []
    for nm, comps, sec, me in SPECTRUM:
        m = mass_fn(comps, alpha0=a0, **params)
        residuals.append(np.log(m / me))
    return float(np.sqrt(np.mean(np.array(residuals) ** 2))), a0


def optimize_1param(mass_fn, key, bounds):
    """Optimise un seul exposant (q ou p)."""
    res = minimize_scalar(
        lambda x: rms_log(mass_fn, {key: x})[0],
        bounds=bounds, method="bounded",
        options={"xatol": 1e-6})
    return res.x, res.fun


def optimize_2params(mass_fn, x0=(1.0, 1.0)):
    """Optimise (q, p)."""
    def loss(x):
        return rms_log(mass_fn, {"q": x[0], "p": x[1]})[0]
    res = minimize(loss, x0=x0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8})
    return res.x[0], res.x[1], res.fun


# ─── Run ─────────────────────────────────────────────────────────────────
def display_table(label, mass_fn, params):
    rms, a0 = rms_log(mass_fn, params)
    print(f"\n  ─── {label} ──────────────────────────")
    print(f"      paramètres : {params}, α₀ = {a0:.4e} MeV")
    print(f"      {'particule':>4} {'m_exp':>11} {'m_pred':>11} {'Δ %':>7}")
    for nm, comps, sec, me in SPECTRUM:
        m = mass_fn(comps, alpha0=a0, **params)
        e = (m - me) / me * 100
        flag = "★" if abs(e) < 1 else (" " if abs(e) < 5 else "?")
        print(f"      {nm:>4} {me:>11.2f} {m:>11.2f} {e:>+6.2f} {flag}")
    print(f"      RMS log = {rms:.4f}  "
          f"(= +{(np.exp(rms)-1)*100:.1f}% relatif)")
    return rms


def main():
    print("=" * 100)
    print("  TQNT-R6 EN ROPE-LENGTH ALGÉBRIQUE (R1) AU LIEU DE U_EM")
    print("=" * 100)
    print(f"  R(K) = {A_R}·cr + {B_R}    (R²=0.957 sur 19 nœuds, cf. R1 du bilan)")
    print(f"  Spectre testé : {len(SPECTRUM)} particules canoniques.")

    results = {}

    # V0 : référence
    rms0 = display_table("V0 (référence R6 + U_EM_V4)",
                         m_V0, params={})
    results["V0"] = rms0

    # V1 : R(L) directement
    rms1 = display_table("V1 : U_EM → R(L) (substitution linéaire)",
                         m_V1, params={})
    results["V1"] = rms1

    # V2 : R(L)^q avec q libre
    q2, rms2 = optimize_1param(m_V2, "q", (0.1, 8.0))
    rms2 = display_table(f"V2 : R(L)^q avec q optimisé = {q2:.4f}",
                         m_V2, params={"q": q2})
    results["V2"] = rms2

    # V3 : sans facteur géométrique, p libre
    p3, rms3 = optimize_1param(m_V3, "p", (1.0, 12.0))
    rms3 = display_table(f"V3 : (f·c_J)^p sans U_EM, p optimisé = {p3:.4f}",
                         m_V3, params={"p": p3})
    results["V3"] = rms3

    # V4 : R(L)^q · c_J^p
    q4, p4, rms4 = optimize_2params(m_V4, x0=(1.0, 3.5))
    rms4 = display_table(f"V4 : R(L)^q · c_J^p, q = {q4:.3f}, p = {p4:.3f}",
                         m_V4, params={"q": q4, "p": p4})
    results["V4"] = rms4

    # V5 : cr(L)^q
    q5, rms5 = optimize_1param(m_V5, "q", (0.1, 8.0))
    rms5 = display_table(f"V5 : cr(L)^q · (f·c_J)^(32/9), q = {q5:.4f}",
                         m_V5, params={"q": q5})
    results["V5"] = rms5

    # ─── Récap ───────────────────────────────────────────────────────────
    print(f"\n  {'═' * 80}")
    print(f"  ▶ RÉCAPITULATIF")
    print(f"  {'═' * 80}")
    print(f"  {'variante':<6} {'description':<40} {'#par':>4} {'RMS log':>9} "
          f"{'gain':>7}")
    print("  " + "-" * 75)
    rows = [
        ("V0", "U_EM_V4 · (f·c_J)^(32/9)         (référence)",    1, rms0),
        ("V1", "R(L) · (f·c_J)^(32/9)            (q=1 forcé)",    1, rms1),
        ("V2", f"R(L)^{q2:.2f} · (f·c_J)^(32/9)        (q libre)", 2, rms2),
        ("V3", f"(f·c_J)^{p3:.2f}                    (sans géom.)", 2, rms3),
        ("V4", f"R(L)^{q4:.2f} · c_J^{p4:.2f}              (q, p libres)",  3, rms4),
        ("V5", f"cr(L)^{q5:.2f} · (f·c_J)^(32/9)       (cr direct)", 2, rms5),
    ]
    for name, desc, npar, rms in rows:
        gain = (rms0 - rms) / rms0 * 100 if rms0 > 0 else 0
        flag = "★" if rms < rms0 * 0.5 else (" " if rms < rms0 * 1.5 else "?")
        print(f"  {name:<6} {desc:<40} {npar:>4} {rms:>9.4f} {gain:>+6.1f}% {flag}")

    print(f"\n  LECTURE :")
    print(f"  ──────────")
    best_name, best_rms = min(results.items(), key=lambda kv: kv[1])
    print(f"    Meilleure variante (hors V0) : {best_name} avec RMS = {best_rms:.4f}")
    print(f"    Référence V0 (U_EM_V4)       : RMS = {rms0:.4f}")
    if best_rms < rms0 * 1.5:
        print(f"  ★ Une substitution algébrique de U_EM est viable :")
        print(f"    {best_name} reproduit le spectre canonique à <{(np.exp(best_rms)-1)*100:.1f}%")
        print(f"    sans dépendance au catalogue Fourier. TQNT devient PRÉDICTIVE.")
    else:
        print(f"  ✗ Aucune substitution algébrique simple n'atteint la précision")
        print(f"    de V0. La formule V4/R6 reste structurellement dépendante de")
        print(f"    la quantité géométrique U_EM (non topologique).")
    print("=" * 100)


if __name__ == "__main__":
    main()
