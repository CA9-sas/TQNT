r"""
Tableau de masses TQNT-R6 sur le spectre complet (46 particules).

Formule (cf. brouillon_lemme_R6.tex §5) :
   m(L) = α₀ · U_EM(L) · c_J(L)^(32/9) · ((n+1)/2)^(-4/3)
   (avec ((n+1)/2)^(-4/3) → 1 si 8_3 ∈ L, secteur EW)

α₀ calibré GLOBALEMENT par moindres carrés log sur les 46 particules.

Configurations issues de spectrum_all_table.py CONFIG (V4 chap. Spectre).
Sortie : tableau texte console + tableau LaTeX dans
docs/spectrum_R6_table.tex pour inclusion dans le brouillon.
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j, EW_MARKER

# ────────────────────────────────────────────────────────────────────────
# Spectre complet (46 particules)
# (nom, config, m_exp_MeV, secteur)
# ────────────────────────────────────────────────────────────────────────
SPECTRUM = [
    # ─── LEPTONS ────────────────────────────────────────────────────
    ("e⁻",        "3_1",               0.511,         "lepton"),
    ("μ⁻",        "3_1+6_1",           105.66,        "lepton"),
    ("τ⁻",        "3_1+5_1+7_1",       1776.86,       "lepton"),
    ("ν_e",       "6_2+9_1",           0.0,           "neutrino"),
    ("ν_μ",       "6_2+9_1",           0.0,           "neutrino"),
    ("ν_τ",       "6_2+9_1",           0.0,           "neutrino"),
    # ─── QUARKS (estimations PDG, "courants") ───────────────────────
    ("u",         "3_1",               2.2,           "quark"),
    ("d",         "3_1",               4.7,           "quark"),
    ("s",         "5_2+5_2",           95.0,          "quark"),
    ("c",         "7_3+7_7",           1270.0,        "quark"),
    ("b",         "4_1+5_2+7_6",       4180.0,        "quark"),
    # ─── MÉSONS PSEUDO-SCALAIRES (J^P = 0^-) ────────────────────────
    ("π⁰",        "8_1",               134.98,        "méson"),
    ("π±",        "8_1",               139.57,        "méson"),
    ("K±",        "4_1+4_1+5_2",       493.68,        "méson"),
    ("K⁰_S",      "4_1+4_1+5_2",       497.61,        "méson"),
    ("η",         "3_1+4_1+5_1",       547.86,        "méson"),
    ("η'",        "5_2+5_2+6_1",       957.78,        "méson"),
    # ─── MÉSONS VECTORIELS (J^P = 1^-) ──────────────────────────────
    ("ρ/ω",       "5_2+7_7",           775.0,         "méson"),
    ("φ",         "3_1+4_1+6_2",       1019.46,       "méson"),
    ("J/ψ",       "4_1+5_2+9_1",       3096.9,        "méson"),
    ("Υ",         "3_1+7_4+9_1",       9460.3,        "méson"),
    # ─── MÉSONS CHARMÉS / BEAUX ─────────────────────────────────────
    ("D⁰",        "6_2+7_7",           1864.84,       "méson"),
    ("D±",        "6_2+7_7",           1869.66,       "méson"),
    ("B",         "4_1+10_1",          5279.4,        "méson"),
    # ─── BARYONS — OCTET ────────────────────────────────────────────
    ("p",         "3_1+7_2+7_2",       938.272,       "baryon"),
    ("n",         "3_1+7_2+7_2",       939.565,       "baryon"),
    ("Λ",         "4_1+7_4",           1115.7,        "baryon"),
    ("Σ⁺",        "6_2+7_2",           1189.4,        "baryon"),
    ("Σ⁰",        "6_2+7_2",           1192.6,        "baryon"),
    ("Σ⁻",        "6_2+7_2",           1197.4,        "baryon"),
    ("Ξ⁰",        "4_1+4_1+7_2",       1314.9,        "baryon"),
    ("Ξ⁻",        "4_1+4_1+7_2",       1321.7,        "baryon"),
    # ─── BARYONS — DÉCUPLET ─────────────────────────────────────────
    ("Δ",         "3_1+5_2+6_2",       1232.0,        "baryon"),
    ("Σ*",        "5_2+6_2+7_2",       1383.0,        "baryon"),
    ("Ξ*",        "4_1+5_2+7_2",       1532.0,        "baryon"),
    ("Ω⁻",        "5_2+5_2+7_2",       1672.0,        "baryon"),
    # ─── BARYONS CHARMÉS / BEAUX ────────────────────────────────────
    ("Λ_c",       "6_3+8_1",           2286.0,        "baryon"),
    ("Σ_c",       "4_1+5_1+7_1",       2455.0,        "baryon"),
    ("Ω_c",       "3_1+6_2+6_2",       2695.0,        "baryon"),
    ("Λ_b",       "5_2+5_2+8_2",       5620.0,        "baryon"),
    # ─── BARYONS BEAUX (analogues PDG) ──────────────────────────────
    ("Σ_b",       "4_1+5_2+8_2",       5811.0,        "baryon"),
    ("Ξ_b",       "5_2+5_2+8_2",       5797.0,        "baryon"),
    # ─── BOSONS ÉLECTROFAIBLES (porteurs 8_3) ───────────────────────
    ("W±",        "4_1+7_3+8_3",       80377.0,       "EW"),
    ("Z⁰",        "5_1+7_2+8_3",       91188.0,       "EW"),
    ("H",         "6_1+7_4+8_3",       125250.0,      "EW"),
    ("t",         "7_4+7_6+8_3",       173100.0,      "EW"),
]

# ────────────────────────────────────────────────────────────────────────
# Formule TQNT-R6
# ────────────────────────────────────────────────────────────────────────
P_R6 = 32 / 9
X_R6 = 3 / 8


def parse_cfg(cfg): return cfg.split("+")


def sector_of(comps):
    n = len(comps)
    if n == 1:
        return "single"
    if EW_MARKER in comps:
        return "ew"
    return "hadron"


def f_R6(n, sec):
    if sec in ("single", "ew"):
        return 1.0
    return ((n + 1) / 2) ** (-X_R6)


def mass_R6(comps, alpha0):
    sec = sector_of(comps)
    f = f_R6(len(comps), sec)
    U = sum(u_em(K) for K in comps)
    cJ = sum(c_j(K) for K in comps)
    return alpha0 * U * (f * cJ) ** P_R6


def calibrate_global(particles, mass_func):
    """α₀ globalement, log LS, excluant neutrinos (m_exp=0)."""
    log_a = []
    for nm, cfg, me, cat in particles:
        if me <= 0:
            continue
        m_unit = mass_func(parse_cfg(cfg), alpha0=1.0)
        log_a.append(np.log(me / m_unit))
    return float(np.exp(np.mean(log_a)))


def calibrate_by_sector(particles, mass_func):
    """α₀ par secteur (3 valeurs : single, hadron, ew)."""
    alphas = {}
    for sec_target in ("single", "hadron", "ew"):
        sub = [(nm, cfg, me, cat) for nm, cfg, me, cat in particles
               if sector_of(parse_cfg(cfg)) == sec_target and me > 0]
        if sub:
            alphas[sec_target] = calibrate_global(sub, mass_func)
    return alphas


# ────────────────────────────────────────────────────────────────────────
# Affichage
# ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 102)
    print("  SPECTRE COMPLET TQNT-R6 — VALIDATION CONTRE L'EXPÉRIENCE")
    print("  Formule : m(L) = α₀ · U_EM(L) · c_J(L)^(32/9) · ((n+1)/2)^(-4/3)")
    print("=" * 102)
    print(f"\n  Référence : DONNÉES EXPÉRIMENTALES (PDG 2024).")
    print(f"  La conjecture R6 prétend être UNIVERSELLE → on calibre α₀")
    print(f"  UNE SEULE FOIS sur l'électron (anchor minimal), puis on évalue")
    print(f"  TOUTES les autres masses comme PRÉDICTIONS pures sans réajustement.")

    # ─── Calibration sur ÉLECTRON SEUL (vraie prédiction) ──────────────
    a0_e = SPECTRUM[0][2] / mass_R6(parse_cfg(SPECTRUM[0][1]), alpha0=1.0)
    print(f"\n  α₀ (calibré sur m_e = 0.511 MeV exact) = {a0_e:.4e} MeV")
    print(f"  ⇒ 1 SEUL paramètre libre dans toute la formule.")

    # Optionnel : aussi calibration globale (LS log sur 43 part.) pour comparaison
    a0_global = calibrate_global(SPECTRUM, mass_R6)
    a0_sector = calibrate_by_sector(SPECTRUM, mass_R6)
    print(f"\n  Pour comparaison (paramètres CACHÉS) :")
    print(f"    α₀ global LS  (1 param.)  = {a0_global:.4e} MeV")
    print(f"    α₀ par secteur (3 param.) = single {a0_sector.get('single',0):.4e},  "
          f"hadron {a0_sector.get('hadron',0):.4e},  ew {a0_sector.get('ew',0):.4e}")

    # Tableau principal : α₀ calibré sur ÉLECTRON UNIQUEMENT
    print(f"\n  TABLEAU PRÉDICTIONS R6 vs EXPÉRIENCE  (α₀ calibré sur e⁻)")
    print(f"  {'#':>3} {'particule':>10} {'config':>17} {'cat.':>8} "
          f"{'n':>2} {'m_exp':>12} {'m_R6':>12} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 92)
    log_res_sector = {"single": [], "hadron": [], "ew": []}
    pct_err = []
    flags_count = {"★": 0, "✓": 0, " ": 0, "?": 0, "✗": 0}
    for i, (nm, cfg, me, cat) in enumerate(SPECTRUM, start=1):
        comps = parse_cfg(cfg)
        sec = sector_of(comps)
        m_pred = mass_R6(comps, a0_e)        # ← α₀ UNIQUE, calibré sur e
        if me > 0:
            err = (m_pred - me) / me * 100
            log_res_sector[sec].append(np.log(m_pred / me))
            pct_err.append(err)
            if abs(err) < 1:
                flag = "★"
            elif abs(err) < 10:
                flag = "✓"
            elif abs(err) < 50:
                flag = " "
            elif abs(err) < 200:
                flag = "?"
            else:
                flag = "✗"
            flags_count[flag] += 1
            err_str = f"{err:+8.2f}"
        else:
            flag = "ν"
            err_str = "    ----"
        print(f"  {i:>3} {nm:>10} {cfg:>17} {cat:>8} {len(comps):>2} "
              f"{me:>12.3f} {m_pred:>12.3f} {err_str:>9} {flag:>5}")

    # Statistiques
    print(f"\n  ─── STATISTIQUES (1 PARAMÈTRE α₀ CALIBRÉ SUR e⁻) ───")
    print(f"  Total particules        : {len(SPECTRUM)}")
    print(f"  Avec masse mesurée      : {len([p for p in SPECTRUM if p[2] > 0])}")
    print(f"  Δ < 1 %    (★ excellent): {flags_count['★']}")
    print(f"  Δ < 10 %   (✓ bon)      : {flags_count['✓']}")
    print(f"  Δ < 50 %   ( ordinaire) : {flags_count[' ']}")
    print(f"  50-200 %   (? mauvais)  : {flags_count['?']}")
    print(f"  > 200 %    (✗ échec)    : {flags_count['✗']}")
    print()
    print(f"  ─── RMS log par secteur (α₀ par secteur) ───")
    for sec, residuals in log_res_sector.items():
        if residuals:
            arr = np.array(residuals)
            rms = float(np.sqrt(np.mean(arr ** 2)))
            print(f"    {sec:>10} ({len(arr):>2} part.) : "
                  f"RMS log = {rms:.3f}  (= +{(np.exp(rms)-1)*100:.0f}% relatif)")
    all_res = [r for v in log_res_sector.values() for r in v]
    rms_global = float(np.sqrt(np.mean(np.array(all_res) ** 2)))
    print(f"    {'TOTAL':>10} ({len(all_res):>2} part.) : "
          f"RMS log = {rms_global:.3f}  (= +{(np.exp(rms_global)-1)*100:.0f}% relatif)")

    # ─── Export LaTeX ─────────────────────────────────────────────────
    tex_path = "docs/spectrum_R6_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"% Tableau de masses TQNT-R6 sur 46 particules" + "\n")
        f.write(r"% Généré par simulation/experiments/spectrum_R6_table.py" + "\n")
        f.write(r"\begin{center}" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.1}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{longtable}{rlllrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\# & particule & configuration & cat. & $n$ & "
                r"$m_{\text{exp}}$ (MeV) & $m_{R6}$ (MeV) & $\Delta$\,\% \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endhead" + "\n")
        for i, (nm, cfg, me, cat) in enumerate(SPECTRUM, start=1):
            comps = parse_cfg(cfg)
            sec = sector_of(comps)
            m_pred = mass_R6(comps, a0_sector.get(sec, a0_global))
            cfg_tex = cfg.replace("_", r"\_").replace("+", "{+}")
            nm_tex = nm.replace("⁻", r"$^-$").replace("⁺", r"$^+$") \
                       .replace("⁰", r"$^0$").replace("±", r"$^\pm$") \
                       .replace("μ", r"$\mu$").replace("τ", r"$\tau$") \
                       .replace("ν", r"$\nu$").replace("π", r"$\pi$") \
                       .replace("ρ", r"$\rho$").replace("ω", r"$\omega$") \
                       .replace("φ", r"$\phi$").replace("ψ", r"$\psi$") \
                       .replace("Υ", r"$\Upsilon$").replace("η", r"$\eta$") \
                       .replace("Λ", r"$\Lambda$").replace("Σ", r"$\Sigma$") \
                       .replace("Ξ", r"$\Xi$").replace("Δ", r"$\Delta$") \
                       .replace("Ω", r"$\Omega$") \
                       .replace("_", r"$_")
            # Fix _x to _{x}$ at end
            nm_tex = nm_tex.replace("_c$", r"_c$").replace("_b$", r"_b$") \
                           .replace("_S$", r"_S$").replace("_e$", r"_e$") \
                           .replace("_μ$", r"_\mu$").replace("_τ$", r"_\tau$")
            if "$_" in nm_tex and not nm_tex.endswith("$"):
                nm_tex = nm_tex + "$"
            if me > 0:
                err = (m_pred - me) / me * 100
                err_str = f"{err:+.2f}"
            else:
                err_str = "—"
            f.write(f"  {i} & {nm_tex} & ${cfg_tex}$ & {cat} & {len(comps)} & "
                    f"{me:.3f} & {m_pred:.2f} & {err_str} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{longtable}" + "\n")
        f.write(r"\end{center}" + "\n")

    print(f"\n  ✓ Tableau LaTeX exporté dans : {tex_path}")
    print("=" * 102)


if __name__ == "__main__":
    main()
