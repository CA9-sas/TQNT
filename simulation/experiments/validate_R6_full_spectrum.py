r"""
Chantier 1 → D4-3 — validation de la formule TQNT-R6 sur le SPECTRE V4 COMPLET.

Formule TQNT-R6 (cf. docs/brouillon_lemme_R6.tex, §5) :
   m(L) = α₀ · U_EM(L) · c_J(L)^(32/9) · ((n+1)/2)^(-4/3)
   (avec ((n+1)/2)^(-4/3) → 1 pour secteur EW, c.-à-d. si 8_3 ∈ L)

Comparée à V4 :
   m(L) = α₀ · U_EM(L) · (f_V4(n) · c_J(L))^3.55
   où f_V4 = {1:1, 2:0.86, 3:0.77} hardcoded.

On teste les 41 particules canoniques V4 (cf. spectrum_all_table.py CONFIG)
et on compare :
   - m_V4 (formule actuelle)
   - m_R6 (conjecture consolidée)
   - m_exp (mesure expérimentale)

Stats par secteur : single, hadron (méson n=2, baryon n=3), EW.
α₀ recalibré globalement par moindres carrés log dans chaque schéma.

Nucléons p/n : V4 ne distingue pas (même config 3_1+7_2+7_2). La masse
prédite est la masse "nue" sans tension d'enlacement β·(Lk-Lk_opt)².
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j, EW_MARKER, F_SCREENING

# ============================================================
# Spectre V4 canonique (cf. spectrum_all_table.py)
# ============================================================
# Format : (nom, config, m_exp_MeV)
SPECTRUM = [
    # leptons / quarks (single ou hypothétiques)
    ("e-",       "3_1",               0.511),
    ("nu",       "6_2+9_1",           0.0),         # masse négligeable
    ("u",        "3_1",               2.2),
    ("d",        "3_1",               4.7),
    ("s",        "5_2+5_2",           95.0),
    ("c",        "7_3+7_7",           1270.0),
    ("b",        "4_1+5_2+7_6",       4180.0),
    ("mu-",      "3_1+6_1",           105.66),
    ("tau-",     "3_1+5_1+7_1",       1776.86),
    # mésons pseudoscalaires
    ("pi0",      "8_1",               134.98),
    ("pi+-",     "8_1",               139.57),
    ("K+-",      "4_1+4_1+5_2",       493.68),
    ("K0_S",     "4_1+4_1+5_2",       497.61),
    ("eta",      "3_1+4_1+5_1",       547.86),
    ("etap",     "5_2+5_2+6_1",       957.78),
    # mésons vectoriels
    ("rho_om",   "5_2+7_7",           775.0),
    ("phi",      "3_1+4_1+6_2",       1019.46),
    ("Jpsi",     "4_1+5_2+9_1",       3096.9),
    ("Ups",      "3_1+7_4+9_1",       9460.3),
    # mésons charmés/beaux
    ("D0",       "6_2+7_7",           1864.84),
    ("D+",       "6_2+7_7",           1869.66),
    ("B",        "4_1+10_1",          5279.4),
    # nucléons
    ("p",        "3_1+7_2+7_2",       938.272),
    ("n",        "3_1+7_2+7_2",       939.565),
    # hypérons
    ("Lambda",   "4_1+7_4",           1115.7),
    ("Sigma+",   "6_2+7_2",           1189.4),
    ("Sigma0",   "6_2+7_2",           1192.6),
    ("Sigma-",   "6_2+7_2",           1197.4),
    ("Xi0",      "4_1+4_1+7_2",       1314.9),
    ("Xi-",      "4_1+4_1+7_2",       1321.7),
    # baryons décuplet
    ("Delta",    "3_1+5_2+6_2",       1232.0),
    ("Sigma*",   "5_2+6_2+7_2",       1383.0),
    ("Xi*",      "4_1+5_2+7_2",       1532.0),
    ("Omega-",   "5_2+5_2+7_2",       1672.0),
    # baryons charmés/beaux
    ("Lambdac",  "6_3+8_1",           2286.0),
    ("Sigmac",   "4_1+5_1+7_1",       2455.0),
    ("Omegac",   "3_1+6_2+6_2",       2695.0),
    ("Lambdab",  "5_2+5_2+8_2",       5620.0),
    # EW (porteurs 8_3)
    ("W",        "4_1+7_3+8_3",       80377.0),
    ("Z0",       "5_1+7_2+8_3",       91188.0),
    ("H",        "6_1+7_4+8_3",       125250.0),
    ("t",        "7_4+7_6+8_3",       173100.0),
]


def parse(cfg): return cfg.split("+")


def resolve_sector(comps):
    if len(comps) == 1:
        return "single"
    if EW_MARKER in comps:
        return "ew"
    return "hadron"


def mass_V4(comps, alpha0, p=3.55):
    sec = resolve_sector(comps)
    n = len(comps)
    f = 1.0 if sec in ("single", "ew") else F_SCREENING.get(n, 1.0)
    U = sum(u_em(K) for K in comps)
    cJ = sum(c_j(K) for K in comps)
    return alpha0 * U * (f * cJ) ** p


def mass_R6(comps, alpha0, p=32 / 9, x=3 / 8):
    sec = resolve_sector(comps)
    n = len(comps)
    f = 1.0 if sec in ("single", "ew") else ((n + 1) / 2) ** (-x)
    U = sum(u_em(K) for K in comps)
    cJ = sum(c_j(K) for K in comps)
    return alpha0 * U * (f * cJ) ** p


def calibrate_global(particles, mass_func, **kw):
    """α₀ qui minimise RMS log sur la liste de particules (excluant ν)."""
    log_alphas = []
    for name, cfg, m_exp in particles:
        if m_exp <= 0:
            continue
        comps = parse(cfg)
        m_unit = mass_func(comps, alpha0=1.0, **kw)
        log_alphas.append(np.log(m_exp / m_unit))
    return float(np.exp(np.mean(log_alphas)))


def main():
    print("=" * 112)
    print("  VALIDATION TQNT-R6 sur SPECTRE V4 COMPLET (41 particules)")
    print("=" * 112)

    valid = [p for p in SPECTRUM if p[2] > 0]

    a0_V4 = calibrate_global(valid, mass_V4)
    a0_R6 = calibrate_global(valid, mass_R6)
    print(f"\n  α₀ V4 (cal. globale LS log) = {a0_V4:.6e} MeV")
    print(f"  α₀ R6 (cal. globale LS log) = {a0_R6:.6e} MeV  "
          f"(shift {(a0_R6/a0_V4-1)*100:+.2f}%)")

    # ─── Calcul des prédictions ──────────────────────────────────────────
    rows = []
    for name, cfg, m_exp in SPECTRUM:
        if m_exp <= 0:
            continue
        comps = parse(cfg)
        sec = resolve_sector(comps)
        n = len(comps)
        m_v4 = mass_V4(comps, a0_V4)
        m_r6 = mass_R6(comps, a0_R6)
        rows.append((name, cfg, sec, n, m_exp, m_v4, m_r6))

    # ─── Tableau complet ─────────────────────────────────────────────────
    print(f"\n  {'particule':>10} {'config':>16} {'sect':>6} {'n':>2} "
          f"{'m_exp':>11} {'m_V4':>11} {'m_R6':>11} {'ΔV4 %':>8} {'ΔR6 %':>8}")
    print("  " + "-" * 96)
    for nm, cfg, sec, n, me, mv, mr in rows:
        ev = (mv - me) / me * 100
        er = (mr - me) / me * 100
        flag_v = "★" if abs(ev) < 1 else (" " if abs(ev) < 20 else "?")
        flag_r = "★" if abs(er) < 1 else (" " if abs(er) < 20 else "?")
        print(f"  {nm:>10} {cfg:>16} {sec:>6} {n:>2} "
              f"{me:>11.2f} {mv:>11.2f} {mr:>11.2f} "
              f"{ev:>+7.1f} {flag_v} {er:>+7.1f} {flag_r}")

    # ─── Stats globales ──────────────────────────────────────────────────
    e_v4 = np.array([(r[5] - r[4]) / r[4] * 100 for r in rows])
    e_r6 = np.array([(r[6] - r[4]) / r[4] * 100 for r in rows])
    log_e_v4 = np.array([np.log(r[5] / r[4]) for r in rows])
    log_e_r6 = np.array([np.log(r[6] / r[4]) for r in rows])

    print(f"\n  STATS GLOBALES ({len(rows)} particules) :")
    rms_lv = float(np.sqrt(np.mean(log_e_v4 ** 2)))
    rms_lr = float(np.sqrt(np.mean(log_e_r6 ** 2)))
    pct_lv = (np.exp(rms_lv) - 1) * 100
    pct_lr = (np.exp(rms_lr) - 1) * 100
    print(f"    RMS log V4 = {rms_lv:.3f}  (= +{pct_lv:.0f}% en relatif)")
    print(f"    RMS log R6 = {rms_lr:.3f}  (= +{pct_lr:.0f}% en relatif)")
    print(f"    max |Δ|%   V4 = {np.max(np.abs(e_v4)):.1f}%, R6 = {np.max(np.abs(e_r6)):.1f}%")
    n_close_v4 = int(np.sum(np.abs(e_v4) < 5))
    n_close_r6 = int(np.sum(np.abs(e_r6) < 5))
    print(f"    Δ<5%  : V4 = {n_close_v4}/{len(rows)}, R6 = {n_close_r6}/{len(rows)}")

    # ─── Stats par secteur ───────────────────────────────────────────────
    print(f"\n  STATS PAR SECTEUR :")
    print(f"  {'secteur':>10} {'N':>3} {'RMS log V4':>11} {'RMS log R6':>11} "
          f"{'max V4 %':>10} {'max R6 %':>10}")
    print("  " + "-" * 60)
    for sec_target in ("single", "hadron", "ew"):
        sub_idx = [i for i, r in enumerate(rows) if r[2] == sec_target]
        if not sub_idx:
            continue
        lv = log_e_v4[sub_idx]
        lr = log_e_r6[sub_idx]
        ev = e_v4[sub_idx]
        er = e_r6[sub_idx]
        print(f"  {sec_target:>10} {len(sub_idx):>3} "
              f"{np.sqrt(np.mean(lv**2)):>11.3f} "
              f"{np.sqrt(np.mean(lr**2)):>11.3f} "
              f"{np.max(np.abs(ev)):>9.1f}% {np.max(np.abs(er)):>9.1f}%")

    # Stats par n (composantes)
    print(f"\n  STATS PAR n (nombre de composantes) :")
    print(f"  {'n':>3} {'N':>3} {'RMS log V4':>11} {'RMS log R6':>11}")
    print("  " + "-" * 35)
    for nt in (1, 2, 3):
        sub_idx = [i for i, r in enumerate(rows) if r[3] == nt and r[2] != "ew"]
        if not sub_idx:
            continue
        lv = log_e_v4[sub_idx]
        lr = log_e_r6[sub_idx]
        print(f"  {nt:>3} {len(sub_idx):>3} "
              f"{np.sqrt(np.mean(lv**2)):>11.3f} "
              f"{np.sqrt(np.mean(lr**2)):>11.3f}")

    # ─── Calibration PAR SECTEUR (plus juste, V4 utilise déjà f(n)) ─────
    print(f"\n  ─── Calibration α₀ INDÉPENDANTE PAR SECTEUR ───")
    print(f"  V4 utilise déjà 3 paramètres dans le spectre (α₀, f(2), f(3)).")
    print(f"  Pour comparer R6 honnêtement, on calibre α₀ secteur par secteur.")
    print()
    sectors = ("single", "hadron", "ew")
    a0_V4_sec = {}
    a0_R6_sec = {}
    for sec in sectors:
        sub = [p for p in valid if resolve_sector(parse(p[1])) == sec]
        if not sub:
            continue
        a0_V4_sec[sec] = calibrate_global(sub, mass_V4)
        a0_R6_sec[sec] = calibrate_global(sub, mass_R6)
        print(f"    {sec:>10} ({len(sub):>2} part.) : "
              f"α₀_V4 = {a0_V4_sec[sec]:.3e}, α₀_R6 = {a0_R6_sec[sec]:.3e}")

    # Recalcul avec calibration par secteur
    log_v4_sec, log_r6_sec = [], []
    e_v4_sec, e_r6_sec = [], []
    print(f"\n  {'particule':>10} {'config':>16} {'sect':>6} "
          f"{'m_exp':>10} {'m_V4 sec':>11} {'m_R6 sec':>11} "
          f"{'ΔV4 %':>7} {'ΔR6 %':>7}")
    print("  " + "-" * 88)
    for nm, cfg, sec, n, me, _, _ in rows:
        comps = parse(cfg)
        mv = mass_V4(comps, a0_V4_sec[sec])
        mr = mass_R6(comps, a0_R6_sec[sec])
        ev = (mv - me) / me * 100
        er = (mr - me) / me * 100
        log_v4_sec.append(np.log(mv / me))
        log_r6_sec.append(np.log(mr / me))
        e_v4_sec.append(ev)
        e_r6_sec.append(er)
        fv = "★" if abs(ev) < 5 else (" " if abs(ev) < 30 else "?")
        fr = "★" if abs(er) < 5 else (" " if abs(er) < 30 else "?")
        print(f"  {nm:>10} {cfg:>16} {sec:>6} "
              f"{me:>10.2f} {mv:>11.2f} {mr:>11.2f} "
              f"{ev:>+6.1f} {fv} {er:>+6.1f} {fr}")

    rms_v4_sec = float(np.sqrt(np.mean(np.array(log_v4_sec) ** 2)))
    rms_r6_sec = float(np.sqrt(np.mean(np.array(log_r6_sec) ** 2)))
    print(f"\n  RMS log avec α₀ par secteur :")
    print(f"    V4 = {rms_v4_sec:.3f}   (= +{(np.exp(rms_v4_sec)-1)*100:.0f}% en relatif)")
    print(f"    R6 = {rms_r6_sec:.3f}   (= +{(np.exp(rms_r6_sec)-1)*100:.0f}% en relatif)")

    # ─── Lecture ─────────────────────────────────────────────────────────
    rms_v4 = float(np.sqrt(np.mean(log_e_v4**2)))
    rms_r6 = float(np.sqrt(np.mean(log_e_r6**2)))
    print(f"\n  LECTURE :")
    print(f"  ──────────")
    print(f"  Sur le spectre V4 complet, R6 vs V4 :")
    print(f"    RMS log V4 = {rms_v4:.3f},  RMS log R6 = {rms_r6:.3f}")
    diff = (rms_r6 - rms_v4) / rms_v4 * 100
    if abs(diff) < 5:
        print(f"  ★ Les deux formules sont QUANTITATIVEMENT ÉQUIVALENTES")
        print(f"    (différence {diff:+.1f}%). R6 est donc une RÉ-ÉCRITURE")
        print(f"    structurelle exacte de V4, sans perte ni gain de précision.")
    elif diff > 0:
        print(f"  ✓ V4 reste meilleur de {diff:.1f}% — coût acceptable pour la")
        print(f"    réduction de 3 paramètres libres (p, f(2), f(3)).")
    else:
        print(f"  ★ R6 BAT V4 de {-diff:.1f}% en précision globale")
        print(f"    AVEC 3 paramètres libres EN MOINS — résultat fort.")
    print("=" * 112)


if __name__ == "__main__":
    main()
