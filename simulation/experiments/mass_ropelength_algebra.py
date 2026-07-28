r"""
Chantier 1 — l'algèbre de la tightness : la rope-length comme invariant.

Insight utilisateur : « plus la boucle est serrée, plus la masse est
importante. » On cherche l'algèbre simple qui détermine cette tightness
géométrique et entre dans la formule de la masse.

CANDIDAT NATUREL : la rope-length R(K), invariant topologique exact —
le ratio L_min/d où L_min est la longueur minimale d'un tube de
diamètre d qui réalise le nœud K en évitant l'auto-intersection.
C'est la mesure DIRECTE de tightness : plus R est grand, plus le tube
doit être long pour s'auto-éviter en réalisant la topologie K.

On teste, depuis les rope-lengths PUBLIÉES (Cantarella-Kusner-Sullivan,
Pieranski, Diao) :
1. Corrélation R(K) ↔ c_J(K) et R(K) ↔ cr(K) (crossing number).
2. La formule m(K) = m_e · (R(K)/R(3_1))^p, p calibré sur (e⁻, π⁰).
3. La cohérence avec le spectre V4.
4. Quelle algèbre simple capture R(K) le mieux : c_J, cr, ou autre ?

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j, mass
from theory.abraham import REFERENCE_N50K
from theory.jones import braid_index

# Rope-lengths publiées (en unités de diamètre de tube ; sources :
# Cantarella-Kusner-Sullivan 2002, Pieranski 1998, Diao 2003).
# Valeurs convergées approximatives sur des minima locaux.
# ⚠️ SUPPLANTÉE 2026-07-22 : audit vs table ACPR (arXiv:math/0508248)
# — 9 bornes lâches (+2 à +8,8 %), 4 valeurs IMPOSSIBLES (sous le
# minimum : 6_2, 6_3, 7_6, 7_7). Source canonique désormais :
# theory/ropelength_acpr.py. Cette table est conservée pour la
# reproduction historique (convention du dépôt).
ROPELENGTH = {
    "3_1":   16.372,
    "4_1":   20.99,
    "5_1":   23.55,
    "5_2":   26.13,
    "6_1":   28.31,
    "6_2":   28.13,
    "6_3":   27.21,
    "7_1":   30.69,
    "7_2":   32.96,
    "7_3":   33.61,
    "7_4":   34.97,
    "7_5":   35.16,
    "7_6":   32.66,
    "7_7":   31.93,
    "8_1":   37.49,
    "8_2":   37.32,
    "8_3":   37.30,
    "9_1":   38.62,
    "10_1":  42.33,
}

M_E   = 0.51099895     # MeV
M_PI0 = 134.9768       # MeV


def main():
    print("=" * 76)
    print("  ALGÈBRE DE LA TIGHTNESS : rope-length R(K) comme invariant fondamental")
    print("=" * 76)

    knots = list(ROPELENGTH.keys())
    R   = np.array([ROPELENGTH[K] for K in knots])
    cJ  = np.array([c_j(K) for K in knots])
    UEM = np.array([u_em(K) for K in knots])
    bi  = np.array([braid_index(K) for K in knots])
    cr  = np.array([int(K.split("_")[0]) for K in knots])
    m_V4 = np.array([mass(K, "single") for K in knots])

    # ----- (1) Corrélations R(K) ↔ algèbres simples ----------------------
    print("\n  (1) Corrélations R(K) avec les algèbres simples :")
    print(f"  {'K':>5} {'R(K)':>8} {'c_J':>6} {'R/c_J':>7} {'cr':>3} "
          f"{'R/cr':>7} {'br':>3} {'R/br':>7}")
    for K, r, j, c, b in zip(knots, R, cJ, cr, bi):
        print(f"  {K:>5} {r:>8.2f} {j:>6.2f} {r/j:>7.2f} {c:>3} "
              f"{r/c:>7.2f} {b:>3} {r/b:>7.2f}")

    print(f"\n  Statistiques R(K) / X(K) :")
    for label, x in [("c_J", cJ), ("cr", cr), ("braid index", bi)]:
        ratio = R / x
        print(f"    R / {label:>12} : moyenne {ratio.mean():>6.2f}, "
              f"écart-type {ratio.std():>5.2f}, "
              f"CV = {ratio.std()/ratio.mean()*100:>5.1f}%")

    # ----- (2) Calibration m ∝ R^p sur (e⁻, π⁰) -------------------------
    print(f"\n  (2) Calibration m = m_e · (R(K)/R(3_1))^p sur (e⁻, π⁰) :")
    R_31 = ROPELENGTH["3_1"]
    R_81 = ROPELENGTH["8_1"]
    p_calib = np.log(M_PI0 / M_E) / np.log(R_81 / R_31)
    print(f"    p = log(m_π0/m_e) / log(R(8_1)/R(3_1))")
    print(f"      = log({M_PI0/M_E:.1f}) / log({R_81/R_31:.3f})")
    print(f"      = {p_calib:.3f}")

    # ----- (3) Prédictions m_pred vs V4 ---------------------------------
    print(f"\n  (3) m_pred = m_e · (R/R(3_1))^{p_calib:.2f}  comparé à V4 :")
    print(f"  {'K':>5} {'R(K)':>8} {'m_pred':>12} {'m_V4':>12} {'ratio':>8}")
    devs = []
    for K, r, m4 in zip(knots, R, m_V4):
        m_pred = M_E * (r / R_31) ** p_calib
        ratio = m_pred / m4 if m4 > 0 else np.nan
        devs.append(np.log10(ratio))
        flag = " " if abs(np.log10(ratio)) < 0.5 else "*"
        print(f"  {K:>5} {r:>8.2f} {m_pred:>12.4f} {m4:>12.4f} {ratio:>8.2f}× {flag}")
    devs = np.array(devs)
    rms = np.sqrt(np.mean(devs ** 2))
    print(f"\n  RMS log10(m_pred / m_V4) = {rms:.3f}  (facteur {10**rms:.1f}× typique)")

    # ----- (4) Refit sur V4 ---------------------------------------------
    log_R = np.log(R)
    log_m = np.log(m_V4)
    p_fit, log_C = np.polyfit(log_R, log_m, 1)
    res = log_m - (p_fit * log_R + log_C)
    rms_fit = np.sqrt(np.mean(res ** 2))
    R2 = 1 - np.sum(res ** 2) / np.sum((log_m - log_m.mean()) ** 2)
    print(f"\n  (4) Refit log m_V4 = p·log R + log C sur 19 nœuds :")
    print(f"    p = {p_fit:.3f}")
    print(f"    C = {np.exp(log_C):.4e}")
    print(f"    R² = {R2:.4f}")
    print(f"    RMS log = {rms_fit:.3f}")
    print(f"    (à comparer à p calibré sur (e,π0) = {p_calib:.2f})")

    # ----- (5) Comparer aux corrélations connues -------------------------
    print(f"\n  (5) Trois algèbres en compétition pour la 'tightness' :")
    for label, x, p_known in [("R (rope-length, table)", R, None),
                               ("c_J (Jones algebra)",  cJ, None),
                               ("cr (crossing number)",  cr, None)]:
        log_x = np.log(x)
        p_x, log_Cx = np.polyfit(log_x, log_m, 1)
        res_x = log_m - (p_x * log_x + log_Cx)
        R2_x = 1 - np.sum(res_x ** 2) / np.sum((log_m - log_m.mean()) ** 2)
        rms_x = np.sqrt(np.mean(res_x ** 2))
        print(f"    {label:>30} : p = {p_x:>5.2f}, R² = {R2_x:.3f}, "
              f"RMS log = {rms_x:.3f}")

    # ----- (6) Lecture ---------------------------------------------------
    print(f"\n  LECTURE :")
    print(f"  ──────────────────────────────────────────────────")
    cv_cJ = (R/cJ).std() / (R/cJ).mean() * 100
    cv_cr = (R/cr).std() / (R/cr).mean() * 100
    if cv_cJ < 15:
        print(f"  R(K) ≈ {(R/cJ).mean():.2f}·c_J(K)  (CV {cv_cJ:.0f}% — bonne corrélation).")
        print(f"  L'algèbre c_J est donc un PROXY ALGÉBRIQUE direct de la tightness.")
    if cv_cr < 15:
        print(f"  R(K) ≈ {(R/cr).mean():.2f}·cr(K)  (CV {cv_cr:.0f}% — bonne corrélation).")
        print(f"  Le simple crossing number est aussi un bon proxy de la tightness.")
    print(f"  ")
    print(f"  La formule m ∝ R^{p_calib:.1f} (calibrée sur 2 particules expérimentales)")
    print(f"  RMS = {rms:.2f} décade vs V4 — c'est la formule la plus PHYSIQUE")
    print(f"  (un seul invariant topologique exact, un seul exposant calibré).")
    print(f"  Si on substitue R ≈ {(R/cJ).mean():.2f}·c_J, on retombe sur c_J^p, mais avec")
    print(f"  une interprétation claire : c_J est un proxy de la tightness géométrique.")
    print("=" * 76)


if __name__ == "__main__":
    main()
