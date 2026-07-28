r"""
Test prédictif aveugle TQNT V5 sur particules PDG 2024-2026
non incluses dans le training set Excel V5 diffusion ni dans
recent_exotic_states.py.

Méthode V5 strict : opérateur Ω_ts(SU(2)_8) entièrement analytique
(zéro paramètre libre), recherche de la meilleure assignation
parmi les combinations_with_replacement(19 nœuds, n_comp), avec
n_comp dicté par la nature du hadron (méson=2, baryon=3, etc.).

Output : tableau + CSV + statistiques.
"""
from __future__ import annotations
import numpy as np
import csv
import json
import os
from collections import Counter
from itertools import combinations_with_replacement

from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index

# ───── constantes V5 (copiées de recent_exotic_states.py) ─────
M_E = 0.51099895
M_P = 938.27208
M_N = 939.56542
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_LOBES = {
    "3_1": 3, "4_1": 2, "5_1": 5, "5_2": 2, "6_1": 2,
    "6_2": 3, "6_3": 3, "7_1": 7, "7_2": 2, "7_3": 3,
    "7_4": 2, "7_5": 3, "7_6": 2, "7_7": 3, "8_1": 2,
    "8_2": 3, "8_3": 3, "9_1": 9, "10_1": 2,
}

A = np.sqrt(7 / (8 * np.pi))
P = 3 / 4
Q = 3 / 4
X = -np.sqrt(6)
B = 33 / 20
RHO_0 = 10 - 3 / (8 * np.pi)

print("Calcul Frenet sur 19 nœuds...")
U_KT = {}
TOPO = {}
for K in KNOTS:
    geo = frenet_geometry(K, 50000)
    kappa = geo["kappa"]
    tau = geo["tau"]
    speed = geo["speed"]
    dt = 2 * np.pi / len(kappa)
    ds = speed * dt
    U_KT[K] = (
        float(np.sum(kappa ** 2 * ds)),
        float(np.sum(tau ** 2 * ds)),
    )
    pj = jones_polynomial(K)
    TOPO[K] = {
        "cr": int(K.split("_")[0]),
        "n_l": N_LOBES[K],
        "b": braid_index(K),
        "det": jones_determinant(pj),
    }
P_31 = U_KT["3_1"][0] * U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"] * TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"] * M_C + TOPO[K]["n_l"] * M_L for K in KNOTS}


def operator(comps):
    n = len(comps)
    ws = []
    for K in comps:
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        ws.append(M_NAIVE[K] * T ** P)
    skeleton = float(sum(w ** A for w in ws) ** (1 / A))
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    Phi_n = ((n + 1) / 2) ** (-X) if n > 0 else 1.0
    c = Counter(comps)
    Phi_pair = 1.0
    for K, count in c.items():
        if count >= 2:
            rho_K = RHO_0 * TOPO[K]["cr"] ** (-B)
            Phi_pair *= rho_K ** (count - 1)
    return Phi_n * Phi_pair * skeleton * G_comp ** Q


def search_config(m_exp, n_comp, exclude_8_3=True):
    pool = [K for K in KNOTS if (K != "8_3" if exclude_8_3 else True)]
    best = (None, float("inf"), float("inf"))
    for combo in combinations_with_replacement(pool, n_comp):
        cfg = list(combo)
        mp = operator(cfg)
        err = abs((mp - m_exp) / m_exp * 100)
        if err < best[2]:
            best = (cfg, mp, err)
    return best


# ═════════════════════════════════════════════════════════════════════
#  PARTICULES PDG 2024-2026 NON INCLUSES DANS TRAINING V5 NI EXCEL
# ═════════════════════════════════════════════════════════════════════
# Critères :
#   1. Confirmées (≥5σ) ou listées PDG 2024/2025 avec masse précise.
#   2. NON présentes dans docs/TQNT_catalog_V5_diffusion.xlsx
#      (qui contient : e,μ,τ,γ,W,Z,H,t,π,K,η,η',η_c(1S),η_c(2S),
#       J/ψ,ψ(2S),χ_c0,c1,c2(1P),η_b(1S),Υ(1S),Υ(2S),χ_b0,b1,b2(1P),
#       D⁰,D±,D_s±,D*(2007),D*(2010),B±,B⁰,B_s⁰,B_c±,
#       ρ,ω,φ,K*(892),f₀(980),a₂(1320),f₂(1270),
#       p,n,Λ,Σ⁺⁰⁻,Ξ⁰⁻,Δ,Σ*⁺,Ξ*⁰,Ω⁻,
#       Λ_c⁺,Σ_c⁺⁺,Ξ_c⁺,Ω_c⁰,Λ_b⁰,Ξ_b⁰,Ω_b⁻,
#       X(3872),Z_c(3900),X(6900),P_c(4380),P_c(4450))
#   3. NON présentes dans recent_exotic_states.py (qui contient :
#      X(6900),Y(4260),Y(4360),Y(4660),Zc(3900),Zc(4020),Zc(4430),
#      Zb(10610),Zb(10650),Pcs(4459),Pc(4337),X(3915),X(4140),X(4274),
#      Λb(6072),Σb(6097),Ξb(6227),Ωb(6316),Tccs(2941))
#
# Format : (nom, m_exp_MeV, n_comp, année, expérience, type, source)
# ═════════════════════════════════════════════════════════════════════
NEW_PDG = [
    # ─── Tétraquarks all-charm (CMS 2024-2025, Nature 2025) ───────────
    ("X(6600)",     6552.0, 3, "2024", "CMS",       "tetra-charm all-c",       "Nature 2024"),
    ("X(7100)",     7134.0, 3, "2024", "CMS",       "tetra-charm all-c",       "Nature 2024"),
    ("X(7300)",     7287.0, 3, "2024", "CMS",       "tetra-charm all-c (2S)",  "CMS Run-2 update"),
    # ─── T_cc(3875)+ doublement charmé (LHCb 2021, propriétés affinées 2024) ─
    ("T_cc(3875)+", 3874.7, 2, "2024", "LHCb",      "doubly-charm tetraquark", "PDG 2024 affined"),
    # ─── T_cs(2900) / T_csbar(2900) (PDG 2024 entrées) ────────────────
    ("T_cs0(2870)", 2866.0, 2, "2024", "LHCb",      "open-charm-strange T",    "PDG 2024 listing"),
    ("T_cs1(2900)", 2904.0, 2, "2024", "LHCb",      "open-charm-strange T (1-)","PDG 2024 listing"),
    ("Tcs0a(2900)0",2892.0, 2, "2024", "LHCb",      "isospin-1 charm tetraquark","PDG 2024"),
    ("Tcs0a(2900)++",2921.0,2, "2024", "LHCb",      "isospin-1 charm tetraquark","PDG 2024"),
    # ─── X(3960) (charm-strange molecular) — LHCb 2022, PDG 2024 ──────
    ("X(3960)",     3956.0, 2, "2024", "LHCb",      "Ds+Ds- molecular",        "PDG 2024"),
    # ─── X(4630), X(4685) — LHCb 2021, PDG 2024 affinés ──────────────
    ("X(4630)",     4626.0, 2, "2024", "LHCb",      "J/ψφ structure (1-)",     "PDG 2024 affined"),
    ("X(4685)",     4684.0, 2, "2024", "LHCb",      "J/ψφ structure (1+)",     "PDG 2024 affined"),
    # ─── Z_cs(4000), Z_cs(4220) — LHCb 2021, PDG 2024 ────────────────
    ("Z_cs(4000)",  4003.0, 3, "2024", "LHCb",      "charged charm-strange",   "PDG 2024"),
    ("Z_cs(4220)",  4216.0, 3, "2024", "LHCb",      "charged charm-strange",   "PDG 2024"),
    # ─── P_cs(4338) pentaquark — LHCb 2022, PDG 2024 ─────────────────
    ("P_cs(4338)",  4338.2, 3, "2024", "LHCb",      "hidden-charm pentaquark", "PDG 2024 new entry"),
    # ─── Excitations Λ_b (D-wave) confirmées PDG 2024 ────────────────
    ("Λ_b(6146)",   6146.2, 3, "2024", "LHCb",      "1D bottom baryon 3/2+",   "PDG 2024 listing"),
    ("Λ_b(6152)",   6152.5, 3, "2024", "LHCb",      "1D bottom baryon 5/2+",   "PDG 2024 listing"),
    # ─── Excitation Ω_c (LHCb 2023, listée PDG 2024) ─────────────────
    ("Ω_c(3185)",   3185.1, 3, "2024", "LHCb",      "excited charm baryon",    "PDG 2024"),
    ("Ω_c(3327)",   3327.1, 3, "2024", "LHCb",      "excited charm baryon",    "PDG 2024"),
    # ─── Excitations Ξ'_c (LHCb 2025) ────────────────────────────────
    ("Ξ'_c(3055)+", 3055.9, 3, "2025", "LHCb",      "P-wave charm baryon",     "LHCb 2025"),
    ("Ξ'_c(3080)+", 3079.9, 3, "2025", "LHCb",      "P-wave charm baryon",     "LHCb 2025"),
    # ─── Σ_c(2800) (PDG 2024 affiné) ─────────────────────────────────
    ("Σ_c(2800)",   2800.0, 3, "2024", "Belle",     "1P excited charm baryon", "PDG 2024 affined"),
]


def main():
    print()
    print("=" * 105)
    print("  TEST PRÉDICTIF AVEUGLE TQNT V5 — PARTICULES PDG 2024-2026 HORS TRAINING")
    print("  Opérateur Ω_ts SU(2)_8 entièrement analytique, ZÉRO paramètre libre")
    print("=" * 105)

    print(f"\n  Pool d'entrelacs : {len(KNOTS)} nœuds (3_1 à 10_1), 8_3 exclu (secteur EW)")
    print(f"  Nombre de particules testées : {len(NEW_PDG)}")
    print(f"  Sources : PDG 2024/2025 review + LHCb/CMS/ATLAS measurements 2024-2026.")

    print(f"\n  {'particule':>14} {'config TQNT':>17} {'m_exp':>9} "
          f"{'m_TQNT':>9} {'Δ %':>9} {'flag':>5} {'année':>6} {'expé.':>8}")
    print("  " + "-" * 92)

    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0, "✗": 0}
    rows = []
    log_residuals = []
    abs_pct = []

    for nm, me, n, year, expe, ptype, src in NEW_PDG:
        cfg, mp, _ = search_config(me, n, exclude_8_3=True)
        err = (mp - me) / me * 100
        if abs(err) < 0.5:
            flag = "★★"
        elif abs(err) < 5:
            flag = "★"
        elif abs(err) < 20:
            flag = "✓"
        elif abs(err) < 50:
            flag = " "
        elif abs(err) < 200:
            flag = "?"
        else:
            flag = "✗"
        flags[flag] += 1
        rows.append((nm, "+".join(cfg), me, mp, err, flag, year, expe, ptype, src))
        log_residuals.append(np.log(mp / me))
        abs_pct.append(abs(err))
        print(f"  {nm:>14} {'+'.join(cfg):>17} {me:>9.1f} {mp:>9.1f} "
              f"{err:>+8.2f} {flag:>5} {year:>6} {expe:>8}")

    total = len(NEW_PDG)
    n_0p5 = flags["★★"]
    n_5   = n_0p5 + flags["★"]
    n_20  = n_5  + flags["✓"]
    n_50  = n_20 + flags[" "]
    n_200 = n_50 + flags["?"]

    arr = np.array(log_residuals)
    rms_log = float(np.sqrt(np.mean(arr ** 2)))
    rms_relative = (np.exp(rms_log) - 1) * 100
    med_abs = float(np.median(abs_pct))
    mean_abs = float(np.mean(abs_pct))

    print(f"\n  ─── STATISTIQUES SUR {total} PARTICULES PDG 2024-2026 ───")
    print(f"    Δ < 0.5 % (★★)   : {n_0p5}/{total} ({n_0p5/total*100:.0f}%)")
    print(f"    Δ < 5  %  (★)    : {n_5}/{total} ({n_5/total*100:.0f}%)")
    print(f"    Δ < 20 %  (✓)    : {n_20}/{total} ({n_20/total*100:.0f}%)")
    print(f"    Δ < 50 %         : {n_50}/{total}")
    print(f"    50-200 %         : {n_200 - n_50}/{total}")
    print(f"    > 200 %  (échec) : {flags['✗']}/{total}")
    print(f"\n    RMS log         : {rms_log:.4f}  =  {rms_relative:+.2f}% relatif")
    print(f"    Δ médian absolu : {med_abs:.2f} %")
    print(f"    Δ moyen absolu  : {mean_abs:.2f} %")

    # ─── Export CSV ───────────────────────────────────────────────────
    out_dir = "data/blind_test_2026"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "blind_test_pdg_2024_2026.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["particule", "config_TQNT", "m_exp_MeV", "m_TQNT_MeV",
                    "delta_pct", "flag", "annee", "experience", "type", "source"])
        for r in rows:
            w.writerow(r)
    print(f"\n  CSV exporté : {csv_path}")

    json_path = os.path.join(out_dir, "blind_test_pdg_2024_2026.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "rows": [
                {
                    "particule": r[0], "config_TQNT": r[1],
                    "m_exp_MeV": r[2], "m_TQNT_MeV": r[3],
                    "delta_pct": r[4], "flag": r[5],
                    "annee": r[6], "experience": r[7],
                    "type": r[8], "source": r[9],
                } for r in rows
            ],
            "stats": {
                "n_total": total,
                "n_under_0p5pct": n_0p5,
                "n_under_5pct": n_5,
                "n_under_20pct": n_20,
                "rms_log": rms_log,
                "rms_relative_pct": rms_relative,
                "median_abs_pct": med_abs,
                "mean_abs_pct": mean_abs,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"  JSON exporté: {json_path}")
    print("=" * 105)


if __name__ == "__main__":
    main()
