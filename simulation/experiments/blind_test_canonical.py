r"""Variante canonique (sans recherche) du test aveugle PDG 2024-2026.

Pour chaque particule, on impose la SEULE configuration dictée par
le quark-content nominal (méson 2 composantes, baryon 3 composantes,
en utilisant les nœuds "canoniques" associés aux quarks dans les
travaux V4 : u/d→3_1, s→5_2, c→7_3 ou 7_7, b→7_4 ou 8_2, etc.).
Aucune recherche parmi les combos.
"""
from __future__ import annotations
import numpy as np

from simulation.experiments.blind_test_pdg_2024_2026 import operator

# Assignations canoniques d'après les attributions V4/V5
# (Excel diffusion : quarks → nœuds élémentaires).
# u,d → 3_1   ;   s → 5_2 ou 4_1   ;   c → 7_7 ou 7_3   ;
# b → 7_4 / 8_2 ; charmonium cc̄ → (7_3 + 7_7) etc.
CANONICAL = [
    # particule, m_exp, config canonique imposée (sans recherche)
    ("X(6600)",     6552.0, ["7_3", "7_7", "5_2"]),    # cc̄cc̄ + s-like
    ("X(7100)",     7134.0, ["7_3", "7_7", "6_1"]),    # cc̄cc̄ orbital
    ("X(7300)",     7287.0, ["7_3", "7_7", "6_2"]),    # cc̄cc̄ 2S
    ("T_cc(3875)+", 3874.7, ["7_3", "7_7"]),           # cc + ūd̄ ≈ ccc̄c̄ document principal
    ("T_cs0(2870)", 2866.0, ["7_7", "5_2"]),           # cs̄
    ("T_cs1(2900)", 2904.0, ["7_7", "5_2"]),           # cs̄ (1−)
    ("Tcs0a(2900)0",2892.0, ["7_3", "5_2"]),           # cs̄ avec ud̄
    ("Tcs0a(2900)++",2921.0,["7_3", "5_2"]),           # cs̄ avec u-bar
    ("X(3960)",     3956.0, ["7_3", "5_2"]),           # Ds+Ds- ≈ cs̄+c̄s
    ("X(4630)",     4626.0, ["7_3", "7_7"]),           # cc̄ + ss̄ (Jψφ)
    ("X(4685)",     4684.0, ["7_3", "7_7"]),           # cc̄ + ss̄ (Jψφ)
    ("Z_cs(4000)",  4003.0, ["7_3", "7_7", "5_2"]),    # cc̄ + s
    ("Z_cs(4220)",  4216.0, ["7_3", "7_7", "5_2"]),    # cc̄ + s
    ("P_cs(4338)",  4338.2, ["7_3", "7_7", "5_2"]),    # ccūds → cc̄+s+u/d
    ("Λ_b(6146)",   6146.2, ["7_4", "3_1", "3_1"]),    # b+u+d
    ("Λ_b(6152)",   6152.5, ["7_4", "3_1", "3_1"]),    # b+u+d
    ("Ω_c(3185)",   3185.1, ["7_3", "5_2", "5_2"]),    # c+s+s
    ("Ω_c(3327)",   3327.1, ["7_3", "5_2", "5_2"]),    # c+s+s 2S
    ("Ξ'_c(3055)+", 3055.9, ["7_3", "5_2", "3_1"]),    # c+s+u
    ("Ξ'_c(3080)+", 3079.9, ["7_3", "5_2", "3_1"]),    # c+s+u
    ("Σ_c(2800)",   2800.0, ["7_3", "3_1", "3_1"]),    # c+u+u (Σ_c++)
]


def main():
    print()
    print("=" * 95)
    print("  TEST CANONIQUE (SANS RECHERCHE) PDG 2024-2026")
    print("  Configuration imposée par identité quark-content → nœud canonique")
    print("=" * 95)
    print()
    print(f"  {'particule':>14} {'config canonique':>18} {'m_exp':>9} "
          f"{'m_TQNT':>9} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 73)

    log_res = []
    abs_pct = []
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0, "✗": 0}
    for nm, me, cfg in CANONICAL:
        mp = operator(cfg)
        err = (mp - me) / me * 100
        a = abs(err)
        if a < 0.5: flag = "★★"
        elif a < 5: flag = "★"
        elif a < 20: flag = "✓"
        elif a < 50: flag = " "
        elif a < 200: flag = "?"
        else: flag = "✗"
        flags[flag] += 1
        log_res.append(np.log(mp / me))
        abs_pct.append(a)
        print(f"  {nm:>14} {'+'.join(cfg):>18} {me:>9.1f} {mp:>9.1f} "
              f"{err:>+8.2f} {flag:>5}")

    arr = np.array(log_res)
    rms_log = float(np.sqrt(np.mean(arr ** 2)))
    print(f"\n  ─── STATISTIQUES CANONIQUES (sans optimisation a posteriori) ───")
    print(f"    N total          : {len(CANONICAL)}")
    print(f"    Δ < 5  % (★)     : {flags['★★']+flags['★']}/{len(CANONICAL)}")
    print(f"    Δ < 20 % (✓)     : {flags['★★']+flags['★']+flags['✓']}/{len(CANONICAL)}")
    print(f"    RMS log          : {rms_log:.3f}  ({(np.exp(rms_log)-1)*100:+.1f}%)")
    print(f"    Δ médian absolu  : {np.median(abs_pct):.2f}%")
    print(f"    Δ moyen absolu   : {np.mean(abs_pct):.2f}%")
    print(f"    Δ max absolu     : {max(abs_pct):.2f}%")
    print("=" * 95)


if __name__ == "__main__":
    main()
