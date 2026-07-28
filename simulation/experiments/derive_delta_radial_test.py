"""Tenter de deriver delta_Koide (2/9) et le ratio radial (1.38) du secteur sqrt(sigma_chi)*S.

RESULTAT :
  - delta_Koide : delta_vrai=0.22227 rad. 2/9=0.22222 (0.02%) ET 2/9 = Q/3 (Q=2/3 Koide exact).
    => delta = Q/3 : RELATION forte (0.02%) qui LIE delta au hint sqrt2/Z3 (Q=2/3 <=> A=sqrt2).
    delta n'est donc PLUS un parametre calibre independant : il se replie sur le hint (6->5).
    MAIS pas de MECANISME pour 'phase = ratio/3' -> statut de HINT (comme sqrt2), pas derive.
  - ratio radial 1.38 : pentes radiales NON universelles (pi 1.47, rho 1.37, charmonium 3.07).
    Pas de nombre propre ; sqrt2=1.41 colle aux legers mais le charmonium casse. Aucune forme
    close < ~2%. => RESTE CALIBRE (effectif light-hadron), non derive.

Bilan : delta reductible au hint sqrt2/Z3 (6->5) ; radial reste calibre. PAS 6->4.
"""
import math
import numpy as np

ME, MM, MT = 0.51099895, 105.6583755, 1776.86   # MeV
SL = 1.096                                       # GeV^2 = 2 pi sigma_chi


def koide_phase():
    r = [math.sqrt(x) for x in (ME, MM, MT)]
    M = sum(r) / 3
    delta = math.acos((max(r) / M - 1) / math.sqrt(2))     # angle du tau
    Q = (ME + MM + MT) / sum(r) ** 2
    print("=== delta_Koide ===")
    print(f"  delta_vrai = {delta:.5f} rad ; Q = {Q:.5f}")
    for name, val in [("2/9", 2 / 9), ("Q/3 (Z3)", Q / 3), ("2/3^2", 2 / 9)]:
        print(f"    {name:10} = {val:.5f}  ecart {abs(val - delta) / delta * 100:+.2f}%")
    print("  => delta = Q/3 = 2/9 a 0.02% : relation liant delta au hint sqrt2/Z3 (Q<=>A=sqrt2).")
    print("     delta se replie sur le hint (plus un parametre libre) MAIS sans mecanisme = HINT.")


def radial_ratio():
    print("\n=== ratio radial (engin = 1.38) ===")
    fams = {'pi': [0.140, 1.300, 1.800], 'rho': [0.775, 1.450, 1.900],
            'psi(cc)': [3.097, 3.686, 4.040]}
    ratios = []
    for nm, ms in fams.items():
        n = np.arange(len(ms))
        a = np.polyfit(n, np.array(ms) ** 2, 1)[0]
        ratios.append(a / SL)
        print(f"  {nm:8}: pente radiale = {a:.3f} GeV^2 -> ratio/SL = {a / SL:.2f}")
    print(f"  NON universel (dispersion {min(ratios):.2f}-{max(ratios):.2f}). Legers ~1.4.")
    for name, val in [("sqrt2", math.sqrt(2)), ("e/2", math.e / 2), ("4/3", 4 / 3),
                      ("Nambu-Goto=1", 1.0)]:
        print(f"    {name:12} = {val:.3f}  ecart vs 1.38 {abs(val - 1.38) / 1.38 * 100:+.1f}%")
    print("  => aucune forme close propre + non universel -> 1.38 RESTE CALIBRE.")


if __name__ == "__main__":
    koide_phase()
    radial_ratio()
    print("\nBilan : delta reductible au hint sqrt2/Z3 (6->5) ; radial reste calibre. Pas 6->4.")
