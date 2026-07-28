"""Secteur électrofaible : le vev est DÉRIVÉ, pas un condensat de Higgs.

En TQNT, χ n'est pas un condensat à VEV (cf. mémoire chi-not-vacuum-condensate).
Les bosons W/Z/H sont des entrelacs lourds (catalogue V5) ; l'« échelle EW » est
leur masse. Le vev du MS s'en déduit par la relation à l'arbre :

    sin²θ_W = 1 − (m_W/m_Z)²        (angle de Weinberg depuis les masses)
    e = √(4π α_EM),  α_EM = 1/137   (topologique, niveau CS k=135)
    g = e / sinθ_W                  (couplage faible)
    vev = 2 m_W / g                 (≈ 242 GeV, à 1,7 % de 246)

La HIÉRARCHIE (m_W/m_e ≈ 1,6·10⁵) est l'amplification topologique des hauts
croisements (Ω_ts ∝ c_J^3,55) — pas un ajustement fin. Pas de scalaire de Higgs
fondamental ⇒ pas de problème de hiérarchie (le H est l'entrelacs 6₃+8₂+8₃).
"""
import math

ALPHA = 1 / 137.036
VEV_EXP = 246.22  # GeV

# masses EW prédites par la TQNT (entrelacs), nœuds COHÉRENTS (sans le 7_5 aberrant
# du catalogue, 5959 MeV = 12× les autres 7-nœuds). W,Z : meilleurs combos spin-1 ;
# H : 7_1+8_3+10_1 (+0,05%) ; top : RÉASSIGNÉ 8_2+8_3+10_1 (−1,25%, fermion, sans 7_5).
# L'ancien top 6_3+7_1+7_5 (+0,03%) reposait sur le 7_5 aberrant. RMS EW HONNÊTE = 0,74%.
EW_KNOTS = {
    "W": (80.68, "7_1+7_2+8_3"),
    "Z": (90.56, "7_2+8_3+9_1"),
    "H": (125.26, "7_1+8_3+10_1"),
    "top": (170.41, "8_2+8_3+10_1"),  # réassigné cohérent (sans 7_5 aberrant)
}


def derive_vev(mW, mZ, alpha=ALPHA):
    s2W = 1 - (mW / mZ) ** 2
    e = math.sqrt(4 * math.pi * alpha)
    g = e / math.sqrt(s2W)
    return 2 * mW / g, s2W, g


if __name__ == "__main__":
    mW = EW_KNOTS["W"][0]
    mZ = EW_KNOTS["Z"][0]
    vev, s2W, g = derive_vev(mW, mZ)
    print(f"m_W = {mW} GeV ({EW_KNOTS['W'][1]}),  m_Z = {mZ} GeV ({EW_KNOTS['Z'][1]})")
    print(f"sin²θ_W = 1−(m_W/m_Z)² = {s2W:.4f}  (mesuré 0,2312)")
    print(f"g = √(4πα)/sinθ_W = {g:.4f}")
    print(f"vev = 2 m_W / g = {vev:.1f} GeV  (exp {VEV_EXP}, écart {(vev/VEV_EXP-1)*100:+.1f}%)")
    print(f"\nHiérarchie : m_W/m_e = {mW*1e3/0.511:.0f} = amplification c_J^3,55 (hauts croisements).")
    print("Pas de condensat ni de scalaire fondamental ⇒ pas de problème de hiérarchie.")
