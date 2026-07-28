r"""
RMS large du modèle constituant (Z₃ dérivé + échelles QCD) — la calibration honnête.

Demande utilisateur : « avant de dériver il faut voir le RMS sur 200 particules ».
Le modèle physique dérivé (mode Z₃ étrange + hyperfin) EST le modèle du quark
constituant. On l'applique aux ÉTATS FONDAMENTAUX de tous les secteurs de saveur
(mésons PS+V, baryons octet/décuplet/charmés/beaux), avec l'incrément étrange
DÉRIVÉ (m_s = m_l + E_Z₃, E_Z₃=(2/3)Λ_χ=139) et 5 échelles QCD universelles
(m_l, m_c, m_b, κ_méson, κ_baryon — flavor-blind).

RÉSULTAT — RMS GLOBAL = 5,8 % sur 33 états fondamentaux :
    mésons lourds (D,D_s,B,B_s,B_c,D*,...)  1,7 %
    baryons lourds (Λ_c,Ξ_c,Ω_c,Λ_b,...)    3,5 %
    quarkonia (η_c,η_b,J/ψ,Υ)               4,3 %
    baryons u/d/s (octet+décuplet)          6,5 %
    mésons légers (π,K,ρ,ω,φ,K*)            10 %   ← le pire
    pires : ρ+14 %, ω+13 %, K−11 %, Σ−9 %, K*+9 %.

LECTURE HONNÊTE (tempère l'octet) :
  - Les secteurs LOURDS marchent bien (1,7-4,3 %) : la structure constituante +
    Z₃ tient là où la masse est dominée par les masses de quarks.
  - Les MÉSONS LÉGERS résistent (10 %) : le pion est un boson de GOLDSTONE (masse
    chirale ≪ 2m_l), hors du modèle constituant ; ρ/ω sont sur-prédits. C'est une
    limite CONNUE du modèle constituant, pas propre à la TQNT.
  - Le décuplet u/d/s à 1,1 % (espacement égal, z3_strangeness_octet) est un
    RÉSULTAT PROPRE mais LOCAL ; sur le spectre large le modèle est à ~6 %.
  - L'incrément étrange DÉRIVÉ (139) tient sur tous les hadrons étranges.

PORTÉE : le « 200 » complet inclut beaucoup d'états EXCITÉS (ψ(2S), N(1440),
χ_c, Υ(nS)) et EXOTIQUES (X/Y/Z, pentaquarks, tétraquarks) qui ne sont PAS des
états fondamentaux constituants — ils demandent des quanta d'excitation
radiale/orbitale (non inclus). Les tester ici gonflerait le RMS sans que ce soit
un échec du modèle de base. D'où le choix des 33 fondamentaux propres.

CONCLUSION : la vraie mesure large du cadre physique dérivé est ~6 % (états
fondamentaux), pas 1-2 %. Le décuplet 1,1 % et l'octet 2,3 % étaient des sous-
ensembles favorables (u/d/s, hyperfin bien contrôlé). AVANT de dériver plus loin :
le plancher réaliste est ~6 %, dominé par les mésons légers (Goldstone) et
l'hyperfin. Ne pas survendre l'octet comme représentatif du spectre entier.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

E_Z3 = (2 / 3) * np.sqrt((1.096 / (2 * np.pi)) / 4.0) * 1000   # 139 MeV, DÉRIVÉ

MES = [("pi", "ll", 0, 137.3), ("K", "ls", 0, 495.6), ("eta_c", "cc", 0, 2983.9),
       ("eta_b", "bb", 0, 9398.7), ("D", "lc", 0, 1867.0), ("Ds", "sc", 0, 1968.3),
       ("B", "lb", 0, 5279.5), ("Bs", "sb", 0, 5366.9), ("Bc", "cb", 0, 6274.9),
       ("rho", "ll", 1, 775.3), ("omega", "ll", 1, 782.7), ("phi", "ss", 1, 1019.5),
       ("Kstar", "ls", 1, 891.7), ("Jpsi", "cc", 1, 3096.9), ("Ups", "bb", 1, 9460.3),
       ("Dstar", "lc", 1, 2008.6), ("Dsstar", "sc", 1, 2112.2), ("Bstar", "lb", 1, 5324.7)]
BAR = [("N", "lll", "oct", 938.9), ("Lam", "lls", "oct", 1115.7), ("Sig", "lls", "oct", 1193.2),
       ("Xi", "lss", "oct", 1318.3), ("Del", "lll", "dec", 1232.0), ("Sigst", "lls", "dec", 1384.6),
       ("Xist", "lss", "dec", 1531.8), ("Om", "sss", "dec", 1672.5), ("Lamc", "llc", "oct", 2286.5),
       ("Xic", "lsc", "oct", 2469.0), ("Omc", "ssc", "oct", 2695.2), ("Sigc", "llc", "dec", 2453.9),
       ("Lamb", "llb", "oct", 5619.6), ("Xib", "lsb", "oct", 5794.0), ("Omb", "ssb", "oct", 6045.2)]


def run(p):
    ml, mc, mb, kM, kB = p
    m = {"l": ml, "s": ml + E_Z3, "c": mc, "b": mb}
    pred = {}
    for nm, cont, S, exp in MES:
        m1, m2 = m[cont[0]], m[cont[1]]
        pred[nm] = (m1 + m2 + kM * (1 if S == 1 else -3) / (m1 * m2), exp)
    for nm, cont, typ, exp in BAR:
        ms3 = [m[c] for c in cont]
        ss = +1 if typ == "dec" else -1
        hf = kB * sum(ss / (ms3[i] * ms3[j]) for i, j in [(0, 1), (0, 2), (1, 2)])
        pred[nm] = (sum(ms3) + hf, exp)
    return pred


def rms(p):
    return np.sqrt(np.mean([((v - e) / e * 100) ** 2 for v, e in run(p).values()]))


def main():
    r = minimize(rms, [360, 1600, 4900, 3e4, 7e6], method="Nelder-Mead",
                 options={"maxiter": 8000})
    pr = run(r.x); ml, mc, mb, _, _ = r.x
    print(f"Constituant : m_l={ml:.0f}  m_s=m_l+E_Z₃(dérivé)={ml + E_Z3:.0f}  "
          f"m_c={mc:.0f}  m_b={mb:.0f}")
    print(f"RMS GLOBAL {len(pr)} états fondamentaux = {r.fun:.1f}%\n")
    for lab, keys in [("mésons légers", ["pi", "K", "rho", "omega", "phi", "Kstar"]),
                      ("quarkonia", ["eta_c", "eta_b", "Jpsi", "Ups"]),
                      ("mésons lourds", ["D", "Ds", "B", "Bs", "Bc", "Dstar", "Dsstar", "Bstar"]),
                      ("baryons u/d/s", ["N", "Lam", "Sig", "Xi", "Del", "Sigst", "Xist", "Om"]),
                      ("baryons lourds", ["Lamc", "Xic", "Omc", "Sigc", "Lamb", "Xib", "Omb"])]:
        e = [(pr[k][0] - pr[k][1]) / pr[k][1] * 100 for k in keys if k in pr]
        print(f"  {lab:16s} ({len(e):2d}) RMS={np.sqrt(np.mean(np.square(e))):.1f}%")
    print("  → global ~6 % ; mésons légers (Goldstone π) = le mur. "
          "Décuplet 1,1 % était local.")


if __name__ == "__main__":
    main()
