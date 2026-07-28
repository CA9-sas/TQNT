r"""
Dérivation du mode Z₃/étrangeté pour la masse fine de l'octet (et du décuplet).

Contexte : l'embedding lié (build_linked_embedding) a montré que l'ordre de masse
de l'octet EST l'étrangeté (corr +0,975), un mode NON-géométrique. On le DÉRIVE
ici comme mode Z₃ (la saveur = excitation de 2e génération), sans paramètre de
saveur.

QUANTUM Z₃ DÉRIVÉ (0 paramètre de saveur) :
    E_Z₃ = (2/3)·Λ_χ = 139 MeV
  - 2/3 : facteur de Koide, DÉRIVÉ (Q=(1+A²/2)/3, A=√2=d_{1/2}(k=2), Ising
    SU(2)_2 ; cf. derive_koide_23). Le mode Z₃ est la structure à 3 états.
  - Λ_χ = √(σ_χ/4) = 209 MeV : échelle de confinement χ, fixée par la PENTE DE
    REGGE (2πσ_χ=1,096 GeV²/J, mesurée une fois) ; σ_χ=4Λ_χ² (facteur 4=D²Ising).
  L'étrangeté = nombre d'excitations du mode s ; chaque s coûte E_Z₃.

RÉSULTAT 1 — DÉCUPLET (règle d'espacement égal, pur Z₃) :
  m = m_Δ + |S|·E_Z₃ →  Δ/Σ*/Ξ*/Ω à RMS 1,1 %. Espacements observés 153/147/141
  (moy 147) vs E_Z₃=139 dérivé. LA SIGNATURE Z₃ PROPRE : l'espacement égal du
  décuplet EST le quantum Z₃ dérivé (pas de param de saveur ; seule base = m_Δ).

RÉSULTAT 2 — OCTET+DÉCUPLET complet (modèle constituant, E_Z₃ FIXÉ) :
  m = Σ m_q + hyperfin spin-spin, avec m_s = m_l + E_Z₃ (le mode Z₃, FIXÉ dérivé)
  et 2 échelles QCD UNIVERSELLES (m_l constituant, κ hyperfin, flavor-blind) :
    RMS 8 baryons = 2,3 %. L'hyperfin donne N-Δ (306 vs 293 MeV) et le split
    Λ-Σ (paire légère spin-0 vs spin-1) — le MÊME τ_⊥ que N-Δ.

RÉSULTAT 3 — PRÉDICTION PURE (m_l,κ fixés sur N,Δ non-étranges → étrange prédit) :
  RMS 6 baryons étranges = 4,2 %, sous-tir SYSTÉMATIQUE (-2 à -6 %). E_Z₃=139 est
  proche mais l'incrément CONSTITUANT veut ~180 : l'hyperfin en 1/m_s réduit la
  masse étrange, qu'il faut compenser. Le décuplet DIRECT (147, pas de cette
  réduction) matche mieux E_Z₃=139. Honnête : le quantum dérivé est bon à ~5-6 %
  en prédiction pure, 1,1 % sur le décuplet direct.

CONCLUSION : l'étrangeté de l'octet EST le mode Z₃, et son quantum
E_Z₃=(2/3)Λ_χ=139 MeV est DÉRIVÉ (Koide + Regge, 0 param de saveur). Le décuplet
(espacement égal) le confirme à 1,1 %. La masse fine de l'octet = Z₃ (étrangeté,
dérivé) + hyperfin universel (N-Δ/Λ-Σ, τ_⊥) → 2,3 %. C'est le mode manquant que
la géométrie (vortex) ne pouvait pas donner : la saveur, DÉRIVÉE via Z₃/Koide.
Complète le programme masse : familles (vortex, topologique) + saveur (Z₃, Koide).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, fsolve

E_Z3 = (2 / 3) * np.sqrt((1.096 / (2 * np.pi)) / 4.0) * 1000   # 139 MeV, DÉRIVÉ

EXP = {"N": 938.9, "Λ": 1115.7, "Σ": 1193.2, "Ξ": 1318.3,
       "Δ": 1232.0, "Σ*": 1384.6, "Ξ*": 1531.8, "Ω": 1672.5}
STRANGE = ["Λ", "Σ", "Ξ", "Σ*", "Ξ*", "Ω"]
SNUM = {"Λ": 1, "Σ": 1, "Ξ": 2, "Σ*": 1, "Ξ*": 2, "Ω": 3}


def constituent(ml, kappa):
    ms = ml + E_Z3
    return {
        "N": 3 * ml - 3 * kappa / ml ** 2,
        "Δ": 3 * ml + 3 * kappa / ml ** 2,
        "Λ": 2 * ml + ms - 3 * kappa / ml ** 2,
        "Σ": 2 * ml + ms + kappa * (1 / ml ** 2 - 4 / (ml * ms)),
        "Ξ": ml + 2 * ms - 3 * kappa / ms ** 2,
        "Σ*": 2 * ml + ms + kappa * (1 / ml ** 2 + 2 / (ml * ms)),
        "Ξ*": ml + 2 * ms + kappa * (2 / (ml * ms) + 1 / ms ** 2),
        "Ω": 3 * ms + 3 * kappa / ms ** 2,
    }


def _rms(m, keys):
    return float(np.sqrt(np.mean([((m[k] - EXP[k]) / EXP[k] * 100) ** 2 for k in keys])))


def main():
    print(f"E_Z₃ = (2/3)·Λ_χ = {E_Z3:.0f} MeV  (DÉRIVÉ : Koide 2/3 + Λ_χ Regge)\n")

    print("1. DÉCUPLET — espacement égal (pur Z₃, base=m_Δ) :")
    dec = {"Δ": 0, "Σ*": 1, "Ξ*": 2, "Ω": 3}
    md = {k: 1232.0 + s * E_Z3 for k, s in dec.items()}
    for k, s in dec.items():
        print(f"   {k:3s} S={s} pred={md[k]:7.1f} exp={EXP[k]:7.1f} ({(md[k]-EXP[k])/EXP[k]*100:+.1f}%)")
    print(f"   RMS décuplet = {_rms(md, dec):.1f}%\n")

    print("2. OCTET+DÉCUPLET — E_Z₃ FIXÉ + 2 échelles QCD universelles :")
    r = minimize(lambda p: _rms(constituent(*p), EXP), [360, 5e6], method="Nelder-Mead")
    m = constituent(*r.x)
    print(f"   m_l={r.x[0]:.0f} MeV, κ={r.x[1]:.2g} ; m_s−m_l=E_Z₃={E_Z3:.0f} (dérivé)")
    print(f"   RMS 8 baryons = {r.fun:.1f}%  (N-Δ : {m['Δ']-m['N']:.0f} vs 293 MeV)\n")

    print("3. PRÉDICTION PURE — m_l,κ de N,Δ ; étrange PRÉDIT (0 param étrange) :")
    ml, kappa = fsolve(lambda x: [constituent(*x)["N"] - EXP["N"],
                                  constituent(*x)["Δ"] - EXP["Δ"]], [360, 6e6])
    mp = constituent(ml, kappa)
    for k in STRANGE:
        print(f"   {k:3s} S={SNUM[k]} pred={mp[k]:7.0f} exp={EXP[k]:7.0f} ({(mp[k]-EXP[k])/EXP[k]*100:+.1f}%)")
    print(f"   RMS 6 étranges prédits = {_rms(mp, STRANGE):.1f}% (sous-tir systématique)")


if __name__ == "__main__":
    main()
