r"""
Durées de vie du CANAL FORT — SANS CALIBRATION.

La force nucléaire forte EST la tension d'enlacement intra-nucléon (TQNT). La largeur d'une
désintégration forte est donc fixée par l'énergie de tension d'enlacement
multipliée par le nombre de reconnexions nécessaires pour atteindre la stabilité
(idée L. Cariou) :

    Γ = E_tension × N_stab ,
    E_tension = β · ⟨√(c_i c_j)⟩ · D   (potentiel harmonique d'enlacement, P21),
    N_stab    = |Lk − Lk_opt|          (reconnexions ΔLk=±1 vers Lk_opt),
    D         = (Lk − Lk_opt)² = N_stab² ,   Lk_opt = C(n,2).

Soit Γ = β · ⟨√cc⟩ · N_stab³. AUCUNE calibration : β=0,210 MeV est un paramètre
global TQNT, ⟨√cc⟩ et N_stab sont topologiques. τ = ħ/Γ.

Les états ÉTROITS (η', J/ψ, Υ : OZI / anomalie U(1)_A) sont supprimés sous cette
largeur forte naturelle par des règles de sélection connues — non capturées par
la tension topologique (signalés \"anom\").

Usage :  python3 -m simulation.experiments.lifetime_strong_nocalib
"""
from itertools import combinations
import numpy as np
from theory.jones import jones_polynomial, jones_retraction_factor

HBAR = 6.582119e-22      # MeV·s
BETA = 0.210             # MeV (tension d'enlacement, paramètre global TQNT)

def cJ(k):
    return jones_retraction_factor(jones_polynomial(k))

def mcc(comps):
    return float(np.mean([np.sqrt(cJ(a)*cJ(b)) for a, b in combinations(comps, 2)]))

# nom : (composantes V4, n, Lk, τ_exp, anomalie)
STRONG = {
  "etap":   (["5_2","5_2","6_1"], 3, 0, 3.50e-21, "U(1)_A"),
  "rho_om": (["5_2","7_7"],       2, 0, 7.75e-23, "-"),
  "phi":    (["3_1","4_1","6_2"], 3, 0, 1.55e-22, "OZI"),
  "Jpsi":   (["4_1","5_2","9_1"], 3, 0, 7.11e-21, "OZI"),
  "Ups":    (["3_1","7_4","9_1"], 3, 0, 1.22e-20, "OZI"),
  "Delta":  (["3_1","5_2","6_2"], 3, 0, 5.63e-24, "-"),
  "Sigma*": (["5_2","6_2","7_2"], 3, 0, 1.83e-23, "-"),
  "Xi*":    (["4_1","5_2","7_2"], 3, 0, 7.23e-23, "-"),
  "Sigmac": (["4_1","5_1","7_1"], 3, 0, 3.50e-22, "seuil"),
}

def _tau(comps, n, lk):
    lkopt = n*(n-1)//2
    N = abs(lk - lkopt)
    D = N*N
    gamma = BETA * mcc(comps) * D * N        # MeV (= E_tension × N_stab)
    return HBAR/gamma if gamma > 0 else float("inf")

def strong_tau():
    """{nom: τ (s)} du canal fort SANS calibration — pour spectrum_all_table."""
    return {nm: _tau(c, n, lk) for nm, (c, n, lk, te, an) in STRONG.items()}

def main():
    print("="*82)
    print(f"  CANAL FORT SANS CALIBRATION — Γ = β·⟨√cc⟩·D·N_stab  (β={BETA} MeV)")
    print("="*82)
    print(f"\n  {'part':>8} {'<cc>':>5} {'N':>2} {'Γ(MeV)':>8} {'τ_TQNT':>10} {'τ_exp':>10} {'Δlog':>6} {'anom':>6}")
    print("  "+"-"*62)
    norm, allr = [], []
    for nm,(c,n,lk,te,an) in STRONG.items():
        N=abs(lk-n*(n-1)//2); M=mcc(c); g=BETA*M*N*N*N; tp=HBAR/g
        dl=np.log10(tp/te); allr.append(dl)
        if an=="-": norm.append(dl)
        print(f"  {nm:>8} {M:>5.2f} {N:>2} {g:>8.2f} {tp:>10.2e} {te:>10.2e} {dl:>+6.2f} {an:>6}")
    print("  "+"-"*62)
    print(f"\n  RMS Δlog (larges, non supprimés) : {np.sqrt(np.mean(np.square(norm))):.2f} déc ({len(norm)} part.)")
    print(f"  RMS Δlog (tous)                  : {np.sqrt(np.mean(np.square(allr))):.2f} déc ({len(allr)} part.)")
    print("  Les 'anom' (η',J/ψ,Υ,φ) sont étroits : OZI / anomalie U(1)_A les")
    print("  suppriment SOUS la largeur forte naturelle (règles de sélection connues).")
    print("="*82)

if __name__ == "__main__":
    main()
