r"""
« RMS sur 118/122 PDG » du nouveau cadre ? — il n'y en a PAS (réponse honnête).

L'ancien 0,7 % RMS = formule PHÉNOMÉNOLOGIQUE Ω_ts, calibrée per-nœud sur ~38
valeurs U_κ/U_τ de courbes FAUSSES (cf. brouillon_Ukappa_Utau) — un artefact, pas
une dérivation.

Le nouveau cadre NE reproduit PAS le spectre absolu : M_topo = E₀ (puits χ) est
non-perturbatif et tracke la COMPLEXITÉ, pas la SAVEUR. Forcé (ajustement
mass = a·|E₀| + b, 2 params, secteur léger) :
  RMS ≈ 144 % (corr(|E₀|,M) = -0,49, ANTI-corrélé : étrangeté rend Ξ lourd mais
  simple ; Goldstone π léger échappe).

Le nouveau cadre TROQUE le 0,7 % illusoire contre trois prédictions STRUCTURELLES
vérifiables (results_vs_experiment.py) : Koide 61 ppm, Regge 1,5 %, Z₃ 1,8 %.
Il assume (comme la QCD) que M₀ est non-perturbatif — pas de spectre absolu en
forme close.
"""
from __future__ import annotations
import numpy as np

# secteur léger : (E₀ entrelacs du puits χ, masse exp MeV)
LIGHT = {
    "π⁰": (-77.7, 135.0), "ρ⁰": (-38.2, 775.3), "ω": (-77.7, 782.7),
    "η'": (-25.6, 957.8), "N": (-67.1, 938.9), "Λ": (-53.7, 1115.7),
    "Σ": (-60.9, 1193.2), "Ξ": (-47.3, 1318.3), "Δ": (-40.0, 1232.0),
}


def forced_rms():
    """RMS% du meilleur ajustement mass = a·|E₀| + b (2 params)."""
    E = np.array([abs(v[0]) for v in LIGHT.values()])
    M = np.array([v[1] for v in LIGHT.values()])
    A = np.vstack([E, np.ones_like(E)]).T
    (a, b), *_ = np.linalg.lstsq(A, M, rcond=None)
    Mfit = a * E + b
    rms = np.sqrt(np.mean(((Mfit - M) / M) ** 2)) * 100
    corr = np.corrcoef(E, M)[0, 1]
    return rms, corr, list(zip(LIGHT, E, M, Mfit))


def main():
    rms, corr, rows = forced_rms()
    print("Forcé (mass = a·|E₀| + b, 2 params, secteur léger) :")
    for n, e, m, mf in rows:
        print(f"  {n:4s} E₀={e:5.1f}  exp={m:5.0f}  fit={mf:5.0f}  {(mf-m)/m*100:+.0f}%")
    print(f"\n  RMS = {rms:.0f}%   corr(|E₀|,M) = {corr:.2f}  (ANTI-corrélé)")
    print("\n  => pas de '0,7% sur 118/122' : le nouveau cadre ne reproduit PAS le")
    print("     spectre absolu (M₀ non-perturbatif, saveur hors topologie).")
    print("     L'ancien 0,7% = Ω_ts phénoménologique (calibré, courbes fausses).")
    print("     Résultats vérifiables = STRUCTURELS : Koide 61 ppm, Regge 1,5%, Z₃ 1,8%.")


if __name__ == "__main__":
    main()
