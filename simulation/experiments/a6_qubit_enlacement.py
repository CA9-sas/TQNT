# -*- coding: utf-8 -*-
"""A-6 — Le qubit d'enlacement : jeu de paramètres et échelles (04/09/2026).

Variable de qubit = le NOMBRE DE FLUXOÏDE (l'enlacement Lk du flux et de
la boucle supraconductrice). Superposition |Lk⟩+|Lk+1⟩ au point de
frustration Φ_ext=(Lk+½)Φ₀', Φ₀'=h/2e. La jonction Josephson = le
portail de reconnexion (P15-effectif) : un saut de phase = ΔLk=±1.
Design de référence : fluxonium (une jonction + super-inductance).
"""
import numpy as np

h = 6.626e-34
Phi0 = 2.0678e-15          # h/2e (Wb)
GHz = 1e9

# ---- jeu de paramètres fluxonium (standard, réalisable) ----
EJ, EC, EL = 4.0, 1.0, 0.8   # GHz
L = (Phi0 / (2 * np.pi))**2 / (EL * GHz * h)
omega_p = np.sqrt(8 * EJ * EC)
# amplitude de saut de phase (WKB, forme standard)
Delta = (8 / np.pi)**0.5 * omega_p * (2 * EJ / EC)**0.25 * np.exp(-np.sqrt(8 * EJ / EC))
Ip = Phi0 / (2 * np.pi * L) * 2 * np.pi * 0.5   # ~ courant persistant à ½ quantum
eps_per_mPhi0 = 2 * (EL * GHz * h) * 2 * np.pi * 0.5 / Phi0 * (1e-3 * Phi0) / h / GHz

print(f"E_J/E_C/E_L            : {EJ}/{EC}/{EL} GHz  (fluxonium)")
print(f"super-inductance L     : {L*1e9:.0f} nH")
print(f"ω_p (plasma)           : {omega_p:.2f} GHz")
print(f"Δ (saut de phase ΔLk=±1 au portail) : {Delta:.2f} GHz — le gap du qubit")
print(f"pente ε hors point doux: ~{eps_per_mPhi0:.3f} GHz/mΦ₀ (1er ordre nul AU point doux)")
print(f"T2 attendus (état de l'art fluxonium) : 0,1–1,5 ms à 10–20 mK")

# ---- l'échelle nucléaire : la MÊME formule quinze ordres plus haut ----
E_fluxoide_nucl = 1.379e6   # eV — E = Φ_χ²/(2L_ind) de la référence (Δm(n−p))
E_fluxoide_squid = 0.5 * EL * GHz * h / 1.602e-19  # eV, ordre E_L/2
print(f"\nE = Φ²/2L : nucléon {E_fluxoide_nucl:.2e} eV (Δm(n−p), référence)")
print(f"            SQUID   {E_fluxoide_squid:.2e} eV")
print(f"            rapport {E_fluxoide_nucl/E_fluxoide_squid:.1e} — même mécanisme,"
      " douze ordres d'écart, AUCUN recouvrement revendiqué (ontologie seule)")

print("""
GARDE-FOUS (document principal) :
- superselection : Ŵ (enroulement total du condensat) est conservé par
  [H_int,Ŵ]=0 SANS lien faible ; la jonction est précisément l'élément
  qui brise cette conservation (portail ΔLk=±1) → la superposition
  |Lk⟩+|Lk+1⟩ est LICITE (et observée depuis Mooij/van der Wal 2000).
- indiscernabilité : aucun gain de taux TQNT — le dispositif EST un
  fluxonium standard ; l'apport du cadre est l'ontologie (h/2e =
  entrelacs 2×3₁, saut de phase = reconnexion, E=Φ²/2L = le mécanisme
  de Δm(n−p)) et la prédiction d'ABSENCE : quantification h/2e exacte,
  aucun canal de décohérence « de nœud » supplémentaire.""")