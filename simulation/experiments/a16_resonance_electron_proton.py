# -*- coding: utf-8 -*-
"""A-16 — Résonance d'un électron dans l'entrelacs du proton (17/09/2026).

Objection de l'auteur à A-15 : « on parle de résonance, donc un électron dans
un entrelacs de proton ne doit pas avoir des comportements aussi tranchés ».
Correction : A-15 traitait le proton SEUL (p ↔ n de charges différentes).
Avec l'électron dans le système, la charge totale est conservée :
|p e⁻⟩ et |n ν⟩ ont Q = 0 → le mélange est permis. En TQNT : le trèfle de
l'électron cède une unité d'enlacement à l'entrelacs (Lk +3 → +2) par une
reconnexion P15 et sa part libre part en onde χ (le neutrino).
Dès lors, tout est affaire de résonance : amplitude de couplage V contre
désaccord ΔE, angle de mélange θ ≈ V/ΔE, continu.

Entrées : G_F (structure de Fermi dérivée, g_topo = 1), g_A = 1,27 (facteur
de matrice du nucléon, déclaré), a0, masses ; ordres de grandeur, facteurs
O(1) de la matrice nucléaire non résolus.
Repères de mesure déclarés : ⁷Be capture électronique T½ = 53,2 j ; ¹⁸⁷Re
neutre T½ = 4,1e10 ans → ¹⁸⁷Re⁷⁵⁺ nu 32,9 ans (désintégration bêta en
état lié) ; Δ(1232) largeur 117 MeV.
"""
import numpy as np

hbarc = 197.3269804            # MeV·fm
GF = 1.1663787e-11             # MeV⁻²
gA = 1.27
me, mp, mn = 0.51099895, 938.27208816, 939.56542052   # MeV
a0_fm = 52917.72109
MeV_Hz = 1e6 * 1.602176634e-19 / 6.62607015e-34

print("=" * 72)
print("1. Ce qui reste exact : la charge du SYSTÈME entier")
print("=" * 72)
print("  proton seul ↔ neutron seul : Q = +1 / 0 → aucun terme ne les couple (exact)")
print("  proton + électron ↔ neutron + neutrino : Q = 0 / 0 → mélange PERMIS")
print("  → le « tranché » de A-15 venait du périmètre, pas de la physique.")

print("\n" + "=" * 72)
print("2. L'hydrogène comme résonance détunée : |p e⁻⟩ ↔ |n ν⟩")
print("=" * 72)
a0 = a0_fm / hbarc                                   # MeV⁻¹
psi0 = 1 / (np.pi * a0**3)                           # MeV³
V = GF / np.sqrt(2) * psi0 * np.sqrt(1 + 3 * gA**2)  # MeV (ordre de grandeur)
dE = mn - mp - me + 13.6e-6                          # MeV
theta = V / dE
print(f"  |ψ1s(0)|² = {psi0:.2e} MeV³ ; couplage V ≈ G_F/√2·|ψ(0)|²·√(1+3g_A²) = {V:.1e} MeV"
      f" = {V*MeV_Hz:.0f} Hz")
print(f"  désaccord ΔE = m_n − m_p − m_e + 13,6 eV = {dE:.4f} MeV = {dE*MeV_Hz:.1e} Hz")
print(f"  angle de mélange θ ≈ V/ΔE = {theta:.1e} ; part de neutron dans l'hydrogène θ² ≈ {theta**2:.0e}")
print("  → le mélange existe, continûment ; il est seulement minuscule parce que le"
      f"\n    désaccord (0,78 MeV) dépasse le couplage de {np.log10(dE/V):.0f} ordres de grandeur.")

print("\n" + "=" * 72)
print("3. Accorder la résonance : ce que l'environnement fait de ΔE")
print("=" * 72)
for Z in (1, 4, 26, 75):
    s = Z**3
    print(f"  électron 1s d'un noyau Z = {Z:>2} : |ψ(0)|² ×{s:.0e} → V ≈ {V*s*MeV_Hz:.1e} Hz (couplage seul ; ΔE dépend du noyau)")
print("  Quand l'énergie le permet, la résonance s'ouvre et devient un TAUX :")
print("    ⁷Be + e⁻ → ⁷Li + ν : T½ = 53,2 jours (mesuré)")
print("    ¹⁸⁷Re neutre : T½ = 4,1e10 ans → ¹⁸⁷Re⁷⁵⁺ nu : 32,9 ans (mesuré) — l'environnement"
      "\n    électronique déplace ΔE et change la durée de vie d'un facteur ~1e9.")
print("  Désintégration inverse (p + e⁻ → n + ν) dès que l'énergie de Fermi des électrons"
      f"\n    dépasse m_n − m_p = {mn-mp:.3f} MeV : neutronisation des matières denses.")

print("\n" + "=" * 72)
print("4. Pourquoi ce n'est jamais un qubit, même à la résonance")
print("=" * 72)
print("  Un qubit exige deux niveaux DISCRETS qui oscillent (Rabi). Ici l'état final"
      "\n  |n ν⟩ est un CONTINUUM : le neutrino part. À la résonance on n'obtient pas"
      "\n  d'oscillation p e⁻ ⇄ n ν, mais une décroissance (règle d'or de Fermi)."
      "\n  Au seuil exact, l'espace des phases du neutrino s'annule (taux ∝ q_ν²) :"
      "\n  plus on s'accorde finement, moins il se passe quelque chose.")
print(f"  Échelle du couplage d'hydrogène : {V*MeV_Hz:.0f} Hz — une oscillation de Rabi à"
      "\n  cette fréquence serait lente mais mesurable, SI l'état final était discret.")

print("\n" + "=" * 72)
print("5. Le Δ⁺ vu comme résonance")
print("=" * 72)
M, G = 1232.0, 117.0
print(f"  Δ(1232) : facteur de qualité M/Γ = {M/G:.1f} ; largeur / écart N–Δ = {G/(M-mp):.2f}")
print("  C'est une résonance LARGE : la réponse du proton autour de 1,2 GeV est continue,"
      "\n  pas une raie. « Inutilisable comme qubit » ≠ « tranché » : le Δ est vu"
      "\n  en diffusion e⁻p comme un pic étalé sur ~120 MeV.")

print("\n" + "=" * 72)
print("SYNTHÈSE")
print("=" * 72)
print("  Exact (symétrie) : la charge TOTALE et l'univalence — rien d'autre."
      "\n  Tout le reste est résonance : θ = V/ΔE et largeurs, continus."
      f"\n  Hydrogène : θ ≈ {theta:.0e} (V ≈ {V*MeV_Hz:.0f} Hz contre ΔE ≈ 0,78 MeV)."
      "\n  Accordée par l'environnement, la résonance devient capture électronique"
      "\n  ou bêta en état lié — un taux, pas une oscillation, car le neutrino part.")
