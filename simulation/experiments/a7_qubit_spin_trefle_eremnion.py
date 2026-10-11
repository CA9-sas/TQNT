# -*- coding: utf-8 -*-
"""A-7 — Qubit de spin du trèfle, et le même rôle confié à un éremnion (17/09/2026).

Question : après l'abandon du qubit chiral (superselection) et de la voie
e/μ/τ (pas de couplage entre générations, état haut instable, écart de
105 MeV impilotable), que vaut le SPIN du trèfle comme qubit, et que
changerait un nœud de matière noire (éremnion) ?

Entrées TQNT (V7) : spin = arité de tresse (T(2,q) : b=2 → s=1/2 ;
T(3,q) : b=3 → s=1) ; éremnions neutres par périphérie, anneau conducteur
verrouillé, dipôle χ EM-supprimé, SEUL canal EM = polarisabilité
α_pol = R_K³ = 3,8e-4 fm³ ; σ_eq ≤ 4,6e-48 cm² (bande haute, bord du
réservoir) ; masse du bord 1,50–1,64 GeV ; τ > 1e347 s.
Entrées de littérature (centres NV, déclarées comme telles) : D = 2,87 GHz,
T1(300 K) ≈ 6 ms, T2 Hahn(300 K, diamant ¹²C) ≈ 1,8 ms, T2 nucléaire ¹³C
(300 K, découplage) > 1 s. Halo local : ρ = 0,4 GeV/cm³, v = 230 km/s
(valeurs standard).
"""
import numpy as np

h = 6.62607015e-34
hbar = h / (2 * np.pi)
kB = 1.380649e-23
muB = 9.2740100783e-24
g_e = 2.00231930436
mu0_4pi = 1e-7
eps0 = 8.8541878128e-12
eV = 1.602176634e-19

print("=" * 72)
print("A. SPIN DU TRÈFLE (électron) — le canal licite")
print("=" * 72)
gamma = g_e * muB / h
print(f"Zeeman g·μ_B/h                  : {gamma/1e9:.3f} GHz/T")
D_NV = 2.870e9
for T in (300.0, 4.0, 0.02):
    kT_h = kB * T / h
    pol = np.tanh(h * D_NV / (2 * kB * T))
    print(f"T = {T:>6} K : kT/h = {kT_h/1e9:10.3f} GHz, polarisation thermique"
          f" à 2,87 GHz = {pol:.2e}")
print("→ à 300 K l'écart (2,87 GHz) est ~2 000 fois sous kT : l'état n'est pas"
      "\n  protégé par un gap ; il est initialisé par pompage optique (NV, >80 %)"
      "\n  et protégé par la FAIBLESSE du couplage au bain (phonons, spins nucléaires).")
print(f"Superposition licite : même charge Q, même masse, même nœud — seul le"
      "\n  sens du ruban (doublet 4π, b=2) change.")

T1_RT, T2_RT, T2n_RT = 6e-3, 1.8e-3, 1.0
print(f"\nCohérences à 300 K (littérature NV) : T1 ≈ {T1_RT*1e3:.0f} ms,"
      f" T2 ≈ {T2_RT*1e3:.1f} ms (électron), T2 > {T2n_RT:.0f} s (¹³C, mémoire)")

print("\nCouplage dipolaire électron–électron et porte à deux qubits :")
C_dip = mu0_4pi * (g_e * muB)**2 / h          # Hz·m³
for r_nm in (5, 10, 20, 30):
    J = C_dip / (r_nm * 1e-9)**3
    t_gate = 1 / (2 * J)
    n_ops = T2_RT / t_gate
    print(f"  r = {r_nm:>2} nm : J = {J/1e3:8.2f} kHz, porte ≈ {t_gate*1e6:8.2f} µs,"
          f" opérations dans T2 ≈ {n_ops:7.0f}")
print(f"  (constante dipolaire μ0/4π·(gμ_B)²/h = {C_dip*1e27/1e6:.1f} MHz·nm³)")

print("\n" + "=" * 72)
print("B. SPIN D'UN ÉREMNION comme qubit")
print("=" * 72)
m_GeV = np.array([1.50, 1.64])
print("Spin : T(2,q≥21) b=2 → s=1/2 (qubit) ; T(3,q≥10) b=3 → s=1 (qutrit)")
rho, v = 0.4, 230e3
n = rho / m_GeV                               # cm^-3
print(f"Densité locale : {n[1]:.2f}–{n[0]:.2f} par cm³ ; vitesse {v/1e3:.0f} km/s"
      f" → traverse 1 cm en {1e-2/v*1e9:.0f} ns")

print("\nCouplage magnétique : dipôle EM-supprimé, seul canal = polarisabilité"
      "\n  → moment magnétique permanent nul dans le cadre → fréquence Zeeman nulle.")
mu_req = h * 1e3 / (2 * 10.0)                 # pour 1 kHz sous 10 T
print(f"  Pour seulement 1 kHz sous 10 T il faudrait μ ≥ {mu_req/muB:.1e} μ_B.")

R3 = 3.8e-4 * 1e-45                           # m³
for E in (1e8, 1e10):
    dE = 2 * np.pi * eps0 * R3 * E**2
    print(f"  Polarisabilité sous E = {E:.0e} V/m : ΔE = {dE/eV:.1e} eV"
          f" = {dE/h:.1e} Hz — et IDENTIQUE pour les deux états de spin")
print("  → aucune porte, aucune lecture, aucune initialisation (spin aléatoire"
      "\n    dans le halo), aucun piège (ni charge ni moment).")
print("Cohérence : τ > 1e347 s et aucun bain couplé → T1, T2 sans borne dans le cadre."
      "\n  Mais le même couplage sert à lire et à décohérer : un état qu'aucun"
      "\n  environnement ne lit, nous ne le lisons pas non plus.")

print("\n" + "=" * 72)
print("C. ÉREMNIONS DU HALO comme bruit sur le qubit de spin NV")
print("=" * 72)
flux = n[0] * v * 1e2                          # cm^-2 s^-1 (borne haute : masse basse)
sig_nucleon = 4.6e-48                          # cm², bande haute
A_C = 12
sig_C = sig_nucleon * A_C**2                   # cohérence A² (majorant)
n_diam = 1.76e23                               # atomes/cm³
V_10nm = 4 / 3 * np.pi * (10e-7)**3            # cm³
N_nuc = n_diam * V_10nm
rate = flux * sig_C * N_nuc
print(f"Flux ≤ {flux:.1e} /cm²/s ; σ(C) ≤ {sig_C:.1e} cm² ; noyaux à 10 nm ≈ {N_nuc:.1e}")
print(f"Collisions dans la sphère de 10 nm ≤ {rate:.1e} /s")
print(f"Comparé à 1/T2 = {1/T2_RT:.0f} /s : rapport ≤ {rate*T2_RT:.0e}")
print("Et chaque collision est indépendante du spin (recul nucléaire, gravitation :"
      "\n  même couplage pour ↑ et ↓) → aucune phase relative : bruit nul en pratique.")
