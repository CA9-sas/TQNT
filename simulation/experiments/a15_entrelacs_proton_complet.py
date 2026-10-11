# -*- coding: utf-8 -*-
"""A-15 — L'entrelacs complet du proton (3₁ + 7₂ + 7̄₂) comme qubit (17/09/2026).

Suite de A-14. Quatre variables candidates :
  (1) l'enlacement Lk (proton Lk=+3 / neutron Lk=+2) ;
  (2) l'excitation interne de même charge : nucléon (J=½) → Δ⁺ (J=3/2) ;
  (3) le spin ½ du proton ;
  (4) l'entrelacs du proton complété par un électron-trèfle : l'atome
      d'hydrogène, et son qubit hyperfin d'horloge F=0,m=0 ↔ F=1,m=0 (ΔM = 0).
Entrées TQNT (V7) : Q = Σχ + Lk (p/n diffèrent par Lk) ; Δm(n−p) = 1,293 MeV
(dérivé) ; spin ½⊗1⊗1 → {½, 3/2, 5/2} ; théorème de la paire miroir : dans
7₂ + 7̄₂ les sens de circulation s'annulent → le moment magnétique du nucléon
est porté par le cœur trèfle.
Entrées de mesure déclarées : Δ(1232) masse 1232 MeV, largeur 117 MeV ;
γ_p ; fréquence hyperfine de l'hydrogène 1 420,405 751 768 MHz ; coefficient
d'Einstein de la raie à 21 cm A = 2,87e-15 s⁻¹.
"""
import numpy as np

h = 6.62607015e-34
hbar = h / (2 * np.pi)
kB = 1.380649e-23
eV = 1.602176634e-19
muB_h = 13.996245e9            # Hz/T  (μ_B/h)
muN_h = 7.6225932e6            # Hz/T  (μ_N/h)
gJ, gI = 2.00231930436, 5.5856946893

print("=" * 72)
print("(1) ENLACEMENT Lk : proton ↔ neutron")
print("=" * 72)
print("  Q = Σχ + Lk : Lk = +3 (Q = +1) et Lk = +2 (Q = 0) → charges différentes"
      "\n  → superposition interdite (superselection). Écart Δm(n−p) = 1,293 MeV.")

print("\n" + "=" * 72)
print("(2) EXCITATION INTERNE DE MÊME CHARGE : N(939) → Δ⁺(1232)")
print("=" * 72)
dE = 1232 - 938.3                              # MeV
Gam = 117.0                                    # MeV
f = dE * 1e6 * eV / h
tau = hbar / (Gam * 1e6 * eV)
print(f"  Même charge, J = ½ et 3/2 (même secteur demi-entier), M = +½ possible : licite.")
print(f"  écart {dE:.0f} MeV → fréquence {f:.1e} Hz ; largeur {Gam:.0f} MeV → τ(Δ) = {tau:.1e} s")
print(f"  périodes avant désintégration ≈ f·τ = {f*tau:.2f} : moins d'une oscillation")
print("  pilotage : photons gamma de ~300 MeV (photoexcitation M1) — pas de contrôle"
      "\n  cohérent. Tous les états internes du nucléon sont à ≳300 MeV et se"
      "\n  désintègrent par voie forte en ~1e-23 s. Verdict : inutilisable.")

print("\n" + "=" * 72)
print("(3) SPIN ½ DU PROTON : le spin du cœur trèfle")
print("=" * 72)
gp_h = gI * muN_h                              # Hz/T pour ΔM=1 : γ_p/2π
print(f"  paire miroir 7₂+7̄₂ : sens annulés → le moment est celui du cœur 3₁"
      f"\n  γ_p/2π = {gp_h/1e6:.3f} MHz/T ; aucun canal de bruit magnétique propre aux compagnons")
dB = 10e-6
sf = gp_h * dB
print(f"  bruit de champ 10 µT (ΔM = 1) : σ_f = {sf:.0f} Hz → T2* = {np.sqrt(2)/(2*np.pi*sf)*1e3:.2f} ms")

print("\n" + "=" * 72)
print("(4) PROTON + ÉLECTRON-TRÈFLE = HYDROGÈNE : qubit d'horloge hyperfin (ΔM = 0)")
print("=" * 72)
f0 = 1420.405751768e6
k_x = (gJ * muB_h + gI * muN_h) / f0           # x = k_x · B
kz = f0 * k_x**2 / 2                            # Hz/T² (second ordre)
print(f"  f0 = {f0/1e6:.6f} MHz ; décalage d'horloge au second ordre = {kz*1e-8:.0f} Hz/G²"
      f" (Breit–Rabi)")


def f_clock(B):
    x = k_x * B
    return f0 * np.sqrt(1 + x**2)


rng = np.random.default_rng(20260917)
print("  Monte-Carlo, bruit de champ vectoriel quasi statique σ par composante :")
for B0, sig, lab in ((10e-6, 10e-6, "non blindé"), (10e-6, 1e-9, "blindé"),
                     (1e-6, 1e-10, "blindage renforcé")):
    Bv = np.array([0, 0, B0]) + rng.normal(0, sig, (200000, 3))
    fs = f_clock(np.linalg.norm(Bv, axis=1))
    s = np.std(fs)
    print(f"    B0 = {B0*1e6:5.1f} µT, σ = {sig*1e9:8.1f} nT ({lab:<18}) : σ_f = {s:.2e} Hz"
          f" → T2* ≈ {np.sqrt(2)/(2*np.pi*s):.2e} s")

A21 = 2.87e-15
nbar = kB * 300 / (h * f0)
print(f"\n  Désexcitation spontanée F=1→0 : 1/A = {1/A21/3.156e7:.1e} ans")
print(f"  Rayonnement du corps noir à 300 K : n̄ = {nbar:.0f} → taux stimulé A·n̄ ="
      f" {A21*nbar:.1e} /s ({1/(A21*nbar)/3.156e7:.0f} ans)")
print(f"  kT/h à 300 K = {kB*300/h/1e9:.0f} GHz ≫ f0 : populations égales à l'équilibre,"
      "\n  mais un atome isolé ne s'y relaxe qu'en milliers d'années → 300 K sans effet"
      "\n  sur le qubit lui-même ; la limite pratique est le piégeage et les collisions.")
print("  Repères expérimentaux (littérature) : maser à hydrogène, ampoule de stockage"
      "\n  à ~300 K, temps d'interaction ~1 s ; qubit d'horloge hyperfin d'un ion piégé"
      "\n  (même principe, ¹⁷¹Yb⁺ 12,6 GHz), cohérence démontrée au-delà de l'heure.")

print("\n" + "=" * 72)
print("SYNTHÈSE")
print("=" * 72)
rows = [
    ("Lk (p/n)", "interdit", "superselection de charge"),
    ("N → Δ⁺", "licite", f"τ = {tau:.0e} s, < 1 oscillation"),
    ("spin du proton", "licite", "T2* ~ 0,5 ms sous 10 µT ; porté par le cœur 3₁"),
    ("hydrogène, horloge ΔM=0", "licite", "second ordre seul ; secondes à heures ; 300 K sans effet"),
]
for r in rows:
    print(f"  {r[0]:<24} {r[1]:<9} {r[2]}")
print("  L'entrelacs complet ne sert pas par ses états internes (trop hauts, trop"
      "\n  brefs) ; il sert par son spin, et surtout couplé à un électron-trèfle :"
      "\n  l'atome d'hydrogène donne le qubit à ΔM = 0 le plus robuste de la série.")
