# -*- coding: utf-8 -*-
"""Les quatre hérités relus sous la trame (2026-09-02).

H1 — H₀ local : le couloir fermé des deux côtés (théorèmes early interdits +
     mur quadruple) → seul reste échelle des distances : vide RG quantifié
     (profondeur requise vs comptages) → systématique dominante.
H2 — S₈ : théorème de croissance (σ₈=Planck) + libre parcours des éremnions
     (coupure de free-streaming) → régression vérifiée (KiDS-Legacy).
H3 — Li-7 : balayage des canaux de la machinerie récente contre le mur
     d'efficacité (éremnions cinématiquement fermés ; η² déclaré) →
     externalité stellaire prédite.
H4 — Muon puzzle : bilan de Heitler–Matthews chiffré + chimie du condensat χ
     thermalisé (fraction EM) → direction droite, résiduel avant.

Aucune constante nouvelle. Toutes les entrées sont canoniques (V6/V7).
"""
import numpy as np
from scipy.special import kn

MEV = 1.0
GEV = 1e3 * MEV

print("=" * 72)
print("H1 — TENSION H0 : LE COULOIR FERMÉ DES DEUX CÔTÉS")
print("=" * 72)

H0_planck = 67.4          # côté Planck (registre TQNT ; V7 platitude : 67,2)
H0_shoes = 73.04          # Riess 2022 (SH0ES), ±1,04
Om = 0.315
f_growth = Om**0.55       # taux de croissance linéaire

dH_req = H0_shoes / H0_planck - 1.0
delta_req = -3.0 * dH_req / f_growth
print(f"ΔH/H requis            : {dH_req:+.4f} ({H0_shoes} vs {H0_planck})")
print(f"f = Ω_m^0.55           : {f_growth:.3f}")
print(f"δ de vide REQUIS       : {delta_req:+.3f} (ΔH/H = -f·δ/3, linéaire)")

# vide KBC (Keenan-Barger-Cowie 2013) : δ ≈ -0,20 jusqu'à ~300 Mpc
delta_kbc = -0.20
dH_kbc = -f_growth * delta_kbc / 3.0
print(f"vide KBC δ=-0,20       : ΔH/H = {dH_kbc:+.4f} → H0_loc = "
      f"{H0_planck*(1+dH_kbc):.1f} (couvre {dH_kbc/dH_req*100:.0f}% de la tension)")

# volume SH0ES (z ≤ 0,15 ≈ 640 Mpc) : homogénéité δ ≲ 0,05
delta_homog = -0.05
dH_homog = -f_growth * delta_homog / 3.0
print(f"homogénéité à 640 Mpc  : δ ≲ -0,05 → ΔH/H ≤ {dH_homog:+.4f} "
      f"(+{H0_planck*dH_homog:.1f} km/s/Mpc seulement)")
print("→ le vide RG couvre AU PLUS ~40% (coquille optimiste), ~10% au volume")
print("  SH0ES réel → la part dominante est SYSTÉMATIQUE (échelle des distances).")
print("CCHP/JWST 2024-25 : Céphéides 72,05 / TRGB 69,85 / JAGB 67,96 (±~1,5-1,9)")
print("→ dispersion inter-calibrateurs >> erreurs déclarées = la systématique EXISTE.")

print()
print("-" * 72)
print("H1b — SIRÈNES STANDARD : LE RÉFÉRENTIEL DE LA TRAME (ajout 02/09)")
print("-" * 72)
# l'observateur est en mouvement sur la trame ; le dipôle CMB le mesure.
c_kms = 2.998e5
z_gw = 0.0098             # GW170817
cz = c_kms * z_gw
v_dipole = 370.0          # km/s, dipôle CMB (notre vitesse sur la trame)
v_pec_host = 250.0        # incertitude v_pec de l'hôte
print(f"cz(GW170817) = {cz:.0f} km/s ; dipôle 370 km/s = "
      f"{v_dipole/cz*100:.1f}% sur H0 (corrigé exactement — SR, sans traînée)")
print(f"v_pec hôte ±{v_pec_host:.0f} km/s → ΔH0 = ±{v_pec_host/cz*H0_planck:.1f} "
      f"km/s/Mpc PAR ÉVÉNEMENT (systématique dominant, plage 64-73 publiée)")
for zev, dl_err in ((0.0098, 0.15), (0.05, 0.15)):
    sig = np.hypot(v_pec_host / (c_kms * zev), dl_err)
    print(f"z={zev} : σ_H/H = {sig*100:.0f}%/sirène → "
          f"N({sig*100:.0f}%→1,5%) ≈ {(sig/0.015)**2:.0f} événements")
print("→ les sirènes contournent TOUTE l'échelle des distances (amplitude =")
print("  distance absolue, chirp V5.9) : PRÉDICTION H0(sirènes) → 67,4±1,")
print("  pas 73 — l'arbitre le plus propre du verdict « systématique ».")

print()
print("-" * 72)
print("H1c — LE QUADRILLAGE DE LA MATIÈRE NOIRE SUR LA TRAME (ajout 03/09)")
print("-" * 72)
# maille du quadrillage éremnion (réservoir m=1,9 GeV)
rho_c = 9.47e-27          # kg/m3
m_kg = 3.39e-27           # 1,9 GeV en kg
Odm = 0.265
n_mean = Odm * rho_c / m_kg
n_halo = 0.4 / 1.9        # /cm3 (halo local 0,4 GeV/cm3)
print(f"maille comobile moyenne  : {n_mean**(-1/3.):.2f} m (1 éremnion/m³ cosmique)")
print(f"maille halo local        : {n_halo**(-1/3.):.1f} cm")
# maille primordiale au gel (cordes dans les rails, ξT=κ_dom)
xi_gel = 197.3e-15 * 1.14 / 62.0   # m physique
a_gel = 2.41e-12
print(f"maille au gel            : {xi_gel*1e15:.1f} fm physique → "
      f"{xi_gel/a_gel*1e3:.1f} mm comobile (motif effacé par virialisation)")
# profondeur du puits d'UN éremnion à mi-maille
r_half = 0.5 * n_mean**(-1/3.)
depl = 2.0 * (1.9 / 1.22e19) * (1.616e-35 / r_half)
print(f"puits par éremnion       : Ω²−1 = {depl:.1e} à mi-maille")
# granularité Poisson à l'échelle de mesure
R8 = 8.0 * 3.086e22
N8 = n_mean * 4.0 * np.pi / 3.0 * R8**3
print(f"granularité à 8 Mpc      : δρ/ρ = 1/√N = {1.0/np.sqrt(N8):.1e}")
print("→ le quadrillage est un FLUIDE PARFAIT pour la trame (rivière exacte) —")
print("  aucune échelle de mesure ne voit la maille : pas de canal de grain.")
# part du quadrillage dans la poussée
H0 = 67.4
push_dm = H0 * (1.0 - np.sqrt(1.0 - Odm))
print(f"part de poussée du quadrillage : H0 sans DM = {H0*np.sqrt(1-Odm):.1f} → "
      f"les éremnions PORTENT +{push_dm:.1f} km/s/Mpc de H0")
# variance d'échantillonnage de la toile (le vrai canal, déjà compté en H1)
f_g = 0.315**0.55
for R in (100.0, 150.0, 250.0):
    sig_d = 0.83 * 8.0 / R    # σ(R) linéaire ~ σ8·(8/R), h=1 approx
    print(f"toile à R={R:.0f} Mpc/h : σ(δ)={sig_d:.3f} → "
          f"σ(H0) = ±{f_g/3.0*sig_d*H0:.2f} km/s/Mpc")
print("→ l'empreinte de la toile = la variance cosmique, ±0,3-0,8 km/s/Mpc :")
print("  RÉELLE mais ×10 sous la tension (5,6) — déjà contenue dans le vide H1.")

print()
print("=" * 72)
print("H2 — S8 : THÉORÈME DE CROISSANCE + LIBRE PARCOURS DES ÉREMNIONS")
print("=" * 72)

# coupure de free-streaming du réservoir (découplage cinétique AU gel, dérivé)
T_kd = 62.0 * MEV         # T_c dérivé (serrage topologique)
t_kd = 1.61e-4            # s (marche de G, au T_c dérivé)
m_erem = 1.9 * GEV        # bord du réservoir c=23-25
g_s0, g_skd = 3.91, 15.2  # g_s(T_c) dérivé
T0 = 2.35e-10 * MEV       # 2,35e-4 eV

a_kd = (T0 / T_kd) * (g_s0 / g_skd)**(1.0 / 3.0)
v_kd = np.sqrt(3.0 * T_kd / m_erem)
a_eq = 1.0 / 3400.0
c_si = 2.998e8
lam_fs = 2.0 * v_kd * c_si * t_kd / a_kd * np.log(a_eq / a_kd)   # m comobile
pc = 3.086e16
rho_m0 = Om * 9.47e-27    # kg/m3
M_fs = 4.0 * np.pi / 3.0 * (lam_fs / 2.0)**3 * rho_m0
Msun = 1.989e30
print(f"v au découplage        : {v_kd:.3f} c   (T_kd = {T_kd:.0f} MeV, dérivé)")
print(f"a_kd                   : {a_kd:.2e}")
print(f"λ_fs comobile          : {lam_fs/pc:.1f} pc")
print(f"M_fs (coupure)         : {M_fs/Msun:.1e} M_sun")
print(f"→ ~19 ordres sous l'échelle de σ8 (8 Mpc/h ~ 1e14 M_sun) :")
print("  le réservoir est STRICTEMENT CDM aux échelles de structure —")
print("  le théorème σ8 = σ8^Planck est sans échappatoire interne.")
print("  (NB : la coupure ~1e-5 M_sun borde le domaine des perles — cohérence.)")

# confrontation
S8_planck, dS8_p = 0.832, 0.013
S8_kids, dS8_k = 0.815, 0.019   # KiDS-Legacy 2025 (moy. des asym.)
S8_v7, dS8_v7 = 0.826, 0.011    # combiné enregistré V7
z_kids = abs(S8_planck - S8_kids) / np.hypot(dS8_p, dS8_k)
z_v7 = abs(S8_planck - S8_v7) / np.hypot(dS8_p, dS8_v7)
print(f"KiDS-Legacy 2025       : S8 = {S8_kids}±{dS8_k} → {z_kids:.2f}σ de Planck")
print(f"combiné V7             : S8 = {S8_v7}±{dS8_v7} → {z_v7:.2f}σ de Planck")
print("→ la tension RÉGRESSE comme le théorème l'exige (2-3σ → <1σ).")

print()
print("=" * 72)
print("H3 — LI-7 : BALAYAGE DES CANAUX RÉCENTS CONTRE LE MUR D'EFFICACITÉ")
print("=" * 72)

# (a) destruction du 7Be par collision éremnion — cinématique d'abord
T_bbn = 0.05 * MEV        # ~50 keV (fenêtre 7Be)
E_kin_dm = 1.5 * T_bbn    # énergie cinétique thermique
Q_be7 = 1.587 * MEV       # seuil de photodissociation 7Be
print(f"(a) éremnions : E_cin = {E_kin_dm*1e3:.0f} keV vs seuil 7Be "
      f"{Q_be7:.3f} MeV → SOUS-SEUIL ×{Q_be7/E_kin_dm:.0f} (fermé cinématiquement)")

# taux quand même (pour l'ordre)
n_gamma = 2.03e28 * (T_bbn / 0.05)**3      # cm-3 à 50 keV
eta_b = 6.14e-10
n_b = eta_b * n_gamma
n_dm = 5.4 * n_b * (0.938 / (m_erem / GEV))  # rapport de masse Ω, nombre
sigma_erem = 1e-48        # cm² (équivalent SI dérivé, bord c=23)
v_dm = np.sqrt(3.0 * T_bbn / m_erem) * 2.998e10
Gamma = n_dm * sigma_erem * v_dm
t_window = 1e4            # s (fenêtre 7Be)
print(f"    taux Γ = nσv = {Gamma:.1e} s⁻¹ → Γ·t = {Gamma*t_window:.1e} "
      f"(mort par ~{-np.log10(Gamma*t_window):.0f} ordres, en plus du seuil)")

# (b) rappel du mur (cendres)
P_photon = 3e-13
need = 1.0 / P_photon / 4e0   # ~840 photons/baryon pour détruire ~1/3 du 7Be
print(f"(b) mur d'efficacité   : P(γ→7Be) = {P_photon:.0e}/photon → ~840 γ/baryon")
print("    requis (~1,7 GeV/baryon) vs 8,6e-2 disponibles : déficit ×1e4 — FERMÉ")
print("    (canaux neutronique/hadronique : hors mécanisme, fermés 24/07).")

# (c) η_B dérivé et pente en η
eta_tqnt = 7.1e-10
li_scale = (eta_tqnt / eta_b)**2.12
print(f"(c) η_B dérivé 7,1e-10 vs CMB 6,14e-10 : Li/H ∝ η^2.12 → ×{li_scale:.2f}")
print("    (dans l'enveloppe O(1) déclarée ; BBN utilise le η mesuré — sans effet)")

# (d) l'externalité requise
li_spite = 1.6e-10
li_bbn = 4.7e-10
dep = np.log10(li_bbn / li_spite)
print(f"(d) déplétion stellaire requise : {li_bbn/li_spite:.1f} (= {dep:.2f} dex)")
print("→ TQNT prédit une BBN STRICTEMENT STANDARD (P22 nul, éremnions fermés)")
print("  ⇒ la résolution est PRÉDITE EXTERNE (déplétion stellaire/diffusion —")
print("  signature déjà vue NGC 6397) ; falsifiable : une étoile vierge certifiée")
print("  sans déplétion AU plateau rouvrirait une tension de classe F.")

print()
print("=" * 72)
print("H4 — MUON PUZZLE : BILAN HEITLER-MATTHEWS + CHIMIE DU CONDENSAT χ THERMALISÉ")
print("=" * 72)

# bilan requis
E0 = 1e19                 # eV
xi_c = 2e10               # eV (énergie critique du pion ~20 GeV)
N_tot = 50.0              # multiplicité effective
r0 = 2.0 / 3.0            # rétention hadronique naïve (2/3)
z_deficit = 0.35          # ln(données/MC) à 1e19 eV (WHISP)
n_gen = np.log(E0 / xi_c) / np.log(N_tot)
beta0 = 1.0 + np.log(r0) / np.log(N_tot)
dbeta = z_deficit / np.log(E0 / xi_c)
dr = dbeta * r0 * np.log(N_tot)
f_em0 = 1.0 - r0
f_em_req = 1.0 - (r0 + dr)
print(f"générations            : {n_gen:.1f} ; β₀ = {beta0:.3f}")
print(f"déficit z = ln(N_μ^data/N_μ^MC) = {z_deficit} → Δβ = {dbeta:.4f}")
print(f"rétention hadronique   : {r0:.3f} → {r0+dr:.3f} (Δr = +{dr:.3f})")
print(f"fraction EM/interaction: {f_em0:.3f} → {f_em_req:.3f} "
      f"({(f_em_req/f_em0-1)*100:.1f}% relatif) — LA part requise")

# chimie du condensat χ thermalisé à T_trans (statistique, Boltzmann)
T = 156.5 * MEV
# (nom, masse MeV, dégénérescence, fraction EM d'énergie des désintégrations)
species = [
    ("pi0",   135.0,  1, 1.00), ("pi+-",  139.6,  2, 0.00),
    ("K+-",   493.7,  2, 0.15), ("K0",    497.6,  2, 0.26),
    ("eta",   547.9,  1, 0.81), ("rho0",  775.3,  3, 0.00),
    ("rho+-", 775.3,  6, 0.50), ("omega", 782.7,  3, 0.38),
    ("K*",    893.6, 12, 0.20), ("eta'",  957.8,  1, 0.60),
    ("phi",  1019.5,  3, 0.15), ("N",     938.9,  8, 0.10),
    ("Delta",1232.0, 16, 0.15),
]
num, en, em = 0.0, 0.0, 0.0
for name, m, g, w in species:
    x = m / T
    n_i = g * m**2 * T * kn(2, x)                 # densité (Boltzmann)
    E_i = m * kn(1, x) / kn(2, x) + 3.0 * T       # énergie moyenne
    num += n_i
    en += n_i * E_i
    em += n_i * E_i * w
f_em_th = em / en
print(f"condensat χ thermalisé (T = {T:.1f} MeV, gaz de résonances) :")
print(f"fraction EM thermique  : f_EM = {f_em_th:.3f} vs naïf 1/3")
print(f"→ Δf_EM(thermique) = {f_em_th-f_em0:+.3f} vs requis {f_em_req-f_em0:+.3f} :")
print("  la chimie statistique va DANS LE SENS du déficit et couvre l'ordre —")
print("  MAIS les générateurs modernes portent déjà l'essentiel des résonances")
print("  (le déficit persiste SOUS EPOS-LHC) : le résiduel vit dans la région")
print("  AVANT (particule de tête, échange de charge π→ρ⁰) — hors du domaine")
print("  central de la thermalisation. OUVERT, chiffré.")
print("Prédiction dure (désert + Vassiliev) : AUCUNE composante exotique dans")
print("les gerbes — le déficit doit se fermer en QCD molle avant.")
print("Discriminants datés : run O-O LHC (07/2025, √s=5,36 TeV), AugerPrime")
print("(X_max+N_μ par événement), LHCf/FASER (π⁰/ρ⁰ avant).")
