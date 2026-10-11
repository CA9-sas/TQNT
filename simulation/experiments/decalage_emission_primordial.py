"""Un décalage d'émission primordial ? (16/09/2026)
Hypothèse : les photons produits tôt naîtraient avec une énergie moindre (« l'énergie
est rare au début »), d'où un décalage vers le rouge d'émission, distinct de l'expansion,
qui altérerait toutes les observations. On paramètre : λ_émis(z) = λ_0 (1+ε(z)),
et l'on demande (i) ce que la trame en dit, (ii) ce que les données bornent.
"""
import math
print("=== (i) La trame : ce qui fixe la longueur d'onde émise ===")
alpha = 1/137.035999; me = 0.51099895e6  # eV
Ry = alpha**2 * me / 2
print(f"  raie atomique : E ∝ α² m_e c² (Rydberg {Ry:.2f} eV) ; α = 1/(k+2), k = 135 ENTIER (verrou : pas de dérive continue)")
print("  m_e = Green retardée sur le tube trèfle (σ_χ, α) ; σ_χ lu sur M_Pl (rigidité, préexiste à t_c) ; h = la dent, jamais écrasée")
print("  → après le gel (t > 1,6e-4 s), une transition donnée émet la MÊME λ à toute époque : ε(z) = 0 est une prédiction du cadre")
print("  → avant le gel : pas de G, T > 62 MeV, tout photon est thermalisé (le CMB découple à 0,26 eV, 3,8e5 ans plus tard)")
print("  → un écrasement UNIFORME de la trame est inobservable (relativité d'échelle) : seul le redshift cumulatif l'est — c'est l'expansion")
print()
print("=== (ii) Les bornes observationnelles sur ε(z) ===")
# CMB T(z)
beta, sb = -0.007, 0.027   # T(z)=T0(1+z)^(1-β), SZ + absorption moléculaire (Hurier+2014 ; Luzzi+2015)
for z in (1, 3):
    print(f"  T_CMB(z={z}) : (1+z)^(1-β), β = {beta}±{sb} → |ε| < {sb*math.log(1+z):.3f} à {z}")
# FIRAS : un décalage sans expansion n'est pas un corps noir
print("  FIRAS : corps noir à |μ| < 9e-5, |y| < 1,5e-5 — un redshift d'émission (nombre de photons ≠ dilution a³) casse la loi de Planck : exclu à ~1e-4")
# SN Ia time dilation
print("  SN Ia : durée des courbes de lumière ∝ (1+z)^b, b = 1,007 ± 0,009 (Blondin+2008) → le décalage est une dilatation du TEMPS, pas un déficit d'émission")
# quasars alpha, mu
print("  Quasars : Δα/α < 1e-6 (z~1-3), Δμ/μ < 1e-7 (H₂) → rapports de raies fixés : ε(z) commun à toutes les raies < 1e-6 sur les rapports")
print("  Oklo : Δα/α < 1e-8 sur 2 Ga ; horloges : α̇/α < 1e-17/an")
print()
print("=== (iii) Ce qu'un ε(z) ≠ 0 ferait, s'il existait ===")
for eps in (0.01, 0.1):
    print(f"  ε = {eps:4.2f} : z_vrai = (1+z_obs)/(1+ε) − 1 → à z_obs = 1100 (CMB) : {(1101)/(1+eps)-1:.0f} ; H0 lu ×(1+ε)^-1 sur les échelles ∝ z : biais {100*eps:.0f} % — exclu par b, β, FIRAS")
print()
print("BILAN : l'énergie disponible ne fixe pas la longueur d'onde d'une raie ; α et m_e la fixent, et le cadre les verrouille. Il n'y a pas de redshift d'émission primordial : ce qui rougit est la trame qui s'étend (Friedmann mécanisé), vérifié par la dilatation du temps des SN, T(z) du CMB et la loi de Planck.")
