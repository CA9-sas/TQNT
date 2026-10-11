"""La fuite ressort en « particules plates » qui épousent la trame — presque des neutrinos (16/09/2026).

(A) Où les placer dans le cadre : le neutrino est une onde χ ouverte auto-rotative, m = ħc/2R (courbure d'auto-rotation) ;
    une onde « plate » est la limite R → ∞, m → 0 : le membre sans courbure de la famille.
(B) Peuvent-elles éclairer l'enveloppe d'un point rouge ? (absorption requise σN ≳ 1)
(C) Ou bien expliquer la FAIBLESSE EN X des points rouges : la fuite emporte l'énergie dissipée au plus près du trou noir
    (couronne, disque interne), là où naissent les rayons X, sans toucher l'optique.
(D) Fond cosmique de particules plates ; (E) compatibilité avec « aucun neutrino stérile ».
"""
import math
hbarc = 197.327e-15   # MeV·m
print("=== (A) La famille des neutrinos du cadre et sa limite plate ===")
for R in (1.94e-6, 9.9e-6, 1.97e-5, 1e-3, 1.0):
    m_meV = hbarc/(2*R)*1e9
    print(f"  R = {R:.2e} m → m = ħc/2R = {m_meV:.2e} meV")
print("  → R → ∞ : m → 0, onde sans courbure propre qui « épouse » la trame : c'est le membre plat de la famille (vitesse c).\n")
print("=== (B) Éclairer l'enveloppe exige une absorption ===")
AU = 1.496e13
for n in (1e9, 1e11):
    for R_AU in (100, 1000):
        N = n*R_AU*AU
        print(f"  enveloppe n = {n:.0e} cm⁻³ sur {R_AU} UA : colonne N = {N:.1e} cm⁻² → σ requis ≳ {1/N:.0e} cm² par nucléon")
GF2E2 = 1.0e-38
print(f"  un quasi-neutrino de 1 GeV : σ ~ {GF2E2:.0e} cm² → absorbé à 1e-14–1e-11 près : il TRAVERSE l'enveloppe")
print("  → presque-neutrinos : ils ne peuvent PAS allumer l'enveloppe ; la lumière du point rouge reste l'accrétion.\n")
print("=== (C) Mais ils peuvent RETIRER les rayons X ===")
def L_out(x):  # fraction de la luminosité d'un disque mince émise au-delà de r = x·r_in (Newtonien)
    return 3/x*(1 - (2/3)*math.sqrt(1/x))
for x in (1.5, 2, 3, 5):
    print(f"  énergie dissipée à l'intérieur de {x:.1f} r_in ({6*x:.0f} r_g) : {100*(1-L_out(x)):.0f} % de L_disque")
print("  la couronne X d'un AGN emporte typiquement 2–10 % de L_bol, produite à ≲ 10 r_g.")
print("  → une fuite qui capture l'énergie dissipée à ≲ 10 r_g (quelques % à ~30 % de L) éteint les X sans toucher l'optique :")
print("    points rouges FAIBLES EN X, raies de Balmer normales — ce qu'on observe. Et invisible : Sgr A* ne la borne pas.\n")
print("=== (D) Fond cosmique de particules plates ===")
rho_BH = 5e5; rho_c = 1.26e11   # Msun/Mpc³ (Soltan) ; critique
for eta in (0.01, 0.05, 0.3):
    print(f"  η_fuite = {eta:<4} de l'énergie de repos accrétée → Ω_plates ≲ {eta*rho_BH/rho_c:.0e} (avant dilution par z) ; Ω_γ = 5e-5 → négligeable")
print()
print("=== (E) Compatibilité avec « aucun neutrino stérile » (prédiction du cadre) ===")
print("  un stérile se trahit par le MÉLANGE avec les neutrinos actifs (oscillations, N_eff à la BBN) ;")
print("  une particule plate SANS saveur, sans mélange, produite seulement à l'intérieur des trous noirs après z ~ 20,")
print("  n'oscille pas, ne change pas N_eff (BBN/CMB bien avant) : elle n'est pas un stérile au sens testé — à consigner comme catégorie distincte.")
print()
print("BILAN : les particules plates (limite m → 0 de la famille des neutrinos) ne peuvent pas éclairer un point rouge,")
print("mais une fuite qui emporte l'énergie des ~10 r_g internes explique naturellement sa FAIBLESSE EN X, reste invisible")
print("(Sgr A*, fond cosmique Ω ≲ 1e-7) et ne viole pas l'absence de stériles. Prédiction : le déficit X croît avec le taux")
print("d'accrétion (régime où la fuite s'ouvre) ; aucun signal dans les détecteurs de neutrinos.")
