"""h plus petit à la naissance de l'univers, étalement progressif : calcul avec ces états.
(16/09/2026 — demande explicite ; exploration hors quatuor V6 « h = la dent, invariant »)

On pose h(t) = g(t)·h₀, g → 1 aujourd'hui, g < 1 tôt, et l'on demande ce qui change.
Le point dur : h est DIMENSIONNÉ. Seul un rapport sans dimension est observable ; il faut
donc dire ce qui reste fixe quand h bouge. Trois lectures cohérentes avec le cadre :
  (A) c, M_Pl et toutes les ÉNERGIES (σ_χ, masses, kT) fixes ; h ∝ g
      → G = ħc/M_Pl² ∝ g, longueurs de la matière ∝ g, temps atomiques ∝ g ;
  (B) idem (A) mais la tension σ_χ fixe en FORCE (N) : énergies ∝ √g → m/M_Pl ∝ √g ;
  (C) h ∝ g, c et M_Pl fixes, G ∝ g, énergies fixes, mais le rougissement compté en
      LONGUEUR D'ONDE (λ_émis = h c/E ∝ g) — la lecture « décalage » de l'objection.
Pour chacune : BBN, CMB, redshift, phase des éremnions (cellule h³, borne de Pauli), perles.
"""
import math
h0 = 6.626e-34; hb0 = h0/(2*math.pi); c = 2.998e8; G0 = 6.6710e-11; kB = 1.381e-23
Msun = 1.989e30; m_er = 1.8*1.783e-27          # kg (1 GeV = 1,783e-27 kg)
MPl = math.sqrt(hb0*c/G0)

print("=== 0. Ce qui est sans dimension dans le cadre (et donc observable) ===")
print(f"  α = 1/(k+2) : topologique, indépendant de h ;  m_er/M_Pl = {m_er/MPl:.2e} ;  m_p/m_e = 1836 (rapport d'énergies)")
print("  h n'apparaît seul dans AUCUN rapport : il convertit énergie ↔ fréquence, action ↔ nombre de cellules.\n")

def bloc(nom, energies_fixes, redshift_en_lambda):
    print(f"=== Lecture {nom} ===")
    # facteurs d'échelle en fonction de g
    E = "1" if energies_fixes else "g^1/2"
    print(f"  énergies ∝ {E} ; h ∝ g ; G ∝ g ; ℓ_P = ħ/(M_Pl c) ∝ g ; λ_Compton = h/(mc) ∝ {'g' if energies_fixes else 'g^1/2'}")
    # thermique : rho_rad ∝ (kT)^4/(ħc)^3 ; H ∝ sqrt(G rho) ; taux Γ_temps = Γ_E/ħ
    if energies_fixes:
        print("  ρ_rad(T) ∝ T⁴/(ħc)³ ∝ g⁻³ ; H = √(8πGρ/3) ∝ √(g·g⁻³) = g⁻¹ ; tout taux Γ = Γ_E/ħ ∝ g⁻¹  →  Γ/H INVARIANT")
        print("  → histoire thermique en T INCHANGÉE (gel des ν, n/p, BBN, recombinaison) ; seules les étiquettes en secondes changent (t ∝ g)")
        print("  → CMB : distribution de Planck conservée (x = hν/kT invariant), FIRAS aveugle ; T(z) par excitation (kT/E) invariant")
        print("  → Y_p, D/H : INVARIANTS (ne dépendent que de rapports d'énergies et de Γ/H)")
    else:
        print("  m/M_Pl ∝ √g : la hiérarchie change — G_eff pour la matière ∝ g (à énergies mesurées en unités de m)")
        print("  → BBN : H/Γ ∝ √g → |Δg| < 0,2 à t = 1 s (Y_p à 1,5 %) ; CMB : |Δg| < 0,09 à z = 1100 ; LLR : |ġ/g| < 1e-13/an")
        print("  → c'est la fourche « G(t) » déjà tranchée (mémoire) : rien ne bouge après BBN à mieux que 10 %")
    if redshift_en_lambda:
        print("  redshift : λ_émis = hc/E ∝ g(t_em) → 1+z_λ = g(t_em)·(1+z_a) — un BLEUISSEMENT apparent des sources anciennes, pas un rougissement")
        print("  test : dilatation du temps des SN Ia (1+z_t = 1+z_a) contre 1+z_λ : b = 1,007±0,009 → |1−g| < 1 % à z = 0,5 (t = 8,6 Ga)")
    print()

bloc("(A) énergies fixes", True, False)
bloc("(B) tension en force fixe", False, False)
bloc("(C) énergies fixes, décalage lu en λ", True, True)

print("=== L'espace des phases sous h(t) : cellule h³ et borne de Pauli ===")
QP0 = 2*m_er**4/h0**3
print(f"  Q_Pauli(aujourd'hui) = 2 m⁴/h₀³ = {QP0:.1e} kg m⁻⁶ s³ (éremnion 1,8 GeV)")
for nom, Qb in (("naissance thermique homogène", 5.15e-17), ("filament 1e5 à 0,03 c", 6e-9), ("seuil perle", 2.3e-6), ("défaut F", 1e-22)):
    gstar = (QP0/Qb)**(1/3)
    print(f"  Q = {Qb:.1e} ({nom:>28}) : dégénéré dès que g > {gstar:.2e}  → {'jamais (g ≤ 1)' if gstar > 1 else 'atteint à g = %.2f' % gstar}")
print("  lecture (A) : Q_Pauli ∝ h⁻³ ∝ g⁻³ DÉCROÎT quand h croît — la trame REFERME les cellules : un gaz dégénéré à densité fixe voit")
print("  p_F = ħ(3π²n)^{1/3} ∝ g CROÎTRE : la croissance de h POMPE de l'énergie cinétique (E_F ∝ g²) — elle dilate, elle ne compacte pas.")
print("  objets dégénérés (perles) : R ∝ ħ²/(G m^{8/3} M^{1/3}) ∝ g²/g = g → nées compactes, elles GONFLENT avec h ; M_Ch ∝ (ħc/G)^{3/2}/m² = M_Pl³/m² INVARIANTE.")
print("  défaut F (non dégénéré) : décrochage exige δ_DM ≳ ρ_rad/ρ_DM — G et h s'y simplifient : INCHANGÉ (1e3-1e4).")
print("  perle primordiale au gel : v_éch = √(2GM/R) ∝ √g → condition v_naissance < 1,6e-4 c·√g : PLUS DURE tôt.\n")

print("=== Ce qui, dans le cadre, ressemble à « l'énergie de la trame plus petite au début » ===")
print("  G = ħc/M_Pl² naît en UNE marche à t_c = 1,6e-4 s (pas de puits avant) ; ℓ_P = ħ/(M_Pl c) est fixé par M_Pl (rigidité, préexiste).")
print("  Une rampe G(t) = G₀·g(t) entre t_c et la BBN (g → 1 avant 1 s) est la seule fenêtre où rien ne la contraint —")
print("  et dans cette fenêtre elle ne change ni δ requis (G se simplifie), ni Q (unités), ni l'issue : elle DURCIT la perle légère (√g).")
