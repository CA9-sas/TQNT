"""Taux de reconnexion tore neutre – lepton libre, depuis les primitives du cadre (16/09/2026).

Ingrédients canoniques (aucun nombre nouveau) :
  - P15 quantitative : une reconnexion = échange de brins au croisement, amplitude de contact
    1/E_relax² avec E_relax = v, g_topo = 1  →  constante de contact G_F (dérivée, v = √(2π)α⁸M_Pl(1+α/4π)).
  - L'érosion e⁺ + T(2,c) → T(2,c−2) est EXOTHERMIQUE (Q = 2·70,1 + m_e ≈ 140,7 MeV) ; l'énergie doit partir
    en un quantum : le photon (quantum de jauge de χ), prix α — le SOMMET RADIATIF à 3 pattes du cadre
    (cendres du gel : Γ = α G_F² M² Q/8π). Pour une capture 2→2 de contact : σv ≈ α G_F² Q²/π.
  - Conservation de la charge : e⁺ + T⁰ → T'⁺ + γ, puis neutralisation ΔLk en 3–40 ms (T-11).
  - Deux lectures de la complexité de l'hôte :
      R1 : aucune (règle du VEV : « tout comptage de croisements de l'hôte est réfuté ») ;
      R2 : suppression de Vassiliev α^(c−18) pour tout événement topologique sur un nœud c > 18
           (règle Bar-Natan v_{c_i+c_j−2}, suppression de création, normalisation σ(c) du chapitre DM).
"""
import math
alpha = 1/137.036; hbarc = 197.327; hbar = 6.582e-22
v = 246220.0; GF = 1/(math.sqrt(2)*v**2)            # MeV⁻²
MPl = 1.221e22
Q = 2*70.1 + 0.511
print(f"G_F = 1/(√2 v²) = {GF:.4e} MeV⁻² ; Q(érosion) = {Q:.1f} MeV")
sv = alpha*GF**2*Q**2/math.pi                        # MeV⁻² (c = 1)
sv_cm3s = sv*(hbarc*1e-13)**2*3e10
print(f"σv(e⁺ + T → T' + γ) = α G_F² Q²/π = {sv:.2e} MeV⁻² = {sv*(hbarc*1e-13)**2:.1e} cm² × c = {sv_cm3s:.1e} cm³/s")
print(f"  sites de contact le long du tube : jusqu'à c ≈ 23 (facteur multiplicatif, ne change pas l'ordre)\n")

def n_epm(T): return 0.365*T**3                      # MeV³, e± relativistes
def gstar(T): return 15.2 if T > 30 else 10.75
def H(T): return 1.66*math.sqrt(gstar(T))*T**2/MPl  # MeV
Tc = 62.0
GH = n_epm(Tc)*sv/H(Tc)
print("=== Γ/H à T_c = 62 MeV (Γ/H ∝ T tant que les paires sont relativistes ; nombre d'événements intégré ≈ Γ/H(T_c)) ===")
print(f"  R1 (pas de comptage de l'hôte)  : Γ/H = {GH:.0f}   (×23 sites : {23*GH:.0e})")
for c in (19, 21, 23, 25, 27):
    print(f"  R2 α^(c−18), c = {c:2d}           : Γ/H = {GH*alpha**(c-18):.1e}   (×23 sites : {23*GH*alpha**(c-18):.1e})")
print()
print("=== Comparaison aux normalisations du T-10 (c = 23) ===")
for lab, r in (("A cadre, baryons, σ_EM α^{c−18}", 0.067), ("B σ(c), bain e±", 2.51e8), ("C contact géométrique", 3.34e5), ("D T-8", 2.18e5)):
    print(f"  {lab:<32} Γ/H = {r:.1e}   ;  R2/norme = {GH*alpha**5/r:.0e}")
print()
print("=== Conséquences ===")
print(f"  R1 : {GH:.0f} événements d'érosion par tore quel que soit c ; chaque événement retire 2 croisements et le suivant est aussi rapide")
print("       → tout le réservoir descend sous c = 19 et se dissout en leptons avant 1 MeV : Ω_DM → 0.")
print("       La lecture R1 est EXCLUE par l'existence même de la matière noire dans le cadre : il faut une protection de l'hôte.")
c_edge = 18 + math.log(GH)/math.log(1/alpha)
print(f"  R2 : bord d'érosion Γ/H = 1 à c = {c_edge:.1f} ; à c = 23, Γ/H = {GH*alpha**5:.0e} → aucune érosion du réservoir")
print("       → la reconnexion tore–lepton est FAIBLE (G_F², α), pas électromagnétique (σ_EM) ni géométrique :")
print(f"         les normalisations B, C, D du T-10 la surestimaient de 13 à 17 ordres ; même A (baryons, σ_EM) de {0.067/(GH*alpha**5):.0e}.")
print("       → c_min = 23 et Ω = 0,6 ne sont PAS menacés par le bain e± ; la tension du T-10 se dissout sous R2.")
