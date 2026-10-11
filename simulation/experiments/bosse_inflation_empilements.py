"""(A) Une bosse du spectre inflationnaire aux petites échelles ? (B) L'empilement avec 4₁, 5₁… (16/09/2026)

(A) Cible : pics de matière noire δ ~ 1e3–3e4 à M = 1e5–1e6 Msun, en place avant T ~ 2–12 keV, rares (~1e-9).
    On chiffre : l'échelle k et le nombre d'e-folds ; ce qu'une bosse ADIABATIQUE peut donner (croissance
    logarithmique, borne μ de FIRAS, seuil des trous noirs primordiaux) ; ce que donne la cascade à marches
    du cadre ; ce qu'il faudrait pour une bosse ISOCOURBE.
(B) Même mécanisme que les tores de trèfles avec d'autres nœuds simples : mots de tresse, familles d'empilement,
    énergie et charge par unité, fraction au gel (spectre des rails), érosion par l'unité miroir, arrêt de Vassiliev,
    survie à la reconnexion faible, abondance.
"""
import math
import numpy as np

print("=" * 100)
print("(A) LA BOSSE INFLATIONNAIRE")
print("=" * 100)
rho_m = 2.775e11*0.143                     # Msun/Mpc³ (Ω_m h²)
def k_of_M(M): return (4*math.pi/3*rho_m*math.pi**3/M)**(1/3)
kpiv, Nstar = 0.05, 57
for M in (1e5, 3e5, 1e6):
    k = k_of_M(M); print(f"  M = {M:.0e} Msun : k ≈ {k:.0f} Mpc⁻¹, sortie d'horizon à N = N* − ln(k/k*) = {Nstar - math.log(k/kpiv):.1f} e-folds avant la fin")
k = k_of_M(3e5)
keq, Teq = 0.0104, 0.80   # Mpc⁻¹, eV
print()
print("  1. Bosse ADIABATIQUE (ce que produit l'inflation à un seul champ)")
for T_keV in (7.0, 2.0):
    a_ratio = Teq/(T_keV*1e3)                  # a/a_eq
    need = 1/a_ratio                            # δ requis ~ ρ_rad/ρ_DM
    x = 0.44*k/keq*a_ratio*2                    # kτ à T (τ ∝ a en radiation)
    grow = 9.6*math.log(max(x, 1.0001))        # δ_DM/ζ (Hu–Sugiyama, sous-horizon)
    print(f"     T = {T_keV:.0f} keV : δ requis ≈ {need:.0e} ; croissance adiabatique δ_DM ≈ {grow:.1f}·ζ → ζ requis ≈ {need/max(grow,1e-3):.0e}")
print("     seuil de formation de trous noirs primordiaux : ζ ≈ 0,5–1 (δ_rad ≈ 0,45) — franchi 3 à 4 ordres AVANT")
print("     borne μ (FIRAS |μ| < 9e-5) : μ ≈ 2,2 P_ζ Δln k pour 50 < k < 1e4 Mpc⁻¹ → P_ζ ≲ 4e-5, ζ_rms ≲ 6e-3")
print("     → une bosse adiabatique ne peut pas produire le grumeau : elle ferait des trous noirs bien avant, et FIRAS l'exclut déjà.")
print()
print("  2. La cascade à marches du cadre (« potentiel plateau, multitude de petites marches »)")
r = 12/Nstar**2; epsH = r/16
print(f"     r = 12/N*² = {r:.4f} → ε_H = r/16 = {epsH:.1e} ; chute de V par e-fold = 2ε_H = {2*epsH:.1e}")
for c_step in (2*epsH, 1/137):
    amp = c_step/epsH
    print(f"     marche ΔV/V = {c_step:.1e} : oscillation de P(k) d'amplitude relative ~ ΔV/(V ε_H) = {amp:.0f}  →  P_ζ ≲ {2.1e-9*(1+amp):.0e}")
print("     → même une marche de 1/137 (toute la chaleur latente d'un pas Δc = 6/137 d'un coup) ne donne qu'un facteur ~30 :")
print("       il faudrait ×1e6 (inflexion / roulement ultra-lent), que la cascade à marches ne produit pas — et ce serait adiabatique.")
print()
print("  3. Une bosse ISOCOURBE exige un champ spectateur léger pendant l'inflation dont la valeur module la FORMATION")
print("     de la matière noire au gel. Sensibilité du cadre : d ln Ω/d ln T_c ≈ ln(195/1,1)/ln(75/60) =", f"{math.log(195/1.1)/math.log(75/60):.0f}")
print("     → il suffirait d'une modulation locale de T_c de ~ ln(1e4)/23 ≈ 40 % dans les pics. Mais T_c = c_eff σ_χ D₀ :")
print("       σ_χ est lue sur M_Pl (rigidité), k = 135 est entier, c_eff est un nombre du réseau — AUCUN champ du cadre ne les module.")
print("     → le cadre n'a pas de spectateur : pas de bosse isocourbe inflationnaire. La voie est FERMÉE en l'état.")

print()
print("=" * 100)
print("(B) L'EMPILEMENT AVEC D'AUTRES NŒUDS SIMPLES")
print("=" * 100)
ML, slope = 310.0, 70.1
L31 = 32.7448
rope = {"3_1": 32.7448, "4_1": 42.0928, "5_1": 47.2016, "5_2": 49.4704, "6_1": 56.7150, "6_2": 57.0271, "6_3": 57.8435, "7_1": 61.4109}
def Mtopo(K): return ML*rope[K]/L31
frac = {"3_1": 58.7, "4_1": 14.1, "5_2": 5.6, "6_1": 6.2, "6_2": 2.7, "6_3": 1.8}
# famille d'empilement : (mot, croisements ajoutés par unité après reconnexion, amphichiral, famille)
fam = {
 "3_1": ("σ₁³", 2, False, "T(2,q)"),
 "4_1": ("(σ₁σ₂⁻¹)²", 4, True, "Turk's head Th(3,2n) (nœud si 3∤2n, amphichiral, c = 4n)"),
 "5_1": ("σ₁⁵", 4, False, "T(2,q)"),
 "7_1": ("σ₁⁷", 6, False, "T(2,q)"),
 "5_2": ("σ₁³σ₂σ₁⁻¹σ₂", None, False, "puissances de tresse sans famille nommée"),
 "6_3": ("σ₁²σ₂⁻¹σ₁σ₂⁻²", None, True, "puissances de tresse sans famille nommée"),
}
print(f"  {'K':>4} {'mot':>14} {'M libre':>8} {'fraction gel':>12} {'Δc/unité':>8} {'ΔE empiler':>11} {'ΔE dépiler (miroir)':>20} {'charge/unité':>12}  famille")
for K, (w, dc, amph, name) in fam.items():
    M = Mtopo(K) if K != "3_1" else 0.511
    f = frac.get(K, None)
    fs = f"{f:.1f} %" if f is not None else ("(dans 4_1)" if K == "5_1" else "< 1 %")
    if dc:
        Estack = M - dc*slope; Eun = M + dc*slope
        es = f"{Estack:+.0f} MeV"; eu = f"{Eun:+.0f} MeV"
    else:
        es = eu = "—"
    ch = "0" if amph else "≠0 (ΔLk)"
    print(f"  {K:>4} {w:>14} {M:8.1f} {fs:>12} {str(dc) if dc else '—':>8} {es:>11} {eu:>20} {ch:>12}  {name}")
print("  (fraction « 4_1 » = classe de déterminant 5, qui contient aussi 5_1 ; ΔE > 0 = énergie libérée ; trèfle libre = électron)")
print()
print("  Deux différences qui comptent :")
print("   - 4₁ est AMPHICHIRAL : charge nulle par unité, pas de goulot de neutralisation (contrairement à 3₁, 5₁, 7₁) ;")
print("   - l'empilement 4₁ ou 5₁ est EXOTHERMIQUE (+118, +167 MeV) — mais le dépilement par l'unité miroir l'est bien plus (+679, +727).")
print()
# Smoluchowski unité → pile, avec dépilement ; classes n = nombre d'unités
xi = 3.36e-15; c = 2.998e8; alpha = 1/137.036
n_knot = 0.304/xi**3
K0 = math.pi*(0.35e-15)**2*0.03*c
Tc = 62.0; MPl = 1.221e22
H = 1.66*math.sqrt(15.2)*Tc**2/MPl/6.582e-22
g_s = 15.2; s = (2*math.pi**2/45)*g_s*(Tc/197.327)**3*1e45
Y_DM = 2.3e-27/(1.8e3*1.783e-30)/2.9e9
def run(K, dc, f_unit, a_f, nmax=12):
    c0 = int(K.split("_")[0])
    cof = lambda j: c0 + dc*(j-1)
    """Piles de n unités (c = dc·n) ; même unité + pile → n+1 (a_f) ; miroir + pile → n−1 (1) ; unité+miroir → annihilation."""
    Y = np.zeros(nmax+1); Y[1] = f_unit*n_knot/s/2          # par orientation (moitié « unité », moitié « miroir » à l'insertion)
    t, tend = 1.6e-4, 1.6e-3
    while t < tend:
        a = (t/1.6e-4)**0.5; n = Y*s/a**3; dt = t*0.002
        dY = np.zeros_like(Y)
        n1 = n[1]
        for j in range(1, nmax+1):
            cc = cof(j)
            sup = alpha**max(0, cc+dc-18)
            if j < nmax:
                r = a_f*K0*sup*n1*n[j]/s*(0.5 if j == 1 else 1.0)
                dY[j] -= r; dY[1] -= r; dY[j+1] += r
            if j >= 2:
                re = K0*alpha**max(0, cc-18)*n1*n[j]/s
                dY[j] -= re; dY[1] -= re; dY[j-1] += re
        ra = K0*n1*n1/s                                  # unité + miroir → annihilation (pas de suppression)
        dY[1] -= ra
        m = np.max(np.abs(dY)*dt/np.maximum(Y, 1e-40))
        if m > 0.2: dt *= 0.2/m
        Y = np.maximum(Y+dY*dt, 0); t += dt
    out = {cof(j): Y[j] for j in range(1, nmax+1) if Y[j] > 0}
    Mdark = sum(Y[j]*2*(Mtopo(K) + slope*(cof(j)-c0)) for j in range(1, nmax+1) if cof(j) >= 20)
    return Mdark/(Y_DM*1800), out
for K, dc, fu in (("4_1", 4, 0.141*0.5), ("5_1", 4, 0.141*0.5)):
    Mu = Mtopo(K)
    for lab, af in (("capture ∝ E³", (Mu-dc*slope)**3/(Mu+dc*slope)**3), ("capture ∝ E", (Mu-dc*slope)/(Mu+dc*slope)), ("a_f = 1", 1.0)):
        om, dist = run(K, dc, fu, af)
        top = sorted(dist.items(), key=lambda x: -x[1])[:4]
        print(f"  {K} {lab:<14} a_f = {af:.3f} : masse sombre (c ≥ 20) / Planck = {om:.1e} ; classes peuplées : " + ", ".join(f"c={cc}:{y:.0e}" for cc, y in top))
c_stop = 18 + math.log(H/(0.07*n_knot*K0))/math.log(alpha)
print(f"\n  arrêt de Vassiliev de l'empilement (n_unité K₀ α^(c−18) = H) : c_stop ≈ {c_stop:.1f} ; piles 4₁ : c = 20 (Th(3,10), nœud), 24 (Th(3,12) : ENTRELACS, instable), 28 ; piles 5₁ : c = 21, 25, 29 (T(2,q))")
print("  survie à la reconnexion faible (c_min = 20) : les piles c = 20 et 28 survivent si elles sont formées.")
print()
print("  Lien avec la dette : l'empilement est LOCAL (portée ξ = 3 fm, horizon du gel 2e-7 Msun) ; il fixe une abondance moyenne,")
print("  il ne crée aucune surdensité cohérente sur 1e5 Msun (1,5e12 horizons du gel) — même conclusion que pour les trèfles.")
