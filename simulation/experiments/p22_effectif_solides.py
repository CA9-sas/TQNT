"""P22 effectif dans les solides — vérification du dictionnaire (T1).

THÉORÈME (algèbre, vérifié ici) : l'exposant de Landau-Zener interbande
    P_LZ = exp[ - pi E_g^2 / (4 hbar v* e E) ]
se réduit EXACTEMENT à l'exposant de Schwinger sous la substitution
(E_g -> 2 m c^2, v* -> c), et la forme canonique P22
    Gamma = Gamma_P exp(-2 pi / xi),   xi = 2 a lambda_C / c^2
se transpose VERBATIM avec (m -> m*, c -> v*) pourvu que E_g = 2 m* v*^2 :
    xi_eff = 2 (eE/m*) (hbar/m* v*) / v*^2 = 8 hbar v* eE / E_g^2.

Conséquence : le champ critique du vide effectif est
    E_c^eff = m*^2 v*^3 / (e hbar)  =  E_c^Schwinger x (m*/m_e)^2 (v*/c)^3.

Table numérique : vides effectifs usuels (Si, Ge, GaAs, pérovskites,
MoS2, graphène) — seuil de paire E_g, vitesse v* = sqrt(E_g/2m*),
champ critique effectif, et le facteur d'abaissement vs le vide
fondamental. Les m* sont des valeurs littérature indicatives (déclaré).
"""
import numpy as np

HBAR = 1.054571817e-34      # J s
E_CH = 1.602176634e-19      # C
ME = 9.1093837015e-31       # kg
C = 2.99792458e8            # m/s

print("=" * 76)
print("P22 EFFECTIF DANS LES SOLIDES — T1 : LE DICTIONNAIRE   (2026-08-31)")
print("=" * 76)

# ---- 1. Réduction exacte LZ -> Schwinger ---------------------------------
E_field = 1.0e18  # V/m, arbitraire pour le test
Eg_fond = 2 * ME * C**2                      # J
expo_LZ = np.pi * Eg_fond**2 / (4 * HBAR * C * E_CH * E_field)
expo_Schw = np.pi * ME**2 * C**3 / (E_CH * E_field * HBAR)
print(f"\n1) LZ(E_g=2mc^2, v*=c) vs Schwinger : "
      f"{expo_LZ:.6e} vs {expo_Schw:.6e}  "
      f"(écart {abs(expo_LZ/expo_Schw-1):.1e})")

# xi canonique (V6 : xi = 2 a lambda_C / c^2) vs xi_eff = 8 hbar v eE / Eg^2
a = E_CH * E_field / ME
lamC = HBAR / (ME * C)
xi_canon = 2 * a * lamC / C**2
xi_eff = 8 * HBAR * C * E_CH * E_field / Eg_fond**2
print(f"   xi canonique vs xi_eff au fondamental : "
      f"{xi_canon:.6e} vs {xi_eff:.6e}  (écart {abs(xi_canon/xi_eff-1):.1e})")

E_c_fond = ME**2 * C**3 / (E_CH * HBAR)
print(f"   E_c (Schwinger) = {E_c_fond:.3e} V/m")

# ---- 2. Table des vides effectifs ---------------------------------------
# (E_g en eV, m*/m_e réduit interbande — valeurs littérature indicatives)
mats = [
    ("vide fondamental", 2 * 511e3, 1.0),
    ("Si",               1.12,      0.19),   # m_r ~ mc*mv/(mc+mv)
    ("Ge",               0.66,      0.12),
    ("GaAs",             1.42,      0.059),
    ("CsPbI3 (perov.)",  1.73,      0.11),
    ("CsPbBr3 (perov.)", 2.00,      0.13),
    ("MoS2 (monoc.)",    1.80,      0.25),
    ("graphene",         0.0,       0.0),
]
print(f"\n2) {'materiau':18s} {'E_g':>10s} {'v* (m/s)':>10s} "
      f"{'E_c^eff (V/m)':>14s} {'abaissement':>12s}")
for nom, Eg_eV, mr in mats:
    if Eg_eV == 0:
        print(f"   {nom:18s} {'0':>10s} {'v_F=1e6':>10s} {'0 (sans seuil)':>14s}"
              f" {'infini':>12s}")
        continue
    Eg = Eg_eV * E_CH
    ms = mr * ME
    v = np.sqrt(Eg / (2 * ms))
    Ec = ms**2 * v**3 / (E_CH * HBAR)
    print(f"   {nom:18s} {Eg_eV:>8.2f}eV {v:>10.2e} {Ec:>14.2e} "
          f"{E_c_fond/Ec:>11.1e}x")

# ---- 3. Champ pour xi_eff = 1 au silicium vs claquage mesuré -------------
Eg = 1.12 * E_CH; ms = 0.19 * ME; v = np.sqrt(Eg / (2 * ms))
E_xi1 = Eg**2 / (8 * HBAR * v * E_CH)
print(f"\n3) Si : champ pour xi_eff = 1 : {E_xi1:.2e} V/m "
      f"(claquage Zener mesuré ~1e8-1e9 V/m : ORDRE COHERENT)")
print("   graphene : Schwinger-analogue OBSERVE (courants critiques hors "
      "equilibre,\n   Berdyugin+ 2022 ; courants pilotes par le champ "
      "optique, Higuchi+ 2017).")

# ---- 4. T6 : le curseur de longueur d'onde (Keldysh <-> xi) --------------
# Brezin-Itzykson (1970) : paires dans un champ alternatif omega ;
# parametre d'adiabaticite gamma_K = m c omega / (e E).
# Avec xi = 2 e E hbar / (m^2 c^3) :  gamma_K = 2 hbar omega / (xi m c^2).
# gamma_K << 1 : tunnel (Schwinger, longue longueur d'onde) ;
# gamma_K >> 1 : multiphoton (n = 2mc^2/hbar omega quanta cooperent).
print("\n4) T6 — curseur Keldysh/xi : gamma_K = 2 hbar omega/(xi mc^2)")
for nom, hw_eV, E_f in [("laser optique @ 1e23 W/cm2", 1.5, 8.7e14),
                        ("laser optique @ E_c", 1.5, 1.32e18),
                        ("gamma 1 MeV @ 1e15 V/m", 1.0e6, 1.0e15)]:
    xi = 2 * E_CH * E_f * HBAR / (ME**2 * C**3)
    gam = 2 * (hw_eV * E_CH) / (xi * ME * C**2)
    n_ph = 2 * ME * C**2 / (hw_eV * E_CH)
    reg = "tunnel (onde longue)" if gam < 1 else "multiphoton"
    print(f"   {nom:28s} xi={xi:.2e} gamma_K={gam:.2e} -> {reg}"
          f" (n_seuil={n_ph:.1e} photons)")
# verification de l'identite : gamma(BI) = m c omega/(eE) vs 2hw/(xi mc2)
om = 1.5 * E_CH / HBAR; E_f = 8.7e14
g1 = ME * C * om / (E_CH * E_f)
g2 = 2 * HBAR * om / ((2 * E_CH * E_f * HBAR / (ME**2 * C**3)) * ME * C**2)
print(f"   identite Brezin-Itzykson vs forme xi : {g1:.6e} vs {g2:.6e} "
      f"(ecart {abs(g1/g2-1):.1e})")
# transpose solide : gamma_eff ~ 2 hbar omega/(xi_eff E_g) (convention O(1))
Eg = 1.12 * E_CH; ms = 0.19 * ME; v = np.sqrt(Eg / (2 * ms))
E_f = 1e8  # V/m, champ de jonction
xi_e = 8 * HBAR * v * E_CH * E_f / Eg**2
hw = 0.5 * E_CH  # photon THz-IR 0.5 eV sous le gap
print(f"   Si (E=1e8 V/m, hw=0,5 eV) : xi_eff={xi_e:.2f}, "
      f"gamma_eff={2*hw/(xi_e*Eg):.2f} -> regime intermediaire accessible "
      f"(Franz-Keldysh)")

# ---- 5. T7 : Casimir dynamique = P22 periodique ; seuil fermionique ------
# DCE photonique (Wilson+ 2011) : miroir module a Omega -> paires de
# photons a Omega/2 (champ SANS GAP : toute frequence resonne).
# Fermions : gap 2mc^2 -> theoreme adiabatique : taux EXACTEMENT nul si
# hbar Omega < 2mc^2 (l'extension periodique du "jamais spontane").
h_pl = 6.62607015e-34
f_seuil = 2 * ME * C**2 / h_pl
f_dce = 10.7e9                       # drive DCE record (SQUID)
print("\n5) T7 — Casimir dynamique et le seuil fermionique")
print(f"   seuil electrons : f = 2mc^2/h = {f_seuil:.2e} Hz")
print(f"   drive DCE record (SQUID) ~ {f_dce:.1e} Hz -> deficit "
      f"x{f_seuil/f_dce:.1e}")
# route miroir plasma relativiste : upshift 4 gamma^2 depuis l'optique
f_opt = C / 800e-9
gam_req = np.sqrt(f_seuil / (4 * f_opt))
a0 = gam_req                          # gamma d'oscillation ~ a0
I_req = 1.37e18 * (a0 / 0.8)**2       # W/cm^2 (lambda = 0,8 um)
print(f"   miroir plasma relativiste : upshift 4gamma^2 depuis 800 nm -> "
      f"gamma requis ~ {gam_req:.0f}")
print(f"   soit a0 ~ {a0:.0f} -> I ~ {I_req:.1e} W/cm^2 "
      f"(records actuels ~1e23 : LA frontiere)")
# vide effectif : le meme DCE fermionique est accessible
for nom, Eg_eV in [("Si", 1.12), ("graphene", 0.0)]:
    if Eg_eV:
        print(f"   vide effectif {nom} : seuil {Eg_eV*E_CH/h_pl:.1e} Hz "
              f"(IR proche) -> trivialement atteint")
    else:
        print(f"   vide effectif {nom} : SANS GAP -> tout Omega resonne "
              f"(paires e-t THz observees : le DCE fermionique du vide "
              f"construit)")

# ---- 6. T4 : l'exciton = positronium du vide effectif --------------------
# Dictionnaire complete d'un 3e nombre renormalise : alpha* = (alpha/eps_r)
# x (c/v*) (l'ecrantage renormalise le couplage que le vide fondamental
# verrouille a k=135). THEOREME (algebre) : sous la convention uniforme
# seuil = 2 mu v^2, positronium ET exciton donnent
#     E_liaison / (2 mu v^2) = alpha_eff^2 / 4     (exactement).
# Verification donnees : E_X = Ry x (mu/m_e)/eps_r^2 vs mesures.
ALPHA = 1 / 137.035999
RY = 13.605693  # eV
print("\n6) T4 — exciton = positronium du vide effectif")
mu_ps = 0.5
print(f"   controle positronium : E_B = alpha^2 mu c^2/2 = "
      f"{ALPHA**2 * mu_ps * 511e3 / 2:.2f} eV (litt. 6,80) ; "
      f"E_B/(2 mu c^2) = {ALPHA**2/4:.3e} = alpha^2/4 exact")
mats = [  # (nom, mu/m_e, eps_r, E_X mesure meV, E_g eV)
    ("Si",    0.12, 11.7,  14.7, 1.12),
    ("GaAs",  0.058, 12.9,  4.2, 1.42),
    ("GaN",   0.16,  9.5,  26.0, 3.40),
    ("Cu2O",  0.41,  7.5,  98.0, 2.17),
]
print(f"   {'mat.':6s} {'E_X pred (meV)':>14s} {'mesure':>8s} {'ecart':>7s}"
      f" {'alpha*':>7s}")
for nom, mu, eps, ex_mes, Eg in mats:
    ex = RY * 1e3 * mu / eps**2
    vstar = C * np.sqrt(Eg / (2 * mu * 511e3))
    astar = (ALPHA / eps) * (C / vstar)
    print(f"   {nom:6s} {ex:14.1f} {ex_mes:8.1f} "
          f"{100*(ex/ex_mes-1):+6.0f}% {astar:7.3f}")
print("""   Cu2O : serie hydrogenoide observee jusqu'a n=25 (Nature 2014) —
   l'« atome » du vide effectif a son spectre de Rydberg complet.
   graphene : alpha* = (e^2/4pi eps0 hbar v_F)/eps ~ 2,2/eps — le vide
   effectif SANS gap est en couplage FORT (la ou le fondamental est
   verrouille a 1/137) : l'ecrantage seul le sauve de l'instabilite
   excitonique (transition isolante non observee : coherent).""")

# ---- 7. T5 : structure de bandes des rails -------------------------------
# (i) secteur ONDES : si les ondes libres diffractaient sur le reseau de
#     rails (pas a_r), bande de Bragg a E ~ pi hbar c/a_r -> teste par
#     l'astronomie photonique (LHAASO PeV) et les nuls GRB.
# (ii) secteur NOEUDS : la structure de bandes EXISTE et est canonique —
#     minima de Diao/Scharein {24,30,34,36,40} -> ratios de masse entiers
#     (V6 sec. "S depuis les rails", 23/07).
# (iii) gel = claquage THERMIQUE : forme de l'exposant de formation vs la
#     lecture Rindler de P22 (4pi/xi = mc^2/k_B T).
HBARC_MEVFM = 197.32698
L_TREFLE = 0.3537      # fm (circuit du trefle, m_l(nu)/sigma_chi)
a_rail = L_TREFLE / 24 # pas de rail (24 pas de Diao)
E_bragg = np.pi * HBARC_MEVFM / a_rail / 1e3  # GeV
print("\n7) T5 — structure de bandes des rails")
print(f"   pas de rail a_r = L(3_1)/24 = {a_rail*1e3:.1f} am "
      f"(= {a_rail/0.0214:.2f} D_tube : coherent)")
print(f"   (i) ONDES : bande de Bragg hypothetique a pi.hbar.c/a_r = "
      f"{E_bragg:.0f} GeV")
print("       -> photons observes propres jusqu'au PeV (LHAASO, x3e4 "
      "au-dessus)\n       + nuls GRB E_QG > 9,3 E_P : le reseau est "
      "INVISIBLE au secteur libre\n       -> la linearite de \\Box chi=0 "
      "n'est pas qu'un postulat, elle est FORCEE par\n       l'astronomie "
      "photonique (toute bande de Bragg rendrait le ciel opaque a "
      f"{E_bragg:.0f} GeV).")
print("   (ii) NOEUDS : les 'bandes' existent et sont deja canoniques —")
for nom, n_min in [("3_1", 24), ("4_1", 30), ("5_1", 34), ("5_2", 36),
                   ("6_x", 40)]:
    from fractions import Fraction
    f = Fraction(n_min, 24)
    print(f"       N_min({nom}) = {n_min:2d} -> ratio de masse "
          f"{f.numerator}/{f.denominator}")
print("       (m_s/m_l = 36/24 = 3/2 : la brique stoikhion arithmetique, "
      "V6 §48)")
# (iii) gel : exposant de formation vs forme Rindler de P22
T_c, m_bord = 62.0, 1800.0  # MeV
print(f"   (iii) GEL : e^(-m/T_c) au bord c=23 : m/T = {m_bord/T_c:.1f} "
      f"-> e^-{m_bord/T_c:.0f} = {np.exp(-m_bord/T_c):.1e}")
print(f"       forme P22 Rindler : 4pi/xi = mc^2/k_B T -> xi_gel = "
      f"4pi T_c/m = {4*np.pi*T_c/m_bord:.2f}")
print("       -> le facteur de formation A la forme du 3e regime de P22 "
      "(thermique) ;\n       identite FORMELLE consignee (T_c reelle, pas "
      "Unruh — aucun taux modifie).")
