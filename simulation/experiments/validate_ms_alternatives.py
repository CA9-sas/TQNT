r"""
Investigation approfondie de m_s en TQNT.

La validation P23' Piste 2 donne m_eff(s) ≈ 231 MeV alors que :
  - m_s MS-bar @ μ=2 GeV  = 93 MeV
  - m_s MS-bar @ μ=1 GeV  ≈ 175 MeV
  - m_s constituant       ≈ 540 MeV

Le ratio TQNT/MS-bar ≈ 2.5 mérite investigation. Tests systématiques :

1. Échelle de référence (running mass) — m_s vs μ
2. Différentes assignations pour Λ et K
3. Baselines alternatives (neutron, mésons K-π)
4. Définitions alternatives de m_eff (Hölder, géométrique, normalisée)
5. Décomposition isospin (Σ⁺ uus vs Σ⁻ dds donne m_s seul)
6. Statistique sur tous les hadrons étranges du catalogue
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


# ─── Constantes V5 ───
M_E = 0.51099895
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3
A_F = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3",
         "7_1","7_2","7_3","7_4","7_5","7_6","7_7",
         "8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}

print("Précalcul Frenet+Jones…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"]*2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj),
               "U": U_KT[K][0]*U_KT[K][1]}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega_ts(cfg):
    n = len(cfg)
    ws = [M_NAIVE[K] * (TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in cfg]
    skel = float(sum(w**A_F for w in ws)**(1/A_F))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in cfg) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(cfg); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2:
            Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ─── Hadrons et leur saveur (compte de quarks) ───
HADRONS_STR = {
    # Baryons légers (uud, udd)
    "p":     {"cfg": ["3_1","7_2","7_2"], "m_PDG": 938.27,  "q": (2,1,0,0,0,0)},  # uud
    "n":     {"cfg": ["3_1","7_2","7_2"], "m_PDG": 939.57,  "q": (1,2,0,0,0,0)},  # udd
    # Mésons légers
    "π+":    {"cfg": ["3_1","6_1"],       "m_PDG": 139.57,  "q": (1,1,0,0,0,0)},  # ud-bar
    "K+":    {"cfg": ["4_1","4_1","4_1"], "m_PDG": 493.68,  "q": (1,0,1,0,0,0)},  # us-bar
    "K0":    {"cfg": ["6_1","6_1","6_1"], "m_PDG": 497.61,  "q": (0,1,1,0,0,0)},  # ds-bar
    "K*+":   {"cfg": ["6_1","6_2"],       "m_PDG": 891.66,  "q": (1,0,1,0,0,0)},  # us-bar (vector)
    # Baryons étranges
    "Λ":     {"cfg": ["5_1","8_1"],       "m_PDG": 1115.68, "q": (1,1,1,0,0,0)},  # uds
    "Σ+":    {"cfg": ["4_1","4_1","5_1"], "m_PDG": 1189.37, "q": (2,0,1,0,0,0)},  # uus
    "Σ0":    {"cfg": ["4_1","4_1","5_1"], "m_PDG": 1192.64, "q": (1,1,1,0,0,0)},  # uds
    "Σ-":    {"cfg": ["4_1","4_1","5_1"], "m_PDG": 1197.45, "q": (0,2,1,0,0,0)},  # dds
    "Ξ0":    {"cfg": ["4_1","5_1","5_2"], "m_PDG": 1314.86, "q": (1,0,2,0,0,0)},  # uss
    "Ξ-":    {"cfg": ["4_1","5_1","5_2"], "m_PDG": 1321.71, "q": (0,1,2,0,0,0)},  # dss
    "Ω-":    {"cfg": ["7_2","7_3"],       "m_PDG": 1672.45, "q": (0,0,3,0,0,0)},  # sss
    # Baryons décuplet
    "Σ*+":   {"cfg": ["6_1","7_3"],       "m_PDG": 1382.83, "q": (2,0,1,0,0,0)},
    "Ξ*0":   {"cfg": ["6_1","6_3"],       "m_PDG": 1531.80, "q": (1,0,2,0,0,0)},
    # Charme
    "D+":    {"cfg": ["3_1","5_1","7_2"], "m_PDG": 1869.66, "q": (0,1,0,1,0,0)},  # cd-bar
    "D0":    {"cfg": ["3_1","5_1","7_2"], "m_PDG": 1864.84, "q": (1,0,0,1,0,0)},  # cu-bar
    "D_s+":  {"cfg": ["3_1","6_1","7_1"], "m_PDG": 1968.34, "q": (0,0,1,1,0,0)},  # cs-bar
    "Λ_c":   {"cfg": ["3_1","3_1","9_1"], "m_PDG": 2286.46, "q": (1,1,0,1,0,0)},
    "Σ_c++": {"cfg": ["4_1","5_1","7_2"], "m_PDG": 2454.0,  "q": (2,0,0,1,0,0)},
    "Ω_c":   {"cfg": ["4_1","4_1","7_7"], "m_PDG": 2695.20, "q": (0,0,2,1,0,0)},
    # Beauté
    "B_s0":  {"cfg": ["6_3","7_6"],       "m_PDG": 5366.92, "q": (0,0,1,0,1,0)},
    "B_c+":  {"cfg": ["4_1","7_7","9_1"], "m_PDG": 6274.9,  "q": (0,0,0,1,1,0)},
    "Λ_b":   {"cfg": ["6_3","7_4"],       "m_PDG": 5619.60, "q": (1,1,0,0,1,0)},
    "Ω_b":   {"cfg": ["7_7","8_2"],       "m_PDG": 6046.0,  "q": (0,0,2,0,1,0)},
}

# q-tuple : (n_u, n_d, n_s, n_c, n_b, n_t)
Q_INDEX = {"u":0, "d":1, "s":2, "c":3, "b":4, "t":5}


def m_TQNT(name): return omega_ts(HADRONS_STR[name]["cfg"])


# ════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" TEST A — m_s par différentielle baryon : 6 paires")
print("═"*78)

# Paire (H_q, H_ref) où H_q a un s de plus que H_ref
# Δm = m(H_q) - m(H_ref) ≈ m_s
pairs_baryon = [
    ("Λ", "n",  1),  # uds vs udd  (+1 s, -1 d)
    ("Σ+", "p", 1),  # uus vs uud
    ("Σ-", "n", 1),  # dds vs udd  (mais aussi -1 u, +1 d)
    ("Ξ0", "Λ", 1),  # uss vs uds
    ("Ω-", "Ξ-", 1), # sss vs dss
]
print(f"\n{'paire':<14} {'TQNT':>9} {'PDG':>9}  signature")
print("─"*78)
m_eff_pairs = []
for hq, hr, ns in pairs_baryon:
    dm_t = m_TQNT(hq) - m_TQNT(hr)
    dm_p = HADRONS_STR[hq]["m_PDG"] - HADRONS_STR[hr]["m_PDG"]
    q_q = HADRONS_STR[hq]["q"]; q_r = HADRONS_STR[hr]["q"]
    signature = " ".join(f"{c}{n:+d}" for c, n in zip("udscbt", np.array(q_q)-np.array(q_r)) if n != 0)
    print(f"{hq+'-'+hr:<14} {dm_t:>9.2f} {dm_p:>9.2f}  ({signature})")
    m_eff_pairs.append((hq+'-'+hr, dm_t, dm_p, signature))


# ════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" TEST B — m_s par différentielle MÉSON : K-π")
print("═"*78)

# K+ = us-bar, π+ = ud-bar : Δm ≈ m_s - m_d
mK = m_TQNT("K+"); mp_PDG_K = HADRONS_STR["K+"]["m_PDG"]
mPi = m_TQNT("π+"); mp_PDG_pi = HADRONS_STR["π+"]["m_PDG"]

print(f"\n  K+   ({' + '.join(HADRONS_STR['K+']['cfg'])}) : TQNT {mK:.2f}, PDG {mp_PDG_K:.2f}")
print(f"  π+   ({' + '.join(HADRONS_STR['π+']['cfg'])}) : TQNT {mPi:.2f}, PDG {mp_PDG_pi:.2f}")
print(f"  ΔTQNT = m(K) - m(π) = {mK-mPi:.2f} MeV")
print(f"  ΔPDG  = m(K) - m(π) = {mp_PDG_K-mp_PDG_pi:.2f} MeV")
print(f"  (théorie : ΔPDG ≈ m_s - m_d à basse énergie)")
print(f"  PDG m_s - m_d (MS-bar @ 2 GeV) = 93 - 4.67 = 88.7 MeV")

# K*-π : variante vecteur
m_Kstar = m_TQNT("K*+"); m_Kstar_PDG = HADRONS_STR["K*+"]["m_PDG"]
# Pour π vecteur on aurait ρ ; on regarde juste K*
print(f"\n  K*+  : TQNT {m_Kstar:.2f}, PDG {m_Kstar_PDG:.2f}")
print(f"  K* - K = différence ground/excité, NON pas une m_s indep")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" TEST C — m_s aux échelles de renormalisation (running)")
print("═"*78)

# m_s(μ) en MS-bar (α_s loop running, PDG values)
# Approximation : m_s(μ) = m_s(2 GeV) · (α_s(μ)/α_s(2 GeV))^(γ_0/β_0)
# γ_0 = 8 ; β_0 = 11 - 2 nf/3 ; nf = 4 pour Λ_QCD ~ 200 MeV
# Approche simplifiée : valeurs PDG directes
m_s_running = {
    "0.5 GeV": 400.0,   # extrapolation basse énergie
    "1.0 GeV": 175.0,   # PDG basse énergie
    "1.5 GeV": 130.0,
    "2.0 GeV":  93.4,   # référence PDG
    "3.0 GeV":  75.0,
    "4.2 GeV":  67.0,   # @ m_b
}
m_eff_avg = np.mean([dm_t for _, dm_t, _, _ in m_eff_pairs])
print(f"\n  m_eff TQNT (moyenne baryons) : {m_eff_avg:.2f} MeV")
print()
print(f"  {'Échelle μ':<12} {'m_s(μ)':>8} {'ratio TQNT/m_s':>15}")
print("─"*40)
for mu, ms in m_s_running.items():
    print(f"  {mu:<12} {ms:>8.2f}  {m_eff_avg/ms:>15.2f}")

# À quelle échelle TQNT = MS-bar ?
mus = sorted(m_s_running.items(), key=lambda x: float(x[0].split()[0]))
mu_vals = np.array([float(m[0].split()[0]) for m in mus])
ms_vals = np.array([m[1] for m in mus])
# Interpolation : trouver μ tel que m_s(μ) = m_eff_avg
from scipy.interpolate import interp1d
f_inv = interp1d(ms_vals[::-1], mu_vals[::-1], bounds_error=False, kind="linear")
mu_match = f_inv(m_eff_avg)
print(f"\n  TQNT m_eff(s) = m_s MS-bar à μ ≈ {mu_match:.2f} GeV")
print("  (échelle où la TQNT capture naturellement la masse étrange)")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" TEST D — Décomposition isospin pure (Σ⁺ vs Σ⁻)")
print("═"*78)

# Σ⁺ uus et Σ⁻ dds : seule différence est 2u→2d
# Donc (Σ⁻ - Σ⁺) = 2(m_d - m_u) (mais pas m_s)
# Et Σ⁺ - Λ : (uus) - (uds) = 1u - 1d → m_u - m_d
m_Sp = HADRONS_STR["Σ+"]["m_PDG"]; m_Sm = HADRONS_STR["Σ-"]["m_PDG"]
m_L = HADRONS_STR["Λ"]["m_PDG"]
print(f"\n  Σ⁻ - Σ⁺ (PDG)  = {m_Sm - m_Sp:.3f} MeV ≈ 2(m_d - m_u) → m_d - m_u ≈ {(m_Sm-m_Sp)/2:.3f} MeV")
print(f"  PDG m_d - m_u (MS-bar) = {4.67 - 2.16:.3f} MeV")
print(f"\n  Σ⁺ - Λ (PDG)   = {m_Sp - m_L:.3f} MeV  (uus vs uds : ≈ m_u - m_d + correction)")
print(f"  Ω⁻ - Σ⁻ (PDG)  = {HADRONS_STR['Ω-']['m_PDG'] - m_Sm:.3f} MeV ≈ 2(m_s - m_d) → m_s - m_d ≈ {(HADRONS_STR['Ω-']['m_PDG']-m_Sm)/2:.3f} MeV")
print(f"  PDG m_s - m_d (MS-bar) = {93.4 - 4.67:.3f} MeV")
print(f"\n  Test TQNT : Ω⁻ - Σ⁻ via Ω_ts → (m_s - m_d) estimé")
omega_dm = m_TQNT("Ω-") - m_TQNT("Σ-")
print(f"  TQNT Ω⁻ - Σ⁻ = {omega_dm:.3f} MeV → m_s_eff - m_d_eff ≈ {omega_dm/2:.3f} MeV")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" TEST E — Hölder share du nœud étrange dans Λ")
print("═"*78)

# Λ = 5_1 + 8_1. Si 5_1 = "le strange-quark-component", quelle est sa part Hölder ?
cfg_Lambda = HADRONS_STR["Λ"]["cfg"]  # 5_1 + 8_1
ws_L = [M_NAIVE[K] * (TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in cfg_Lambda]
w_a = [w**A_F for w in ws_L]
share_5_1 = w_a[0] / sum(w_a)
share_8_1 = w_a[1] / sum(w_a)
print(f"\n  Λ = 5_1 + 8_1")
print(f"  Ω_ts(Λ) = {m_TQNT('Λ'):.2f} MeV")
print(f"  Part Hölder de 5_1 : {share_5_1*100:.1f}% → m_eff = {share_5_1 * m_TQNT('Λ'):.2f} MeV")
print(f"  Part Hölder de 8_1 : {share_8_1*100:.1f}% → m_eff = {share_8_1 * m_TQNT('Λ'):.2f} MeV")

# Mais le 5_1 dans Λ ne peut pas TOUT être "s" puisque Λ = uds (u+d+s)
# Si on identifie 5_1 = s et 8_1 = ud composite :
print(f"\n  Si 5_1 ≡ s seul : m_eff(s | Λ, Hölder) = {share_5_1 * m_TQNT('Λ'):.2f} MeV")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(" VERDICT — Quelle valeur pour m_s ?")
print("═"*78)
print(f"\n  Méthode                                              m_s (MeV)   ratio/93")
print("─"*72)
print(f"  Différentielle Λ-n (baryon)                          {m_TQNT('Λ')-m_TQNT('n'):>8.2f}   {(m_TQNT('Λ')-m_TQNT('n'))/93.4:>5.2f}")
print(f"  Différentielle Σ+-p (baryon)                         {m_TQNT('Σ+')-m_TQNT('p'):>8.2f}   {(m_TQNT('Σ+')-m_TQNT('p'))/93.4:>5.2f}")
print(f"  Différentielle K+-π+ (méson)                         {mK-mPi:>8.2f}   {(mK-mPi)/93.4:>5.2f}")
print(f"  Demi-différentielle Ω⁻-Σ⁻ (pure 2s)                  {omega_dm/2:>8.2f}   {omega_dm/2/93.4:>5.2f}")
print(f"  Hölder share du 5_1 dans Λ                           {share_5_1*m_TQNT('Λ'):>8.2f}   {share_5_1*m_TQNT('Λ')/93.4:>5.2f}")
print(f"  PDG MS-bar @ 2 GeV (référence)                        {93.4:>8.2f}    1.00")
print(f"  PDG MS-bar @ 1 GeV (basse énergie)                   {175.0:>8.2f}    1.87")
print(f"  Constituant (basse énergie)                          {540.0:>8.2f}    5.78")
