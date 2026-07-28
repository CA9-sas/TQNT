r"""
TEST DÉCISIF — γ(L) sur résidus V6.

Pour isoler la déformation plastique des autres corrections QM
(isospin k_iso, mélange η-η', Coulomb k_EM, spin-orbite α_SO),
on utilise des masses ISOSPIN-MOYENNÉES qui éliminent le bruit
des corrections V6 et révèlent le résidu structurel "pur".

Hypothèse à tester :
  Sur les résidus isospin-moyennés (proxy V6),
  γ(L) suit-il une loi universelle simple ?

Si OUI avec |gain| > 30% : principe plastique validé
Si NON : il manque encore une brique

Particules avec masses isospin-moyennées :
  π̄ = (π⁰ + π±)/2 = 137.27 MeV
  K̄ = (K⁰ + K±)/2 = 495.65 MeV
  N̄ = (p + n)/2 = 938.92 MeV
  Autres : singlets, identiques à leur valeur PDG
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import brentq
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index
from theory.knot_catalog import _CURVE_REGISTRY

M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_FREN = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)),
               float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}

_CACHE = {}
def U_K(K, s_z):
    key = (K, round(s_z, 4))
    if key not in _CACHE:
        pts = _CURVE_REGISTRY[K](15000)
        pts = pts * np.array([1.0, 1.0, s_z])
        dt = 2*np.pi / len(pts)
        r1 = np.gradient(pts, dt, axis=0)
        r2 = np.gradient(r1, dt, axis=0)
        r3 = np.gradient(r2, dt, axis=0)
        speed = np.linalg.norm(r1, axis=1)
        cross = np.cross(r1, r2)
        kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
        triple = np.einsum('ij,ij->i', cross, r3)
        tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
        ds_arr = speed * dt
        _CACHE[key] = float(np.sum(kappa**2 * ds_arr)) * float(np.sum(tau**2 * ds_arr))
    return _CACHE[key]


def omega_def(components, s_list):
    n = len(components); ws=[]; Us=[]
    for K, s in zip(components, s_list):
        Us.append(U_K(K, s))
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── Particules avec masses isospin-moyennées (proxy V6) ─────────
# Pour les multiplets d'isospin, on prend la moyenne des charges.
# Cela élimine les contributions k_EM·Q² + k_iso·I_3 de V6
# et révèle le résidu STRUCTUREL "pur" attribué à la déformation plastique.
PARTICULES = [
    # (nm, cfg, m_iso_avg, 2J, classe)
    ("e⁻",    ["3_1"],                  0.51099895, 1, "lepton"),
    ("μ⁻",    ["3_1","6_1"],            105.6584,   1, "lepton"),
    ("τ⁻",    ["4_1","5_1","6_1"],     1776.86,     1, "lepton"),
    ("π̄",     ["3_1","3_1","5_2"],      137.27,     0, "pseudo"),  # avg π⁰+π±
    ("K̄",     ["6_1","6_1","6_1"],      495.65,     0, "pseudo"),  # avg K⁰+K±
    ("η",     ["3_1","3_1","6_1"],      547.86,     0, "pseudo"),
    ("φ",     ["3_1","5_1","5_2"],     1019.46,     2, "vector"),
    ("J/ψ",   ["3_1","4_1","6_3"],     3096.9,      2, "vector"),
    ("D̄",     ["3_1","5_1","7_2"],     1867.0,      0, "pseudo"),  # avg D⁰+D±
    ("B̄",     ["3_1","6_2","7_7"],     5279.5,      0, "pseudo"),  # B avg
    ("Λ_c",   ["3_1","3_1","9_1"],     2286.46,     1, "baryon"),
    ("N̄",     ["3_1","7_2","7_2"],      938.92,     1, "baryon"),  # avg p+n
]


def sigma_S7(two_J):
    return +1 if two_J == 0 else -1


def gradient_cr(components):
    crs = np.array([TOPO[K]["cr"] for K in components], dtype=float)
    return (crs - crs.mean()) / crs.mean()


def m_pred_with_gamma(components, sigma, gamma, k_anchor=1.0):
    grad = gradient_cr(components)
    s_eq = 1 + (sigma * gamma / k_anchor) * grad
    s_eq = np.clip(s_eq, 0.3, 3.0)
    return omega_def(components, s_eq), s_eq


def find_gamma_required(components, sigma, m_exp, k_anchor=1.0):
    def err(gamma):
        m_p, _ = m_pred_with_gamma(components, sigma, gamma, k_anchor)
        return m_p - m_exp
    try:
        return brentq(err, -3.0, 3.0, xtol=1e-5)
    except ValueError:
        return None


# ───────── Étape 1 — Calculer V5 et résidus sur ces masses iso-moyennées ─────────
print("\n" + "═"*100)
print("  Résidus V5 sur masses isospin-moyennées (≈ V6 sans isospin splitting)")
print("═"*100)

clean_data = []
for nm, cfg, m_exp, two_J, cls in PARTICULES:
    m_v5 = omega_def(cfg, [1.0]*len(cfg))
    d_v5 = (m_v5 - m_exp)/m_exp*100
    sigma = sigma_S7(two_J)
    sign_obs = 0 if abs(d_v5) < 0.1 else int(np.sign(d_v5))
    sign_match = "✓" if sigma == sign_obs else ("·" if sign_obs == 0 else "✗")
    print(f"  {nm:<5} {'+'.join(cfg):<22} {cls:<8} V5={m_v5:>8.2f} "
          f"exp={m_exp:>8.2f} Δ={d_v5:>+6.2f}% σ={sigma:>+2} obs={sign_obs:>+2} {sign_match}")
    clean_data.append({
        "nm": nm, "cfg": cfg, "m_exp": m_exp, "m_v5": m_v5,
        "d_v5": d_v5, "two_J": two_J, "cls": cls, "sigma": sigma,
        "sign_obs": sign_obs,
    })


# ───────── Étape 2 — Extraire γ(L) requis avec masses iso ─────────
print("\n" + "═"*100)
print("  Extraction γ(L) requis sur masses isospin-moyennées")
print("═"*100)

print(f"\n  {'nm':<5} {'config':<22} {'σ':>3} {'γ requis':>10} {'s_attracteur':<28}")
print("  " + "-" * 80)
for d in clean_data:
    if abs(gradient_cr(d["cfg"])).max() < 0.01:
        d["gamma"] = None
        print(f"  {d['nm']:<5} {'+'.join(d['cfg']):<22} {d['sigma']:>+3} "
              f"{'gradient=0':>10}  (config symétrique)")
        continue
    gamma = find_gamma_required(d["cfg"], d["sigma"], d["m_exp"])
    d["gamma"] = gamma
    if gamma is None:
        print(f"  {d['nm']:<5} {'+'.join(d['cfg']):<22} {d['sigma']:>+3} {'no_sol':>10}")
        continue
    _, s_eq = m_pred_with_gamma(d["cfg"], d["sigma"], gamma)
    s_str = ",".join(f"{x:.3f}" for x in s_eq)
    print(f"  {d['nm']:<5} {'+'.join(d['cfg']):<22} {d['sigma']:>+3} "
          f"{gamma:>+10.5f} {s_str:<28}")


# ───────── Étape 3 — Recherche de pattern γ(L) ─────────
print("\n" + "═"*100)
print("  Recherche du pattern γ(L)")
print("═"*100)

# Garder seulement les particules avec gradient non-trivial et γ extrait
usable = [d for d in clean_data
          if abs(gradient_cr(d["cfg"])).max() > 0.01 and d["gamma"] is not None]
print(f"  N = {len(usable)} particules utilisables (gradient non-nul, γ extrait)")

if len(usable) >= 4:
    gammas = np.array([d["gamma"] for d in usable])
    abs_gamma = np.abs(gammas)

    features = {
        "m_exp": np.array([d["m_exp"] for d in usable]),
        "log(m_exp)": np.array([np.log10(d["m_exp"]) for d in usable]),
        "1/m_exp": np.array([1/d["m_exp"] for d in usable]),
        "1/√m_exp": np.array([1/np.sqrt(d["m_exp"]) for d in usable]),
        "1/log(m_exp)": np.array([1/np.log10(d["m_exp"]) for d in usable]),
        "Σ U_κU_τ": np.array([sum(U_KT[K][0]*U_KT[K][1] for K in d["cfg"]) for d in usable]),
        "log(Σ U)": np.array([np.log10(sum(U_KT[K][0]*U_KT[K][1] for K in d["cfg"])) for d in usable]),
        "1/Σ U": np.array([1/sum(U_KT[K][0]*U_KT[K][1] for K in d["cfg"]) for d in usable]),
        "Σ M_naive": np.array([sum(M_NAIVE[K] for K in d["cfg"]) for d in usable]),
        "n_components": np.array([len(d["cfg"]) for d in usable], dtype=float),
    }
    print(f"\n  γ values : {gammas}")
    print(f"  |γ| range : [{abs_gamma.min():.4f}, {abs_gamma.max():.4f}]")
    print(f"\n  Corrélations Pearson |γ| vs invariants :")
    print(f"  {'invariant':<14} {'r vs |γ|':>10} {'r vs γ':>10}")
    print("  " + "-" * 40)
    best_inv = None; best_r = 0
    for label, v in features.items():
        if np.std(v) > 1e-10:
            r_abs = np.corrcoef(v, abs_gamma)[0, 1]
            r_signed = np.corrcoef(v, gammas)[0, 1]
            flag = "★★" if abs(r_abs) > 0.85 else ("★" if abs(r_abs) > 0.65 else "")
            print(f"  {label:<14} {r_abs:>+9.3f}  {r_signed:>+9.3f}  {flag}")
            if abs(r_abs) > abs(best_r):
                best_r = r_abs; best_inv = label
    print(f"\n  → Invariant le mieux corrélé : {best_inv} (r = {best_r:+.3f})")


# ───────── Étape 4 — Test du principe complet ─────────
print("\n" + "═"*100)
print("  Test du principe variationnel complet (γ(L) = C/m_exp)")
print("═"*100)

# Calibrer C sur μ⁻
gamma_mu = next(d["gamma"] for d in usable if d["nm"] == "μ⁻")
m_mu = next(d["m_exp"] for d in usable if d["nm"] == "μ⁻")
C_mass_inv = gamma_mu * m_mu
print(f"\n  Calibration : γ(L) = C/m_exp avec C = γ_μ × m_μ = {C_mass_inv:.4f}")
print(f"\n  {'nm':<5} {'σ':>3} {'γ_req':>9} {'γ(L)=C/m':>10} {'m_pred':>9} "
      f"{'Δ V5':>7} {'Δ V6+plast':>11}")
print("  " + "-" * 80)
rms_v5 = 0; rms_p = 0; n_improved = 0
for d in clean_data:
    cfg = d["cfg"]; sigma = d["sigma"]
    if abs(gradient_cr(cfg)).max() < 0.01:
        # config symétrique → s = 1
        m_p = d["m_v5"]
        d_p = d["d_v5"]
    else:
        gamma_pred = C_mass_inv / d["m_exp"]
        m_p, _ = m_pred_with_gamma(cfg, sigma, gamma_pred)
        d_p = (m_p - d["m_exp"])/d["m_exp"]*100
    g_req = d.get("gamma", None)
    g_req_str = f"{g_req:+.4f}" if g_req is not None else "-----"
    g_pred = C_mass_inv / d["m_exp"]
    better = "✓" if abs(d_p) < abs(d["d_v5"]) else "✗"
    if abs(d_p) < abs(d["d_v5"]): n_improved += 1
    print(f"  {d['nm']:<5} {sigma:>+3} {g_req_str:>9} {g_pred:>+10.5f} "
          f"{m_p:>9.2f} {d['d_v5']:>+6.2f}% {d_p:>+10.2f}% {better}")
    rms_v5 += d["d_v5"]**2
    rms_p += d_p**2

rms_v5 = float(np.sqrt(rms_v5/len(clean_data)))
rms_p = float(np.sqrt(rms_p/len(clean_data)))
gain = (rms_v5 - rms_p)/rms_v5 * 100 if rms_v5 > 0 else 0
print(f"\n  RMS V5             : {rms_v5:.3f}%")
print(f"  RMS V5 + plastique : {rms_p:.3f}%")
print(f"  Gain               : {gain:+.1f}%")
print(f"  Améliorées         : {n_improved}/{len(clean_data)}")


# ───────── Diagnostic final ─────────
print("\n" + "═"*100)
print("  DIAGNOSTIC FINAL")
print("═"*100)
print(f"""
  Test conduit sur masses ISOSPIN-MOYENNÉES (proxy V6 sans k_iso).

  Si γ(L) = C/m_exp est correct :
    ✓ La corrélation r > 0.85 avec 1/m_exp ou équivalent
    ✓ Le gain RMS > 30% sur le spectre nettoyé
    ✓ Les configurations symétriques (K = 6_1×3) restent à m_V5

  Résultats observés ci-dessus.

  Si gain > 30% : principe plastique VALIDÉ comme correction
                  finale au-delà de V6. La déformation plastique
                  fixée par σ(L)·γ(L)·gradient(cr) explique le
                  résidu structurel ~1-2% de V5+V6.

  Si gain modéré (5-30%) : direction correcte, raffinement requis
                            sur la forme exacte de γ(L).

  Si pas de gain : l'amplitude γ(L) dépend d'invariants au-delà
                   des masses/U_κU_τ — probablement nombres
                   quantiques cachés.
""")
