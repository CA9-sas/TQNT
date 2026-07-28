import numpy as np
GF=1.1663787e-5; alpha=1/137.036
# Γ = 𝒢_class · ρ(Q). On teste chaque classe avec sa loi d'espace de phase,
# Γ_0 calibré UNE fois par classe. Delta = log10(τ_pred/τ_exp) en DÉCADES.
hbar=6.582119e-25  # GeV·s
# ---------- (nom, τ_exp[s], classe, Q[GeV], type_PS) ----------
# type_PS: 'lep3'=leptonique 3-corps Q⁵; 'sl'=semileptonique Q⁵; '2bh'=2-corps hélicité;
#          'em'=radiatif α·Q³; 'strong'=fort p_cm
P=[
# --- FAIBLE leptonique (Sargent Q⁵) ---
('mu',    2.197e-6, 'W','lep3', 0.10566),
('tau',   2.903e-13,'W','lep3', 1.77686),
# --- FAIBLE semileptonique/hadronique (Q⁵) ---
('n',     8.79e2,  'W','sl',   0.000782),
('Lambda',2.63e-10,'W','sl',   0.0377),
('Sigma-',1.48e-10,'W','sl',   0.117),
('Sigma+',0.802e-10,'W','sl',  0.116),
('Xi-',   1.64e-10,'W','sl',   0.0599),
('Xi0',   2.90e-10,'W','sl',   0.135),
('Omega-',0.821e-10,'W','sl',  0.062),
('Lc',    2.03e-13,'W','sl',   1.0),
('Lb',    1.47e-12,'W','sl',   3.0),
('D0',    4.10e-13,'W','sl',   1.37),
('Dp',    1.04e-12,'W','sl',   1.37),
('Dsp',   5.04e-13,'W','sl',   1.47),
('Bp',    1.64e-12,'W','sl',   5.0),
('B0',    1.52e-12,'W','sl',   5.0),
('Bs',    1.52e-12,'W','sl',   5.1),
# --- FAIBLE 2-corps hélicité ---
('piC',   2.603e-8, 'W','2bh',  0.0339),
('KC',    1.238e-8, 'W','2bh',  0.388),
('KL',    5.12e-8,  'W','2bh',  0.30),
# --- EM (radiatif) ---
('pi0',   8.52e-17,'E','em',   0.135),
('eta',   5.0e-19, 'E','em',   0.548),
('Sigma0',7.4e-20, 'E','em',   0.074),
('etap',  3.2e-21, 'E','em',   0.958),
# --- FORT (largeur ~ p_cm) ---
('rho',   4.45e-24,'S','strong',0.363),
('omega', 7.75e-23,'S','strong',0.327),
('Delta', 5.63e-24,'S','strong',0.229),
('Kstar', 1.30e-23,'S','strong',0.291),
('phi',   1.55e-22,'S','strong',0.032),
('Sigmastar',1.83e-23,'S','strong',0.208),
('Xistar',7.0e-23, 'S','strong',0.145),
('f2',    3.6e-24, 'S','strong',0.62),
('a2',    4.4e-24, 'S','strong',0.60),
('Deltarad',5.5e-24,'S','strong',0.22),
('rho1450',2.2e-24,'S','strong',0.7),
]
def rho(t,Q,mP=1):
    if t in ('lep3','sl'): return Q**5
    if t=='2bh': return Q**3      # hélicité ~ m_ℓ² p ; approx Q³
    if t=='em':  return Q**3
    if t=='strong': return Q**3   # p_cm³ (P-wave) approx
    return Q**5
# calibrer Γ_0 par classe sur la MÉDIANE (offset log) → teste la DISPERSION intra-classe
import collections
byc=collections.defaultdict(list)
for nm,tau,cl,ty,Q in P:
    G_exp=hbar/tau
    G_shape=rho(ty,Q)
    byc[cl].append((nm,np.log10(G_exp/G_shape),tau,Q,ty,G_exp))
print("VÉRIFICATION DURÉE DE VIE — Γ=Γ_0·ρ(Q), Γ_0 calibré par classe\n")
print(f"{'part':9s} {'classe':6s} {'τ_exp[s]':>10s} {'τ_pred[s]':>10s} {'Δ(déc)':>7s}")
alldelta=[]
for cl in ['W','E','S']:
    items=byc[cl]
    # Γ_0 tel que la médiane du log soit 0
    off=np.median([x[1] for x in items])
    for nm,logGr,tau,Q,ty,Gexp in items:
        Gpred=10**(off)*rho(ty,Q)
        tpred=hbar/Gpred
        d=np.log10(tpred/tau); alldelta.append((cl,d))
        print(f"{nm:9s} {cl:6s} {tau:10.2e} {tpred:10.2e} {d:+7.2f}")
    print()
# RMS par classe et global
for cl in ['W','E','S']:
    d=[x[1] for x in alldelta if x[0]==cl]
    print(f"  RMS classe {cl}: {np.sqrt(np.mean(np.square(d))):.2f} décades ({len(d)} part.)")
dall=[x[1] for x in alldelta]
print(f"\n  RMS GLOBAL = {np.sqrt(np.mean(np.square(dall))):.2f} décades ({len(dall)} particules)")
print(f"  → sur ~{len(dall)} part., le cadre (classe × phase-space) reproduit τ à ~{np.sqrt(np.mean(np.square(dall))):.1f} décade près.")

# CSV
import csv
with open("docs/verification_durees_de_vie.csv","w",newline="") as fcsv:
    w=csv.writer(fcsv); w.writerow(["particule","classe","tau_exp_s","tau_pred_s","delta_decades","Q_GeV","loi_PS"])
    for cl in ['W','E','S']:
        off=np.median([x[1] for x in byc[cl]])
        for nm,logGr,tau,Q,ty,Gexp in byc[cl]:
            Gpred=10**off*rho(ty,Q); tpred=hbar/Gpred
            w.writerow([nm,cl,f"{tau:.3e}",f"{tpred:.3e}",f"{np.log10(tpred/tau):+.2f}",Q,ty])
print("\nCSV -> docs/verification_durees_de_vie.csv")
