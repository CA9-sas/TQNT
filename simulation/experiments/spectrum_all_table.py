r"""
TABLEAU COMPLET — toutes les particules (schéma V4) : structure + masse + durée de vie.

Colonnes : groupe n, spin (2J), Lk, D=(Lk-Lk_opt)² (stabilité), m_TQNT/m_exp,
canal, τ_TQNT/τ_exp. Durées de vie par CANAL :
  weak  : formule raffinée (3 sous-classes : ondes-libres Q^ν·N_ch·(1+3g_A²)·|V|²,
          2-corps hélicité, non-leptonique Q-plat)
  fort  : Γ = C_f · Q^{p}        (résonances ; OZI -> bruit)
  EM    : Γ = C_em · Q^{p}
  EW    : Γ = C_ew · M^{p}       (bosons lourds ; H atypique)
Chaque canal calibré séparément. Stables : D=0 et le plus léger.

Usage :  python3 -m simulation.experiments.spectrum_all_table
"""
from __future__ import annotations

import numpy as np

from theory.mass_formula import mass

# Secteur électrofaible : configs canoniques (source unique, V4 chap. Spectre)
EW_CFG = {"W": "4_1+7_3+8_3", "Z0": "5_1+7_2+8_3",
          "H": "6_1+7_4+8_3", "t": "7_4+7_6+8_3"}

def mass_ew(name):
    """Masse (MeV) d'un entrelacs électrofaible — formule 6.8 (theory.mass_formula).
    Secteur EW NON confiné => f = 1 (pas d'écrantage gluonique f(3)=0,77) ;
    masse lourde car U_EM dominé par le 8_3 (U_EM≈1492)."""
    return mass(EW_CFG[name], sector="ew")

GA = 1.2723
W_N = 1.0 + 3.0 * GA**2
Vud2, Vus2, Vcs2, Vcb2 = 0.949, 0.0497, 0.949, 0.00169
M_MU, M_PI, M_K = 105.66, 139.57, 493.68
F_PI, F_K = 130.2, 155.7

# nom, n, 2J, Lk, m_TQNT, m_exp, canal, Q(MeV)|0, Nch|0, theta|0, tau_exp|None|"conf"
# canal : stable / conf / wA(ondes libres) / wB(2-corps) / wC(non-lept) / fort / EM / EW
P = [
 ("e-",     1,1, 0, 0.51,    0.511,   "stable","",0,0, None),
 ("nu",     2,1, 1, 0.0,     0.0,     "stable","",0,0, None),
 ("u",      1,1, 0, 2.2,     2.2,     "stable","",0,0, None),
 ("d",      1,1, 0, 4.8,     4.7,     "conf",  "",0,0, "conf"),
 ("s",      2,1, 0, 93.,     95.,     "conf",  "",0,0, "conf"),
 ("c",      2,1, 0, 1275.,   1270.,   "conf",  "",0,0, "conf"),
 ("b",      3,1, 0, 4180.,   4180.,   "conf",  "",0,0, "conf"),
 ("mu-",    2,1, 0, 102.,    105.7,   "wA",105.7, 1.0, 1.0, 2.197e-6),
 ("tau-",   3,1, 0, 1781.,   1776.9,  "wA",1777., 5.0, 1.0, 2.903e-13),
 ("pi0",    1,0, 0, 134.,    135.,    "EM",135., 0,0, 8.43e-17),
 ("pi+-",   1,0, 0, 134.,    139.6,   "wB",0,0,0, 2.603e-8),
 ("K+-",    3,0, 1, 494.,    493.7,   "wB",0,0,0, 1.238e-8),
 ("K0_S",   3,0, 0, 494.,    497.6,   "wC",0,0,Vus2, 8.954e-11),
 ("eta",    3,0, 0, 545.,    547.9,   "EM",548., 0,0, 5.02e-19),
 ("etap",   3,0, 0, 955.,    957.8,   "fort",130., 0,0, 3.50e-21),
 ("rho_om", 2,2, 0, 782.,    775.,    "fort",440., 0,0, 7.75e-23),
 ("phi",    3,2, 0, 1015.,   1019.,   "fort",32.,  0,0, 1.55e-22),
 ("Jpsi",   3,2, 0, 3097.,   3097.,   "fort",3097.,0,0, 7.11e-21),
 ("Ups",    3,2, 0, 9428.,   9460.,   "fort",9460.,0,0, 1.22e-20),
 ("D0",     2,0, 0, 1869.,   1865.,   "wA",1270., 5.0, Vcs2, 4.10e-13),
 ("D+",     2,0, 1, 1869.,   1870.,   "wA",1270., 5.0, Vcs2, 1.033e-12),
 ("B",      2,0, 0, 5277.,   5279.,   "wA",4180., 5.0, Vcb2, 1.519e-12),
 ("p",      3,1, 3, 940.,    938.3,   "stable","",0,0, None),
 ("n",      3,1, 2, 941.,    939.6,   "wA",0.78, W_N, Vud2, 878.4),
 ("Lambda", 2,1, 0, 1123.,   1115.7,  "wC",0,0,Vus2, 2.617e-10),
 ("Sigma+", 2,1, 1, 1193.,   1189.4,  "wC",0,0,Vus2, 8.018e-11),
 ("Sigma0", 2,1, 0, 1193.,   1192.6,  "EM",77., 0,0, 7.4e-20),
 ("Sigma-", 2,1,-1, 1199.,   1197.4,  "wC",0,0,Vus2, 1.479e-10),
 ("Xi0",    3,1, 0, 1307.,   1314.9,  "wC",0,0,Vus2, 2.90e-10),
 ("Xi-",    3,1,-1, 1307.,   1321.7,  "wC",0,0,Vus2, 1.639e-10),
 ("Delta",  3,3, 0, 1232.,   1232.,   "fort",156., 0,0, 5.63e-24),
 ("Sigma*", 3,3, 0, 1384.,   1383.,   "fort",130., 0,0, 1.83e-23),
 ("Xi*",    3,3, 0, 1532.,   1532.,   "fort",79.,  0,0, 7.23e-23),
 ("Omega-", 3,3, 0, 1672.,   1672.,   "wC",0,0,Vus2, 8.21e-11),
 ("Lambdac",2,1, 1, 2285.,   2286.,   "wA",1270., 5.0, Vcs2, 2.03e-13),
 ("Sigmac", 3,1, 0, 2453.,   2455.,   "fort",27.,  0,0, 3.50e-22),
 ("Omegac", 3,1, 0, 2697.,   2695.,   "wA",1270., 5.0, Vcs2, 2.68e-13),
 ("Lambdab",3,1, 0, 5613.,   5620.,   "wA",4180., 5.0, Vcb2, 1.471e-12),
 ("W",      3,2, 0, mass_ew("W"),  80377.,  "EW",0,0,0, 3.157e-25),
 ("Z0",     3,2, 0, mass_ew("Z0"), 91188.,  "EW",0,0,0, 2.638e-25),
 ("H",      3,0, 0, mass_ew("H"),  125250., "EW",0,0,0, 1.6e-22),
 ("t",      3,1, 0, mass_ew("t"),  173100., "EW",0,0,0, 5.0e-25),
]


def lkopt(n): return n*(n-1)//2


def main():
    # --- calibrations par canal ---
    def coll(cn): return [p for p in P if p[6] == cn and isinstance(p[10], float)]
    pred = {}

    # canaux faibles (wA/wB/wC) : SANS CALIBRATION — échelle absolue G_F de la
    # relaxation 246 GeV (lifetime_weak_nocalib). Plus aucune ordonnée ajustée.
    from simulation.experiments.lifetime_weak_nocalib import weak_tau
    pred.update(weak_tau())

    # canal FORT : SANS CALIBRATION — Γ = β·⟨√cc⟩·D·N_stab (tension d'enlacement
    # × nombre de reconnexions vers la stabilité), lifetime_strong_nocalib.
    from simulation.experiments.lifetime_strong_nocalib import strong_tau
    pred.update(strong_tau())

    # canaux EM (α, anomalie chirale / M1) et EW (G_F de la relaxation) : SANS
    # CALIBRATION. -> les QUATRE canaux instables sont désormais calibration-free.
    from simulation.experiments.lifetime_ew_em_nocalib import ewem_tau
    pred.update(ewem_tau())

    # --- table ---
    print("="*108)
    print("  SPECTRE COMPLET — structure (groupe/spin/stabilité) + masse + durée de vie ; TQNT vs mesuré")
    print("="*108)
    hdr=f"  {'part.':>8} {'n':>2} {'2J':>2} {'Lk':>3} {'D':>2} {'mT':>9} {'mexp':>9} {'Δm%':>5} {'canal':>6} {'τ_TQNT':>10} {'τ_exp':>10} {'Δlogτ':>6}"
    print(hdr); print("  "+"-"*104)
    res_by={}; dm=[]
    for p in P:
        nm,n,j2,lk,mT,me,cn,Q,Nc,th,te=p
        D=(lk-lkopt(n))**2
        dmp=abs(mT-me)/me*100 if me>0 else 0; dm.append(dmp)
        if isinstance(te,float):
            tp=pred[nm]; dl=np.log10(tp/te); res_by.setdefault(cn,[]).append(dl)
            tps=f"{tp:.2e}"; tes=f"{te:.2e}"; dls=f"{dl:+.2f}"
        else:
            tps="stable" if te is None else "confiné"; tes=tps; dls="--"
        print(f"  {nm:>8} {n:>2} {j2:>2} {lk:>3} {D:>2} {mT:>9.1f} {me:>9.1f} {dmp:>5.1f} {cn:>6} {tps:>10} {tes:>10} {dls:>6}")

    print("\n  RMS Δlogτ par canal :")
    for cn,v in res_by.items():
        print(f"    {cn:>6} : {np.sqrt(np.mean(np.square(v))):.2f} déc ({len(v)} part.)")
    allr=[x for v in res_by.values() for x in v]
    print(f"    GLOBAL : {np.sqrt(np.mean(np.square(allr))):.2f} déc ({len(allr)} instables)")
    print(f"  Masse : Δ moyen {np.mean(dm):.1f}%, max {np.max(dm):.1f}%  ({len(dm)} particules)")
    print("="*108)

    # --- LaTeX longtable ---
    DISP={"e-":"e^-","nu":r"\nu","u":"u","d":"d","s":"s","c":"c","b":"b",
      "mu-":r"\mu^-","tau-":r"\tau^-","pi0":r"\pi^0","pi+-":r"\pi^\pm",
      "K+-":r"K^\pm","K0_S":"K^0_S","eta":r"\eta","etap":r"\eta'",
      "rho_om":r"\rho/\omega","phi":r"\phi","Jpsi":"J/\\psi","Ups":r"\Upsilon",
      "D0":"D^0","D+":"D^+","B":"B","p":"p","n":"n","Lambda":r"\Lambda",
      "Sigma+":r"\Sigma^+","Sigma0":r"\Sigma^0","Sigma-":r"\Sigma^-",
      "Xi0":r"\Xi^0","Xi-":r"\Xi^-","Delta":r"\Delta","Sigma*":r"\Sigma^*",
      "Xi*":r"\Xi^*","Omega-":r"\Omega^-","Lambdac":r"\Lambda_c","Sigmac":r"\Sigma_c",
      "Omegac":r"\Omega_c","Lambdab":r"\Lambda_b","W":"W","Z0":"Z^0","H":"H^0","t":"t"}
    CN={"stable":"---","conf":"conf.","wA":"faible","wB":"faible","wC":"faible",
        "fort":"fort","EM":"EM","EW":"EW"}
    SPIN={0:"0",1:r"\tfrac12",2:"1",3:r"\tfrac32"}
    # Configuration entrelacs V4 (configs CANONIQUES du chap. Spectre complet,
    # recherche exhaustive sur 1539 configs avec masse + spin + charge Q=Σχ+Lk ;
    # cf. TQNT_detaillee_V4_FR.tex, \chap:spectre). ν : 6_2+9_1 (brouillon).
    CONFIG={"e-":"3_1","nu":"6_2+9_1","u":"3_1","d":"3_1","s":"5_2+5_2",
      "c":"7_3+7_7","b":"4_1+5_2+7_6","mu-":"3_1+6_1","tau-":"3_1+5_1+7_1",
      "pi0":"8_1","pi+-":"8_1","K+-":"4_1+4_1+5_2","K0_S":"4_1+4_1+5_2",
      "eta":"3_1+4_1+5_1","etap":"5_2+5_2+6_1","rho_om":"5_2+7_7",
      "phi":"3_1+4_1+6_2","Jpsi":"4_1+5_2+9_1","Ups":"3_1+7_4+9_1",
      "D0":"6_2+7_7","D+":"6_2+7_7","B":"4_1+10_1","p":"3_1+7_2+7_2",
      "n":"3_1+7_2+7_2","Lambda":"4_1+7_4","Sigma+":"6_2+7_2",
      "Sigma0":"6_2+7_2","Sigma-":"6_2+7_2","Xi0":"4_1+4_1+7_2",
      "Xi-":"4_1+4_1+7_2","Delta":"3_1+5_2+6_2","Sigma*":"5_2+6_2+7_2",
      "Xi*":"4_1+5_2+7_2","Omega-":"5_2+5_2+7_2","Lambdac":"6_3+8_1",
      "Sigmac":"4_1+5_1+7_1","Omegac":"3_1+6_2+6_2","Lambdab":"5_2+5_2+8_2",
      "W":"4_1+7_3+8_3","Z0":"5_1+7_2+8_3","H":"6_1+7_4+8_3","t":"7_4+7_6+8_3"}
    CONFIG.update(EW_CFG)  # configs EW : source unique (cohérence masse/affichage)
    def cfmt(s): return "$"+s.replace("+","{+}")+"$"
    def tx(x):
        e=int(np.floor(np.log10(x))); m=x/10**e; return f"${m:.1f}{{\\times}}10^{{{e}}}$"
    def mfmt(x): return f"{x:.1f}" if x<1e4 else f"{x/1000:.1f}\\,\\mathrm{{G}}"
    print("\n  --- LIGNES LATEX (longtable) ---")
    for p in P:
        nm,n,j2,lk,mT,me,cn,Q,Nc,th,te=p
        D=(lk-lkopt(n))**2
        if isinstance(te,float):
            tp=pred[nm]; tps=tx(tp); tes=tx(te)
        else:
            tps="stable" if te is None else "conf."; tes=tps
        print(f"${DISP[nm]}$ & {cfmt(CONFIG[nm])} & {n} & ${SPIN[j2]}$ & {D} "
              f"& ${mfmt(mT)}$ & ${mfmt(me)}$ & {CN[cn]} & {tps} & {tes} \\\\")


if __name__ == "__main__":
    main()
