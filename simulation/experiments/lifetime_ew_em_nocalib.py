r"""
Durées de vie des canaux EM et électrofaible (EW) — SANS CALIBRATION.

  • EW (W, Z, t, H) : MÊME échelle que le faible — G_F = 1/(√2 E_relax²),
    E_relax = 246 GeV (relaxation topologique, masse topologique TQNT) :
      Γ(W) = 9·G_F m_W³/(6√2π)            (3 leptons + 2 doublets quark ×3 couleurs)
      Γ(Z) = G_F m_Z³/(6√2π)·Σ(g_V²+g_A²) (g_{V,A} avec sin²θ_W)
      Γ(t) = G_F m_t³/(8√2π)(1−x)²(1+2x)  (t→Wb, x=m_W²/m_t²)
      Γ(H) = Γ(bb̄)/BR(bb̄)               (Yukawa ∝ m_b courant)
  • EM (π⁰, η, Σ⁰) : échelle α (anomalie chirale / M1), AUCUN préfacteur ajusté :
      Γ(π⁰→γγ) = α² m_π³/(64π³ f_π²)       (anomalie, EXACT)
      Γ(η→γγ)  = (1/3) α² m_η³/(64π³ f_π²)  (coeff. octet ; /BR pour le total)
      Γ(Σ⁰→Λγ) = (4/3) α μ_ΣΛ² k³/m_p²      (M1, μ_ΣΛ quark-model)

Constantes physiques mesurées (masses, sin²θ_W, f_π, μ_ΣΛ, α) — pas de calibration
de durée de vie. τ = ħ/Γ.

Usage :  python3 -m simulation.experiments.lifetime_ew_em_nocalib
"""
import numpy as np

PI = np.pi
HBAR = 6.582119e-25            # GeV·s
E_RELAX = 246.0
G_F = 1.0/(np.sqrt(2)*E_RELAX**2)
ALPHA = 1.0/137.036
SW2 = 0.2312                    # sin²θ_W
# masses (GeV)
mW, mZ, mt, mH = 80.379, 91.188, 172.76, 125.25
mb_run, mc, mtau = 2.79, 1.27, 1.77686     # m_b courant à m_H
mpi0, meta, mSig0, mLam, mp = 0.13498, 0.54786, 1.19264, 1.11568, 0.93827
FPI = 0.0922                   # GeV (constante de désintégration, conv. physique)
MU_SL = 1.61                   # moment de transition Σ⁰Λ (magnétons nucléaires)
BR_Hbb = 0.581                 # BR(H->bb̄) mesuré

def _gva(T3, Q):
    gv = T3 - 2*Q*SW2; ga = T3
    return gv*gv + ga*ga

def ewem_tau():
    """{nom: τ (s)} des canaux EM+EW SANS calibration — pour spectrum_all_table."""
    # --- EW (G_F de la relaxation) ---
    GamW = 9*G_F*mW**3/(6*np.sqrt(2)*PI)
    SZ = 3*_gva(0.5,0) + 3*_gva(-0.5,-1) + 3*2*_gva(0.5,2/3) + 3*3*_gva(-0.5,-1/3)
    GamZ = G_F*mZ**3/(6*np.sqrt(2)*PI)*SZ
    x = mW**2/mt**2
    Gamt = G_F*mt**3/(8*np.sqrt(2)*PI)*(1-x)**2*(1+2*x)
    Gambb = 3*G_F*mH*mb_run**2/(4*np.sqrt(2)*PI)*(1-4*mb_run**2/mH**2)**1.5
    GamH = Gambb/BR_Hbb
    # --- EM (α, anomalie / M1) ---
    GamPi0 = ALPHA**2*mpi0**3/(64*PI**3*FPI**2)
    GamEta_gg = (1.0/3)*ALPHA**2*meta**3/(64*PI**3*FPI**2)
    GamEta = GamEta_gg/0.394                       # /BR(η->γγ)
    k = (mSig0**2-mLam**2)/(2*mSig0)
    GamSig0 = (4.0/3)*ALPHA*MU_SL**2*k**3/mp**2
    G = {"W":GamW,"Z0":GamZ,"t":Gamt,"H":GamH,
         "pi0":GamPi0,"eta":GamEta,"Sigma0":GamSig0}
    return {nm: HBAR/g for nm, g in G.items()}

def main():
    print("="*78)
    print(f"  EM (α) & EW (G_F relaxation = {G_F:.3e}) SANS CALIBRATION")
    print("="*78)
    exp = {"W":3.157e-25,"Z0":2.638e-25,"t":5.0e-25,"H":1.6e-22,
           "pi0":8.43e-17,"eta":5.02e-19,"Sigma0":7.4e-20}
    note = {"W":"9 canaux","Z0":"Σ(gv²+ga²)","t":"t->Wb","H":"bb̄/BR (m_b courant)",
            "pi0":"anomalie EXACT","eta":"octet (mixing η-η')","Sigma0":"M1 μ_ΣΛ"}
    tau = ewem_tau()
    print(f"\n  {'part':>7} {'τ_TQNT':>11} {'τ_exp':>11} {'Δlog':>6}   note")
    print("  "+"-"*60)
    ew, em = [], []
    for nm in ["W","Z0","t","H","pi0","eta","Sigma0"]:
        dl = np.log10(tau[nm]/exp[nm])
        (ew if nm in ("W","Z0","t","H") else em).append(dl)
        print(f"  {nm:>7} {tau[nm]:>11.3e} {exp[nm]:>11.3e} {dl:>+6.2f}   {note[nm]}")
    print("  "+"-"*60)
    print(f"\n  RMS EW (W,Z,t,H)   : {np.sqrt(np.mean(np.square(ew))):.2f} déc")
    print(f"  RMS EM (π⁰,η,Σ⁰)  : {np.sqrt(np.mean(np.square(em))):.2f} déc")
    print("  EM : π⁰ exact (anomalie) ; η sous-estimé (mixing η-η' non inclus) ;")
    print("       Σ⁰ via moment de transition quark-model. EW : même G_F que le faible.")
    print("="*78)

if __name__ == "__main__":
    main()
