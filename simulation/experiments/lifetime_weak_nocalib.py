r"""
Durées de vie du CANAL FAIBLE — SANS CALIBRATION.

L'échelle absolue vient de la relaxation topologique 246 GeV (masse topologique
TQNT) : G_F = 1/(√2 E_s²). Aucune ordonnée à l'origine ajustée sur les durées de
vie : seules entrent des constantes physiques (masses, |V_CKM|², f_π, f_K, g_A).

Topologies (toutes Γ = ... avec le MÊME G_F) :
  • 3-corps leptonique/semi-lept. (Sargent) : Γ = G_F² Q⁵ N_eff |V|² /(192π³)
      - μ : Q=m_μ, N_eff=1 ;  τ : Q=m_τ, N_eff=5 ;
      - charme/beauté (D,B,Λc,Ωc,Λb) : Q=m_quark, N_eff=5, |V|²=Vcs²|Vcb² ;
  • neutron (bas Q, fonction de Fermi) : Γ = G_F²Vud²(1+3g_A²)m_e⁵f_N/(2π³) ;
  • 2-corps hélicité (π±,K±) : Γ = (G_F²/8π) f² |V|² m_P m_μ² (1−m_μ²/m_P²)² ;
  • non-leptonique (hypérons, K⁰_S) : Γ = p_cm (G_F Vud Vus M² f_π)²/(8π M²)
    (estimation dimensionnelle ; l'élément de matrice hadronique O(1) est le
     seul résidu d'interaction nucléaire forte, PAS une calibration de durée de vie).

Usage :  python3 -m simulation.experiments.lifetime_weak_nocalib
"""
import numpy as np

# ---------- échelle absolue : la relaxation ----------
E_RELAX = 246.0                       # GeV (masse topologique EW, formule TQNT)
G_F = 1.0/(np.sqrt(2)*E_RELAX**2)     # GeV^-2  (DÉRIVÉ, pas mesuré)
HBAR = 6.582119e-25                    # GeV·s
PI3 = 192*np.pi**3

# ---------- constantes physiques (pas de durée de vie) ----------
GA = 1.2723; W_N = 1+3*GA**2
Vud, Vus, Vcs, Vcb = 0.9740, 0.2243, 0.974, 0.0411
F_PI, F_K = 0.1302, 0.1557              # GeV
M_E, M_MU = 0.51099895e-3, 0.105658     # GeV
F_N, RC = 1.6887, 1.039                 # neutron : phase-space + corr. radiative
MeV = 1e-3

def pcm(M, m1, m2):
    return np.sqrt((M**2-(m1+m2)**2)*(M**2-(m1-m2)**2))/(2*M)

def tau_lept3(Q_GeV, Neff, V2):
    return HBAR/(G_F**2 * Q_GeV**5 * Neff * V2 / PI3)

def tau_neutron():
    rate = (G_F**2 * Vud**2 * W_N/(2*np.pi**3)) * M_E**5 * F_N / HBAR
    return 1.0/rate/RC

def tau_helicity(fP, V2, mP, BRmu):
    rate = (G_F**2/(8*np.pi)) * fP**2 * V2 * mP * M_MU**2 * (1-M_MU**2/mP**2)**2 / HBAR
    return BRmu/rate            # τ_total = BR(μν)/Γ(μν)

def tau_nonlept(M, m1, m2):
    A = G_F*Vud*Vus*M**2*F_PI          # GeV
    rate = pcm(M, m1, m2)*A**2/(8*np.pi*M**2)/HBAR
    return 1.0/rate

# ---------- non-leptonique : (parent, daughter1, daughter2) en GeV ----------
_NL = {
  "K0_S":  (0.49761, 0.13957, 0.13957),
  "Lambda":(1.11568, 0.93827, 0.13957),
  "Sigma+":(1.18937, 0.93827, 0.13498),
  "Sigma-":(1.19745, 0.93957, 0.13957),
  "Xi0":   (1.31486, 1.11568, 0.13498),
  "Xi-":   (1.32171, 1.11568, 0.13957),
  "Omega-":(1.67245, 1.11568, 0.49368),
}


def _rows():
    """(nom, τ_TQNT sans calibration, τ_exp) pour les 18 particules faibles."""
    rows = [
      # 3-corps leptonique/semi-leptonique (Sargent), Q en GeV
      ("mu-",    tau_lept3(M_MU, 1, 1.0),       2.197e-6),
      ("tau-",   tau_lept3(1.7769, 5, 1.0),     2.903e-13),
      ("D0",     tau_lept3(1.270, 5, Vcs**2),   4.10e-13),
      ("D+",     tau_lept3(1.270, 5, Vcs**2),   1.033e-12),
      ("B",      tau_lept3(4.180, 5, Vcb**2),   1.519e-12),
      ("Lambdac",tau_lept3(1.270, 5, Vcs**2),   2.03e-13),
      ("Omegac", tau_lept3(1.270, 5, Vcs**2),   2.68e-13),
      ("Lambdab",tau_lept3(4.180, 5, Vcb**2),   1.471e-12),
      ("n",      tau_neutron(),                 878.4),
      # 2-corps hélicité
      ("pi+-",   tau_helicity(F_PI, Vud**2, 0.13957, 1.0000), 2.603e-8),
      ("K+-",    tau_helicity(F_K,  Vus**2, 0.49368, 0.6356), 1.238e-8),
    ]
    EXPNL = {"K0_S":8.954e-11,"Lambda":2.617e-10,"Sigma+":8.018e-11,
             "Sigma-":1.479e-10,"Xi0":2.90e-10,"Xi-":1.639e-10,"Omega-":8.21e-11}
    for nm,(M,m1,m2) in _NL.items():
        rows.append((nm, tau_nonlept(M,m1,m2), EXPNL[nm]))
    return rows


def weak_tau():
    """{nom: τ (s)} du canal faible SANS calibration — pour spectrum_all_table."""
    return {nm: tp for nm, tp, te in _rows()}


def main():
    print("="*86)
    print(f"  CANAL FAIBLE SANS CALIBRATION — G_F = 1/(√2·{E_RELAX:.0f}²) = {G_F:.3e} GeV^-2")
    print("="*86)
    rows = _rows()

    print(f"\n  {'particule':>10} {'τ_TQNT':>11} {'τ_exp':>11} {'Δlog10':>7}")
    print("  "+"-"*44)
    res = {"lept":[], "héli":[], "nonl":[]}
    grp = {"mu-":"lept","tau-":"lept","D0":"lept","D+":"lept","B":"lept",
           "Lambdac":"lept","Omegac":"lept","Lambdab":"lept","n":"lept",
           "pi+-":"héli","K+-":"héli"}
    for nm,tp,te in rows:
        dl = np.log10(tp/te); res.setdefault(grp.get(nm,"nonl"),[]).append(dl)
        print(f"  {nm:>10} {tp:>11.3e} {te:>11.3e} {dl:>+7.2f}")

    print("\n  RMS Δlog10 (sans calibration) :")
    allr=[]
    for g,v in res.items():
        allr += v
        print(f"    {g:>5} : {np.sqrt(np.mean(np.square(v))):.2f} déc ({len(v)} part.)")
    print(f"    TOTAL : {np.sqrt(np.mean(np.square(allr))):.2f} déc ({len(allr)} part.)")
    print("="*86)


if __name__ == "__main__":
    main()
