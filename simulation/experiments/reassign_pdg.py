"""Réassignation PDG par les 4 observables (masse, spin, charge, durée de vie).

Recalcule les masses de tous les entrelacs (≤3 nœuds) avec l'opérateur Ω_ts
(n_ℓ = formule de genre, loi universelle) et cherche, pour chaque particule PDG,
la (ou les) meilleure(s) assignation(s) compatible(s) avec :
  - MASSE   : m(Ω_ts) ≈ m_PDG (tolérance) ;
  - SPIN    : parité de tresse → fermion/boson ;
  - CHARGE  : Q = Σχ + Lk, χ = σ·b ; Lk entier physique (|Lk| ≤ 4) ;
  - VIE     : Γ = G_F² Φ_n(Q) calculée (propriété dérivée).

Sortie : docs/TQNT_reassignation_V6.xlsx
"""
from __future__ import annotations
import itertools
import numpy as np

import theory.omega_ts as OT
from theory.omega_ts import omega_ts_axiomatic, n_lobes_formula, _N_LOBES, frenet_energies
from theory.jones import braid_index
from theory.knot_catalog import knot_curve

KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]

# bascule l'opérateur sur la formule de genre (loi universelle)
OT._N_LOBES.clear()
OT._N_LOBES.update({k: n_lobes_formula(k) for k in KNOTS})
OT._axiom_data.cache_clear()


def sigma_chi(name: str, N: int = 2048) -> tuple[float, int]:
    """σ (saturation de courbure) et χ = σ·b. σ=0 amphichiral, ±1 chiral torique."""
    P = np.asarray(knot_curve(name, N))
    if len(P) > 1 and np.allclose(P[0], P[-1]):
        P = P[:-1]
    m = len(P)
    ik = np.fft.fftfreq(m, d=1.0/m) * 1j
    F = np.fft.fft(P, axis=0)
    r1 = np.real(np.fft.ifft(ik[:, None]*F, axis=0))
    r2 = np.real(np.fft.ifft((ik**2)[:, None]*F, axis=0))
    r3 = np.real(np.fft.ifft((ik**3)[:, None]*F, axis=0))
    cr = np.cross(r1, r2); nc = np.linalg.norm(cr, axis=1); nr = np.linalg.norm(r1, axis=1)
    kap = nc/np.clip(nr**3, 1e-12, None)
    tau = np.einsum('ij,ij->i', cr, r3)/np.clip(nc**2, 1e-12, None)
    ismax = (kap > np.roll(kap, 1)) & (kap > np.roll(kap, -1))
    tp = int(np.sum(ismax & (tau > 0))); tn = int(np.sum(ismax & (tau < 0)))
    sig = (tp - tn)/(tp + tn) if (tp + tn) > 0 else 0.0
    return sig, int(round(sig * braid_index(name)))


# table des données par nœud
AMPHI = {"4_1", "6_3", "8_3"}
KDATA = {}
for k in KNOTS:
    s, chi = sigma_chi(k)
    if k in AMPHI:
        chi = 0
    KDATA[k] = {"m": omega_ts_axiomatic(k), "b": braid_index(k), "chi": chi}


def link_mass(comp): return omega_ts_axiomatic("+".join(comp))
def link_spin_parity(comp):
    """fermion (demi-entier) ⟺ Σ(b_i−1) impair ⟺ (Σb − n) impair."""
    s = sum(KDATA[k]["b"] for k in comp) - len(comp)
    return "fermion" if s % 2 == 1 else "boson"
def link_chi(comp): return sum(KDATA[k]["chi"] for k in comp)

# ---- catalogue PDG : (nom, masse MeV, spin, charge, vie s, catégorie) ----
# catégorie fixe le nombre de composantes (filtre physique V5) :
#   lep=1, mes=2-3, bar=3, ew=3
PDG = [
    ("e-",0.511,0.5,-1,1e40,"lep"), ("mu-",105.66,0.5,-1,2.197e-6,"lep"), ("tau-",1776.9,0.5,-1,2.903e-13,"lep"),
    ("pi+",139.57,0,1,2.603e-8,"mes"), ("pi0",134.98,0,0,8.5e-17,"mes"), ("K+",493.68,0,1,1.238e-8,"mes"),
    ("K0",497.61,0,0,8.95e-11,"mes"), ("eta",547.86,0,0,5e-19,"mes"), ("rho",775.26,1,1,4.5e-24,"mes"),
    ("omega",782.66,1,0,7.75e-23,"mes"), ("eta'",957.78,0,0,3.3e-21,"mes"), ("phi",1019.46,1,0,1.55e-22,"mes"),
    ("D+",1869.66,0,1,1.04e-12,"mes"), ("D0",1864.84,0,0,4.1e-13,"mes"), ("Ds+",1968.34,0,1,5.0e-13,"mes"),
    ("J/psi",3096.9,1,0,7.1e-21,"mes"), ("B+",5279.34,0,1,1.638e-12,"mes"), ("B0",5279.66,0,0,1.519e-12,"mes"),
    ("Bs0",5366.92,0,0,1.515e-12,"mes"), ("Ups",9460.3,1,0,1.2e-20,"mes"),
    ("p",938.27,0.5,1,1e40,"bar"), ("n",939.57,0.5,0,879.0,"bar"), ("Lambda",1115.68,0.5,0,2.63e-10,"bar"),
    ("Sigma+",1189.37,0.5,1,8.02e-11,"bar"), ("Sigma-",1197.45,0.5,-1,1.48e-10,"bar"),
    ("Xi0",1314.86,0.5,0,2.9e-10,"bar"), ("Xi-",1321.71,0.5,-1,1.64e-10,"bar"), ("Omega-",1672.45,1.5,-1,8.2e-11,"bar"),
    ("Lambda_c",2286.46,0.5,1,2.0e-13,"bar"), ("Lambda_b",5619.6,0.5,0,1.47e-12,"bar"),
    ("Delta",1232.0,1.5,1,5.6e-24,"bar"),
    ("W",80377.0,1,1,3.2e-25,"ew"), ("Z",91187.6,1,0,2.6e-25,"ew"), ("H",125250.0,0,0,1.6e-22,"ew"),
    ("top",172690.0,0.5,1,5e-25,"ew"),
]
NRANGE = {"lep":(1,1), "mes":(2,3), "bar":(3,3), "ew":(3,3)}
GF, HBAR, PI = 1.1664e-5, 6.582e-25, np.pi


# canoniques V5 connus (pour comparaison)
CANON = {"e-":"3_1","mu-":"3_1+6_1","p":"3_1+7_2+7_2","n":"3_1+7_2+7_2",
         "pi+":"3_1+3_1+5_2","K+":"4_1+4_1+4_1","D0":"3_1+5_1+7_2","B+":"3_1+6_2+7_7",
         "Bs0":"6_3+7_6","Lambda":"5_1+8_1","H":"6_3+8_2+8_3","Sigma+":"4_1+4_1+5_1"}


def search(target_m, parity, charge, nrange, tol=0.03, lk_max=3):
    """Filtres DURS : catégorie (nb composantes), parité, |Lk|≤lk_max.
    Retourne le meilleur (masse) et le nombre de candidats restants."""
    nmin, nmax = nrange
    best = None; ncand = 0
    for n in range(nmin, nmax+1):
        for comb in itertools.combinations_with_replacement(KNOTS, n):
            if link_spin_parity(comb) != parity:
                continue
            lk = charge - link_chi(comb)
            if abs(lk) > lk_max:                # charge : Lk physique petit
                continue
            m = link_mass(comb); err = abs(m/target_m - 1)
            if err <= tol:
                ncand += 1
                if best is None or err < best[0]:
                    best = (err, "+".join(comb), m, link_chi(comb), lk)
    return best, ncand


def estimate_lifetime(m_parent, cat, charge):
    """(a) Vie estimée : Q = écart à la PDG compatible la plus proche en dessous,
    Γ via la classe (lep/baryon→3-corps Sargent Q⁵ ; méson→2-corps).
    Rugueux : la vraie vie exige le canal de désintégration."""
    # nearest lighter PDG particle as decay product proxy
    lighter = [p for p in PDG if p[1] < m_parent - 1]
    if not lighter:
        return None
    md = max(p[1] for p in lighter)
    Q = (m_parent - md) / 1000.0   # GeV
    if Q <= 0:
        return None
    if cat in ("lep", "bar"):      # faible 3-corps (Sargent)
        G = GF**2 * Q**5 / (192*PI**3)
    else:                          # méson 2-corps (échelle p_cm ~ Q)
        G = GF**2 * (0.13**2) * Q**3 / (8*PI)
    return HBAR / G if G > 0 else None


if __name__ == "__main__":
    import openpyxl
    print("χ:", {k: KDATA[k]["chi"] for k in ["3_1","4_1","5_1","6_3","7_2"]})
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Réassignation V6"
    hdr = ["Particule","m_PDG(MeV)","spin","charge","cat","vie_mes(s)","vie_estim(s)",
           "assignation V6","m_Ωts","Δm%","Σχ","Lk","#candidats","canonique V5","retrouve?"]
    ws.append(hdr)
    nfound = 0; ncan = 0; deg = []
    for nm, m, s, q, tau, cat in PDG:
        parity = "fermion" if (s*2) % 2 == 1 else "boson"
        tol = 0.03 if m < 3000 else 0.08
        res, ncand = search(m, parity, q, NRANGE[cat], tol=tol)
        deg.append(ncand)
        tau_est = estimate_lifetime(m, cat, q) if tau < 1e30 else None
        document principal = CANON.get(nm, ""); retro = ""
        if res:
            nfound += 1
            err, cfg, mp, chi, lk = res
            if document principal:
                retro = "OUI" if "+".join(sorted(document principal.split("+")))=="+".join(sorted(cfg.split("+"))) else "non"
                if retro=="OUI": ncan += 1
            ws.append([nm,m,s,q,cat,tau,(round(tau_est,abs(int(np.log10(tau_est)))+2) if tau_est else ""),
                       cfg,round(mp,1),round(err*100,2),chi,lk,ncand,document principal,retro])
        else:
            ws.append([nm,m,s,q,cat,tau,"","AUCUNE","","","","",0,document principal,""])
        print(f"  {nm:>9}: {(res[1] if res else 'AUCUNE'):<22} {ncand:>2} cand | document principal:{retro}")
    print(f"\n  {nfound}/{len(PDG)} assignées ; canonique retrouvé {ncan}/{len(CANON)}")
    print(f"  Liberté (filtres DURS catégorie+parité+|Lk|≤3) : médiane {int(np.median(deg))}, max {max(deg)}")
    wb.save("docs/TQNT_reassignation_V6.xlsx")
    print("Écrit : docs/TQNT_reassignation_V6.xlsx")
