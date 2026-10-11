# -*- coding: utf-8 -*-
"""Portage MC de gerbe — cascade hadronique 1D avec la chimie TQNT (03/09).

Au-delà du moment-Z (qui supposait x_L et N fixes, sans fluctuations ni
compétition désintégration/interaction) : un Monte-Carlo de cascade —
- compétition désintégration/interaction par espèce : P_dec = 1/(1+E/ξ),
  ξ_π = 20 GeV, ξ_K = 850 GeV (effectifs, altitude moyenne) ;
- multiplicité N(E) = 6·E^0.18, partage central fluctuant, x_L ~ U[0,15, 0,55] ;
- espèces suivies : π±, K±(+K⁰), nucléons ; π⁰/η → EM (perdu) ;
- muons : π± → 1 μ ; K → 0,85 μ effectif (BR μν + ππ→μ) — commun aux
  variantes (s'annule dans les rapports).

CHIMIES (seule différence entre variantes) :
- naïf : centre isospin-démocratique (f_EM = 1/3, pas de K/N thermiques) ;
         case de tête échangée = π⁰ toujours.
- gen  : centre = condensat thermalisé ; case échangée vectorielle à 30 %
         (le manque de ρ⁰ avant −60 % mesuré par NA61 sur les générateurs).
- tqnt : centre = condensat thermalisé ; case échangée 3:1 vecteur
         (demi-tour porté par la reconnexion) ; q = (2−√2)/(2(3−√2)) dérivé.

Sorties : N_μ(E0) par chimie, β mesuré entre 1e5 et 1e6 GeV, rapports
extrapolés à 1e19 eV — confrontés au moment-Z et au déficit WHISP requis.
Limites déclarées : 1D (pas d'atmosphère/géométrie), ξ effectifs, transport
des muons ignoré (commun), carte de désintégration thermique approchée.
"""
import numpy as np
from scipy.special import kn

rng = np.random.default_rng(20260903)

Q_EXCH = (2.0 - np.sqrt(2.0)) / (2.0 * (3.0 - np.sqrt(2.0)))   # 0,1847 dérivé
XI_PI, XI_K = 20.0, 850.0            # GeV
MU_PER_K = 0.85

# ---- partition d'énergie du condensat thermalisé (T=156,5), avec carte
# de désintégration → (EM, π±, K, N) ----
T = 156.5
# nom, m, g, (part EM, part π±, part K, part N) de l'énergie de l'espèce
SPECIES = [
    ("pi0",   135.0,  1, (1.00, 0.00, 0.0, 0.0)),
    ("pi+-",  139.6,  2, (0.00, 1.00, 0.0, 0.0)),
    ("K+-",   493.7,  2, (0.00, 0.00, 1.0, 0.0)),
    ("K0",    497.6,  2, (0.25, 0.15, 0.6, 0.0)),
    ("eta",   547.9,  1, (0.81, 0.19, 0.0, 0.0)),
    ("rho0",  775.3,  3, (0.00, 1.00, 0.0, 0.0)),
    ("rho+-", 775.3,  6, (0.50, 0.50, 0.0, 0.0)),
    ("omega", 782.7,  3, (0.38, 0.62, 0.0, 0.0)),
    ("K*",    893.6, 12, (0.20, 0.20, 0.6, 0.0)),
    ("eta'",  957.8,  1, (0.60, 0.40, 0.0, 0.0)),
    ("phi",  1019.5,  3, (0.15, 0.35, 0.5, 0.0)),
    ("N",     938.9,  8, (0.10, 0.10, 0.0, 0.8)),
    ("Delta",1232.0, 16, (0.15, 0.25, 0.0, 0.6)),
]

def partition(thermal):
    if not thermal:
        return np.array([1/3, 2/3, 0.0, 0.0])
    tot = np.zeros(4)
    for _, m, g, frac in SPECIES:
        x = m / T
        e_i = g * m**2 * T * kn(2, x) * (m * kn(1, x) / kn(2, x) + 3.0 * T)
        tot += e_i * np.array(frac)
    return tot / tot.sum()

PART_TH = partition(True)
PART_NF = partition(False)
F_VEC = {"naif": 0.0, "gen": 0.30, "tqnt": 0.75}

def central(E_pool, part):
    """Production centrale : répartit chaque énergie en N(E) fragments,
    espèces tirées selon la partition (EM, π, K, N). Retourne (π, K, N)."""
    if len(E_pool) == 0:
        return (np.array([]),) * 3
    n = np.maximum(3, (6.0 * E_pool**0.18).astype(int))
    Ec = np.repeat(E_pool / n, n) * rng.uniform(0.4, 1.6, int(n.sum()))
    u = rng.random(len(Ec))
    c = np.cumsum(part)
    return (Ec[(u >= c[0]) & (u < c[1])],      # π±
            Ec[(u >= c[1]) & (u < c[2])],      # K
            Ec[(u >= c[2])])                   # N

def shower(E0, chem):
    part = PART_NF if chem == "naif" else PART_TH
    fv = F_VEC[chem]
    pi, K, N = np.array([]), np.array([]), np.array([E0])
    n_mu = 0.0
    while len(pi) + len(K) + len(N) > 0:
        pools, lead_pi, lead_K, lead_N = [], [], [], []
        # π± : désintégration vs interaction
        if len(pi):
            dec = rng.random(len(pi)) < 1.0 / (1.0 + pi / XI_PI)
            dec |= pi < 0.5
            n_mu += dec.sum()
            act = pi[~dec]
            xL = rng.uniform(0.15, 0.55, len(act))
            El, pools_pi = act * xL, act * (1 - xL)
            exch = rng.random(len(El)) < Q_EXCH
            vec = exch & (rng.random(len(El)) < fv)
            lead_pi.append(El[~exch])                       # retenu froid
            lead_pi.append(np.repeat(El[vec] / 2, 2))       # ρ⁰→π⁺π⁻
            pools.append(pools_pi)                          # π⁰ échangé → EM
        # K : désintégration vs interaction
        if len(K):
            dec = rng.random(len(K)) < 1.0 / (1.0 + K / XI_K)
            dec |= K < 1.0
            n_mu += MU_PER_K * dec.sum()
            act = K[~dec]
            xL = rng.uniform(0.15, 0.55, len(act))
            lead_K.append(act * xL)
            pools.append(act * (1 - xL))
        # nucléons : interagissent tant que E > 2 GeV
        if len(N):
            act = N[N > 2.0]
            xL = rng.uniform(0.3, 0.7, len(act))
            lead_N.append(act * xL)
            pools.append(act * (1 - xL))
        cp, cK, cN = central(np.concatenate(pools) if pools else np.array([]), part)
        pi = np.concatenate(lead_pi + [cp]) if (lead_pi or len(cp)) else np.array([])
        K = np.concatenate(lead_K + [cK]) if (lead_K or len(cK)) else np.array([])
        N = np.concatenate(lead_N + [cN]) if (lead_N or len(cN)) else np.array([])
        # nucléons sous le seuil : l'énergie cinétique finit en pions (pas
        # de puits silencieux — les baryons ne nourrissent jamais l'EM :
        # l'« anomalie baryonique », amplificateur de muons connu)
        low = N[(N <= 2.0) & (N > 1.02)]
        if len(low):
            e_kin = low - 0.94
            n_pi = np.maximum(1, np.round(e_kin / 0.5).astype(int))
            pi_low = np.repeat(e_kin / n_pi, n_pi)
            pi = np.concatenate([pi, pi_low[pi_low > 0.12]])
        pi, K, N = pi[pi > 0.12], K[K > 0.5], N[N > 2.0]
    return n_mu

E_LO, E_HI, N_EV = 1e5, 1e6, 40
print(f"{'chimie':8s} {'N_μ(1e14 eV)':>16} {'N_μ(1e15 eV)':>16} {'β_MC':>13}")
res = {}
for chem in ("naif", "gen", "tqnt"):
    lo = np.array([shower(E_LO, chem) for _ in range(N_EV)])
    hi = np.array([shower(E_HI, chem) for _ in range(N_EV)])
    ml, sl = lo.mean(), lo.std(ddof=1) / np.sqrt(N_EV)
    mh, sh = hi.mean(), hi.std(ddof=1) / np.sqrt(N_EV)
    beta = np.log(mh / ml) / np.log(E_HI / E_LO)
    sbeta = np.hypot(sl / ml, sh / mh) / np.log(E_HI / E_LO)
    res[chem] = (ml, sl, mh, sh, beta, sbeta)
    print(f"{chem:8s} {ml:10.0f} ±{sl:4.0f} {mh:10.0f} ±{sh:5.0f} "
          f"{beta:7.3f}±{sbeta:.3f}")

E_TGT = 1e10   # GeV = 1e19 eV
print(f"\nrapports (direct à 1e15 eV, extrapolé à 1e19 eV) :")
for a, b, lab in (("tqnt", "naif", "TQNT/naïf  "),
                  ("tqnt", "gen",  "TQNT/génér."),
                  ("gen",  "naif", "génér./naïf")):
    A, B = res[a], res[b]
    r_sim = A[2] / B[2]
    sr = r_sim * np.hypot(A[3] / A[2], B[3] / B[2])
    dbeta = A[4] - B[4]
    sdb = np.hypot(A[5], B[5])
    r_ext = r_sim * (E_TGT / E_HI)**dbeta
    sre = r_ext * np.log(E_TGT / E_HI) * sdb
    print(f"  {lab} : ×{r_sim:.2f}±{sr:.2f} → ×{r_ext:.2f}±{sre:.2f} à 1e19")
print("\nrappels : moment-Z donnait TQNT/gen ×1,37 ; requis (WHISP vs")
print("QGSJet/EPOS) ×1,42. Limites déclarées : 1D, ξ effectifs, carte de")
print("désintégration approchée, transport μ ignoré (commun aux variantes).")