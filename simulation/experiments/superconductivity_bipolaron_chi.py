"""
Supraconductivité TQNT : bipolaron auto-piégé dans le puits chi + seuil de masse.

Reproduit :
  1) échec de tous les canaux DIRECTS entre deux électrons (statiques)
  2) la danse = auto-piégeage, couplage de Frohlich alpha_F ~ sqrt(m*)
  3) le SEUIL DE MASSE m_c (au-delà : bipolaron -> supra)
  4) énergie de liaison du bipolaron

Usage : PYTHONPATH=. python3 -m simulation.experiments.superconductivity_bipolaron_chi
"""
from __future__ import annotations
import numpy as np

e = 1.602176634e-19
eps0 = 8.854e-12
mu0 = 1.257e-6
hbar = 1.055e-34
m_e = 9.109e-31
kB = 1.380649e-23
mu_B = 9.274e-24
G = 6.674e-11
meV = 1e-3 * e

PREF = e ** 2 / (4 * np.pi * eps0 * hbar)   # = alpha_EM * c = 2.19e6 m/s
ALPHA_BP = 6.8                               # seuil de liaison bipolaron (fort couplage)


def direct_channels(a: float = 2.5e-10) -> dict:
    """Canaux directs entre deux électrons (statiques) -- tous échouent."""
    return dict(
        gravite_CGQ_meV=G * m_e ** 2 / a / meV,
        magnetique_meV=mu0 / (4 * np.pi) * 3 * mu_B ** 2 / a ** 3 / meV,  # singulet, tout attractif
        coulomb_meV=e ** 2 / (4 * np.pi * eps0 * a) / meV,                # REPULSIF, domine
    )


def alpha_F(m_ratio: float, hw_meV: float, kappa_diel: float) -> float:
    """Couplage de Frohlich de la 'danse' (auto-piégeage). alpha_F ~ sqrt(m*)."""
    return PREF * np.sqrt(m_ratio * m_e / (2 * hw_meV * meV)) * kappa_diel


def mass_threshold(hw_meV: float, kappa_diel: float) -> float:
    """Masse critique m_c/m_e : au-delà, alpha_F > seuil bipolaron -> supra."""
    return (ALPHA_BP / alpha_F(1.0, hw_meV, kappa_diel)) ** 2


def binding_meV(m_ratio: float, hw_meV: float, kappa_diel: float) -> float:
    """Énergie de liaison (self-énergie polaron, Pekar fort couplage)."""
    aF = alpha_F(m_ratio, hw_meV, kappa_diel)
    return 0.106 * aF ** 2 * hw_meV


def omega_chi_eV(n: float, m_ratio: float) -> float:
    """Pulsation du puits chi = fréquence plasma (backreaction auto-cohérente linéarisée).

    omega_chi^2 = n e^2 / (eps0 m*).  Résout la tension Tc : ~eV (rapide) -> bipolaron léger.
    À densité atomique -> ~échelle d'ionisation (unification).
    """
    omega = np.sqrt(n * e ** 2 / (eps0 * m_ratio * m_e))
    return hbar * omega / e


def bipolaron_mass_ratio(E_p_eV: float, n: float, m_ratio: float) -> float:
    """m_bp/m* = exp(g^2), g^2 = E_p/(hbar omega_chi). Antiadiabatique (chi rapide) -> léger."""
    g2 = E_p_eV / omega_chi_eV(n, m_ratio)
    return float(np.exp(g2))


def Tc_mcmillan(hw_log_eV: float, lam: float, mustar: float = 0.1) -> float:
    """Tc McMillan/Allen-Dynes (K). Prefacteur plasmon (~eV) -> haute Tc possible.

    ATTENTION : omega_p ~ E_F => Migdal violé => non fiable pour plasmons
    (corrections de vertex abaissent Tc). Estimation d'ordre seulement.
    """
    denom = lam - mustar * (1 + 0.62 * lam)
    if denom <= 0:
        return 0.0
    T_log = hw_log_eV * e / kB   # prefacteur en K
    return (T_log / 1.2) * float(np.exp(-1.04 * (1 + lam) / denom))


def lambda_pl_from_rs(rs: float) -> float:
    """Couplage électron-plasmon DÉRIVÉ (pôle plasmon, écrantage Thomas-Fermi).

    lambda_pl = (y/4) ln(1+4/y), y=(k_TF/k_F)^2 ≈ 0.66 r_s.
    Métaux (rs 2-6) : ~0.46-0.69. Mais net nu lambda_pl - mu* ≈ 0 (wp~EF) ;
    c'est le vertex non-adiabatique (forward,+) qui donne net ~0.2 (réaliste).
    """
    y = 0.663 * rs
    return (y / 4) * np.log(1 + 4 / y)


def net_coupling_nonadiabatic(rs: float, Pavg: float = 0.4) -> float:
    """Net effectif après quasi-annulation Coulomb + vertex non-adiabatique."""
    lp = lambda_pl_from_rs(rs)
    lp_na = lp * (1 + 2 * lp * Pavg)        # vertex forward/dynamique (>0)
    mustar = lp                             # mu* ≈ mu ≈ lambda_pl (wp~EF, pas de retardation)
    return lp_na - mustar


def eliashberg_rho(T_eV: float, wp_eV: float, lam0: float, mustar: float,
                   wc_eV: float, N: int = 300) -> float:
    """Plus grand valeur propre de l'équation de gap linéarisée d'Eliashberg.

    Eliashberg isotrope, mode bosonique unique à wp (axe de Matsubara).
    Tc = T où rho = 1.  T, wp, wc en eV.
    """
    nn = np.arange(N)
    wn = (2 * nn + 1) * np.pi * T_eV                 # Matsubara > 0
    def lam_k(d):
        return lam0 * wp_eV ** 2 / (wp_eV ** 2 + (2 * np.pi * T_eV * d) ** 2)
    Mf = np.arange(-N, N)
    sgn = np.sign(2 * Mf + 1)
    Z = 1 + (np.pi * T_eV / wn) * np.sum(lam_k(nn[:, None] - Mf[None, :]) * sgn[None, :], axis=1)
    mu = np.where(wn < wc_eV, 2 * mustar, 0.0)
    A = (np.pi * T_eV / Z[:, None]) * (
        lam_k(nn[:, None] - nn[None, :]) + lam_k(nn[:, None] + nn[None, :] + 1)
        - mu[None, :]) / wn[None, :]
    return float(np.max(np.real(np.linalg.eigvals(A))))


def eliashberg_Tc_K(wp_eV: float, lam0: float, mustar: float = 0.1,
                    wc_fac: float = 10.0, N: int = 300, hi_eV: float = 1.5) -> float:
    """Tc (K) par bissection sur rho=1. ATTENTION : single-mode + Migdal marginal (wp~E_F)."""
    wc = wc_fac * wp_eV
    lo, hi = 1e-4, hi_eV
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if eliashberg_rho(mid, wp_eV, lam0, mustar, wc, N) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * 11604.5


if __name__ == "__main__":
    print("1) CANAUX DIRECTS (statiques) entre 2 électrons -- tous échouent")
    for k, v in direct_channels().items():
        print(f"   {k:18s} = {v:.3e} meV")
    print("   -> Coulomb (répulsif) domine ; aucun canal direct ne lie.\n")

    print("2) LA DANSE : alpha_F(m*) ~ sqrt(m*)  (hw=30 meV, kappa=0.3)")
    for mr in [0.5, 1, 2, 5, 10, 20]:
        aF = alpha_F(mr, 30, 0.30)
        state = "BIPOLARON" if aF > ALPHA_BP else "metal"
        print(f"   m*={mr:4.1f} m_e : alpha_F = {aF:5.2f}  [{state}]")

    print("\n3) SEUIL DE MASSE m_c (prédiction : supra <=> m* > m_c)")
    for hw, k in [(30, 0.30), (50, 0.15), (30, 0.10)]:
        print(f"   hw={hw} meV, kappa={k:.2f} : m_c = {mass_threshold(hw, k):5.2f} m_e")

    print("\n4) ÉNERGIE DE LIAISON (bipolaron, fort couplage)")
    for mr, hw, k in [(2, 30, 0.30), (5, 30, 0.30), (10, 50, 0.15)]:
        print(f"   m*={mr:2d} m_e : E_p = {binding_meV(mr, hw, k):4.0f} meV  (>> meV : viable)")

    print("\n5) RÉSOLUTION TENSION Tc : omega_chi = freq. plasma (dérivée)")
    a0 = 0.529e-10
    for label, n, mr in [("supra (metal)", 3e28, 2.0),
                         ("supra (lourds)", 1e28, 10.0),
                         ("ATOMIQUE (ionisation)", 1 / a0 ** 3, 1.0)]:
        hw_chi = omega_chi_eV(n, mr)
        mbp = bipolaron_mass_ratio(0.300, n, mr)
        tag = "-> m_bp/m*=%.2f (leger)" % mbp if hw_chi < 50 else "-> ~energie d'IONISATION"
        print(f"   {label:22s}: hbar w_chi = {hw_chi:6.2f} eV  {tag}")
    print("   -> chi RAPIDE (eV) => bipolaron leger => haute Tc. Meme w_chi => ionisation.")

    print("\n6) Tc CHIFFRE (McMillan, prefacteur plasmon ~eV, mu*=0.1)")
    hwp = omega_chi_eV(3e28, 2.0)   # ~4.55 eV
    print(f"   prefacteur hbar w_p = {hwp:.2f} eV ~ {hwp*e/kB:.0f} K (vs Debye ~350 K)")
    for lam in [0.3, 0.5, 0.7]:
        print(f"   lambda_pl={lam:.1f} : Tc ~ {Tc_mcmillan(hwp, lam):.0f} K")
    print("   CAVEAT : w_p/E_F~1 => Migdal VIOLE => au-dela-de-Migdal requis.")

    print("\n7) AU-DELA DE MIGDAL : vertex non-adiabatique (GPS)")
    EF_eV = 5.0
    Qc = hwp / (4 * EF_eV)              # cutoff de transfert (forward si <0.5)
    print(f"   Migdal m=w_p/E_F={hwp/EF_eV:.2f} ; Q_c=hbar w_p/4E_F={Qc:.2f} (forward si <0.5)")
    print(f"   -> forward+dynamique => vertex P>0 (GPS) => lambda_eff>lambda => Tc MONTE")
    Pavg = 0.4   # representatif (forward, m~1) ; non derive
    for lam in [0.3, 0.5]:
        leff = lam * (1 + 2 * lam * Pavg)
        print(f"   lambda={lam:.1f} -> lambda_eff={leff:.2f} : "
              f"Tc {Tc_mcmillan(hwp, lam):.0f}K -> {Tc_mcmillan(hwp, leff):.0f}K")
    print("   CAVEAT : expansion marginale a w_p~E_F ; SIGNE robuste, MAGNITUDE = Eliashberg")
    print("            non-adiabatique numerique (P+C+self-energie). <P>=0.4 representatif.")

    print("\n8) SOLVEUR ELIASHBERG NUMERIQUE (Tc vs lambda_pl, wp=4.55 eV, mu*=0.1)")
    for lam0 in [0.20, 0.25, 0.30, 0.40, 0.50]:
        Tc = eliashberg_Tc_K(4.55, lam0)
        reg = "REALISTE" if Tc < 150 else ("eleve" if Tc < 2000 else "NON PHYSIQUE")
        print(f"   lambda_pl={lam0:.2f} : Tc = {Tc:5.0f} K  [{reg}]")
    print("   -> prefacteur eV => Tc HYPER-sensible a lambda_pl.")
    print("   -> Tc realiste (1-100 K) <=> lambda_pl ~ 0.2-0.3.")
    print("   -> VRAI inconnu restant : lambda_pl (couplage plasmon-electron), a DERIVER.")

    print("\n9) DERIVATION lambda_pl (pole plasmon / ecrantage) -> CHAINE REFERMEE")
    print(f"   {'r_s':>4} | {'lambda_pl':>9} | {'net nu (l-mu*)':>14} | {'+vertex':>8} | {'Tc (K)':>8}")
    for rs in [2, 3, 4, 5, 6]:
        lp = lambda_pl_from_rs(rs)
        net = net_coupling_nonadiabatic(rs)
        Tc = eliashberg_Tc_K(4.55, net) if net > 0.05 else 0.0
        print(f"   {rs:4d} | {lp:9.2f} | {0.0:14.2f} | {net:8.2f} | {Tc:6.0f}")
    print("   -> lambda_pl DERIVE ~0.5 ; net nu ~0 (quasi-annulation, wp~EF) ;")
    print("      VERTEX non-adiab. (forward,+) donne net ~0.2 -> Tc 1-100 K.")
    print("   => SUPRA INTRINSEQUEMENT NON-ADIABATIQUE (sans vertex : net~0, pas de supra).")
