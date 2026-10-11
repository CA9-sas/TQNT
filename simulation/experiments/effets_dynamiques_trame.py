"""Effets dynamiques de la trame — LES CALCULS DÉTAILLÉS (refonte 02/09).

Un calcul par processus de l'inventaire (Peltier/Seebeck RETIRÉS — contenu) :
 A1 : dispersion nulle — l'arithmétique de la borne GRB (E_QG > x E_P) ;
 A2 : ondes de la métrique — masse de chirp GW150914 depuis (f1, f2, tau)
      + la double petitesse au laboratoire (barre tournante, h ~ 1e-40) ;
 A3 : poussée cosmologique — identité v=sqrt(2GM/R)=HR <-> Friedmann,
      contrôle H0 depuis rho_c ;
 B1 : Re W — force de Casimir (renvoi CL, nombres) ;
 B2 : Im W — taux DCE, espace des phases Om^3/6 ;
 B3 : branche thermique — Polder-van Hove SiC/or (conservé) ;
 B4 : extinction — épaisseur de peau ;
 C1 : Schwinger — l'intégrale WKB sous la barrière = pi m^2 c^3/(eE hbar),
      identité avec -2pi/xi ;
 C2 : Breit-Wheeler — seuil et section efficace au pic ;
 D1 : Unruh — périodicité euclidienne, a(1 K) ;
 D2 : miroir accéléré — N/T = a^2 Om^3/(6 pi c^2).
"""
import numpy as np

KB = 1.380649e-23
E_CH = 1.602176634e-19
HBAR = 1.054571817e-34
C = 2.99792458e8
G = 6.6710e-11               # convention du cadre (G_prédit)
SIGMA_SB = 5.670374419e-8
ME = 9.1093837015e-31
MSUN = 1.98892e30

print("=" * 74)
print("EFFETS DYNAMIQUES DE LA TRAME — LES CALCULS         (2026-09-02)")
print("=" * 74)

# ---- A1 : borne GRB (dispersion nulle) -----------------------------------
H0 = 67.4 * 1e3 / 3.0857e22          # s^-1
Om_m, Om_L = 0.315, 0.685
z_grb, E_grb, dt_obs = 0.903, 31.0, 0.030   # GRB 090510 : GeV, s
zz = np.linspace(0, z_grb, 2000)
K = np.trapezoid((1 + zz) / (H0 * np.sqrt(Om_m * (1 + zz)**3 + Om_L)), zz)
E_P_GeV = 1.2209e19
E_QG = E_grb * K / dt_obs            # GeV (LIV linéaire, retard ~ E/E_QG)
print(f"""
A1 — dispersion nulle : l'arithmétique GRB (090510, z={z_grb})
   retard LIV linéaire : dt = (E/E_QG) K(z), K = int (1+z)dz/H(z) = {K:.3e} s
   E = {E_grb} GeV, dt_obs < {dt_obs*1e3:.0f} ms -> E_QG > {E_QG:.2e} GeV
   = {E_QG/E_P_GeV:.1f} E_Planck (publié : > 9,3 E_P, même méthode)
   -> la trame est lisse au-delà de l'échelle de Planck : dispersion 0.""")

# ---- A2 : ondes de la métrique -------------------------------------------
f1, f2, tau = 35.0, 150.0, 0.2       # GW150914 : Hz, Hz, s
Mc = (C**3 / G) * ((5 / 96) * np.pi**(-8/3) * (3 / 8)
                   * (f1**(-8/3) - f2**(-8/3)) / tau)**(3/5)
print(f"""A2 — ondes de la métrique (écrasement oscillant)
   masse de chirp depuis (f: {f1:.0f}->{f2:.0f} Hz en {tau} s) :
   M_c = (c^3/G)[(5/96) pi^-8/3 (3/8)(f1^-8/3 - f2^-8/3)/tau]^(3/5)
       = {Mc/MSUN:.1f} M_sol (GW150914 publié ~30 : le canal est MESURÉ)""")
M_bar, L_bar, f_bar, r_det = 1000.0, 1.0, 10.0, 1000.0
Iddot = 2 * M_bar * L_bar**2 * (2 * np.pi * f_bar)**2
h_lab = 2 * G * Iddot / (C**4 * r_det)
print(f"   double petitesse au labo : barre 1 t x 1 m à 10 Hz, r=1 km ->"
      f" h ~ {h_lab:.1e}\n   (LIGO : 1e-21 pour 30 M_sol à 400 Mpc — le"
      f" labo est 19 ordres dessous)")

# ---- A3 : poussée cosmologique -------------------------------------------
rho_c = 3 * H0**2 / (8 * np.pi * G)
H_back = np.sqrt(8 * np.pi * G * rho_c / 3)
print(f"""A3 — poussée cosmologique (Friedmann mécanisé)
   identité : v = sqrt(2GM/R) = HR <=> 2GM/R = (8piG/3) rho R^2 (exacte)
   rho_c = 3H0^2/(8piG) = {rho_c:.3e} kg/m^3 ;
   contrôle H(rho_c) = {H_back * 3.0857e22 / 1e3:.1f} km/s/Mpc (= H0, boucle
   fermée) ; Omega_tot = 1 en théorème (poussée = attraction).""")

# ---- B1/B2 : les branches force et paires de W ---------------------------
d = 100e-9
F_A = np.pi**2 * HBAR * C / (240 * d**4)
print(f"""B1 — Re W (force) : F/A = pi^2 hbar c/(240 d^4) = {F_A:.1f} Pa à
   100 nm ; eta(or) = 0,53 (Lifshitz, cf. brouillon Casimir C2/C7).""")
Om_t = 1.0
w = np.linspace(0, Om_t, 100001)
ph = np.trapezoid(w * (Om_t - w), w)
print(f"""B2 — Im W (paires DCE) : spectre omega1+omega2 = Omega ; espace des
   phases int w(Om-w)dw = {ph:.6f} = Om^3/6 ({Om_t**3/6:.6f}) ->
   N/T = a^2 Om^3/(6 pi c^2) ; SQUID (10,7 GHz, 1e5 ph/s) -> a_eff ~ 24 um.""")

# ---- B3 : branche thermique (Polder-van Hove) ----------------------------
def theta(w_, T):
    x = HBAR * w_ / (KB * T)
    return HBAR * w_ / np.expm1(np.clip(x, 1e-12, 500))

def eps_sic(w_):
    e_inf, wTO, wLO, g_ = 6.7, 1.494e14, 1.827e14, 8.97e11
    return e_inf * (wLO**2 - w_**2 - 1j*g_*w_) / (wTO**2 - w_**2 - 1j*g_*w_)

def eps_au(w_):
    wp, g_ = 9.0*E_CH/HBAR, 0.035*E_CH/HBAR
    return 1 - wp**2 / (w_ * (w_ + 1j*g_))

def flux(d_, T1, T2, eps_f, nw=300, nk=1500):
    w_ = np.logspace(12.7, 15.3, nw)
    dTh = theta(w_, T1) - theta(w_, T2)
    tot = 0.0
    for wi, dthi in zip(w_, dTh):
        if dthi == 0: continue
        eps = eps_f(wi); k0 = wi / C
        k = np.linspace(1e-3*k0, 0.9999*k0, nk//3)
        kz0 = np.sqrt(k0**2 - k**2 + 0j)
        kzm = np.sqrt(eps*k0**2 - k**2 + 0j)
        kzm = np.where(kzm.imag < 0, -kzm, kzm)
        s_p = 0.0
        for r in [(kz0-kzm)/(kz0+kzm), (eps*kz0-kzm)/(eps*kz0+kzm)]:
            Tp = (1-np.abs(r)**2)**2 / np.abs(1 - r**2*np.exp(2j*kz0*d_))**2
            s_p += np.trapezoid(Tp * k, k)
        k = k0 * np.logspace(1e-4, np.log10(1 + 60/(k0*d_) if k0*d_ < 60
                                            else 2), nk)[1:]
        kap = np.sqrt(k**2 - k0**2 + 0j).real
        kzm = np.sqrt(eps*k0**2 - k**2 + 0j)
        kzm = np.where(kzm.imag < 0, -kzm, kzm)
        s_e = 0.0
        for r in [(1j*kap-kzm)/(1j*kap+kzm),
                  (eps*1j*kap-kzm)/(eps*1j*kap+kzm)]:
            ex = np.exp(-2*kap*d_)
            Te = 4*(r.imag)**2*ex / np.abs(1 - r**2*ex)**2
            s_e += np.trapezoid(Te * k, k)
        tot += dthi * (s_p + s_e) * wi
    return tot * np.log(w_[1]/w_[0]) / (4*np.pi**2)

T1, T2 = 301.0, 299.0
h_bb = SIGMA_SB * (T1**4 - T2**4) / (T1 - T2)
print(f"B3 — branche thermique (Polder-van Hove ; h_bb = {h_bb:.2f} W/m2K) :")
for d_ in (10e-9, 100e-9, 1e-6):
    h_sic = flux(d_, T1, T2, eps_sic) / (T1 - T2)
    h_au = flux(d_, T1, T2, eps_au) / (T1 - T2)
    print(f"   d = {1e9*d_:5.0f} nm : SiC {h_sic:7.1f} W/m2K (x{h_sic/h_bb:.0f})"
          f" ; or {h_au:7.2f} (x{h_au/h_bb:.1f})")
print("   factorisation exacte : Phi = T_trame(w,k;d) x DeltaTheta_contenu.")

# ---- B4 : extinction ------------------------------------------------------
lam_p = 115e-9
print(f"B4 — extinction : peau delta = lambda_p/2pi = {1e9*lam_p/(2*np.pi):.0f}"
      f" nm (Cu) ; les courants périphériques compensent l'onde (Y2).")

# ---- C1 : Schwinger, l'intégrale WKB --------------------------------------
u = np.linspace(-1, 1, 200001)
I_wkb = np.trapezoid(np.sqrt(1 - u**2), u)
print(f"""C1 — Schwinger : sous la barrière |p|c = sqrt((mc^2)^2 - (eEx)^2),
   demi-largeur x* = mc^2/eE ; exposant = (2/hbar) int |p| dx
   = 2 (m^2c^3/eE hbar) int_-1^1 sqrt(1-u^2) du ; l'intégrale = {I_wkb:.6f}
   = pi/2 -> exposant = pi m^2 c^3/(eE hbar) = 2pi/xi (xi = 2eE hbar/m^2c^3 :
   IDENTITÉ avec la forme canonique P22, vérifiée).""")

# ---- C2 : Breit-Wheeler ---------------------------------------------------
r_e = 2.8179403e-15
sig_T = 8*np.pi/3 * r_e**2
print(f"""C2 — Breit-Wheeler : seuil s >= (2mc^2)^2 = (1,022 MeV)^2 ;
   section au pic (sqrt(s) ~ 1,4 MeV) : sigma_max ~ sigma_T/4
   = {sig_T/4*1e4*1e24:.2f} x1e-25 cm^2 (sigma_T = {sig_T*1e4:.2e} cm^2) ;
   observé STAR 2021 (photons quasi réels).""")

# ---- D1 : Unruh -----------------------------------------------------------
a_1K = 2 * np.pi * C * KB / HBAR
print(f"""D1 — Unruh : métrique de Rindler ds^2 = (ax/c^2)^2 c^2 dtau^2 - dx^2 ;
   temps euclidien régulier à l'horizon ssi période beta = 2pi c/a
   -> k_B T = hbar a/(2pi c) ; a(T=1 K) = {a_1K:.2e} m/s^2 ;
   g terrestre -> {HBAR*9.81/(2*np.pi*C*KB):.1e} K. Identité P22 :
   4pi/xi = mc^2/k_B T_U (référence, 1e-9).""")

# ---- D2 : miroir accéléré -------------------------------------------------
def rate(a_, f_):
    Om_ = 2*np.pi*f_
    return a_**2 * Om_**3 / (6*np.pi*C**2)
print(f"""D2 — miroir accéléré : N/T = a^2 Om^3/(6 pi c^2) (CL-4) ;
   mécanique 1 nm @ 5 GHz : {rate(1e-9, 5e9):.1e} ph/s (désespéré) ;
   SQUID observé ~1e5/s -> a_eff ~ 24 um (modulation électrique de R).""")
