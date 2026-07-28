"""Le calcul SANS BOUTON du 5,6% (P22 forme WKB) — test de verite de la reattribution.

CIBLE : rho_nucl,bord = rho_Lambda^TQNT - rho_Lambda^obs = 0.30e-10 J/m^3 (5,6%).
FORME (derivee, zero parametre) : Gamma = Gamma_P * exp(-2*pi/xi), xi = 2*a*lambdabar_C/c^2,
avec a l'acceleration CGQ de la metrique emergente CANONIQUE (P5, gravitationnelle :
Omega^2 = 1 + kappa_g |Psi|^2 ; g = (c^2/2) grad(Omega^2-1) = gravite newtonienne locale).

SCHEMA (borne SUPERIEURE volontairement genereuse) :
  rho_bord <= Gamma(xi_max) * m_e c^2 * tau_pers * f_vol,  avec tau_pers = t_Hubble
  et f_vol = 1 (tout le volume de l'univers a la frontiere la plus favorable).
  Si MEME cette borne est infiniment sous la cible, le verdict est inattaquable.

BILAN ATTENDU : sur la metrique gravitationnelle, xi est ridicule partout
(peau hadronique ~1e-37, halos ~1e-81 [audit JWST], horizon BH solaire ~1e-16)
-> Gamma = Gamma_P * e^{-10^k} = 0 strict -> la reattribution du 5,6% ECHOUE.
Question inverse : quel xi* faudrait-il ? -> xi* ~ 0.02 (a ~ 1e27 m/s^2) soutenu
cosmologiquement -- aucune frontiere physique ne le fournit.
NB : l'hypothese canonique « xi ~ 1 a la peau baryonique » exigerait un puits
metrique DeltaOmega^2 ~ r_p/lambdabar_C ~ 2.6e-3 au bord du hadron : la metrique
gravitationnelle donne ~1e-39 (36 decades sous le requis). L'ancienne forme
Gamma_0 e^{eta xi} masquait ce trou derriere Gamma_0 et eta libres.
"""
import math

G = 6.67430e-11
C = 2.99792458e8
HBAR = 1.054571817e-34
ME = 9.1093837015e-31
MP = 1.67262192e-27
QE = 1.602176634e-19
LAMBDABAR_E = HBAR / (ME * C)                 # 3.86e-13 m
T_P = 5.391247e-44
L_P = 1.616255e-35
T_HUBBLE = 4.35e17                            # s
GAMMA_P = 1.0 / (T_P * L_P**3)                # taux d'essai Planck [m^-3 s^-1]

RHO_TARGET = 0.30e-10                         # J/m^3 (5,6% de rho_Lambda^TQNT)


def xi_from_a(a):
    return 2 * a * LAMBDABAR_E / C**2


def run():
    print("=== Calcul sans bouton : rho_nucl,bord (P22 WKB, metrique gravitationnelle) ===")
    print(f"Gamma_P = {GAMMA_P:.3e} /m^3/s ; cible rho = {RHO_TARGET:.2e} J/m^3\n")

    # frontieres candidates : (nom, acceleration gravitationnelle locale)
    fronts = [
        ("peau hadronique (g = G m_p / r_p^2)",      G * MP / (0.84e-15)**2),
        ("surface terrestre",                         9.81),
        ("surface solaire",                           274.0),
        ("naine blanche",                             1e6),
        ("etoile a neutrons",                         1.2e12),
        ("horizon BH solaire (kappa_s)",              C**4 / (4 * G * 1.989e30)),
        ("halo galactique (audit JWST : xi=4.3e-81)", None),
    ]
    print(f"{'frontiere':44}{'xi':>12}   ln(Gamma/Gamma_P)")
    for name, a in fronts:
        xi = 4.3e-81 if a is None else xi_from_a(a)
        print(f"{name:44}{xi:12.3e}   {-2*math.pi/xi:12.3e}")

    # borne superieure la plus genereuse : la frontiere au xi maximal (etoile a
    # neutrons / horizon), f_vol = 1, tau = t_Hubble
    xi_best = max(4.3e-81, *[xi_from_a(a) for _, a in fronts if a is not None])
    ln_gamma = math.log(GAMMA_P) - 2 * math.pi / xi_best
    ln_rho_max = ln_gamma + math.log(ME * C**2 * T_HUBBLE)   # f_vol = 1
    print(f"\nBORNE SUP (xi_max = {xi_best:.2e}, f_vol=1, tau=t_H) :")
    print(f"  ln(rho_bord) <= {ln_rho_max:.3e}   (il faudrait ln = {math.log(RHO_TARGET):.1f})")
    print(f"  deficit : exp(-{2*math.pi/xi_best:.2e}) ... rho_bord = 0 strict a toute precision.")

    # question inverse : quel xi* pour atteindre la cible ?
    # rho = Gamma_P e^{-2pi/xi} m_e c^2 t_H * f  ->  2pi/xi = ln(Gamma_P m c^2 t_H f / rho)
    for f_vol in (1.0, 0.05, 1e-30):
        need = math.log(GAMMA_P * ME * C**2 * T_HUBBLE * f_vol / RHO_TARGET)
        xi_star = 2 * math.pi / need
        a_star = xi_star * C**2 / (2 * LAMBDABAR_E)
        print(f"  inverse (f_vol={f_vol:g}) : xi* = {xi_star:.4f} -> a* = {a_star:.2e} m/s^2")

    print("\n=== Hypothese canonique « xi ~ 1 a la peau baryonique » : audit ===")
    dOmega2_needed = (0.84e-15) / LAMBDABAR_E
    print(f"  xi ~ 1 au bord du hadron exigerait DeltaOmega^2 ~ r_p/lambdabar_C = {dOmega2_needed:.2e}")
    dOmega2_grav = 2 * G * MP / ((0.84e-15) * C**2)
    print(f"  la metrique gravitationnelle donne DeltaOmega^2 ~ 2Gm_p/(r_p c^2) = {dOmega2_grav:.2e}")
    print(f"  trou : {dOmega2_needed/dOmega2_grav:.1e} (l'hypothese n'a jamais ete justifiee)")

    print("\n=== VERDICT ===")
    print("Sous la metrique emergente canonique (gravitationnelle, P5), la nucleation P22")
    print("est NULLE partout dans le cosmos (double exponentielle) : la reattribution du")
    print("residu de 5,6% a une contribution de surface est REFUTEE en l'etat, sans bouton.")
    print("L'ancienne forme Gamma_0 e^{eta xi} masquait ce trou derriere deux parametres")
    print("libres. Il faudrait xi ~ 0.02 (a ~ 1e27 m/s^2) soutenu cosmologiquement, ou un")
    print("puits metrique fort ~2.6e-3 au bord des hadrons : rien ne le fournit dans le")
    print("cadre canonique. Options : (a) le 5,6% redevient un ecart ouvert (revision de")
    print("la Regle 7) ; (b) postuler un couplage de xi a la couche chi FORTE (nouveau")
    print("contenu, a formuler et a payer) ; (c) revoir rho_Lambda^TQNT lui-meme.")


if __name__ == "__main__":
    run()
