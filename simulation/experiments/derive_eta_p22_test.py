"""Tenter de deriver eta (P22, nucleation aux frontieres) — le libre non pince.

FORME CANONIQUE : Gamma = Gamma_0 * exp(eta * xi), xi = |grad Omega^2| lambdabar_C / Omega^2,
eta ~ 1 jamais calibre. P22 PRETEND recouvrir Hawking ET Schwinger a xi >> 1.

OBSERVATION CLE (prise au serieux quantitativement) :
  via la CGQ (P5) : g = (c^2/2) grad(Omega^2 - 1)  =>  xi = 2 a lambdabar_C / c^2
  (a = acceleration locale ; lambdabar_C = Compton reduite de l'espece nucleee).
  Alors les taux semi-classiques EXACTS se reecrivent en xi :
    Schwinger (paires chargees, champ E)   : Gamma ~ exp(-pi m^2 c^3/(qE hbar)) = exp(-2*pi/xi)
    Unruh/Hawking (facteur de Boltzmann)   : exp(-m c^2 / k T_U) = exp(-4*pi/xi)
  Ce sont des IDENTITES (verifiees numeriquement ci-dessous), pas des analogies.

CONSEQUENCES pour eta :
  (a) Tangence : exp(eta*xi) est la tangente logarithmique de exp(-2pi/xi) au point de
      raccord xi* : eta = 2pi/xi*^2. Au point d'articulation canonique xi*=1 : eta = 2pi
      (canal Schwinger) ou 4pi (canal thermique). MAIS xi* est une convention -> eta
      n'est pas uniquement pince par tangence (honnete).
  (b) REMPLACEMENT DE FORME (la voie forte) : adopter Gamma = Gamma_P * exp(-2*pi/xi)
      elimine eta ET Gamma_0 (zero parametre libre : Gamma_P = taux d'essai de Planck,
      exposant derive). Les trois regimes survivent (xi->0 : Gamma->0 ; xi~1 : e^{-2pi} ;
      xi>>1 : ->Gamma_P) et l'unification Hawking/Schwinger devient EXACTE au lieu de
      qualitative. Par espece : xi_m = 2 a lambdabar_C(m)/c^2 -> canal dominant = la
      masse la plus legere (electron), ce qui DERIVE le choix canonique lambdabar_C(e).
  Downstream insensible : les audits (JWST xi=4e-81, NICER, NANOGrav) ne dependent pas
  de la valeur d'eta (toujours jointe a Gamma_0, jamais pincee).
"""
import math

HBAR = 1.054571817e-34
C = 2.99792458e8
KB = 1.380649e-23
QE = 1.602176634e-19
ME = 9.1093837015e-31
G = 6.67430e-11
MSUN = 1.98847e30

LAMBDABAR_E = HBAR / (ME * C)          # 3.8616e-13 m


def xi_from_a(a, lambdabar=LAMBDABAR_E):
    return 2 * a * lambdabar / C**2


def run():
    print("=== Identites exactes (exposants semi-classiques en xi) ===")
    # --- Schwinger : champ E quelconque (ici E critique pour la lisibilite)
    E_crit = ME**2 * C**3 / (QE * HBAR)
    for frac in (1.0, 0.1):
        E = frac * E_crit
        S_schwinger = math.pi * ME**2 * C**3 / (QE * E * HBAR)
        a = QE * E / ME
        xi = xi_from_a(a)
        print(f"  Schwinger E={frac:>4.1f} E_crit : S = {S_schwinger:10.4f} ; "
              f"2*pi/xi = {2*math.pi/xi:10.4f} ; xi = {xi:.4f}")

    # --- Unruh/Hawking : trou noir solaire et acceleration arbitraire
    for M in (1.0, 10.0):
        kappa = C**4 / (4 * G * M * MSUN)          # gravite de surface
        T_H = HBAR * kappa / (2 * math.pi * C * KB)
        S_th = ME * C**2 / (KB * T_H)
        xi = xi_from_a(kappa)
        print(f"  Hawking M={M:>4.1f} Msun : mc^2/kT_H = {S_th:.4e} ; "
              f"4*pi/xi = {4*math.pi/xi:.4e} ; ratio = {S_th*xi/(4*math.pi):.6f}")

    print("\n=== Table des regimes (forme derivee Gamma = Gamma_P exp(-2 pi / xi)) ===")
    rows = [
        ("Terre (g = 9.81)",                     9.81),
        ("naine blanche (g ~ 1e6)",              1e6),
        ("etoile a neutrons (g ~ 1e12)",         1e12),
        ("champ critique Schwinger (a_c)",       QE * E_crit / ME),
        ("xi = 1 (peau baryonique CGQ)",         C**2 / (2 * LAMBDABAR_E)),
    ]
    for name, a in rows:
        xi = xi_from_a(a)
        expo = -2 * math.pi / xi
        print(f"  {name:36}: xi = {xi:9.3e} ; ln(Gamma/Gamma_P) = {expo:12.4g}")

    print("\n=== Consequences pour eta ===")
    print("  (a) tangence log au raccord xi* : eta = 2*pi/xi*^2 ;")
    print("      xi* = 1 -> eta = 2*pi = %.3f (Schwinger) ou 4*pi = %.3f (thermique)" %
          (2 * math.pi, 4 * math.pi))
    print("      MAIS xi* est une convention : eta non pince par la seule tangence.")
    print("  (b) REMPLACEMENT : Gamma = Gamma_P exp(-2*pi/xi) -> ZERO parametre libre")
    print("      (eta ET Gamma_0 elimines) ; Hawking/Schwinger exacts ; regimes preserves ;")
    print("      canal dominant = espece la plus legere -> lambdabar_C(e) DERIVE.")
    print("  Downstream : audits JWST/NICER/NANOGrav insensibles (xi astrophysique ~ 0).")


if __name__ == "__main__":
    run()
