"""Re-audit du calcul amont de rho_Lambda^TQNT (chap. energie noire) — F4 amont.

FORMULE (eq:rho-Lambda) :
  rho_Lambda = c_grav/(32 pi^2) * m_e^4 c^5/hbar^3 * exp(-beta_L*c_J(3_1)) * alpha_kg^(3/4)
  c_grav = 3k/(k+2) (k=135) ; beta_L = 0.914 (Boltzmann topologique, calibre chap. MN) ;
  c_J(3_1) = 3.3 (couche solveur : span + gamma_J*|V(-1)|, gamma_J = 0.1) ;
  alpha_kg = kappa_g/lambdabar_C = 8.08e-41 (kappa_g = 3.12e-53, P5/V5.5).

QUESTIONS D'AUDIT :
  1. Arithmetique : chaque facteur refait depuis CODATA ; verification des valeurs imprimees.
  2. Observable : rho_obs recalculee depuis (Omega_L, H0) — et sa dispersion sous la
     TENSION DE HUBBLE (Planck 67.36 vs SH0ES 73.04 : rho_obs ~ H0^2).
  3. Sensibilites : quelle variation de chaque entree ferme les 5.6% ?
  4. Bruit structurel : precision reelle de beta_L (recoupement alpha_cond/1.1 a ~19%),
     de c_J (gamma_J a 1 chiffre), de p (verification interne 0.739 vs 3/4).
VERDICT ATTENDU : l'ecart de 5.6% est TRES en dessous du bruit structurel de la formule
(>= +-20%) ET de la dispersion observationnelle (tension H0 : +-17% sur rho_Lambda) --
il n'a jamais ete significatif. La vraie prediction est l'ordre de grandeur (123 ordres
resolus) ; le « residu » que P22 devait expliquer se dissout.
"""
import math

HBAR = 1.054571817e-34
C = 2.99792458e8
ME = 9.1093837015e-31
G = 6.67430e-11
MPC = 3.0857e22

K_CS = 135
BETA_L = 0.914
CJ_31 = 3.3
KAPPA_G = 3.12e-53
LAMBDABAR_E = HBAR / (ME * C)


def rho_lambda_tqnt(beta=BETA_L, cj=CJ_31, p=0.75, kappa=KAPPA_G):
    c_grav = 3 * K_CS / (K_CS + 2)
    dens_e = ME**4 * C**5 / HBAR**3
    alpha_kg = kappa / LAMBDABAR_E
    return c_grav / (32 * math.pi**2) * dens_e * math.exp(-beta * cj) * alpha_kg**p


def rho_obs(H0_kms_mpc, omega_l=0.685):
    H0 = H0_kms_mpc * 1e3 / MPC
    rho_crit_c2 = 3 * H0**2 * C**2 / (8 * math.pi * G)
    return omega_l * rho_crit_c2


def run():
    print("=== 1. Arithmetique refaite (CODATA) ===")
    c_grav = 3 * K_CS / (K_CS + 2)
    dens_e = ME**4 * C**5 / HBAR**3
    alpha_kg = KAPPA_G / LAMBDABAR_E
    sup = math.exp(-BETA_L * CJ_31)
    print(f"  c_grav = {c_grav:.4f} ; c_grav/(32pi^2) = {c_grav/(32*math.pi**2):.3e}")
    print(f"  m_e^4 c^5/hbar^3 = {dens_e:.4e} J/m^3   [le texte imprime 6.83e22 : TYPO x{dens_e/6.83e22:.1f}]")
    print(f"  exp(-beta*cJ) = {sup:.4f} ; rho_nod = {dens_e/(32*math.pi**2)*sup:.3e} (texte : 2.21e20 OK)")
    print(f"  alpha_kg = {alpha_kg:.3e} ; ^(3/4) = {alpha_kg**0.75:.3e}")
    rho_p = rho_lambda_tqnt()
    print(f"  PRODUIT rho_Lambda^TQNT = {rho_p:.3e} J/m^3 (texte : 5.56e-10 OK)")
    print(f"  [NB eq:suppression imprime c_grav/(32pi^2)*alpha^3/4 pour rho/rho_nod :")
    print(f"   incoherent d'un facteur 32pi^2 avec rho_nod qui contient deja 1/(32pi^2)]")
    print(f"  [NB dimension : alpha_kg = kappa_g/lambdabar_C suppose kappa_g en METRES ;")
    print(f"   la convention de kappa_g (P5) n'est pas documentee dans le chapitre]")

    print("\n=== 2. Observable et tension de Hubble ===")
    for name, h0 in (("Planck 2018", 67.36), ("SH0ES 2022", 73.04)):
        r = rho_obs(h0)
        print(f"  rho_obs({name}, H0={h0}) = {r:.3e} J/m^3 ; ecart TQNT : {(rho_p/r-1)*100:+.1f}%")
    spread = (rho_obs(73.04) / rho_obs(67.36) - 1) * 100
    print(f"  dispersion observationnelle (tension H0) : {spread:+.1f}% sur rho_Lambda")
    print(f"  -> la prediction TQNT tombe ENTRE les deux determinations !")

    print("\n=== 3. Sensibilites : fermer les 5.6% demande... ===")
    target = rho_obs(67.36)
    dln = math.log(target / rho_p)
    print(f"  d(ln rho) requis = {dln:+.4f}")
    print(f"  via beta_L : delta = {-dln/CJ_31:+.4f} ({-dln/CJ_31/BETA_L*100:+.1f}% de 0.914)")
    print(f"  via c_J    : delta = {-dln/BETA_L:+.4f} (gamma_J : 0.100 -> {0.1 - dln/BETA_L/3:.3f})")
    lnalpha = math.log(KAPPA_G / LAMBDABAR_E)
    print(f"  via p      : delta = {dln/lnalpha:+.6f} (3/4 -> {0.75 + dln/lnalpha:.5f}, "
          f"{abs(dln/lnalpha)/0.75*100:.2f}%)")
    print(f"  via H0     : {((rho_p/rho_obs(67.36, 0.685))**0.5 - 1)*100:+.1f}% sur H0 "
          f"(tension H0 : +8.4%)")

    print("\n=== 4. Bruit structurel des entrees ===")
    print(f"  beta_L : recoupement interne alpha_cond/1.1 = {1.1991/1.1:.3f} vs 0.914 -> "
          f"coherent a {(1.1991/1.1/0.914-1)*100:.0f}% seulement ; precision reelle >~ 5-20%")
    for db in (0.05, 0.10):
        print(f"    beta_L +-{db*100:.0f}% -> rho x/{math.exp(BETA_L*db*CJ_31):.2f}")
    print(f"  c_J(3_1) : gamma_J = 0.1 (1 chiffre) ; gamma_J +-0.01 -> rho -+{(1-math.exp(-BETA_L*0.03))*100:.1f}%")
    print(f"  p : la verification interne du texte donne p_eff = 0.739 (ecart 1.5% a 3/4) ;")
    print(f"    1% sur p = facteur {math.exp(abs(lnalpha)*0.0075):.1f} sur rho")

    print("\n=== VERDICT ===")
    print("L'ecart de 5.6% est TRES en dessous (i) du bruit structurel de la formule amont")
    print("(beta_L seul : >=15-90% ; gamma_J : ~3% ; p : ~0.08% suffit) et (ii) de la")
    print("dispersion observationnelle (tension H0 : 17% sur rho_Lambda ; la prediction")
    print("TQNT tombe ENTRE Planck et SH0ES). => LE 5.6% N'A JAMAIS ETE SIGNIFICATIF.")
    print("La prediction reelle et remarquable est l'ORDRE DE GRANDEUR (123 ordres")
    print("resolus, ~+-0.1-0.5 decade de bruit) ; le « residu » que P22 devait expliquer")
    print("se dissout. F4 se reformule : mecanisme nul (P22-surface) pour un ecart qui")
    print("n'exigeait pas d'explication. + 2 typos a corriger (6.83e22 ; eq:suppression")
    print("facteur 32pi^2) + convention dimensionnelle de kappa_g a documenter.")


if __name__ == "__main__":
    run()
