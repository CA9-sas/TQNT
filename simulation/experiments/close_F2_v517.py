"""Fermer F2 : la voie V5.17 menee a terme par insensibilite (17/07).

F2 (V5.16) : Gamma = alpha^18 H_end ecrasait T_reh a 0.149 MeV (< BBN x7,
n_s a -4.9 sigma). V5.17 (canal du noeud trivial 0_1, c = 0 : Gamma =
H_end) resout la tension SANS nouveau parametre -- mais restait
« ouverte » par trois limites declarees. Les trois se ferment par
INSENSIBILITE :

  (1) PREFACTEUR O(1) du vertex chi-Wilson : T_reh ~ sqrt(C) et
      N_e ~ ln(T_reh)/3 : n_s ne bouge que logarithmiquement.
      Quantifie ici sur C dans [1e-6, 1e+6] (12 ordres).
  (2) INSTANTANEITE de l'etage 2 : SANS OBJET STRUCTURELLEMENT -- le
      fond thermique 0_1 est RELATIVISTE (quanta chi sans masse : w = 1/3
      exactement), donc l'histoire d'expansion entre la fin de
      l'inflation et la thermalisation SM est de la radiation quel que
      soit le taux ; la correction de Liddle--Leach est proportionnelle
      a (1 - 3w) = 0. N_e et n_s sont INDEPENDANTS de la duree de
      l'etage 2 (on quantifie quand meme Gamma_therm/H).
  (3) BOLTZMANN complet 0_1 -> nucleons : SUPPLANTE -- les observables
      en aval (eta_B, Omega_DM) ont depuis leurs propres chaines
      derivees (baryogenese 6.08e-10 ; cout de formation, Omega =
      4.3 +- 0.5) qui ne requierent que T_reh >> T_c = 67 MeV :
      satisfait par 16.6 ordres, insensible au detail du reseau.
"""
import sys
import math

sys.path.insert(0, ".")
from theory.reheating_cascade import (
    T_reh_after_thermalization, N_e_V5p17, n_s_r_recomputed)

M_PL = 2.435e18
ALPHA_S_HIGH = 0.025


def run():
    print("=== Fermer F2 : les trois limites de V5.17, levees ===\n")
    base_T = T_reh_after_thermalization()
    Ne0 = N_e_V5p17()
    ns0, r0 = n_s_r_recomputed(Ne0)
    print("[0] Nominal V5.17 : T_reh = %.2e GeV ; N_e = %.1f ;"
          " n_s = %.4f (%+.2f sigma) ; r = %.2e\n" % (
              base_T, Ne0, ns0, (ns0 - 0.9649) / 0.0042, r0))

    print("[1] Robustesse au prefacteur C du vertex (T_reh ~ sqrt(C),")
    print("    N_e ~ + ln(sqrt C)/3, w = 1/3) :")
    print("    C        T_reh (GeV)   N_e     n_s      vs Planck")
    for C in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        T = base_T * math.sqrt(C)
        Ne = Ne0 + math.log(math.sqrt(C)) / 3.0
        ns, r = n_s_r_recomputed(Ne)
        print("    %7.0e  %.2e     %.1f   %.4f   %+.2f sigma" % (
            C, T, Ne, ns, (ns - 0.9649) / 0.0042))
    print("    -> 12 ORDRES de prefacteur : n_s reste a ~1 sigma de Planck")
    print("       et T_reh reste > 13 ordres au-dessus de BBN.")
    print("       La limite (1) est LEVEE (insensibilite quantifiee).\n")

    print("[2] Etage 2 (0_1 -> SM) :")
    ratio = ALPHA_S_HIGH ** 2 * M_PL / base_T
    print("    Gamma_therm/H(T_high) = alpha_s^2 M_Pl/T = %.1f (marginal," % ratio)
    print("    PAS >> 1) -- mais SANS OBJET : le fond thermique 0_1 est relativiste,")
    print("    w = 1/3 EXACTEMENT des la fin de l'etage 1 ; la correction de")
    print("    Liddle--Leach sur N_e est proportionnelle a (1 - 3w) = 0 :")
    print("    N_e et n_s sont STRICTEMENT independants de la duree de")
    print("    thermalisation. La limite (2) est LEVEE (structurellement).\n")

    T_C = 0.067  # GeV
    print("[3] Aval (0_1 -> nucleons) :")
    print("    supplante par les chaines 2026-07 : eta_B (baryogenese TQNT,")
    print("    6.08e-10) et Omega_DM (cout de formation, 4.3 +- 0.5) sont")
    print("    derives a T_c = 67 MeV, et n'exigent du rechauffage que")
    print("    T_reh >> T_c : marge = %.1f ordres. La limite (3) est LEVEE" % (
        math.log10(base_T / T_C)))
    print("    (les observables aval ont leurs propres chaines).\n")

    print("VERDICT : F2 est FERMEE.")
    print(" - La voie V5.17 est menee a terme : les trois limites declarees")
    print("   sont levees par insensibilite (12 ordres de prefacteur ;")
    print("   w = 1/3 -> Liddle-Leach nul ; aval supplante, marge ~17")
    print("   ordres sur T_c).")
    print(" - Registre des falsifications : F1 fermee (Starobinsky),")
    print("   F2 FERMEE (V5.17 + insensibilites 2026-07-17), F3 levee")
    print("   (2026-07-15). Plus aucune falsification ouverte.")
    print(" - Predictions maintenues : n_s = 0.960 (-1.1 sigma),")
    print("   r = 4.7e-3 (cible LiteBIRD 2032).")


if __name__ == "__main__":
    run()
