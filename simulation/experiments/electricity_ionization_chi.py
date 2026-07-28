"""
Électricité, ionisation et transfert de charge depuis la danse du puits χ.

Reproduit :
  1) profondeur du puits chi (valence) = hbar w_chi local ~ energie d'ionisation
  2) seuils du puits NUCLEAIRE (Coulomb) : valence ~eV, coeur ~keV
  3) chaine d'ionisation inter-moleculaire (echappement + transfert/impact)

Usage : PYTHONPATH=. python3 -m simulation.experiments.electricity_ionization_chi
"""
from __future__ import annotations
import numpy as np

e = 1.602176634e-19
eps0 = 8.854e-12
hbar = 1.055e-34
m_e = 9.109e-31
Ry = 13.606   # eV


def hw_chi_eV(n: float) -> float:
    """hbar * frequence plasma a la densite n (eV) = echelle du puits chi local."""
    return hbar * np.sqrt(n * e ** 2 / (eps0 * m_e)) / e


def well_depth_molecular(L_angstrom: float) -> float:
    """Profondeur du puits chi de valence (electron sur volume L^3) en eV."""
    return hw_chi_eV(1 / (L_angstrom * 1e-10) ** 3)


def nuclear_threshold_eV(Z: int, n: int = 1, screen: float = 0.0) -> float:
    """Seuil d'echappement du puits de Coulomb nucleaire : Ry (Z-screen)^2/n^2 (eV)."""
    return Ry * (Z - screen) ** 2 / n ** 2


def impact_kinetic_eV(E_field_Vm: float, d_nm: float) -> float:
    """Energie cinetique gagnee par un electron dans un champ E sur distance d (eV)."""
    return E_field_Vm * (d_nm * 1e-9)


if __name__ == "__main__":
    print("1) PUITS chi DE VALENCE = energie d'ionisation moleculaire")
    for L in [1.5, 2.0, 3.0]:
        print(f"   ({L} A)^3 : I = hbar w_chi = {well_depth_molecular(L):5.1f} eV")

    print("\n2) SEUILS DU PUITS NUCLEAIRE (Coulomb)")
    print(f"   {'atome':>5} | {'Z':>3} | {'coeur K (Z-1)^2':>16} | {'valence (mes.)':>14}")
    for nom, Z, IE in [("H", 1, 13.6), ("C", 6, 11.3), ("Fe", 26, 7.9), ("Pb", 82, 7.4)]:
        IK = nuclear_threshold_eV(Z, 1, screen=1.0 if Z > 1 else 0.0)
        unit = f"{IK/1000:.1f} keV" if IK > 1000 else f"{IK:.0f} eV"
        print(f"   {nom:>5} | {Z:3d} | {unit:>16} | {IE:11.1f} eV")
    print(f"   He : I2 (dernier e-, puits nu Z=2) = {nuclear_threshold_eV(2):.1f} eV")

    print("\n3) CHAINE D'IONISATION (danse inter-moleculaire)")
    print("   (a) echappement A : monter le flanc, seuil I_A ~ 10 eV (valence) -> A+")
    for Ef, d in [(1e9, 15), (1e10, 15)]:
        print(f"   (b) champ {Ef:.0e} V/m sur {d} nm -> K = {impact_kinetic_eV(Ef, d):.0f} eV")
    print("   (c) impact B : K >= I_B (~12 eV) -> B+  ;  (d) 2 e- -> avalanche = COURANT")
