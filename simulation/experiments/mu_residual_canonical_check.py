"""Le residu mu- (base de calibration de P23/C_plast) existe-t-il encore sous sqrt(sigma_chi)*S ?

CONTEXTE : sous Omega_ts, le muon etait 'l'unique anomalie persistante' (-2.92 MeV = -2.76%),
et C_plast (P23) a ete calibre dessus. Omega_ts est abandonne/archive : la base tient-elle ?

VERDICT : NON -- le residu mu- N'EXISTE PAS sous la formule canonique.
  1. Les residus leptoniques de la loi Koide sont UNIFORMES : e -0.198%, mu -0.197%,
     tau -0.191% (dispersion 0.007 point de %) -> le -0.2% est un decalage d'ECHELLE
     (valeur de sqrt(sigma_chi)), PARTAGE par les trois leptons. Le muon n'est PAS
     specifiquement anormal.
  2. Au niveau de la FORME : Koide Q=2/3 predit m_mu depuis (m_e, m_tau) a -67 ppm.
  3. Comparaison : l'anomalie Omega_ts (-2.92 MeV) est 14x le residu canonique d'echelle
     (0.21 MeV, partage) et ~400x le residu de forme (0.007 MeV).
=> La base de calibration de C_plast = -2.92 MeV etait un ARTEFACT d'Omega_ts. S'ajoute le
   verdict torsion (couplage B_ext = neV : P23 pas magnetique-direct). La decision sur P23
   (garder tel quel / recalibrer / archiver, comptage 15->14) appartient a l'utilisateur.
"""
import math

from scipy.optimize import brentq

ME, MMU, MTAU = 0.51099895, 105.6583755, 1776.86
SQSIG = 417.65


def koide_mass(k):
    return 0.75 * (1 + math.sqrt(2) * math.cos(2 / 9 + 2 * math.pi * k / 3)) ** 2 * SQSIG


def run():
    print("=== residus canoniques (loi Koide) ===")
    res = {}
    for k, nm, exp in ((1, 'e', ME), (2, 'mu', MMU), (0, 'tau', MTAU)):
        m = koide_mass(k)
        res[nm] = (m - exp) / exp * 100
        print(f"  {nm:3}: {m:9.3f} vs {exp:9.3f} -> {res[nm]:+.3f}%")
    print(f"  dispersion {max(res.values())-min(res.values()):.3f} pt de % -> UNIFORME (echelle partagee)")

    mmu_pred = brentq(lambda m: (ME + m + MTAU) - 2 / 3 * (math.sqrt(ME) + math.sqrt(m) + math.sqrt(MTAU)) ** 2, 50, 200)
    print(f"\n=== forme : Q=2/3 predit m_mu = {mmu_pred:.4f} ({(mmu_pred-MMU)/MMU*1e6:+.0f} ppm) ===")

    print(f"\n=== vs Omega_ts : C_plast=-2.92 MeV (-2.76%) ; canonique 0.21 MeV partage (14x moins),")
    print(f"    forme 0.007 MeV (~400x moins). LE RESIDU MU- ETAIT UN ARTEFACT OMEGA_TS.")


if __name__ == "__main__":
    run()
