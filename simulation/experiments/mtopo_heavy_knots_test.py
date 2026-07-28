"""Deriver M_topo = sqrt(sigma_chi)*l(K) (ropelength) pour les noeuds lourds ?

Cible nommee au §Comptage V6 : deriver les 5 masses de noeuds calibrees (m_l=310 trefle,
m_s=480, m_c=1557, m_b=4900, f_pi) depuis la masse statique M_topo = tension x longueur
(Etape 2 de brouillon_masse_demonstration_TQNT ; ropelengths Pieranski/CKS archives).

RESULTAT (deux etages) :
  1. ECHELLE ABSOLUE : rayon naturel 1/sqrt(sigma) -> M_topo(3_1)=6.8 GeV (x22 trop grand).
     Le rayon de tube reste UNE normalisation calibree (memoire : tube radius not pinned).
     On calibre sur m_l(3_1) -> scale = 18.93 MeV/unite de ropelength.
  2. PAS ETRANGE : MARCHE. M_topo(5_2)=495 MeV vs m_s=480 (+3.1%). Le PLATEAU leger
     (310-500 MeV) est GEOMETRIQUE : echelle de confinement x ropelength O(16-28).
  3. CHARME/BEAUTE : ECHEC STRUCTUREL. l requis = 82 (c~21) et 259 (c~68), or :
     (a) la ropelength croit ~3.71/croisement (lineaire = cas GENEREUX ; asymptotique
         O(c^{3/4})..O(c) -> pire) ;
     (b) COLLISION avec le seuil matiere noire : c>18 -> suppression EM alpha^(c-18)
         (charme ~1.7e-6, beaute ~1e-107) -> noeud INVISIBLE ; mais D+/B+ sont CHARGES
         et EM-visibles. Contradiction structurelle.
  => PLAFOND GEOMETRIQUE de la matiere visible : M_topo(c<=18) <~ 1.4 GeV.
     Le charme (1557) est JUSTE au-dela, la beaute (4900) x3.6 au-dela.
  4. 'M ~ l^p, p=generation' (0.94/1.95/2.91) : NON robuste a l'assignation
     (p charme 1.95->2.34, beaute 2.91->5.10 selon le noeud choisi). Numerologie. Ecarte.

VERDICT : M_topo=sqrt(sigma_chi)*l(K) explique le PLATEAU LEGER (m_l, m_s a ~3%, 1 seule
normalisation) mais est REFUTE comme origine de m_c, m_b (plafond 1.4 GeV). Les masses
lourdes exigent un autre moteur (restent calibrees). Le comptage V6 est inchange (15+2).
"""
import math

SQSIG = 417.65
RL = {"3_1": 16.372, "4_1": 20.99, "5_1": 23.55, "5_2": 26.13, "6_1": 28.31, "6_2": 28.13,
      "6_3": 27.21, "7_1": 30.69, "7_2": 32.96, "7_3": 33.61, "7_4": 34.97, "7_5": 35.16,
      "7_6": 32.66, "7_7": 31.93, "8_1": 37.49, "8_2": 37.32, "8_3": 37.30, "9_1": 38.62,
      "10_1": 42.33}
ML, MS, MC, MB = 310., 480., 1557., 4900.
ALPHA = 1 / 137.036


def run():
    print("=== 1) echelle absolue ===")
    print(f"  rayon naturel 1/sqrt(sigma): M_topo(3_1)={SQSIG*RL['3_1']/1000:.1f} GeV vs 310 MeV (x{SQSIG*RL['3_1']/ML:.0f})")
    scale = ML / RL['3_1']
    print(f"  -> rayon calibre sur m_l : scale={scale:.2f} MeV/unite")

    print("\n=== 2) pas etrange ===")
    for K in ('4_1', '5_2', '6_1'):
        m = scale * RL[K]
        print(f"  {K}: {m:.0f} MeV vs m_s=480 ({(m-MS)/MS*100:+.1f}%)")

    print("\n=== 3) charme/beaute : structurellement hors de portee ===")
    slope = (RL['10_1'] - RL['3_1']) / 7
    for nm, m in (('charme', MC), ('beaute', MB)):
        lreq = m / scale
        creq = 3 + (lreq - RL['3_1']) / slope
        sup = ALPHA ** (creq - 18)
        print(f"  {nm}: l requis={lreq:.0f} -> c~{creq:.0f} (>18) -> suppression EM ~{sup:.1e} : "
              f"noeud INVISIBLE, or D+/B+ charges. CONTRADICTION.")
    l18 = RL['3_1'] + slope * 15
    print(f"  PLAFOND geometrique (c<=18) : M_topo <= {scale*l18:.0f} MeV ~ 1.4 GeV "
          f"(charme juste au-dela, beaute x3.6)")

    print("\n=== 4) 'p=generation' non robuste ===")
    for nm, K, m in (('charme 7_6 (cat.)', '7_6', MC), ('charme 8_1', '8_1', MC),
                     ('beaute 6_2 (cat.)', '6_2', MB), ('beaute 10_1', '10_1', MB)):
        p = math.log(m / ML) / math.log(RL[K] / RL['3_1'])
        print(f"  {nm:18}: p={p:.2f}")
    print("  -> depend de l'assignation (1.95-2.34 ; 2.91-5.10). Numerologie, ecarte.")

    print("\nVERDICT : plateau leger GEOMETRIQUE (m_s a 3%) ; m_c, m_b REFUTES comme")
    print("          tension x longueur (plafond 1.4 GeV au seuil DM). Restent calibres.")


if __name__ == "__main__":
    run()
