"""Identifier l'autre moteur des masses de noeuds lourds (m_c=1557, m_b=4900) ?

Contexte : M_topo=sqrt(sigma_chi)*l(K) refute pour c/b (plafond geometrique ~1.4 GeV au
seuil DM c>18 ; mtopo_heavy_knots_test.py). Chasse aux candidats TQNT, avec discipline
anti-numerologie (cf. memoire combi-overfit).

CANDIDATS TESTES :
  1. Z3/Koide NATIF (triplets de noeuds lourds) : (m_l,m_c,m_b) A=0.72 ; (m_s,m_c,m_b) A=0.64
     -> PAS la signature sqrt2. REFUTE.
  2. Echelle vibrationnelle/radiale M^2=M0^2+n*pente : n_c=1.5/2.1, n_b=15.8/21.8 (radiale/
     orbitale) -> aucune echelle entiere. REFUTE.
  3. Enlacement harmonique (beta=0.21 MeV) : m_b exigerait |dLk|~60 (nucleon: 3). ABSURDE.
  4. TRANS-SECTORIEL (c,b,t), masses courantes : Q=0.6694 (+0.42%), A=1.420 (+0.4% de sqrt2),
     delta=0.069 rad. SEUL SURVIVANT (fait connu de la litterature, Rodejohann-Zhang 2011).
     Echo structurel : A=sqrt2 PARTAGE leptons<->(c,b,t), delta DIFFERENT (0.222 vs 0.069)
     = exactement la structure du brouillon delta (A=polaire universel, delta=azimut modulus
     par secteur). MAIS tension : en TQNT top = entrelacs EW (8_2+8_3+10_1 -> 170.41 GeV,
     -1.25%) ; deux moteurs revendiquent m_t a ~1% -> liberte d'assignation, non tranche.
     Avec constituantes (1557,4900) le signal s'affaiblit (A=1.378, -2.5%).
  5. Echelle geometrique x3.2 / m_c=sqrt(m_s m_b) : PAS stable courant/constituant. ECARTE.

VERDICT : moteur NON identifie parmi les mecanismes natifs. La seule signature survivante
est le Z3/sqrt2 ETENDU trans-sectoriellement a (c,b,t) -- hypothese qui exigerait de relire
le top (3e mode Z3 lourd vs entrelacs EW), tension ouverte. m_c, m_b restent calibres.
"""
import math

ML, MS, MC, MB, MT = 310., 480., 1557., 4900., 172760.
MC_CUR, MB_CUR, MT_CUR = 1270., 4180., 172690.


def koide(m1, m2, m3):
    r = [math.sqrt(x) for x in (m1, m2, m3)]
    Q = (m1 + m2 + m3) / sum(r) ** 2
    A2 = 6 * Q - 2
    A = math.sqrt(A2) if A2 > 0 else float('nan')
    M = sum(r) / 3
    delta = math.acos(min((max(r) / M - 1) / A, 1.0)) if A == A else float('nan')
    return Q, A, delta


def run():
    print("=== 1) Z3/Koide natif ===")
    for nm, t in (('(m_l,m_c,m_b)', (ML, MC, MB)), ('(m_s,m_c,m_b)', (MS, MC, MB))):
        Q, A, _ = koide(*t)
        print(f"  {nm}: Q={Q:.3f}, A={A:.3f} (sqrt2? non) -> REFUTE")

    print("\n=== 2) echelle radiale/orbitale ===")
    for nm, sl in (('radiale', 1.38 * 1.096e6), ('orbitale', 1.096e6)):
        print(f"  {nm}: n_c={(MC**2-ML**2)/sl:.2f}, n_b={(MB**2-ML**2)/sl:.2f} -> pas entiers, REFUTE")

    print("\n=== 3) enlacement harmonique ===")
    print(f"  |dLk| requis pour m_b : {math.sqrt((MB-ML)/(0.21*6.15)):.0f} (nucleon: 3) -> ABSURDE")

    print("\n=== 4) trans-sectoriel (c,b,t) ===")
    for nm, t in (('courantes', (MC_CUR, MB_CUR, MT_CUR)), ('constituantes', (MC, MB, MT))):
        Q, A, d = koide(*t)
        print(f"  {nm}: Q={Q:.4f} ({(Q-2/3)*150:+.2f}% de 2/3), A={A:.3f}, delta={d:.4f} rad")
    print("  A=sqrt2 partage leptons<->(c,b,t) ; delta different (0.222 vs 0.069) = structure")
    print("  'A polaire universel / delta azimut par secteur'. TENSION : top = entrelacs EW en TQNT.")

    print("\n=== 5) echelle x3.2 ===")
    print(f"  constituantes m_c/m_s={MC/MS:.2f}, m_b/m_c={MB/MC:.2f} ; courantes 13.6, 3.29 -> instable, ECARTE")

    print("\nVERDICT : moteur non identifie ; seul survivant = Z3/sqrt2 trans-sectoriel (c,b,t),")
    print("          en tension avec top=EW. m_c, m_b restent calibres (comptage 15+2 inchange).")


if __name__ == "__main__":
    run()
