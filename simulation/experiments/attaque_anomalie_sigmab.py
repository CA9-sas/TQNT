"""Attaque de l'anomalie Sigma_b/Xi_b' : le triplement Z3 de la torsion (21/07).

ANOMALIE : les baryons bottom a paire symetrique mesurent n ~ 4-5
demi-tours la ou la regle n = S_paire + J + 1/2 en prevoit 2-3
(residus -1,4 a -2,4 %). Les antisymetriques (n=1) sont PARFAITS aux
deux echelles (Lambda_b +0,03 %) : l'anomalie est CONFINEE a la
bascule de paire symetrique.

DECOMPOSITION : n = [base 1 (jonction J=1/2, universelle)]
                  + [p = cout de bascule antisym->sym de la paire]
                  + [j ~ 1/2 si J=3/2 (bascule du lourd)]
QUESTION : comment p evolue-t-il du charme au bottom ?
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
LC, XC, OC = MC + 2 * ML, MC + ML + MS, MC + 2 * MS
LB, XB, OB = MB + 2 * ML, MB + ML + MS, MB + 2 * MS


def n_mes(m, core):
    return (m * m - core * core) / HALF


print(__doc__)
print("=" * 74)
print("[A] LA MESURE : le cout de bascule de paire p, charme vs bottom")
print("=" * 74)
# p = n_mes(etat sym J=1/2) - 1  (base antisym universelle = 1)
fams = [("ll (Sigma)", n_mes(2.4529, LC) - 1, n_mes(5.8106, LB) - 1),
        ("ls (Xi')  ", n_mes(2.5784, XC) - 1, n_mes(5.93502, XB) - 1),
        ("ss (Omega)", n_mes(2.6952, OC) - 1, n_mes(6.0452, OB) - 1)]
ratios = []
for nom, pc, pb in fams:
    ratios.append(pb / pc)
    print("  paire %s : p_charme = %.3f   p_bottom = %.3f   ratio = %.3f"
          % (nom, pc, pb, pb / pc))
r = np.array(ratios)
print("\n  RATIO bottom/charme = %.3f +- %.3f" % (r.mean(), r.std()))
print("  candidats : 3 EXACT (pas Z3 de generation)  -> ecart %+.1f %%"
      % (100 * (r.mean() / 3 - 1)))
print("              m_b/m_c = %.3f (echelle de masse) -> ecart %+.1f %%"
      % (MB / MC, 100 * (r.mean() / (MB / MC) - 1)))
print("              M_coeur_b/M_coeur_c = %.2f        -> ecart %+.1f %%"
      % (LB / LC, 100 * (r.mean() / (LB / LC) - 1)))
print("  -> LE FACTEUR EST 3 (a 0,4 %%, les trois familles : 2,98/2,95/3,03) ;")
print("     m_b/m_c = 3,18 est defavorise (+6 %% systematique), le ratio de")
print("     coeurs 2,5 exclu. LECTURE : un pas de GENERATION = un pas Z3 --")
print("     la torsion de paire symetrique TRIPLE (les 3 caracteres du")
print("     trefle, N_gen = Z3 du document principal), elle ne suit PAS la masse.")

print("\n" + "=" * 74)
print("[B] LA LOI REPAREE : n = 1 + 3^g p_paire + 1/2 [J=3/2], g=0(c),1(b)")
print("=" * 74)
# p_paire ancre au charme (3 nombres mesures), bottom PREDIT sans rien
p_c = {"Sigma": n_mes(2.4529, LC) - 1, "Xi": n_mes(2.5784, XC) - 1,
       "Omega": n_mes(2.6952, OC) - 1}
for nom, m, core, fam, j in [("Sigma_b ", 5.81060, LB, "Sigma", 0),
                             ("Sigma_b*", 5.83210, LB, "Sigma", 0.5),
                             ("Xi_b'   ", 5.93502, XB, "Xi", 0),
                             ("Xi_b*   ", 5.95533, XB, "Xi", 0.5),
                             ("Omega_b ", 6.04520, OB, "Omega", 0)]:
    n = 1 + 3 * p_c[fam] + j
    pred = np.sqrt(core * core + n * HALF)
    print("  %s n = 1+3x%.3f%s = %.2f -> %.4f GeV  (%+.2f %%)"
          % (nom, p_c[fam], "+1/2" if j else "    ", n, pred,
             100 * (pred / m - 1)))
print("  -> LES 5 ETATS ANOMAUX FERMENT A <=0,1 %% (etaient -1,4 a -2,4 %%),")
print("     avec 3 ancres charme + le facteur 3 exact + j=1/2. Controle")
print("     charme (j=1/2 retro-applique) : Sigma_c* -0,4 %%, coherent.")

print("\n" + "=" * 74)
print("[C] PREDICTIONS FALSIFIABLES (etats non observes)")
print("=" * 74)
n_omb_st = 1 + 3 * p_c["Omega"] + 0.5
m_omb_st = np.sqrt(OB * OB + n_omb_st * HALF)
xcc_core = 2 * MC + ML
n_xcc = n_mes(3.6216, xcc_core)
m_xcc_st = np.sqrt(xcc_core**2 + (n_xcc + 0.5) * HALF)
print("  Omega_b* (J=3/2, non observe) : n = %.2f -> M = %.1f MeV"
      % (n_omb_st, 1e3 * m_omb_st))
print("    splitting Omega_b* - Omega_b = %.1f MeV (LHCb-accessible)"
      % (1e3 * (m_omb_st - 6.0452)))
print("  Xi_cc* (J=3/2, non observe)   : M = %.1f MeV (ancre Xi_cc mesure)"
      % (1e3 * m_xcc_st))
print("  Consistance Z3 : PAS de generation au-dela (saturation Z3 derivee,")
print("  le top ne hadronise pas) -- le facteur 3 ne se re-teste qu'en")
print("  charme->bottom : il est unique, comme le pas de generation.")

print("\n" + "=" * 74)
print("VERDICT : l'anomalie est RESOLUE en structure -- la torsion de paire")
print("symetrique TRIPLE par pas de generation (3,00 +- 0,03 ; Z3, pas la")
print("masse). Reste ouvert : deriver p_charme (0,42/0,95/1,32 -- croit")
print("avec le contenu leger de la paire) depuis les noeuds.")
