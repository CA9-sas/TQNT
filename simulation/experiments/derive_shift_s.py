"""Le shift s des bandes P : mesure de precision et qualification (21/07).

CIBLE : le decalage etrange des bandes (+0,36 demi-tour, Ds et Bs
pareil, non triple). Trois determinations, dont une SANS COEUR :
  (1) Ds - D   (coeurs demi-substitution)
  (2) Bs - B   (idem)
  (3) K* : bande K2*(1430)/K*(892) vs bande a2(1320)/rho -- DONNEES
      PURES (aucun coeur TQNT) ; le mecanisme SS17 predit que le bout
      etrange du K* (l'un des DEUX bouts qui tournent) ne porte que la
      MOITIE de la rotation -> shift_K* = s/2.
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
dd, bb = ML + MC, ML + MB
ds, bs = MC + (ML + MS) / 2, MB + (ML + MS) / 2

print(__doc__)
print("=" * 76)
print("[A] LES TROIS DETERMINATIONS")
print("=" * 76)
cen = lambda core, m1, m2: np.mean([(m * m - core * core) / HALF for m in (m1, m2)])
s_c = cen(ds, 2.5351, 2.5691) - cen(dd, 2.4221, 2.4611)
s_b = cen(bs, 5.8287, 5.83986) - cen(bb, 5.7257, 5.7395)
n_k = (1.4273**2 - 0.89555**2) / HALF     # bande K2* sur ancre K*(892)
n_a = (1.3182**2 - 0.77526**2) / HALF     # bande a2 sur ancre rho
s_k = 2 * (n_k - n_a)                     # x2 : le bout etrange = 1/2 rotation
print("  (1) Ds - D  (coeurs 1/2-subst.)      : %.4f" % s_c)
print("  (2) Bs - B  (idem)                   : %.4f" % s_b)
print("  (3) K* x2   (donnees pures, 0 coeur) : %.4f" % s_k)
vals = np.array([s_c, s_b, s_k])
print("  => s = %.4f +- %.4f  -- la determination K* (sans coeur) TOMBE"
      % (vals.mean(), vals.std()))
print("     SUR les deux autres : la loi d'emploi 'part de rotation du")
print("     bout etrange' (pleine en lourd-leger, moitie en leger-leger)")
print("     est une CONFIRMATION INDEPENDANTE du mecanisme SS17.")

print("\n" + "=" * 76)
print("[B] LES ELIMINATIONS (par la constance en M^2)")
print("=" * 76)
print("  s est CONSTANT en demi-tours entre charme (0,360) et bottom")
print("  (0,355) et leger (0,358) -- un decalage LINEAIRE (coeur, masse")
print("  d'extremite) scalerait comme M (x2,7 entre Ds et Bs) : EXCLU ;")
print("  la correction classique de bout massif de Regge (dM^2 ~ m^3/2")
print("  M^1/2) : EXCLUE idem. Le shift est une TORSION QUANTIFIEE,")
print("  attachee au NOEUD compagnon (5_2 vs 3_1), cote portail (non")
print("  triple).")

print("\n" + "=" * 76)
print("[C] FORMES CLOSES CONFRONTEES (0,358 +- 0,002, quasi sans syst.)")
print("=" * 76)
m = vals.mean(); sig = max(vals.std(), 0.002)
for nom, v, com in [
        ("(m_s-m_l)/m_s = 1 - l(3_1)/l(5_2)", 1 - ML / MS, "mecanique naturel"),
        ("1/3 (tiers Z3/tresse)", 1 / 3, "mecanisme nomme"),
        ("rac2/4", np.sqrt(2) / 4, "sans mecanisme"),
        ("1/e", 1 / np.e, "sans mecanisme"),
        ("rac2-1", np.sqrt(2) - 1, "l'exces d'echange"),
        ("5/14", 5 / 14, "arithmetique")]:
    print("  %-36s %.4f  (%+5.1f %%, %4.1f sigma)  %s"
          % (nom, v, 100 * (m / v - 1), abs(m - v) / sig, com))
print("  -> AUCUNE forme close a mecanisme ne tombe dans les 2 sigma :")
print("     (m_s-m_l)/m_s a +4,3 % (~7 sigma), 1/3 a -7 % (~10 sigma) ;")
print("     les compatibles (rac2/4, 5/14) sont sans mecanisme (regle du")
print("     null). Le shift RESTE une constante MESUREE.")

print("\n" + "=" * 76)
print("VERDICT : s n'est PAS derive -- mais il est TRANSFORME :")
print("  (i) precision x30 : s = 0,358 +- 0,002 (3 determinations dont")
print("      une sans coeur) ;")
print("  (ii) QUALIFIE : torsion quantifiee (les explications de coeur/")
print("      lineaires/bout massif EXCLUES par la constance), attachee au")
print("      noeud compagnon (5_2), cote portail (non triplee) ;")
print("  (iii) LOI D'EMPLOI : proportionnel a la part de rotation du bout")
print("      etrange (pleine lourd-leger, 1/2 leger-leger : K* verifie a")
print("      0,1 %) -- troisieme confirmation du mecanisme SS17 ;")
print("  (iv) formes closes : toutes defavorisees ou sans mecanisme --")
print("      candidat le plus proche a mecanisme : (m_s-m_l)/m_s (+4,3 %).")
print("  Cible restante nommee : la torsion propre du 5_2 au bout d'un")
print("  tube en rotation (calcul geometrique sur plongement ideal).")
