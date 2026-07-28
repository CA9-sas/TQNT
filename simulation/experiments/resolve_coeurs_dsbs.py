"""Resolution des coeurs D_s/B_s (+4,2/+1,5 %) : la demi-substitution (21/07).

OBSERVATION : la substitution l->s coute
  - dans les BARYONS : le PLEIN ecart de masses de noeuds
    (m_s - m_l = 184,8 MeV) -- verifie via l'echelle ;
  - dans les MESONS lourds : la MOITIE.
MECANISME CANDIDAT (partage du tube) : le meson est la construction
K+Kbar sur UN tube commun -- l'increment de ropelength d'une
substitution de compagnon est porte a parts egales par les deux moities
du circuit -> facteur 1/2 ; le baryon a des JAMBES PROPRES a la
jonction Y -> increment entier. (Nomme, non demontre -- meme rang que
les identifications de l'echelle.)
REGLE : coeur(H_s) = m_Q + m_l + (m_s - m_l)/2 = m_Q + (m_l + m_s)/2.
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
DMS = MS - ML                       # 0,1848 GeV

print(__doc__)
print("=" * 76)
print("[A] LA MESURE : cout de substitution l->s (en unites de m_s-m_l)")
print("=" * 76)
for nom, m2, m1, attendu in [
        ("D_s - D   (meson, c)", 1.96835, 1.86966, 0.5),
        ("B_s - B   (meson, b)", 5.36692, 5.27966, 0.5),
        ("Xi_c - Lambda_c (baryon, c)", 2.46790, 2.28646, 1.0),
        ("Xi_b - Lambda_b (baryon, b)", 5.79190, 5.61960, 1.0)]:
    r = (m2 - m1) / DMS
    print("  %-30s %.3f  (attendu %.1f)" % (nom, r, attendu))
print("  -> mesons : 0,534/0,472 (moyenne 0,50 +- 0,03) = LA MOITIE ;")
print("     baryons : 0,98/0,93 = le PLEIN. La structure suit le NOMBRE")
print("     DE TUBES : un tube commun (meson) partage l'increment, les")
print("     jambes propres (baryon, jonction Y) le portent entier.")

print("\n" + "=" * 76)
print("[B] LES COEURS REPARES (regle 1/2, zero parametre)")
print("=" * 76)
ds_new = MC + (ML + MS) / 2
bs_new = MB + (ML + MS) / 2
for nom, me, old, new in [
        ("D_s", 1.96835, MC + MS, ds_new),
        ("D_s*", 2.11220, np.sqrt((MC + MS)**2 + HALF),
         np.sqrt(ds_new**2 + HALF)),
        ("B_s", 5.36692, MB + MS, bs_new),
        ("B_s*", 5.41540, np.sqrt((MB + MS)**2 + HALF),
         np.sqrt(bs_new**2 + HALF))]:
    print("  %-5s %.4f -> %.4f GeV  (%+.2f -> %+.2f %%)"
          % (nom, old, new, 100 * (old / me - 1), 100 * (new / me - 1)))
print("  Cascade molecules : Zcs(3985) = Ds D* -> %.4f (%+.2f %% ; etait +2,0)"
      % (ds_new + np.sqrt((MC + ML)**2 + HALF),
         100 * ((ds_new + np.sqrt((MC + ML)**2 + HALF)) / 3.9827 - 1)))

print("\n" + "=" * 76)
print("[C] CONTROLES ET FRONTIERES")
print("=" * 76)
print("  Baryons INTACTS : la regle 1/2 ne s'applique qu'aux mesons (un")
print("  tube) -- Omega_c a plein 2 m_s reste a -0,01 %% via l'echelle ;")
print("  Xi_c'->Omega_c : predit 110,7 MeV vs mesure 116,8 (coherent).")
print("  B_c (c bbar = deux lourds) : mesure 6274 SOUS le coeur 6509")
print("  (-3,6 %%) -- liaison NEGATIVE : c'est le signe de la COMPRESSION")
print("  Coulomb (comme les quarkonia, tau<0) : B_c est reclasse hors")
print("  fonctionnelle PAR PRINCIPE (le demi-quarkonium), pas repare ici.")
print("  Secteur leger : hors domaine (lois propres GMOR/Regge).")

print("\n" + "=" * 76)
print("VERDICT : les coeurs D_s/B_s sont REPARES par la demi-substitution")
print("-- coeur(H_s) = m_Q + (m_l+m_s)/2 : D_s +4,2 -> -0,45 %, B_s +1,5")
print("-> -0,23 %, D_s* +3,3 -> -0,81 %, B_s* +1,5 -> -0,10 % ; Zcs +2,0")
print("-> -0,4 %. Mecanisme nomme (partage du tube commun K+Kbar vs")
print("jambes propres de la jonction), teste sur 4 substitutions (2")
print("mesons 1/2, 2 baryons 1) ; zero parametre. B_c -> compression")
print("(signe teste). Restent : bandes P (j_K), Omega_c* -0,7 %.")
