"""Derivation de la base 1 (jonction Y) : le demi-tour de trialite (21/07).

CIBLE : le dernier nombre non derive de l'echelle de recouplage -- la
base n = 1 (demi-tour universel des fondamentaux baryoniques au-dessus
de la somme constituante nue ; mesuree 0,88 +- 0,09).

DERIVATION (theoreme conditionnel, 5 pas -- tous les ingredients sont
DEJA aux regles etablies) :
  (1) TRIALITE (document principal, par:trialite-porte) : les trois brins d'un
      baryon portent les trois caracteres Z3 DISTINCTS de la porte
      (1, w, w^2) -- c'est ce qui permet a la jonction Y d'exister
      (neutralite Z3 : 2pi/3 + 2pi/3 + 2pi/3 = 2pi = 0 mod 2pi).
  (2) ENROULEMENT : autour de la jonction, la phase du condensat chi
      fait donc UN TOUR COMPLET (winding entier = 1, pas 0 -- la
      neutralite est mod 2pi, le chemin reel enroule 2pi).
  (3) DOUBLE COUVERTURE (document principal, sec:v6-lorentz-socle) : le repere de
      ruban est l'objet 4pi -- il tourne de la MOITIE de la phase du
      condensat : 2pi/2 = pi = UN DEMI-TOUR.
  (4) QUANTUM (S4'', sec:v6-torsion-socle) : un demi-tour des anneaux
      peripheriques = pi sigma_chi dans M^2  =>  n_base = 1 EXACTEMENT.
  (5) MESON = caractere + conjugue (document principal : phase Z3 RELATIVE qui se
      compense) : enroulement total 0 => base 0. L'ANTI-baryon
      enroule -2pi => demi-tour oppose, MEME energie (paire en angle).
COROLLAIRES SANS FRAIS :
  - INVARIANCE de generation (pas de x3) : les caracteres sont ceux de
    la PORTE, identiques pour tout contenu -- la base est GEOMETRIQUE
    (portail), la ou l'exces de paire est DYNAMIQUE (couple au noeud).
  - Comptage de jonctions : molecules/tetraquarks (2 mesons) base 0 ;
    pentaquarque = baryon+meson : 1 jonction (deja dans le coeur) ;
    dibaryons lourds : base 2 (prediction).
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
R2 = np.sqrt(2)
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952

print(__doc__)
print("=" * 76)
print("[A] TEST : les bases mesurees, baryons vs mesons (meme construction)")
print("=" * 76)
ETATS = [("Lambda_c (jonction)", 2.28646, MC + 2 * ML, 1),
         ("Xi_c     (jonction)", 2.46790, MC + ML + MS, 1),
         ("Lambda_b (jonction)", 5.61960, MB + 2 * ML, 1),
         ("Xi_b     (jonction)", 5.79190, MB + ML + MS, 1),
         ("D   (pas de jonction)", 1.86966, MC + ML, 0),
         ("D0  (pas de jonction)", 1.86484, MC + ML, 0),
         ("B   (pas de jonction)", 5.27934, MB + ML, 0)]
for nom, m, core, att in ETATS:
    n = (m * m - core * core) / HALF
    sens = 2 * core * core * 0.01 / HALF
    print("  %s n = %+.3f  (attendu %d ; sensibilite %.2f par %% de coeur)"
          % (nom, n, att, sens))
bary = [(m * m - c * c) / HALF for _, m, c, a in ETATS if a == 1]
mes = [(m * m - c * c) / HALF for _, m, c, a in ETATS if a == 0]
print("\n  BARYONS : base = %.2f +- %.2f  (predit 1) ;  MESONS : %.2f +- %.2f"
      % (np.mean(bary), np.std(bary), np.mean(mes), np.std(mes)))
print("  (predit 0). La SEPARATION baryon/meson = la jonction, comme le")
print("  demande le pas (5) -- et l'ecart residuel est DANS le bruit des")
print("  coeurs (sensibilite ~0,2/%% au charme, ~1,1/%% au bottom : les")
print("  0,74-0,97 correspondent a des coeurs a +-0,2 %%).")

print("\n" + "=" * 76)
print("[B] INVARIANCE DE GENERATION (la signature geometrique)")
print("=" * 76)
rb = np.mean(bary[2:]) / np.mean(bary[:2])
print("  base_bottom / base_charme = %.2f  -- PAS 3 (l'exces de paire," % rb)
print("  lui, triple : 2,98). La base ne voit pas le contenu : PORTAIL")
print("  (caracteres Z3 de la porte), pas dynamique. Coherent pas (1).")

print("\n" + "=" * 76)
print("[C] COROLLAIRES / PREDICTIONS (multi-lourds, non observes)")
print("=" * 76)
occ_core = 2 * MC + MS          # Omega_cc (ccs)
occc_core = 3 * MC              # Omega_ccc
for nom, core, n1, n2 in [
        ("Omega_cc (ccs, J=1/2)", occ_core, R2, 1 + R2),
        ("Omega_ccc (J=3/2)", occc_core, R2 + 0.5, 1.5 + R2)]:
    m1 = np.sqrt(core * core + n1 * HALF)
    m2 = np.sqrt(core * core + n2 * HALF)
    print("  %s : M = %.0f a %.0f MeV  (verrou pur n=%.2f vs bascule n=%.2f)"
          % (nom, 1e3 * m1, 1e3 * m2, n1, n2))
print("  (fourchette = l'ambiguite Xi_cc declaree : la paire cc mesuree")
print("   prefere 1+rac2 -- le verrou des identiques lourds reste ouvert ;")
print("   repere externe : reseau ~4,8 GeV pour Omega_ccc, DANS la")
print("   fourchette [4782, 4840].)")
print("  Omega_bbb : HORS CADRE torsion attendu -- 3 b = liaison Coulomb")
print("  dominante (compression, tau<0, comme les quarkonia) : le reseau")
print("  donne ~14,4 GeV < 3 m_b = %.2f GeV (liaison NEGATIVE) --" % (3 * MB))
print("  la categorisation torsion/compression PREVOIT ce signe.")
print("  Dibaryons lourds (2 jonctions) : base 2 (prediction structurelle).")

print("\n" + "=" * 76)
print("VERDICT : la base 1 est DERIVEE (theoreme conditionnel) --")
print("demi-tour de TRIALITE : trois caracteres Z3 distincts => un tour")
print("de condensat autour de la jonction => le ruban (4pi) tourne de pi")
print("=> n_base = 1. Conditions, TOUTES deja aux regles etablies : (a) trialite de")
print("la porte ; (b) double couverture du ruban ; (c) quantum pi sigma_chi.")
print("Gratuits : base(meson) = 0 (compensation K+Kbar), invariance de")
print("generation (portail), anti-baryon pair. Mesure : 0,88+-0,09 vs 1")
print("(-1,3 sigma, residu dans le bruit des coeurs). L'echelle de")
print("recouplage est desormais SANS AUCUN element non derive hormis")
print("l'indice rac2-1 (Koide).")
