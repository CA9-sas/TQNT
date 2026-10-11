"""Mécanisme du 1/pi — dérivation de B·L = 1/pi (cible fille, brouillon Higgs).

Assemblage de TROIS pièces canoniques + UNE identification :

  [P1, établi] Le vertex à 4 opérateurs du cadre vit AU CROISEMENT
       (théorème fusion/scission : l'intermédiaire a UN croisement défait ;
       « vertex 4-opérateurs = G_F dérivé »). C'est aussi pourquoi
       l'accrual est PAR BARREAU (par croisement) — cohérent étage (3).
  [P2, établi] UN CROISEMENT = UN DEMI-TOUR (pi radians) : le générateur
       de tresse est un demi-twist ; l'échange de deux brins = rotation
       de pi (le même fait que « un demi-tour de l'anneau ÉCHANGE les
       membres », vérifié 1e-16 dans la chaîne rac2-1). Re-vérifié
       numériquement ci-dessous (comptage de croisements vs enroulement).
  [P3, identification déclarée] L'APERTURE DU PORTAIL = 1 RADIAN,
       UNIVERSELLE : l'aperture mesurée au portail de jonction est
       l'identité (a = 0,9965 ± 0,0078 rad) et sa sélection était
       « l'identité = UNIQUE taux sans échelle de la trame graduée en
       action » — un argument d'unicité : TOUT portail de la même trame
       lit au même taux. Transplant au portail de croisement = la seule
       identification nouvelle (rang canonique).
  [P4, rang linéarité] Chaque patte du quartique s'attache au portail au
       PREMIER ORDRE, avec le poids angulaire du portail sur le cycle du
       croisement : (aperture)/(cycle) = 1 rad / pi rad. TROISIÈME
       apparition du même rang de linéarité (vertex ∝ m ; course ∝ m ;
       pattes additives).

  =>  Delta lambda par barreau = 4 pattes x (1/pi) = 4/pi
      B·L = Delta lambda /4 = (aperture)/(cycle) = 1/pi EXACT
      — les deux 4 se SIMPLIFIENT : B·L est le pur rapport angulaire.

  Constance par barreau (requise par la forme a^4 ln pure) : chaque
  croisement offre le même demi-tour, le portail lit au même taux sans
  échelle -> accrual CONSTANT -> beta constant -> log pur (boucle fermée
  avec le contrôle C3 du brouillon).

DISCRIMINANT : les couples angulaires (aperture, cycle) concurrents sont
tués par la mesure de m_H — table ci-dessous.
"""
import numpy as np

# ---------------------------------------------------------------- entrées
alpha_inv = 137.035999
L = np.log(alpha_inv)
v_GF = 246.21965
mH_pdg, s_pdg = 125.20, 0.11

print("=" * 74)
print("MÉCANISME DU 1/pi : B·L = aperture/cycle = 1 rad / pi rad")
print("=" * 74)

# ------------------------------------------ P2 : croisement = demi-tour
# deux brins torsadés d'un angle total theta ; projection x-z ;
# nombre de croisements = nombre de demi-tours de l'enroulement relatif
print("\n[P2] Re-vérification : nombre de croisements vs enroulement relatif")
print(f"   {'theta':>8s} {'croisements':>12s} {'theta/pi':>9s}")
z = np.linspace(0, 1, 200001)
for theta in (0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi, 3 * np.pi):
    xA = np.cos(theta * z)
    xB = -np.cos(theta * z)
    sgn = np.sign(xA - xB)
    ncross = int(np.sum(sgn[1:] * sgn[:-1] < 0))
    print(f"   {theta/np.pi:6.1f}pi {ncross:12d} {theta/np.pi:9.1f}")
th = np.pi
pA_end = np.array([np.cos(th), np.sin(th)])
print(f"   échange des bouts à theta=pi : |p_A(1) - p_B(0)| = "
      f"{np.linalg.norm(pA_end - np.array([-1.0, 0.0])):.1e}  (EXACT)")
print("   -> un croisement se forme par DEMI-TOUR (pi) d'enroulement ;"
      " à pi pile,\n      les brins sont ÉCHANGÉS : le générateur de"
      " tresse EST le demi-tour.")

# ------------------------------------------ l'assemblage
print(f"""
[P1]+[P2]+[P3]+[P4] — l'assemblage :
   Delta lambda par barreau = 4 pattes x (1 rad)/(pi rad) = 4/pi = {4/np.pi:.5f}
   B·L = Delta lambda / 4 = 1/pi = {1/np.pi:.6f}   (les 4 se simplifient)
   mesuré : {(mH_pdg**2/(4*v_GF**2))*L:.6f} +/- {((mH_pdg**2/(4*v_GF**2))*L)*2*s_pdg/mH_pdg:.6f}  (0,5 sigma)""")

# ------------------------------------------ discriminant angulaire
print("\nDISCRIMINANT — couples (aperture, cycle) concurrents -> m_H :")
print(f"   {'lecture':44s} {'B·L':>8s} {'m_H (GeV)':>10s} {'écart':>8s}")
CANDS = [
    ('1 rad / demi-tour pi  [LA dérivation]',    1/np.pi),
    ('1 rad / tour complet 2pi',                  1/(2*np.pi)),
    ('1 rad / cycle du ruban 4pi',                1/(4*np.pi)),
    ('aperture pi/3 (ruban x Z3) / demi-tour',    1/3),
    ('aperture pi/2 / demi-tour',                 1/2),
    ('moyenne <|cos|> = 2/pi par patte',          2/np.pi),
]
for name, c in CANDS:
    mh = 2 * v_GF * np.sqrt(c / L)
    dev = (mh - mH_pdg) / s_pdg
    print(f"   {name:44s} {c:8.4f} {mh:10.2f} {dev:+7.1f} s")
print("   -> seul le couple IDENTITÉ (1 rad) / DEMI-TOUR (pi) survit"
      " (0,5 sigma) ;\n      tous les concurrents canoniques morts à"
      " >= 27 sigma. Les candidats (ii)\n      <|cos|>=2/pi et (iii)"
      " 1/2pi du brouillon reçoivent leur verdict : EXCLUS.")

# ------------------------------------------ chaîne finale
B = 1 / (np.pi * L)
mh = 2 * v_GF * np.sqrt(B)
print(f"""
CHAÎNE FERMÉE : m_H = 2v/sqrt(pi ln(1/alpha)) = {mh:.4f} GeV
   ({100*(mh/mH_pdg-1):+.3f} % du PDG 2024, {abs(mh-mH_pdg)/s_pdg:.2f} sigma)

Statut : le 1/pi est DÉRIVÉ CONDITIONNEL — pièces : [P1] vertex au
croisement (établi), [P2] croisement = demi-tour (établi, re-vérifié),
[P3] aperture universelle 1 rad (identification, par l'argument d'unicité
du taux sans échelle — le rang de l'aperture elle-même), [P4] pattes
additives au premier ordre (3e apparition du rang de linéarité).
La constance par barreau (même demi-tour, même taux) FERME la forme log
pure. Ne plus dire « résidu 1/pi mesuré » : dérivé conditionnel, avec
discriminant angulaire à >= 27 sigma sur tous les concurrents.""")
