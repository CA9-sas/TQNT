"""Facteur bosonique g_HVV = 2 m_V^2/v — dérivation (brouillon Higgs, tâche 2).

Chaîne en trois pièces :

  [T1, théorème de réponse] Le vertex scalaire d'une espèce X est la
       RÉPONSE de son terme de paire à l'amplitude locale :
          g_X = d(coefficient du terme de paire de X)/da |_(a=v).
       (La généralisation naturelle de la pièce 2 du brouillon : le
       vertex lit la prise ; ici, il lit sa DÉRIVÉE.)
  [T2, canonique] L'ORDRE des actions décide du coefficient :
       - FERMION = PREMIER ordre (Dirac FORCÉ, hérité de P10 : une action
         CS est du 1er ordre par définition) -> terme de paire m_f psi-bar psi
         -> coefficient LINÉAIRE en m ;
       - BOSON = SECOND ordre (équation d'onde de la trame □ + m^2 ;
         « masse = fréquence confinée » : la prise entre AU CARRÉ dans la
         dispersion) -> terme de paire (1/2) m_V^2 V^2 -> coefficient
         QUADRATIQUE en m.
  [T3, identification] Le QUARTET est le secteur de l'amplitude :
       m_V(a) = m_V·(a/v) (les asthénions sont « au VEV » — le W est le
       PLANCHER de la barrière ; leur prise est l'amplitude elle-même).

  =>  g_HVV = d/da [ m_V^2 (a/v)^2 ] · (1/2 champ^2 -> règle de Feynman x2)
            = 2 m_V^2 a / v^2 |_v  =  2 m_V^2 / v
      La PUISSANCE 2 = l'ordre de l'opérateur d'onde (T2).
      Le COEFFICIENT 2 = d(a^2)/da (dérivée du carré).

  Cas fermion (contrôle de cohérence) : coefficient m_f(a) -> réponse
  m_f/v = EXACTEMENT le résultat de la pièce 2 (création de paire) — la
  sur-détermination exige que l'échelle hadronique soit routée par le
  barreau EW (indécidé, consigné) ; la voie création-de-paire suffit.

  Cas photon : m_gamma = 0 -> réponse nulle -> couplage d'arbre NUL (le
  corollaire du brouillon re-dérivé par la même règle).

  Cas H : EXCEPTION DÉCLARÉE — le H est le quantum de l'amplitude
  elle-même (il SATURE le mode) : son auto-couplage vient des dérivées de
  la fonctionnelle (E'''(v) = 5 m_H^2/v, kappa_lambda = 5/3), PAS de la
  règle de réponse (qui donnerait 2 m_H^2/v, kappa_lambda = 2/3 — FAUX
  et incohérent avec la fonctionnelle). La distinction quartet-porté /
  quantum-du-mode est structurelle.

DISCRIMINANT : la puissance p du couplage (g = 2v (m_V/v)^p) est pincée
par les kappa mesurés — p = 1 et p = 3 morts à >= 10 sigma.
"""
import numpy as np

# ---------------------------------------------------------------- entrées
v = 246.21965
m_W, m_Z = 80.3692, 91.1880          # PDG
KAPPA = {'W': (m_W, 1.05, 0.05), 'Z': (m_Z, 0.99, 0.06)}

print("=" * 74)
print("FACTEUR BOSONIQUE g_HVV = 2 m_V^2/v : ordre de l'action + réponse")
print("=" * 74)

print(f"""
[T1]+[T2]+[T3] — l'assemblage :
   fermion (1er ordre)  : coeff = m_f(a)          -> g = m_f/v   (= pièce 2)
   boson   (2e ordre)   : coeff = m_V^2 (a/v)^2   -> g = 2 m_V^2/v
   photon  (m = 0)      : réponse nulle           -> g_arbre = 0
   H       (le mode)    : E'''(v) = 5 m_H^2/v     -> kappa_lambda = 5/3
   La puissance 2 = l'ordre de l'opérateur d'onde ; le 2 = d(a^2)/da.

Valeurs prédites (masses PDG, v = G_F) :
   g_HWW = 2 m_W^2/v = {2*m_W**2/v:.2f} GeV
   g_HZZ = 2 m_Z^2/v = {2*m_Z**2/v:.2f} GeV
   lambda_WZ = kappa_W/kappa_Z = 1 EXACTEMENT (même règle, même quartet)
   mesuré : {1.05/0.99:.3f} +/- ~0.078  ({abs(1.05/0.99-1)/0.078:.1f} sigma)""")

# ------------------------------------------ discriminant : la puissance p
print("\nDISCRIMINANT — puissance p du couplage, g = 2v (m_V/v)^p :")
print(f"   {'p':>3s} {'kappa_W':>9s} {'écart_W':>9s} {'kappa_Z':>9s} {'écart_Z':>9s}")
for p in (1, 2, 3):
    row = []
    for name, (m, k, s) in KAPPA.items():
        kappa_p = (m / v) ** (p - 2)
        row.append((kappa_p, abs(kappa_p - k) / s))
    print(f"   {p:3d} {row[0][0]:9.3f} {row[0][1]:8.1f}s {row[1][0]:9.3f} "
          f"{row[1][1]:8.1f}s")
print("""   -> p = 2 seul survit (1,0/0,2 sigma) ; p = 1 mort à 40/29 sigma,
   p = 3 mort à 14/10 sigma. La donnée PINCE le carré — donc l'ordre 2
   de l'action bosonique, indépendamment de l'ontologie.

Corollaires et statuts :
 - la règle re-dérive le couplage fermionique m_f/v (cohérence pièce 2 —
   sur-détermination CONDITIONNELLE : elle exigerait que l'échelle
   hadronique soit routée par le barreau EW ; indécidé, la voie
   création-de-paire suffit et reste LA dérivation fermionique) ;
 - couplage d'arbre nul au photon re-dérivé (réponse d'une prise nulle) ;
 - lambda_WZ = 1 exact : toute violation custodiale tuerait [T3] ;
 - l'exception H est STRUCTURELLE (quantum du mode, pas passager du
   quartet) — c'est la même distinction que « l'asthénion sature le mode
   d'amplitude », et elle est REQUISE : la règle de réponse appliquée au
   H donnerait kappa_lambda = 2/3, incompatible avec la fonctionnelle
   (5/3) ET discriminable au FCC ;
 - statut : [T1] généralisation de la pièce 2 (rang identique) ;
   [T2] canonique (Dirac 1er ordre / onde 2e ordre) ; [T3] identification
   au rang du secteur EW (asthénions au VEV). Le facteur bosonique n'est
   PLUS « identifié » : dérivé conditionnel, puissance pincée par les
   données à >= 10 sigma contre toute alternative.""")
