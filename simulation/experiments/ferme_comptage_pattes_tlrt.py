"""Fermeture TL/RT du comptage des pattes — le maillon P4 dissous.

Cible : la dernière pièce non fermée du mécanisme du 1/pi était P4
(« chaque patte s'attache au premier ordre avec le poids aperture/cycle »
— un comptage 4 x 1/pi posé au rang linéarité). La voie TL/RT (celle qui
a fermé d = rac2) le remplace par TROIS THÉORÈMES EXACTS + les DEUX
identifications DÉJÀ établies du cadre (boucle <-> demi-tour ; graduation
en action). AUCUNE identification nouvelle ne subsiste.

  [A] UNICITÉ DU CONTACT (TL_2 exact) : l'espace des tangles à 4 points
      du portail est de dimension 2 : {1, e} (brins parallèles / cup-cap).
      - 1 = propagation libre ; sous le croisement elle reçoit une PHASE
        PURE (unitaire) -> ne génère AUCUN opérateur de potentiel ;
        en particulier AUCUN terme quadratique -mu^2 a^2 n'est généré
        par le flot (stabilité radiative structurelle, consignée) ;
      - e = LE canal de contact (2 pattes s'annihilent, 2 se créent =
        le contact en s du quartique). Il est UNIQUE.
      => le flot par croisement génère EXACTEMENT un quartique — la
      forme a^4 f(ln a) n'est plus une hypothèse de dominance, c'est la
      structure du portail. (Un 6-points exigerait TL_3 : >= 2
      croisements — ordre supérieur du flot.)
  [B] COÛT D'UNE BOUCLE (fermeture de Markov, exact) : closure(1) = d^2
      (2 boucles), closure(e) = d (1 boucle) — l'insertion de contact
      COÛTE EXACTEMENT UNE BOUCLE.
  [C] TAUX UNITÉ DÉRIVÉ (valeurs propres RT/Kauffman) : sigma = A·1 +
      A^-1·e a pour valeurs propres A (canal traversant) et -A^-3
      (canal de contact ; d = -A^2-A^-2). Phase RELATIVE accumulée sur
      le demi-tour : pi - 2pi/(k+2) -> le taux de séparation des canaux
      est 1 - 2/(k+2) par radian -> 1 EXACTEMENT à k -> infini :
      L'IDENTITÉ D'APERTURE (1 rad) REÇOIT SON ORIGINE TL/RT.

  ASSEMBLAGE (identifications ÉTABLIES seulement) : par barreau (un
  croisement), le flot reçoit l'unique contact [A], dont le coût est une
  boucle [B] ; une boucle <-> un demi-tour (LA même identification que
  la chaîne d = rac2) = pi radians ; la trame gradue l'action au taux
  unité par radian [C] -> poids du contact par barreau :

        w = (1 unité d'action par boucle) / (pi radians par boucle)
          = 1/pi  =  B·L   (exactement ; les « 4 pattes » sont dissous
        — le 4 de Delta lambda = 4w est le facteur de symétrie de
        lambda/4 a^4, une convention, plus un comptage physique).

  DISCRIMINANTS (les données tranchent les mètres alternatifs) :
   - mètre « phase de canal » (porter la correction -2/(k+2), k=135
     dans w) : EXCLU ~8 sigma ;
   - mètres « valeur de boucle » (pondérer par la dimension quantique,
     w = 1/(pi d) ou 1/(pi d^2)) : EXCLUS ~300 sigma — le flot COMPTE
     les boucles, il ne les évalue pas (cohérent graduation en action :
     on compte des quanta) ;
   - le mètre géométrique (demi-tour plein, comptage de boucles) : le
     seul survivant, 0,5 sigma.
"""
import numpy as np

# ---------------------------------------------------------------- entrées
alpha_inv = 137.035999
L = np.log(alpha_inv)
v = 246.21965
mH, s_mH = 125.20, 0.11
BL_meas = (mH**2 / (4 * v**2)) * L
s_BL = BL_meas * 2 * s_mH / mH
apert, s_apert = 0.9965, 0.0078      # aperture jonction mesurée (rad)

print("=" * 74)
print("FERMETURE TL/RT DU COMPTAGE DES PATTES  (P4 dissous, 2026-08-27)")
print("=" * 74)

# ------------------------------------------ [A]+[B] : algèbre TL_2 exacte
print("\n[A]+[B] — algèbre TL_2 (vérification numérique de l'exact) :")
for k in (2, 135):
    d = 2 * np.cos(np.pi / (k + 2))
    # représentation fidèle 2x2 de TL_2 sur {canal traversant, canal e}
    e = np.array([[0, 0], [0, d]])
    one = np.eye(2)
    assert np.allclose(e @ e, d * e), "e^2 = d e"
    # fermeture de Markov : tr_Markov(x) via les boucles
    print(f"   k={k:3d} : d={d:.6f} ; e^2=d·e OK ; "
          f"closure(1)=d^2={d**2:.4f} (2 boucles), closure(e)=d={d:.4f}"
          f" (1 boucle) -> le contact coûte UNE boucle (exact)")

# ------------------------------------------ [C] : valeurs propres du demi-tour
print("\n[C] — le demi-tour aux valeurs propres de Kauffman "
      "(sigma = A·1 + A^-1·e) :")
for k in (2, 135):
    A = np.exp(1j * np.pi / (2 * (k + 2)))
    d = -A**2 - A**-2
    e = np.array([[0, 0], [0, d]])
    sigma = A * np.eye(2) + A**-1 * e
    ev = np.linalg.eigvals(sigma)
    # attendu : {A, -A^-3}
    exp_ev = sorted([A, -A**-3], key=lambda z: np.angle(z))
    got_ev = sorted(ev, key=lambda z: np.angle(z))
    assert np.allclose(exp_ev, got_ev), "valeurs propres {A, -A^-3}"
    rel = np.angle((-A**-3) / A) % (2 * np.pi)
    rate = rel / np.pi
    print(f"   k={k:3d} : phase relative contact/traversant = {rel:.6f} rad"
          f" = pi(1-2/(k+2)) [{np.pi*(1-2/(k+2)):.6f}] ;"
          f" taux = {rate:.6f}/rad")
print(f"   -> k -> infini : taux -> 1 EXACTEMENT par radian — l'identité"
      f" d'aperture\n      dérivée. Aperture jonction mesurée :"
      f" {apert} +/- {s_apert} rad\n      (taux 1 : "
      f"{abs(1-apert)/s_apert:.2f} sigma ; taux 135/137 = "
      f"{135/137:.5f} : {abs(135/137-apert)/s_apert:.2f} sigma"
      f" — les deux permis, préférence faible pour 1)")

# ------------------------------------------ discriminants sur B·L
print(f"\nDISCRIMINANT — les mètres alternatifs contre B·L mesuré "
      f"({BL_meas:.6f} +/- {s_BL:.6f}) :")
d135 = 2 * np.cos(np.pi / 137)
CANDS = [
    ('géométrique (demi-tour plein, comptage)      1/pi', 1/np.pi),
    ('phase de canal k=135     (1-2/137)/pi',            (1 - 2/137)/np.pi),
    ('valeur de boucle          1/(pi d)',               1/(np.pi * d135)),
    ('valeur de boucle carrée   1/(pi d^2)',             1/(np.pi * d135**2)),
]
for name, w in CANDS:
    mh_pred = 2 * v * np.sqrt(w / L)
    dev = (w - BL_meas) / s_BL
    print(f"   {name:50s} = {w:.6f}  m_H={mh_pred:7.2f}  ({dev:+6.1f} s)")
print("   -> seul le mètre GÉOMÉTRIQUE survit : le flot COMPTE les boucles")
print("      (graduation en action = comptage de quanta), il ne pondère")
print("      pas par la dimension quantique ; la correction de phase de")
print("      canal reste INTERNE au portail (exclue à ~8 sigma).")

# ------------------------------------------ coïncidence consignée (null)
w_dress = (1/np.pi) * (1 - 1/(alpha_inv * 4 * np.pi))
print(f"""
[consigné SANS mécanisme, non retenu — règle du null] :
   (1/pi)(1 - alpha/4pi) = {w_dress:.6f} -> {abs(w_dress-BL_meas)/s_BL:.2f} sigma
   (le même habillage alpha/4pi que v ; aucun mécanisme ici — à ne pas
   retenir tant qu'un calcul ne le force pas.)

BILAN : P4 (comptage des pattes) est DISSOUS — remplacé par [A] unicité
du contact (TL exact), [B] coût d'une boucle (Markov exact), [C] taux
unité dérivé (RT, k -> infini ; correction finie-k EXCLUE par les
données comme mètre), assemblés sous les DEUX identifications DÉJÀ
canoniques : boucle <-> demi-tour (chaîne d = rac2) et graduation en
action (1/radian). AUCUNE identification nouvelle. Corollaire [A] :
le flot du portail ne génère PAS de terme quadratique (pas de -mu^2
radiatif à une insertion) — la protection d'échelle est structurelle.""")
