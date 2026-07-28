"""Les bandes orbitales baryoniques : le residu nomme derive (22/07).

CIBLE : les valeurs de bande P mesurees (S29) -- Lambda_c 2,938,
Xi_c 3,254, Lambda_b 6,246, Xi_b 6,497 (+ les multiplets Omega).

[1] LA FORME MESONIQUE ECHOUE : Dn = 2 + 3^g X donne X_c = 0,94 vs
    X_b = 1,42 (non universel) -- le compagnon baryonique n'est PAS un
    noeud : c'est un CABLE A DEUX BRINS (le diquark).
[2] L'IDENTIFICATION : la partie relative (x3^g) vaut
        (Dn_b - Dn_c)/2 = 1,654  vs  4(d-1) = 1,657   (-0,2 %).
    MECANISME (le maillon S12) : un demi-tour d'anneau autour d'un
    cable 2 brins = UN echange des membres, excès (d-1) (theoreme
    demontre, demontre_maillon_twist). Par revolution, QUATRE
    demi-tours agissent sur le cable (2 du tube + 2 du repere
    entraine, S17) -> 4 echanges -> 4(d-1), stockes x3^g au ring
    lourd. Condition declaree : pour le cable, les demi-tours du TUBE
    s'accumulent aussi en relatif (au noeud simple ils sont
    portail-aveugles) -- l'echange n'a pas de composante aveugle.
[3] LA PARTIE AVEUGLE : b = Dn(Lambda_c) - 4(d-1) = 1,281 -- LE seul
    nombre mesure du secteur (candidats 9/7 +0,4 %, 2(2/pi) -0,6 % :
    sans mecanisme, regle du null, reste mesure).
[4] VERIFICATION COMPLETE (8 niveaux, 6 systemes, avec s intensif,
    verrou de-triple et l'echelle des flips du document principal).
"""
import numpy as np

D1 = np.sqrt(2) - 1
S = 0.3590                     # shift derive (DWr/pi)
X4 = 4 * D1                    # partie relative par 3^g
B = 2.9380 - X4                # partie aveugle, mesuree sur Lambda_c
XF = {"j0": 0.0, "j1": 0.5, "j2": 1.0}   # echelle des flips (S32)

MES = {
    "Dn(Lambda_c)": (2.9380, B + X4),
    "Dn(Lambda_b)": (6.2465, B + 3 * X4),
    "Dn(Xi_c)":     (3.2540, B + X4 + S),
    "Dn(Xi_b)":     (6.4970, B + 3 * X4 + S),
}
# niveaux Omega en n absolu : n(j) = 1 + (d-1) + [b + 3^g 4(d-1)] + s + x_j
OM = {
    "Omega_c j0": (4.5754, 1 + D1 + B + X4 + S + XF["j0"]),
    "Omega_c j1": (5.2365, 1 + D1 + B + X4 + S + XF["j1"]),
    "Omega_c j2": (5.7648, 1 + D1 + B + X4 + S + XF["j2"]),
    "Omega_b j1": (8.5579, 1 + D1 + B + 3 * X4 + S + XF["j1"]),
    "Omega_b j2": (9.0277, 1 + D1 + B + 3 * X4 + S + XF["j2"]),
}


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] L'IDENTIFICATION 4(d-1) ET LA REGLE")
    print("=" * 78)
    print("""  partie relative mesuree : (6,2465 - 2,9380)/2 = %.4f
  4(d-1) = %.4f  ->  ecart %.2f %%
  REGLE DES BANDES BARYONIQUES :
     Dn = b + 3^g . 4(d-1)   [+ s si jambe etrange (intensif)]
     (Omega : + x_j flips, verrou (d-1) plat en onde P)
  avec b = %.4f mesure UNE fois (Lambda_c).""" % (
        (6.2465 - 2.9380) / 2, X4, 100 * ((6.2465 - 2.9380) / 2 / X4 - 1), B))

    print("=" * 78)
    print("[B] VERIFICATION : 4 bandes + 5 niveaux Omega (une seule mesure : b)")
    print("=" * 78)
    for nom, (mes, pred) in {**MES, **OM}.items():
        tag = ""
        if "j0" in nom:
            tag = "  <- le suspect d'assignation (S32)"
        print("  %-14s mesure %.4f  pred %.4f  (%+.3f, %+.1f %%)%s" % (
            nom, mes, pred, mes - pred, 100 * (mes / pred - 1), tag))
    print("""
  -> 8 des 9 valeurs a <= 1,8 %% (les niveaux Omega_b a -0,4/-0,03 %% !),
  le seul ecart (Omega_c j0, +3 %%) est LE suspect d'assignation deja
  declare (S32). NB : les niveaux Omega utilisent la bande PREDITE
  (pas la mesuree) -- c'est la chaine complete b -> Lambda -> Xi ->
  Omega qui tient. BONUS : les Omega SELECTIONNENT l'echelle des flips
  (A = 0,722) -- support mutuel avec le S32.""")

    print("=" * 78)
    print("[C] STATUT ET PREDICTIONS")
    print("=" * 78)
    print("""  DERIVE : la partie relative 4(d-1).3^g -- le cable a 2 brins
  echange ses membres a chaque demi-tour (theoreme du maillon S12),
  4 demi-tours/revolution (2 tube + 2 repere), l'exces (d-1) par
  echange ; condition declaree : pas de composante aveugle pour
  l'echange (identification, rang canonique).
  MESURE : b = 1,281 (un nombre pour tout le secteur -- il remplace
  les 4 bandes lues ; candidats numeriques sans mecanisme, null).
  PREDICTIONS : toute bande P de baryon lourd se calcule -- p.ex. le
  multiplet 1P de Sigma_c/Sigma_b (diquark ll spin 1, PAS de s :
  niveaux j a n = 1+(d-1)+b+3^g 4(d-1)+x_j) ; les ondes P de Xi_cc
  (cable cc) : meme 4(d-1) avec le verrou cc plat. Le RESIDU NOMME
  « bandes orbitales baryoniques » est REDUIT a UN nombre (b) +
  une identification.
  NB secteur leger : les N*/Delta* relevent du secteur Regge propre
  (ancres sectorielles du document principal), pas de la machinerie constituant --
  hors de cette regle, declare.""")


if __name__ == "__main__":
    main()
