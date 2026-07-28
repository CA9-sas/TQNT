"""Pincer r_link pour fermer les +6.6% de la chaine beta_otimes (fluxoide) ?

DOUBLE CONTRAINTE : r_link est sur-determine --
  (i)  chaine du MOMENT   : mu_p - mu_n = (r/lambdabar_C(p)) mu_N a v=c
       -> r_mu = 4.70589 x 0.210309 = 0.9898 fm (PDG, exacte par construction) ;
  (ii) chaine du FLUXOIDE : E(dLk=1) = Phi_chi^2/(2 mu0 r [ln(8r/a)-2]) = 1.293 MeV
       -> r_E = 1.040 fm (a = d_tube = 0.209 fm).
Ecart r_E / r_mu = +5.1% : UN filament unique a v=c ne peut PAS satisfaire les deux.

CANDIDATS de pincement testes (chacun contre LES DEUX contraintes) :
  ristretto des echelles derivees : hbar*c/Lambda_chi, hbar*c/E_K(200), 5*d_tube,
  cercle-ropelength du trefle, moyenne geometrique (numerologique, control).

RECONCILIATIONS possibles (testees) :
  (a) v < c : v/c = r_mu/r_E ~ 0.952 reconcilie exactement (le moment porte v*r,
      l'inductance porte r seul) -- mais v n'est pas pince par la theorie.
  (b) coherence croisee du courant induit : le moment du courant de fluxoide
      mu_ind = (Phi/L)*pi*r^2 vaut pi/(ln(8r/a)-2) = 1.92x le moment de la chaine (i)
      -> les deux chaines ne partagent PAS un meme filament simple de toute facon.
  (c) courant distribue radialement : a <r^2> fixe (moment), l'etalement REDUIT L
      (rayon moyen plus petit + section effective plus grande) -> E AUGMENTE :
      mauvaise direction, refute comme fermeture.

VERDICT attendu : r_link ne se pince pas sans casser la chaine du moment a ~5% ;
la tension 5% entre les deux determinations est REELLE (frontiere actuelle).
La resolution de principe est l'outillage d'entrelacs ideal (mesurer le circuit
d'enlacement reel dans le 3_1+7_2+7_2 relaxe).
"""
import math

HBARC = 197.3269804          # MeV.fm
ALPHA = 1.0 / 137.0
H_OVER_E = 4.135667696e-15   # Wb
MU0 = 4e-7 * math.pi
J_PER_MEV = 1.602176634e-13
FM = 1e-15

D_TUBE = 0.209               # fm
LAMBDABAR_P = 0.210309       # fm (Compton reduite du proton)
DMU_PDG = 4.70589            # mu_p - mu_n en mu_N
TARGET = 1.293               # MeV
LAMBDA_CHI = 208.8            # MeV
SQSIG = 417.65
R_ROPE_31 = 16.372 * 2 * D_TUBE / (2 * math.pi)   # cercle a la ropelength du trefle

PHI = ALPHA * H_OVER_E


def E_fluxoid(r_fm, a_fm=D_TUBE):
    L = MU0 * r_fm * FM * (math.log(8 * r_fm / a_fm) - 2)
    return PHI**2 / (2 * L) / J_PER_MEV


def r_star(a_fm=D_TUBE):
    lo, hi = 0.5, 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if E_fluxoid(mid, a_fm) > TARGET:
            lo = mid
        else:
            hi = mid
    return lo


def run():
    r_mu = DMU_PDG * LAMBDABAR_P
    r_E = r_star()
    print("=== Double contrainte sur r_link ===")
    print(f"  (i)  moment   : r_mu = {r_mu:.4f} fm (exacte par construction)")
    print(f"  (ii) fluxoide : r_E  = {r_E:.4f} fm (E = {TARGET} MeV exact)")
    print(f"  tension r_E/r_mu - 1 = {r_E/r_mu - 1:+.1%} -> un filament unique a v=c ne peut pas tout faire\n")

    print("=== Candidats de pincement (E vs cible ; moment implique a v=c vs PDG) ===")
    cands = [
        ("r_mu (chaine moment, statu quo)", r_mu),
        ("hbar c / Lambda_chi",             HBARC / LAMBDA_CHI),
        ("hbar c / 200 MeV (E_K ronde)",    HBARC / 200.0),
        ("5 x d_tube",                      5 * D_TUBE),
        ("cercle ropelength 3_1",           R_ROPE_31),
        ("sqrt(r_mu x R_rope)  [controle numerologique]", math.sqrt(r_mu * R_ROPE_31)),
        ("r_E (cible exacte)",              r_E),
    ]
    for name, r in cands:
        E = E_fluxoid(r)
        dmu = r / LAMBDABAR_P
        print(f"  {name:46}: r = {r:6.4f} fm ; E = {E:6.3f} MeV ({(E/TARGET-1)*+100:+5.1f}%) ;"
              f" dmu = {dmu:5.2f} mu_N ({(dmu/DMU_PDG-1)*100:+5.1f}%)")

    print("\n=== Reconciliations ===")
    v_over_c = r_mu / r_E
    print(f"  (a) v < c : v/c = r_mu/r_E = {v_over_c:.4f} (beta^2 = {v_over_c**2:.3f})")
    print(f"      -> les DEUX contraintes exactes avec r = {r_E:.4f} fm et v = {v_over_c:.3f} c ;")
    print(f"      mais v n'est pince par rien (la chaine du moment posait v=c par principe).")
    ratio = math.pi / (math.log(8 * r_mu / D_TUBE) - 2)
    print(f"  (b) moment du courant induit / moment de la chaine (i) = pi/(ln(8r/a)-2) = {ratio:.2f}")
    print(f"      -> pas le meme filament : la geometrie effective differe entre les deux reponses.")
    print(f"  (c) courant distribue a <r^2> fixe : r_moyen diminue ET la section effective")
    print(f"      augmente -> L diminue -> E augmente : REFUTE comme fermeture du +6.6%.")

    print("\n=== VERDICT ===")
    print("  r_link ne se pince pas : tout choix fermant le fluxoide casse le moment a ~+5%")
    print("  (et reciproquement). La tension de 5% entre les deux determinations est REELLE.")
    print("  Voie de principe : outillage d'ENTRELACS ideal (contraintes inter-composantes)")
    print("  pour mesurer le circuit d'enlacement du 3_1+7_2+7_2 relaxe, au lieu d'un cercle.")


if __name__ == "__main__":
    run()
