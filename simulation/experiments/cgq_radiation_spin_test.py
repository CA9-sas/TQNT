"""Test : la rotation d'un noeud emet-elle des 'ondelettes de CGQ' perturbant l'experience ?

La CGQ est le champ metrique emergent (secteur gravitationnel ; memoire champ-gravitationnel-
quantique). Un coeur anisotrope en rotation est une source variable -> il rayonne comme un
quadrupole de masse : des ondelettes de CGQ = ondes gravitationnelles a l'echelle d'UNE particule.

RESULTAT (estimation electron) : rayonnement gravitationnel, donc supprime de ~10^43.
  - P ~ 7e-37 W ; temps pour rayonner l'energie de repos ~ 1e23 s = 2.6e5 x age de l'Univers.
  - alpha_G/alpha_EM ~ 2.4e-43 : le canal EM (moment magnetique) est 10^43 fois plus fort.
  - pour un lepton (3_1, C_{q>=3}) le quadrupole m=2 est NUL par symetrie -> encore + supprime.
=> les ondelettes de CGQ NE perturbent PAS l'experimentation (inobservables) et ne sont PAS
   l'origine du spin (secteur gravitationnel mass-gradue, meme mur que le deficit h~10^-45).
   Le vrai 'la rotation perturbe l'experience' EST reel mais EM (moment magnetique = observable
   du spin, Stern-Gerlach/g=2), 10^43 fois plus fort -- pas la CGQ.
"""
import numpy as np

G, c, hbar = 6.674e-11, 2.998e8, 1.055e-34
me, mc2 = 9.109e-31, 8.187e-14           # electron : masse (kg), energie repos (J)
tHubble = 4.35e17                        # s


def estimate(eps=1.0):
    lamC = hbar / (me * c)               # Compton reduite
    wC = mc2 / hbar                       # frequence interne (Compton/zitterbewegung)
    I = me * lamC ** 2                    # moment d'inertie ~ m lambda_C^2
    P = (32 / 5) * (G / c ** 5) * I ** 2 * eps ** 2 * wC ** 6   # Einstein quadrupole
    tau = mc2 / P
    alphaG = G * me ** 2 / (hbar * c)
    alphaEM = 1 / 137.036
    return dict(lamC=lamC, wC=wC, I=I, P=P, tau=tau, ratio=alphaG / alphaEM)


def run():
    r = estimate()
    print(f"lambda_C={r['lamC']:.2e} m  w_C={r['wC']:.2e} rad/s  I={r['I']:.2e} kg m^2")
    print(f"P(ondelettes-CGQ, quadrupole eps=1) = {r['P']:.2e} W")
    print(f"tau = E_repos/P = {r['tau']:.2e} s = {r['tau']/tHubble:.2e} x age Univers")
    print(f"alpha_G/alpha_EM = {r['ratio']:.2e}  (EM = moment magnetique = 10^43 x plus fort)")
    print("Lepton torique : quadrupole m=2 NUL (symetrie C_{q>=3}) -> octupole m=3, + supprime.")
    print("CONCLUSION : ondelettes-CGQ inobservables (grav.), pas l'origine du spin. Q REFUTEE.")


if __name__ == "__main__":
    run()
