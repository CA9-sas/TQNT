"""Exces GeV du centre galactique (GCE) : execution chiffree de la
confrontation TQNT (23/07).

LE PHENOMENE (non resolu, ~15 ans) : exces diffus de photons gamma
dans les ~10 degres interieurs, spectre pique a E ~ 2-3 GeV
(E^2 dN/dE ~ 1-2e-6 GeV/cm^2/s/sr au pic, Calore+ 2015 ; morphologie
quasi spherique ~ NFW generalise gamma ~ 1,2). Deux interpretations
s'affrontent depuis 2009 : annihilation DM (~30-70 GeV b bbar) vs
population non resolue de pulsars milliseconde (granularite
photonique : soutenue puis contestee -- statut communautaire OUVERT).

TROIS CALCULS TQNT (zero ajustement) :
 [A] sigma_v REQUIS pour que le reservoir TQNT (M = 1,9 GeV) produise
     le flux GCE -- vs le sigma_v TQNT (ecrasement Vassiliev) ;
 [B] MUR CINEMATIQUE : une paire M <= 1,9 GeV ne fait AUCUN photon
     au-dessus de 1,9 GeV -- or le GCE s'etend a ~10+ GeV et pique a
     2-3 GeV : meme a sigma_v arbitraire, le reservoir ne peut pas
     reproduire la FORME ;
 [C] LE SEUL CANAL GAMMA TQNT AU CENTRE : les 6 raies fines des etats
     stricts (1,07-1,64 GeV, F3 levee : x8-13 SOUS les limites de
     raies 2015 au Delta_c = 22 derive) -- des RAIES, pas un
     continuum : le discriminant est une recherche de raies profonde
     dans la region centrale.
Entrees publiees (declarees, pas de donnees Fermi locales) :
J-bar(10 deg, gNFW 1,2) ~ 1e23 GeV^2 cm^-5 sr^-1 ;
E^2 dN/dE|_pic ~ 1,5e-6 GeV cm^-2 s^-1 sr^-1.
"""
import math

ALPHA = 1 / 137.0
SV_THERMAL = 3e-26                 # cm^3/s
SV_TQNT = SV_THERMAL * ALPHA**10   # ecrasement de paire (motif etabli)
M_EDGE = 1.9                       # GeV
JBAR = 1e23                        # GeV^2 cm^-5 sr^-1 (gNFW 1,2, 10 deg)
E2DNDE_PIC = 1.5e-6                # GeV cm^-2 s^-1 sr^-1 a E ~ 2 GeV
E_EVAL = 1.0                       # GeV (sous le mur cinematique)
DNDE_ANN = 5.0                     # ph/GeV/annihilation a 1 GeV (M=1,9 :
                                   # ~10 photons repartis sous 1,9 GeV)
RAIES = [1.07, 1.10, 1.22, 1.35, 1.47, 1.64]


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] LE sigma_v REQUIS (a la masse TQNT, M = %.1f GeV)" % M_EDGE)
    print("=" * 78)
    # dPhi/dE = sigma_v/(8 pi m^2) * Jbar * dN/dE
    dphide_obs = E2DNDE_PIC / E_EVAL**2      # a 1 GeV (ordre du pic)
    sv_req = 8 * math.pi * M_EDGE**2 * dphide_obs / (JBAR * DNDE_ANN)
    print("""  flux GCE a %.0f GeV : dPhi/dE ~ %.1e ph/cm^2/s/sr/GeV
  sigma_v requis = 8 pi m^2 (dPhi/dE)/(J dN/dE) = %.1e cm^3/s
  sigma_v TQNT (alpha^10 x thermique)          = %.1e cm^3/s
  -> DEFICIT : %.0f dex. Le reservoir ne produit RIEN d'observable
  (NB : meme le requis %.0e serait deja en tension CMB/Planck pour
  une masse de 2 GeV -- l'interpretation « DM legere » est coincee
  des deux cotes, TQNT ou pas).""" % (
        E_EVAL, dphide_obs, sv_req, SV_TQNT,
        math.log10(sv_req / SV_TQNT), sv_req))

    print("=" * 78)
    print("[B] LE MUR CINEMATIQUE (independant de sigma_v)")
    print("=" * 78)
    print("""  Une paire du reservoir libere s^(1/2) = 2M = %.1f GeV ; chaque
  photon emporte E < M = %.1f GeV. Or le GCE :
   - pique a 2-3 GeV (au-dessus du mur) ;
   - s'etend jusqu'a >= 10 GeV (tres au-dessus).
  MEME avec un sigma_v arbitrairement grand, le reservoir TQNT ne
  peut pas reproduire la forme du GCE : l'exclusion est DOUBLE
  (taux par [A], forme par [B]). L'interpretation TQNT du GCE est
  donc : PAS de la matiere noire -- population de sources ponctuelles
  (pulsars milliseconde), position sur laquelle la TQNT n'a aucun
  enjeu (elle est compatible avec l'issue astrophysique standard).""" % (
        2 * M_EDGE, M_EDGE))

    print("=" * 78)
    print("[C] LE SEUL CANAL TQNT AU CENTRE : LES 6 RAIES FINES")
    print("=" * 78)
    print("""  Etats stricts (amphichiraux, population secondaire) : raies a
  %s GeV
  (masses ACPR canoniques ; largeur ~0 -- Wr ideaux nuls, aucun
  mecanisme d'etalement). Statut etabli (F3 LEVEE, re-chiffrage
  2026-07-22) : taux x8-13 SOUS les limites de raies Fermi-LAT 2015
  au Delta_c = 22 derive.
  DISCRIMINANT NET POUR LE GCE :
   - si le GCE est TQNT-lie (il ne peut pas l'etre par continuum,
     cf. [A]+[B]) il n'y a QU'UNE signature possible : des RAIES
     fines a 1,07-1,64 GeV dans la region centrale -- une analyse de
     raies profonde 15+ ans y est sensible ;
   - un GCE purement continu qui se resout en sources ponctuelles =
     l'issue predite (pulsars) ;
   - une raie fine detectee AILLEURS qu'a 1,07-1,64 GeV refuterait
     le spectre strict.
  Reserve declaree : le rehaussement eventuel du taux de raies vers
  le bulbe (densite de perles Class E accrue) n'est pas re-derive
  ici -- les taux etablis sont des moyennes ciel entier.""" % (
        "/".join("%.2f" % r for r in RAIES)))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  GCE vs TQNT :
  (1) l'interpretation « annihilation DM » est exclue pour le
      reservoir TQNT par %.0f dex de taux ET par la forme (mur a
      %.1f GeV vs pic 2-3 GeV, queue >= 10 GeV) ;
  (2) prediction : le GCE se resoudra en pulsars milliseconde
      (issue astrophysique -- aucune tension TQNT) ;
  (3) falsifiable deux fois : (i) un GCE demontre lisse ET DM
      (morphologie NFW stricte, pas de granularite) a >= 2 GeV ne
      pourrait PAS etre le reservoir TQNT -> tension pour toute
      l'interpretation sombre TQNT ; (ii) les raies 1,07-1,64 GeV
      sont la SEULE fenetre gamma TQNT au centre galactique --
      cible : analyses de raies Fermi-LAT profondes region centrale.""" % (
        math.log10(8 * math.pi * M_EDGE**2 * (E2DNDE_PIC / E_EVAL**2)
                   / (JBAR * DNDE_ANN) / SV_TQNT), M_EDGE))


if __name__ == "__main__":
    main()
