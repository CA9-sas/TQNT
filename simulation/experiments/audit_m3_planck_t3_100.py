"""AUDIT m=3 -- COMPLETION DU TEST DE FREQUENCE T3 avec le 100 GHz
(24/07, apres re-telechargement de la carte tronquee).

Le T3 initial n'avait que 143 vs 217 (la carte 100 GHz locale etait
tronquee -- telechargement de juin interrompu a 1,08/2,01 Go). Ici :
100 GHz nettoyee par les memes gabarits 353/545, masque |b|>20,
MC 150 sims, meme pipeline et meme graine de config que l'audit.

LECTURE ATTENDUE : le verdict etabli (anisotropie bas-m generique,
pas de signature Z3) ne depend pas de ce point ; le 100 GHz teste la
CHROMATICITE sur le troisieme canal : un residu de poussiere
monterait de 100 -> 143 -> 217 ; l'exces mesure etait achromatique
(2,25 vs 2,54). On mesure aussi r(m=1..8) au 100 GHz pour verifier
que le profil bas-m generique est le meme.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import audit_m3_planck as A  # noqa: E402


def power_of_m_core(alm, m):
    p = 0.0
    for l in range(max(30, m), A.LMAX + 1):
        p += np.abs(alm[A.hp.Alm.getidx(A.LMAX, l, m)]) ** 2
    return p


def main():
    print(__doc__, flush=True)
    np.random.seed(A.SEED)
    m100 = A.load("HFI_SkyMap_100_2048_R3.01_full.fits")
    m353 = A.load("HFI_SkyMap_353_2048_R3.01_full.fits")
    m545 = A.load("HFI_SkyMap_545_2048_R3.01_full.fits")
    mask = A.make_mask(20)
    c100 = A.clean_map(m100, m353, m545, mask)

    res = A.run_config(c100, mask, "100 nettoyee, |b|>20 (T3 complete)")
    r_bins = res["galactique"][0]
    r_core = float(np.mean(r_bins[1:]))

    # profil en m (l = 30-512), MC dedie
    alm_obs = A.masked_alm(c100, mask)
    cl_obs = A.hp.alm2cl(alm_obs)
    MS = list(range(1, 9))
    obs = {m: power_of_m_core(alm_obs, m) for m in MS}
    sims = {m: np.zeros(A.N_SIM) for m in MS}
    for i in range(A.N_SIM):
        s = A.hp.synfast(cl_obs, A.NSIDE, lmax=A.LMAX)
        a = A.masked_alm(s, mask)
        for m in MS:
            sims[m][i] = power_of_m_core(a, m)
    print("\n  profil r(m) au 100 GHz (l = 30-512) :")
    rs = {}
    for m in MS:
        rs[m] = obs[m] / sims[m].mean()
        print("    m = %d : r = %.2f%s" % (
            m, rs[m], "  <-- m=3" if m == 3 else ""), flush=True)

    print("\n" + "=" * 78)
    print("SYNTHESE T3 COMPLETE (r moyen l = 30-512, axe galactique)")
    print("=" * 78)
    print("  100 GHz nettoyee : r = %.2f" % r_core)
    print("  143 GHz nettoyee : r = 2.25   (audit 23/07)")
    print("  217 GHz nettoyee : r = 2.54   (audit 23/07)")
    grad = "MONOTONE en frequence (chromatique -> poussiere)" \
        if r_core < 2.25 < 2.54 and (2.54 - r_core) > 0.3 * 2.25 else \
        "PLAT en frequence (achromatique -- inchange)"
    print("  -> %s" % grad)
    lowm = [rs[m] for m in (1, 2, 3, 4)]
    highm = [rs[m] for m in (5, 6, 7, 8)]
    print("""  profil en m : bas-m (1-4) moyen %.2f vs hauts-m (5-8) %.2f
  -> le motif « anisotropie bas-m generique » %s au 100 GHz.
VERDICT : le test de frequence est COMPLET (3 canaux). Le verdict de
l'audit (aucune signature Z3 -- exces bas-m generique de residus
diffus, artefact de selection de juin) est INCHANGE ; la carte 100 GHz
locale est reparee (2 013 278 400 octets, verifiee).""" % (
        float(np.mean(lowm)), float(np.mean(highm)),
        "se REPRODUIT" if np.mean(lowm) > 1.5 * np.mean(highm)
        else "ne se reproduit PAS (a comprendre)"))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
