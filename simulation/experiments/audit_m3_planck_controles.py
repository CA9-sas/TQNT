"""AUDIT m=3 -- CONTROLES DECISIFS T4/T5 (23/07, suite).

Le premier passage (audit_m3_planck.py) montre que l'exces m=3 SURVIT
aux tests axe/masque/frequence (r = 2,2-4,3, p ~ 0). AVANT toute
conclusion, deux controles qui peuvent tuer le signal :
  [T4] SPECIFICITE EN m : mesurer r(m) pour m = 1..8 (meme pipeline,
       meme MC). Si TOUS les m sont eleves, le null (ciel isotrope
       gaussien) est simplement FAUX pour une carte reelle (bruit
       anisotrope, residus) -- aucune signature Z3 ; le « m=3 » de
       juin n'etait special que parce qu'on ne regardait que lui.
  [T5] NULL A BRUIT ANISOTROPE : le bruit Planck n'est PAS isotrope
       (profondeur en anneaux ecliptiques, trous aux poles). La
       colonne II_cov de la carte 143 donne la variance par pixel ->
       sims = synfast(C_l signal) + bruit gaussien anisotrope reel.
       Si r retombe a ~1, l'exces etait le BRUIT ANISOTROPE mal
       modelise par le null isotrope de juin.
"""
import os
import sys
import time
import gc

import numpy as np
import healpy as hp

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                    "planck_cmb")
NSIDE = 256
LMAX = 512
N_SIM = 150
BINS = [2, 30, 100, 300, 512]
SEED = 20260724
F143 = "HFI_SkyMap_143_2048_R3.01_full.fits"
F353 = "HFI_SkyMap_353_2048_R3.01_full.fits"
F545 = "HFI_SkyMap_545_2048_R3.01_full.fits"


def load_field(name, field, power=0):
    print("  charge %s [champ %d]" % (name, field), flush=True)
    m = hp.read_map(os.path.join(DATA, name), field=field,
                    dtype=np.float64)
    if hp.npix2nside(len(m)) != NSIDE:
        m = hp.ud_grade(m, NSIDE, power=power)
    gc.collect()
    return m


def make_mask(b_cut):
    theta, _ = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    b = 90 - np.degrees(theta)
    return (np.abs(b) > b_cut).astype(float)


def masked_alm(m, mask):
    mm = m * mask
    mm -= mm[mask > 0].mean()
    mm *= mask
    return hp.map2alm(mm, lmax=LMAX)


def power_of_m(alm, m):
    p = 0.0
    for l in range(max(30, m), LMAX + 1):   # cœur l = 30-512
        p += np.abs(alm[hp.Alm.getidx(LMAX, l, m)]) ** 2
    return p


def main():
    print(__doc__, flush=True)
    np.random.seed(SEED)
    m143 = load_field(F143, 0)
    m353 = load_field(F353, 0)
    m545 = load_field(F545, 0)
    # variance II par pixel (moyenne de variances en degradant :
    # var(moyenne de N pixels) = var/N -> power=2 la gere en ud_grade)
    var143 = load_field(F143, 4, power=2)
    var143 = np.clip(var143, 0, None)
    sig143 = np.sqrt(var143)

    mask = make_mask(20)
    sel = mask > 0
    X = np.column_stack([m353[sel], m545[sel]])
    a, b = np.linalg.solve(X.T @ X, X.T @ m143[sel])
    c143 = m143 - a * m353 - b * m545

    alm_obs = masked_alm(c143, mask)
    cl_obs = hp.alm2cl(alm_obs)

    MS = list(range(1, 9))
    obs_m = {m: power_of_m(alm_obs, m) for m in MS}

    # bruit anisotrope : C_l signal = C_l obs - C_l bruit moyen
    npix = hp.nside2npix(NSIDE)
    cl_noise_mean = 4 * np.pi * np.mean((sig143[sel]) ** 2) * (
        sel.mean()) / npix * np.ones(LMAX + 1)
    cl_signal = np.clip(cl_obs - cl_noise_mean, 0, None)

    print("\n== T4 + T5 : MC jumeaux (isotrope vs bruit anisotrope) ==",
          flush=True)
    iso = {m: np.zeros(N_SIM) for m in MS}
    aniso = {m: np.zeros(N_SIM) for m in MS}
    t0 = time.time()
    for i in range(N_SIM):
        # null de juin : ciel isotrope aux C_l observes
        s_iso = hp.synfast(cl_obs, NSIDE, lmax=LMAX)
        a_iso = masked_alm(s_iso, mask)
        # null corrige : signal isotrope + bruit gaussien ANISOTROPE reel
        s_ani = hp.synfast(cl_signal, NSIDE, lmax=LMAX) \
            + np.random.randn(npix) * sig143
        a_ani = masked_alm(s_ani, mask)
        for m in MS:
            iso[m][i] = power_of_m(a_iso, m)
            aniso[m][i] = power_of_m(a_ani, m)
    print("  %d sims x2 en %.0f s" % (N_SIM, time.time() - t0), flush=True)

    print("\n  m   r_iso (null juin)   r_aniso (bruit reel)   p_aniso")
    r_iso_all, r_ani_all = [], []
    for m in MS:
        r_i = obs_m[m] / iso[m].mean()
        r_a = obs_m[m] / aniso[m].mean()
        p_a = (aniso[m] >= obs_m[m]).mean()
        r_iso_all.append(r_i)
        r_ani_all.append(r_a)
        tag = " <-- le m de juin" if m == 3 else ""
        print("  %d   %6.2f              %6.2f              %.3f%s" % (
            m, r_i, r_a, p_a, tag), flush=True)

    r3_iso = r_iso_all[2]
    others_iso = [r for j, r in enumerate(r_iso_all) if j != 2]
    r3_ani = r_ani_all[2]
    print("\n" + "=" * 78)
    print("LECTURE")
    print("=" * 78)
    spec = r3_iso > 1.5 * np.median(others_iso)
    print("""  [T4] specificite : r_iso(m=3) = %.2f vs mediane des autres m
       = %.2f -> %s""" % (
        r3_iso, float(np.median(others_iso)),
        "m=3 SPECIFIQUE" if spec else
        "PAS SPECIFIQUE : tous les m sont eleves -- le null isotrope"
        " est simplement faux pour une carte reelle (anisotropies"
        " generiques), il n'y a AUCUNE signature Z3 particuliere"))
    print("""  [T5] bruit anisotrope : r(m=3) passe de %.2f (null juin) a
       %.2f sous le null a bruit reel -> %s""" % (
        r3_iso, r3_ani,
        "l'exces EST le bruit anisotrope (signal mort)" if r3_ani < 1.3
        else "un exces subsiste au-dela du bruit anisotrope"))
    print("""
VERDICT COMBINE : le « signal m=3 » de juin est-il une detection ?
  - s'il n'est PAS specifique (T4) OU s'il meurt sous le bruit reel
    (T5) : NON -- artefact de null mal pose ; la requalification du
    portefeuille tient (par un chemin different de l'avant-plan pur).
  - s'il est specifique ET survit : anomalie d'anisotropie a
    documenter serieusement (toujours PAS une signature TQNT : la V6
    predit zero empreinte -- ce serait une anomalie contre LCDM ET
    contre la TQNT a la fois).""")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
