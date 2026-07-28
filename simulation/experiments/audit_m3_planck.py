"""AUDIT DECISIF du « signal m=3 » Planck de juin (23/07).

JUIN (monte_carlo_planck_v4) : exces de puissance azimutale m=3
(|a_{l,3}|^2, coordonnees GALACTIQUES, masque |b|>20, nettoyage par
regression 353/545) declare « ROBUSTE » (r ~ 2,3, p ~ 0) -- avec la
consigne « detection paper ». Le MC (synfast isotrope sur les C_l
observes, meme masque) teste L'ISOTROPIE ; il ne distingue PAS un
signal cosmologique d'un residu d'avant-plan (anisotrope, correle au
plan galactique). QUATRE TESTS TRANCHENT :
  [T0] REPRODUCTION (resolution reduite, NSIDE 256 / LMAX 512) --
       retrouver l'exces de juin dans la config galactique ;
  [T1] ROTATION D'AXE : le m=3 est mesure autour de l'axe du repere.
       Un signal COSMOLOGIQUE n'a pas d'axe prefere -> son exces
       moyen ne peut pas etre special a l'axe galactique. Un residu
       d'avant-plan est ACCROCHE au plan galactique -> l'exces
       s'effondre autour des axes equatorial/ecliptique ;
  [T2] SEVERITE DU MASQUE (|b| > 20/30/40) : un residu de plan
       decroit quand on coupe plus large ; un signal du ciel profond
       reste ;
  [T3] FREQUENCE (143 vs 217 nettoyees pareil ; la carte 100 GHz
       locale est TRONQUEE -- telechargement de juin incomplet,
       1,08/2,01 Go -- exclue et signalee) : un signal CMB est
       achromatique ; un residu de poussiere suit la loi d'emission.
Chaque config a son propre Monte-Carlo (150 simulations isotropes,
pipeline identique). Verdict par le motif conjoint.
"""
import os
import sys
import time

import numpy as np
import healpy as hp

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                    "planck_cmb")
NSIDE = 256
LMAX = 512
N_SIM = 150
BINS = [2, 30, 100, 300, 512]
SEED = 20260723


def load(name):
    import gc
    print("  charge %s" % name, flush=True)
    for attempt in range(3):
        try:
            m = hp.read_map(os.path.join(DATA, name), field=0,
                            dtype=np.float64)
            break
        except ValueError as e:
            print("    lecture ratee (tentative %d : %s), gc + retry"
                  % (attempt + 1, e), flush=True)
            gc.collect()
            time.sleep(5)
    else:
        raise RuntimeError("lecture impossible : %s" % name)
    if hp.npix2nside(len(m)) != NSIDE:
        m = hp.ud_grade(m, NSIDE)
    gc.collect()
    return m.astype(np.float64)


def make_mask(b_cut):
    theta, _ = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    b = 90 - np.degrees(theta)
    return (np.abs(b) > b_cut).astype(float)


def clean_map(m, m353, m545, mask):
    sel = mask > 0
    X = np.column_stack([m353[sel], m545[sel]])
    y = m[sel]
    a, b = np.linalg.solve(X.T @ X, X.T @ y)
    return m - a * m353 - b * m545


def masked_alm(m, mask):
    mm = m * mask
    mm -= mm[mask > 0].mean()
    mm *= mask
    return hp.map2alm(mm, lmax=LMAX)


def power_m(alm, m):
    p = np.zeros(LMAX + 1)
    for l in range(max(3, m), LMAX + 1):
        p[l] = np.abs(alm[hp.Alm.getidx(LMAX, l, m)]) ** 2
    return p


def binned(p):
    return np.array([p[lo:hi].sum() for lo, hi in zip(BINS[:-1], BINS[1:])])


def run_config(m_obs, mask, label, rotators=None, rng=None):
    """MC complet d'une config ; rotators : {nom: Rotator} pour mesurer
    le m=3 autour d'autres axes (obs ET sims tournes pareil)."""
    print("\n== CONFIG %s ==" % label, flush=True)
    alm_obs = masked_alm(m_obs, mask)
    cl_obs = hp.alm2cl(alm_obs)
    axes = {"galactique": None}
    if rotators:
        axes.update(rotators)
    obs = {}
    for ax, rot in axes.items():
        a = alm_obs
        if rot is not None:
            a = alm_obs.copy()
            hp.rotate_alm(a, *rot)
        obs[ax] = binned(power_m(a, 3))
    sims = {ax: np.zeros((N_SIM, len(BINS) - 1)) for ax in axes}
    t0 = time.time()
    for i in range(N_SIM):
        m_sim = hp.synfast(cl_obs, NSIDE, lmax=LMAX)
        alm_sim = masked_alm(m_sim, mask)
        for ax, rot in axes.items():
            a = alm_sim
            if rot is not None:
                a = alm_sim.copy()
                hp.rotate_alm(a, *rot)
            sims[ax][i] = binned(power_m(a, 3))
    print("  %d sims en %.0f s" % (N_SIM, time.time() - t0), flush=True)
    out = {}
    for ax in axes:
        r = obs[ax] / sims[ax].mean(axis=0)
        p = (sims[ax] >= obs[ax]).mean(axis=0)
        out[ax] = (r, p)
        rs = "  ".join("%.2f" % x for x in r)
        ps = "  ".join("%.3f" % x for x in p)
        print("  axe %-12s r par bin [%s]  p [%s]" % (ax, rs, ps),
              flush=True)
    return out


def main():
    print(__doc__, flush=True)
    np.random.seed(SEED)
    m143 = load("HFI_SkyMap_143_2048_R3.01_full.fits")
    m353 = load("HFI_SkyMap_353_2048_R3.01_full.fits")
    m545 = load("HFI_SkyMap_545_2048_R3.01_full.fits")
    m217 = load("HFI_SkyMap_217_2048_R3.01_full.fits")

    mask20 = make_mask(20)
    c143 = clean_map(m143, m353, m545, mask20)

    # rotations galactique -> equatorial / ecliptique (angles d'Euler zyz)
    rots = {"equatorial": hp.rotator.coordsys2euler_zyz(coord=["G", "C"]),
            "ecliptique": hp.rotator.coordsys2euler_zyz(coord=["G", "E"])}

    # T0 + T1 : reproduction + axes (meme MC)
    res_axes = run_config(c143, mask20, "143 nettoyee, |b|>20 (T0+T1 axes)",
                          rotators=rots)

    # T2 : masque 30 / 40 (axe galactique)
    res_m30 = run_config(clean_map(m143, m353, m545, make_mask(30)),
                         make_mask(30), "143 nettoyee, |b|>30 (T2)")
    res_m40 = run_config(clean_map(m143, m353, m545, make_mask(40)),
                         make_mask(40), "143 nettoyee, |b|>40 (T2)")

    # T3 : 217 nettoyee pareil, |b|>20 (100 GHz : fichier tronque, exclu)
    res_217 = run_config(clean_map(m217, m353, m545, mask20), mask20,
                         "217 nettoyee, |b|>20 (T3)")

    print("\n" + "=" * 78, flush=True)
    print("SYNTHESE DE L'AUDIT (r moyen des bins l=30-512, le cœur"
          " de l'exces de juin)")
    print("=" * 78)

    def rmean(res, ax="galactique"):
        return float(np.mean(res[ax][0][1:]))

    rows = [
        ("143 |b|>20, axe galactique (juin)", rmean(res_axes)),
        ("143 |b|>20, axe equatorial", rmean(res_axes, "equatorial")),
        ("143 |b|>20, axe ecliptique", rmean(res_axes, "ecliptique")),
        ("143 |b|>30, axe galactique", rmean(res_m30)),
        ("143 |b|>40, axe galactique", rmean(res_m40)),
        ("217 |b|>20, axe galactique", rmean(res_217)),
    ]
    for lab, r in rows:
        print("  %-38s r = %.2f" % (lab, r))

    r_gal = rows[0][1]
    r_axes = (rows[1][1] + rows[2][1]) / 2
    r_40 = rows[4][1]
    r_217 = rows[5][1]
    print("""
LECTURE :
  - axe : galactique %.2f vs autres axes %.2f -> %s
  - masque : |b|>20 %.2f vs |b|>40 %.2f -> %s
  - frequence : 143 %.2f vs 217 %.2f -> %s""" % (
        r_gal, r_axes,
        "SPECIFIQUE A L'AXE GALACTIQUE (avant-plan)" if r_gal > 1.3 and
        r_axes < 0.75 * r_gal else "pas specifique a l'axe",
        r_gal, r_40,
        "DECROIT avec la coupure (avant-plan)" if r_40 < 0.75 * r_gal
        else "stable sous la coupure",
        r_gal, r_217,
        "CHROMATIQUE (avant-plan)" if abs(r_217 - r_gal) > 0.3 * r_gal
        else "achromatique"))
    n_fg = sum([
        r_gal > 1.3 and r_axes < 0.75 * r_gal,
        r_40 < 0.75 * r_gal,
        abs(r_217 - r_gal) > 0.3 * r_gal,
    ])
    print("VERDICT : %d/3 signatures d'avant-plan presentes." % n_fg)
    if n_fg >= 2:
        print("""  => RESIDU D'AVANT-PLAN CONFIRME : le « signal m=3 » de juin
  n'est pas cosmologique. La requalification du portefeuille est
  VALIDEE PAR LE CALCUL ; aucune detection, pas de paper. Coherent
  avec la prediction V6 (zero empreinte Z3 : domaines sous-horizon).""")
    elif n_fg == 1:
        print("""  => INDETERMINE : une seule signature -- approfondir (gabarits
  d'avant-plans par region, fonction de faisceau) avant tout verdict.""")
    else:
        print("""  => LE SIGNAL SURVIT AUX TROIS TESTS : surprise majeure a
  investiguer serieusement (et en tension avec la prediction V6
  d'absence d'empreinte -- dans les deux sens, pas de paper hatif).""")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
