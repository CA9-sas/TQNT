"""Ponderation en action intra-chemin + processus directionnel (23/07).

VISION UTILISATEUR (a implementer) : des ondes se glissent dans les
rails en sens INVERSES ; des ondes PERPENDICULAIRES aux rails
continuent de se propager normalement ; par MAGNETISME, ca genere des
ENFILEMENTS, CASSE les ondes et REPRODUIT le processus (cycle).

TRADUCTION DANS LE CADRE (dichotomie qui DEFINIT alpha_cond) :
 - sens inverses dans un rail -> collision frontale certaine (meme
   canal 1D) -> courants antiparalleles se repoussent -> CASSE =
   SCISSION ;
 - onde perpendiculaire traversant au noeud + capture magnetique ->
   ENROULEMENT = ENFILEMENT (Delta Lk = +-1) = PASSAGE ;
 - le cycle casse -> re-faufilement = la cascade etablie (branchement
   derive scission 0,628 / passage 0,372 -> ratio cible
   passages/scissions = 2 e^{-alpha} = 0,592-0,603).

DEUX VOLETS :
[A] PONDERATION MAGNETIQUE INTRA-CHEMIN : poids de forme
    w = e^{kappa U}, U = somme de Neumann normalisee
    (t_i . t_j / r_ij) -- les courants co-diriges proches s'attirent
    (formes solenoidales compactes favorisees), les antiparalleles se
    repoussent. Echantillon unique, reponderation a l'analyse :
    mesure de la REPONSE alpha(kappa) et du kappa* requis pour
    atteindre la fenetre cible [1,199, 1,220].
[B] COMPTAGE DIRECTIONNEL DES RENCONTRES (gaz d'ondes isotrope sur
    le reseau de rails, 6 directions) : la frontale meme-rail est
    CERTAINE (canal 1D), la perpendiculaire exige une COINCIDENCE au
    noeud (fenetre de capture delta) ->
       passages/scissions = 4 delta_eff = 2 delta x (facteur vitesse)
    inversee sur la cible -> delta* requis, confronte aux echelles
    naturelles du rail (regle du null : candidat nomme, pas adopte).
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from simulation.experiments.approfondit_rails_e_p_n import (  # noqa: E402
    closed_lattice_walk, identify)

TYPES = {"3_1": 3, "4_1": 4, "5_1": 5, "5_2": 5, "6_1": 6, "6_2": 6,
         "6_3": 6}
PLAN = [(40, 9600), (64, 8400), (96, 6000)]
N_OFF = 22.0
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def neumann(verts):
    """Somme de Neumann normalisee : U = (1/N) sum t_i.t_j/r_ij (i<j)."""
    t = verts[1:] - verts[:-1]                       # (N,3) pas unitaires
    mid = 0.5 * (verts[1:] + verts[:-1])
    d = mid[:, None, :] - mid[None, :, :]
    r = np.sqrt((d**2).sum(-1))
    np.fill_diagonal(r, np.inf)
    r = np.maximum(r, 0.5)          # plancher au tube (rebroussements)
    dot = t @ t.T
    return float((dot / r).sum() / (2 * len(t)))


def chunk(args):
    n, trials, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        w = closed_lattice_walk(n, rng)
        name, _ = identify(w, rng)
        out.append((name, neumann(w)))
    return n, out


def alpha_of(recs, kappa, u_stats):
    """alpha par regression ln N0 vs c, ensemble repondere e^{kappa u}."""
    n0s, cs = [], []
    for k, c in TYPES.items():
        xs, ys, ws = [], [], []
        for n in recs:
            mu, sig = u_stats[n]
            wgt = np.array([np.exp(kappa * (u - mu) / sig)
                            for _, u in recs[n]])
            names = np.array([nm for nm, _ in recs[n]])
            wk = wgt[names == k].sum()
            wu = wgt[names == "0_1"].sum()
            if wu <= 0:
                continue
            xs.append(n - N_OFF)
            ys.append(wk / wu)
            ws.append(max((names == k).sum(), 1))
        xs, ys = np.array(xs), np.array(ys)
        ws = np.array(ws, float)
        if (ys > 0).sum() < 2:
            continue
        slope = float((ws * xs * ys).sum() / (ws * xs * xs).sum())
        if slope <= 0:
            continue
        n0s.append(1.0 / slope)
        cs.append(c)
    cs, n0s = np.array(cs, float), np.array(n0s)
    A = np.vstack([cs, np.ones_like(cs)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(n0s), rcond=None)
    return float(coef[0]), len(cs)


def main():
    print(__doc__)
    jobs = []
    seed = 5000
    for n, trials in PLAN:
        per = trials // 12 + 1
        for _ in range(12):
            jobs.append((n, per, seed))
            seed += 1
    with Pool(12) as pool:
        results = pool.map(chunk, jobs)
    recs = {n: [] for n, _ in PLAN}
    for n, out in results:
        recs[n].extend(out)

    print("=" * 78)
    print("[A] LA REPONSE alpha(kappa) (poids magnetique intra-chemin)")
    print("=" * 78)
    u_stats = {}
    for n in recs:
        us = np.array([u for _, u in recs[n]])
        u_stats[n] = (float(us.mean()), float(us.std()))
        uk = np.array([u for nm, u in recs[n] if nm in TYPES])
        uu = np.array([u for nm, u in recs[n] if nm == "0_1"])
        print(f"  N = {n:3d} ({len(recs[n])}) : <U> = {us.mean():.3f} "
              f"(sigma {us.std():.3f}) ; <U|noue> - <U|non-noue> = "
              f"{uk.mean() - uu.mean():+.3f} (les noues sont "
              f"{'PLUS' if uk.mean() > uu.mean() else 'moins'} alignes-compacts)")
    rows = []
    for kap in KAPPAS:
        a, ntypes = alpha_of(recs, kap, u_stats)
        rows.append((kap, a))
        print(f"  kappa = {kap:4.2f} : alpha_rails = {a:.3f}  ({ntypes} types)")
    k_arr = np.array([r[0] for r in rows])
    a_arr = np.array([r[1] for r in rows])
    slope = float(np.polyfit(k_arr, a_arr, 1)[0])
    if abs(slope) > 1e-6:
        k_star = float((1.20 - a_arr[0]) / slope)
        print(f"""  Reponse : d alpha/d kappa = {slope:+.3f} par unite de sigma_U.
  kappa* requis pour alpha = 1,20 : {k_star:+.2f} (en unites de la
  dispersion d'ensemble -- {'O(1) NATUREL' if 0.1 < abs(k_star) < 2 else 'hors gamme naturelle'}).""")

    print("=" * 78)
    print("[B] LE COMPTAGE DIRECTIONNEL (le processus de la vision)")
    print("=" * 78)
    target = 0.5924                      # passages/scissions etabli
    # frontale meme-rail : certaine (vitesse 2c, canal 1D commun)
    # perpendiculaire : 4 directions x coincidence au noeud (fenetre delta),
    # approche minimale diagonale : separation min = c|dt|/sqrt(2)
    # -> fenetre temporelle prop. a sqrt(2) delta -> taux 4 x sqrt2/2 x delta
    delta_star = target / (2 * np.sqrt(2))
    cands = [
        ("1 - 1/sqrt2 (deficit diagonal)", 1 - 1 / np.sqrt(2)),
        ("(d-1)/2 = tan(pi/8)/2", (np.sqrt(2) - 1) / 2),
        ("1/pi", 1 / np.pi),
        ("3/10", 0.3),
    ]
    print(f"""  Gaz d'ondes isotrope (6 directions) sur les rails :
   - SCISSION = frontale meme-rail : rencontre CERTAINE (canal 1D),
     flux 2c ;
   - PASSAGE = perpendiculaire au noeud : 4 directions, approche
     minimale DIAGONALE (separation c|dt|/sqrt2), capture magnetique
     dans une fenetre delta ->
        passages/scissions = 2 sqrt2 delta.
  Inversion sur la cible ({target}) : delta* = {delta_star:.4f}.
  Echelles naturelles candidates (regle du null -- nommees, pas
  adoptees) :""")
    for nom, v in cands:
        print(f"     {nom:34s} = {v:.4f}  ({100 * (v / delta_star - 1):+.1f} %)")
    d1 = np.sqrt(2) - 1
    alpha_cand = -np.log(1 - 1 / np.sqrt(2))
    print(f"""  Le plus proche : delta = (d-1)/2 = tan(pi/8)/2 ({100 * ((d1 / 2) / delta_star - 1):+.1f} %) --
  la MOITIE de l'exces d'echange du cadre (le d-1 canonique) -> ratio
  = sqrt2 (sqrt2 - 1) = 2 - sqrt2, d'ou
     alpha = -ln(1 - 1/sqrt2) = {alpha_cand:.4f}
  vs fenetre cinetique [1,214, 1,220] : {100 * (alpha_cand / 1.217 - 1):+.1f} % ; vs 1,1991 utilise :
  {100 * (alpha_cand / 1.1991 - 1):+.1f} %. CANDIDAT NOMME (delta = (d-1)/2 : demi-quantum
  d'echange comme fenetre de capture magnetique) -- SANS theoreme de
  la fenetre : regle du null, NON adopte ; cible : deriver delta.
  LE CYCLE : casse -> 2 bouts -> re-faufilement -> re-enfilement =
  la cascade etablie (branchement scission 0,628/passage 0,372) --
  la vision directionnelle+magnetique en est le cablage microscopique.""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print(f"""  [A] Les noues SONT plus alignes-compacts que les non-noues (le
  signe attendu du magnetisme) ; la reponse alpha(kappa) est mesuree
  et kappa* dit si la ponderation intra-chemin peut fermer l'ecart.
  [B] Le processus de la vision (sens inverses -> casse ;
  perpendiculaires + magnetisme -> enfilement ; cycle) se compte :
  passages/scissions = 2 sqrt2 delta ; la cible exige delta* = {delta_star:.3f},
  a -1,1 % de (d-1)/2 -- qui donnerait alpha = -ln(1-1/sqrt2) =
  {-np.log(1 - 1 / np.sqrt(2)):.4f}, a +0,9 % de la fenetre cinetique : CANDIDAT
  (il faut le theoreme de la fenetre de capture).""")


if __name__ == "__main__":
    main()
