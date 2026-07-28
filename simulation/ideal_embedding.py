"""Outillage de plongement idéal : minimisation de ropelength (algorithme type SONO).

Le plongement idéal d'un nœud K est celui qui minimise la ropelength
rho(K) = longueur / rayon de tube, sous les deux contraintes d'épaisseur :
  (1) globale : distance >= 2R entre points non voisins le long de la courbe ;
  (2) locale  : rayon de courbure >= R partout.

Algorithme (Pieranski, SONO — Shrink-On-No-Overlaps) :
  itérer { égaliser les arêtes ; contrôler la courbure ; repousser les
  recouvrements ; rétrécir si (presque) aucun recouvrement }.
Le rétrécissement est adaptatif : suspendu tant que la violation maximale
dépasse la tolérance. La topologie est préservée par construction (les brins
sont repoussés avant de pouvoir se traverser, pas de pas brusque).

Convention : R = 1 (unités de rayon de tube). ropelength = longueur.
Valeurs publiées (unités de rayon) : 3_1 = 32.74, 4_1 = 42.09, 7_2 ~ 65.9.

Usage :
    from simulation.ideal_embedding import sono, ropelength, min_gap
    P_ideal, info = sono(P0, iters=20000)
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Géométrie de base (courbes fermées = polygones cycliques N x 3)            #
# --------------------------------------------------------------------------- #
def curve_length(P):
    return float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())


def resample(P, n):
    """Rééchantillonnage uniforme en abscisse curviligne (courbe fermée)."""
    P = np.asarray(P, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    Q = np.empty((n, 3))
    ext = np.vstack([P, P[:1]])
    for c in range(3):
        Q[:, c] = np.interp(t, s, ext[:, c])
    return Q


def _pair_data(P, R):
    """Matrice des distances + masque des paires non locales.

    Deux points sont « non locaux » si leur écart curviligne dépasse pi*R
    (l'arc minimal qu'un tube de rayon R peut replier sur lui-même).
    """
    n = len(P)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    ledge = curve_length(P) / n
    g = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    g = np.minimum(g, n - g)
    skip = max(2, int(np.ceil(np.pi * R / ledge)))
    return D, g > skip


def min_gap(P, R=1.0):
    """Distance minimale entre points non locaux (contrainte globale).
    Retourne +inf si aucune paire non locale (petit cercle a la limite)."""
    D, mask = _pair_data(P, R)
    if not mask.any():
        return float("inf")
    return float(D[mask].min())


def ropelength(P, R=1.0):
    return curve_length(P) / R


# --------------------------------------------------------------------------- #
#  Les trois opérateurs SONO                                                  #
# --------------------------------------------------------------------------- #
def _control_curvature(P, R, frac=0.5):
    """Lisse les sommets où le rayon de courbure < R ; retourne (P, excès max).

    Le lissage local redistribue les angles (répare les coudes) mais la somme
    des angles de virage est topologiquement contrainte : si la violation est
    globale (courbe trop courte), c'est au rétrécissement de s'arrêter --- d'où
    l'excès retourné, traité comme une profondeur de recouvrement par sono().
    """
    prev, nxt = np.roll(P, 1, 0), np.roll(P, -1, 0)
    e1 = P - prev
    e2 = nxt - P
    l1 = np.linalg.norm(e1, axis=1)
    l2 = np.linalg.norm(e2, axis=1)
    cosang = np.clip((e1 * e2).sum(1) / (l1 * l2), -1.0, 1.0)
    theta = np.arccos(cosang)                      # angle de virage au sommet
    ledge = 0.5 * (l1 + l2)
    theta_max = 2.0 * np.arcsin(np.clip(ledge / (2 * R), 0, 1))
    ratio = theta / np.maximum(theta_max, 1e-12)
    bad = ratio > 1.0
    if not bad.any():
        return P, 1.0
    mid = 0.5 * (prev + nxt)
    L0 = curve_length(P)
    P = P.copy()
    P[bad] += frac * (mid[bad] - P[bad])
    # lissage isopérimètre : le déplacement vers la corde raccourcit la courbe ;
    # on restaure la longueur pour que seul le rétrécissement explicite rétrécisse
    c = P.mean(0)
    P = c + (P - c) * (L0 / curve_length(P))
    return P, float(ratio.max())


def _remove_overlaps(P, R, push=0.6, tol=1e-9):
    """Repousse symétriquement les paires non locales à distance < 2R."""
    D, mask = _pair_data(P, R)
    viol = mask & (D < 2 * R)
    iu = np.triu(viol)
    ii, jj = np.nonzero(iu)
    if len(ii) == 0:
        return P, 0.0
    disp = np.zeros_like(P)
    cnt = np.zeros(len(P))
    vec = P[ii] - P[jj]
    d = np.linalg.norm(vec, axis=1)
    d = np.where(d < tol, tol, d)
    corr = 0.5 * push * (2 * R - d)
    u = vec / d[:, None]
    np.add.at(disp, ii, u * corr[:, None])
    np.add.at(disp, jj, -u * corr[:, None])
    np.add.at(cnt, ii, 1)
    np.add.at(cnt, jj, 1)
    nz = cnt > 0
    disp[nz] /= cnt[nz, None]
    depth = float((2 * R - d).max())
    return P + disp, depth


def sono(P0, R=1.0, n=None, iters=20000, shrink=0.9995, push=0.6,
         depth_tol=0.02, resample_every=10, log_every=0, loosen_start=True):
    """Minimise la ropelength de la courbe fermée P0 (SONO).

    Retourne (P, info) avec info = dict(ropelength, min_gap, iters, history).
    La topologie de P0 est préservée (répulsion avant traversée).
    """
    P = resample(np.asarray(P0, float), n or len(P0))
    P = P - P.mean(0)
    # départ : desserrer jusqu'à satisfaire les contraintes (sauf en réchauffe,
    # où l'on laisse la répulsion résoudre les recouvrements en place)
    if loosen_start:
        while min_gap(P, R) < 2 * R:
            P *= 1.05
    hist = []
    best, best_rho = None, np.inf
    for it in range(iters):
        if it % resample_every == 0:
            P = resample(P, len(P))
            rho = ropelength(P, R)
            if rho < best_rho and min_gap(P, R) >= 2 * R * (1 - 1.2 * depth_tol):
                best, best_rho = P.copy(), rho
        P, curv_ratio = _control_curvature(P, R)
        P, depth = _remove_overlaps(P, R, push=push)
        if depth < depth_tol * R and curv_ratio < 1.05:
            c = P.mean(0)
            P = c + (P - c) * shrink
        if log_every and it % log_every == 0:
            hist.append((it, ropelength(P, R), min_gap(P, R)))
    if best is not None:
        P = best
    # polissage final : plus de rétrécissement, seulement les contraintes
    for _ in range(200):
        P = resample(P, len(P))
        P, _ = _control_curvature(P, R)
        P, depth = _remove_overlaps(P, R, push=push)
        if depth < 1e-4:
            break
    info = {"ropelength": ropelength(P, R), "min_gap": min_gap(P, R),
            "iters": iters, "history": hist}
    return P, info


# --------------------------------------------------------------------------- #
#  Entrelacs (multi-composantes) : contraintes inter-composantes              #
# --------------------------------------------------------------------------- #
def linking_number(P, Q):
    """Nombre d'enlacement de Gauss entre deux courbes fermées (entier)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    dlP = np.roll(P, -1, 0) - P
    dlQ = np.roll(Q, -1, 0) - Q
    mP = P + 0.5 * dlP
    mQ = Q + 0.5 * dlQ
    tot = 0.0
    for i in range(len(P)):
        rij = mQ - mP[i]
        d3 = np.linalg.norm(rij, axis=1) ** 3
        tot += float(((np.cross(dlP[i], dlQ) * rij).sum(1) / d3).sum())
    return int(round(tot / (4 * np.pi)))


def _link_concat(comps):
    X = np.vstack(comps)
    comp = np.concatenate([np.full(len(C), k) for k, C in enumerate(comps)])
    offs = np.cumsum([0] + [len(C) for C in comps])
    return X, comp, offs


def _link_pair_mask(comps, R):
    """Masque des paires contraintes : inter-composantes (toutes) +
    intra-composante non locales (écart curviligne > pi*R)."""
    X, comp, offs = _link_concat(comps)
    n = len(X)
    mask = comp[:, None] != comp[None, :]
    for k, C in enumerate(comps):
        m = len(C)
        ledge = curve_length(C) / m
        skip = max(2, int(np.ceil(np.pi * R / ledge)))
        g = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
        g = np.minimum(g, m - g)
        sl = slice(offs[k], offs[k + 1])
        mask[sl, sl] = g > skip
    return X, mask, offs


def link_min_gap(comps, R=1.0):
    X, mask, _ = _link_pair_mask(comps, R)
    if not mask.any():
        return float("inf")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return float(D[mask].min())


def link_ropelength(comps, R=1.0):
    return sum(curve_length(C) for C in comps) / R


def _link_remove_overlaps(comps, R, push=0.6, tol=1e-9):
    X, mask, offs = _link_pair_mask(comps, R)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    viol = mask & (D < 2 * R)
    ii, jj = np.nonzero(np.triu(viol))
    if len(ii) == 0:
        return comps, 0.0
    disp = np.zeros_like(X)
    cnt = np.zeros(len(X))
    vec = X[ii] - X[jj]
    d = np.linalg.norm(vec, axis=1)
    d = np.where(d < tol, tol, d)
    corr = 0.5 * push * (2 * R - d)
    u = vec / d[:, None]
    np.add.at(disp, ii, u * corr[:, None])
    np.add.at(disp, jj, -u * corr[:, None])
    np.add.at(cnt, ii, 1)
    np.add.at(cnt, jj, 1)
    nz = cnt > 0
    disp[nz] /= cnt[nz, None]
    Xn = X + disp
    return [Xn[offs[k]:offs[k + 1]] for k in range(len(comps))], float((2 * R - d).max())


def sono_link(comps0, R=1.0, iters=20000, shrink=0.9995, push=0.6,
              depth_tol=0.02, resample_every=10, log_every=0, loosen_start=True):
    """SONO pour entrelacs : minimise la ropelength totale sous les contraintes
    intra (non-local >= 2R, courbure >= R) ET inter-composantes (>= 2R partout).

    La topologie (types de nœuds + nombres d'enlacement) est préservée par la
    répulsion avant traversée. Retourne (comps, info).
    """
    comps = [resample(np.asarray(C, float), len(C)) for C in comps0]
    ctr = np.vstack(comps).mean(0)
    comps = [C - ctr for C in comps]
    if loosen_start:
        while link_min_gap(comps, R) < 2 * R:
            comps = [C * 1.05 for C in comps]
    hist = []
    best, best_rho = None, np.inf
    for it in range(iters):
        if it % resample_every == 0:
            comps = [resample(C, len(C)) for C in comps]
            rho = link_ropelength(comps, R)
            if rho < best_rho and link_min_gap(comps, R) >= 2 * R * (1 - 1.2 * depth_tol):
                best, best_rho = [C.copy() for C in comps], rho
        curv_ratio = 1.0
        new = []
        for C in comps:
            C2, cr = _control_curvature(C, R)
            new.append(C2)
            curv_ratio = max(curv_ratio, cr)
        comps, depth = _link_remove_overlaps(new, R, push=push)
        if depth < depth_tol * R and curv_ratio < 1.05:
            c = np.vstack(comps).mean(0)
            comps = [c + (C - c) * shrink for C in comps]
        if log_every and it % log_every == 0:
            hist.append((it, link_ropelength(comps, R), link_min_gap(comps, R)))
    if best is not None:
        comps = best
    for _ in range(200):
        comps = [resample(C, len(C)) for C in comps]
        comps = [_control_curvature(C, R)[0] for C in comps]
        comps, depth = _link_remove_overlaps(comps, R, push=push)
        if depth < 1e-4:
            break
    info = {"ropelength": link_ropelength(comps, R),
            "min_gap": link_min_gap(comps, R), "iters": iters, "history": hist}
    return comps, info


# --------------------------------------------------------------------------- #
#  Réchauffes graduées (topologiquement sûres)                                #
# --------------------------------------------------------------------------- #
def squeeze(P, u, s):
    """Compression affine anisotrope de facteur s le long de la direction u.

    Homéomorphisme (jamais de passage de brin). Avec s >= 0.5 le gap minimal
    reste >= s*2R : la répulsion qui suit est sans ambiguïté de direction.
    """
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    c = P.mean(0)
    X = P - c
    return c + X - (1 - s) * (X @ u)[:, None] * u[None, :]


def anneal(P0, R=1.0, guard=None, target=None,
           rungs=(0.5, 0.55, 0.62, 0.7, 0.8), trials_per_rung=3,
           iters=15000, n=None, sweeps=2, rng=None, log=None):
    """Réchauffes graduées : compressions anisotropes (fortes -> douces) + SONO.

    - Chaque essai : squeeze le long d'une direction aléatoire, re-relaxation
      SONO sans desserrage initial, validation par `guard` (ex. Alexander).
    - Garde le meilleur VALIDE ; arrêt anticipé si ropelength <= target.
    - `sweeps` répétitions de l'échelle complète (redémarrées du meilleur).
    Retourne (P_best, rho_best, journal).
    """
    rng = rng or np.random.default_rng(0)
    log = log or (lambda *a, **k: None)
    P, info = sono(P0, R=R, n=n, iters=iters)
    best, best_rho = P, info["ropelength"]
    if guard is not None and not guard(best):
        raise ValueError("la configuration de départ ne passe pas la garde")
    journal = [("start", 1.0, best_rho, True)]
    log(f"depart : rho = {best_rho:.2f}")
    for sweep in range(sweeps):
        for s in rungs:
            for t in range(trials_per_rung):
                u = rng.normal(size=3)
                Q = squeeze(best, u, s)
                Q, inf2 = sono(Q, R=R, n=n or len(best), iters=iters,
                               loosen_start=False)
                ok = guard(Q) if guard is not None else True
                rho = inf2["ropelength"]
                journal.append((f"sweep{sweep}", s, rho, ok))
                log(f"  sweep {sweep} s={s:.2f} essai {t} : rho = {rho:7.2f} "
                    f"gap = {inf2['min_gap']:.3f} {'OK' if ok else 'REJET'}")
                if ok and rho < best_rho - 1e-3:
                    best, best_rho = Q, rho
                    log(f"    -> nouveau meilleur : {best_rho:.2f}")
                    if target is not None and best_rho <= target:
                        return best, best_rho, journal
    return best, best_rho, journal


# --------------------------------------------------------------------------- #
#  Minimiseur global : polissage de gradient + multi-départs + continuation   #
# --------------------------------------------------------------------------- #
def _length_gradient(P):
    """Gradient polygonal de la longueur : somme des arêtes sortantes unitaires.

    Descendre le long de -g raccourcit la courbe là où il y a du mou local —
    le complément du rétrécissement UNIFORME de sono (qui ne voit pas le mou
    non uniforme)."""
    prev = P - np.roll(P, 1, 0)
    nxt = P - np.roll(P, -1, 0)
    prev /= np.linalg.norm(prev, axis=1)[:, None]
    nxt /= np.linalg.norm(nxt, axis=1)[:, None]
    return prev + nxt


def polish_gradient(P0, R=1.0, iters=600, step0=0.02, push=0.6, gap_tol=0.02):
    """Descente de longueur par sommet sous contraintes (pas adaptatif).

    Chaque passe : pas le long de -grad(longueur), restauration (courbure +
    recouvrements), acceptation si la ropelength baisse et le gap reste sain.
    Le pas est borné (<= 0.05 R) : jamais de traversée de brin (même garantie
    que sono). Retourne (P, ropelength)."""
    best = resample(np.asarray(P0, float), len(P0))
    best_rho = ropelength(best, R)
    s = step0
    for _ in range(iters):
        Q = best - s * R * _length_gradient(best)
        for _ in range(4):
            Q = resample(Q, len(Q))
            Q, _ = _control_curvature(Q, R)
            Q, depth = _remove_overlaps(Q, R, push=push)
            if depth < 1e-3 * R:
                break
        rho = ropelength(Q, R)
        if rho < best_rho and min_gap(Q, R) >= 2 * R * (1 - gap_tol):
            best, best_rho = Q, rho
            s = min(s * 1.3, 0.05)
        else:
            s *= 0.5
            if s < 1e-4:
                break
    return best, best_rho


def polish_gradient_link(comps0, R=1.0, iters=600, step0=0.02, push=0.6,
                         gap_tol=0.02):
    """Polissage de gradient pour entrelacs (contraintes intra + inter).

    Même principe que polish_gradient ; préserve les nombres d'enlacement
    (pas borné, répulsion avant traversée). Retourne (comps, ropelength)."""
    best = [resample(np.asarray(C, float), len(C)) for C in comps0]
    best_rho = link_ropelength(best, R)
    s = step0
    for _ in range(iters):
        Q = [C - s * R * _length_gradient(C) for C in best]
        for _ in range(4):
            Q = [resample(C, len(C)) for C in Q]
            Q = [_control_curvature(C, R)[0] for C in Q]
            Q, depth = _link_remove_overlaps(Q, R, push=push)
            if depth < 1e-3 * R:
                break
        rho = link_ropelength(Q, R)
        if rho < best_rho and link_min_gap(Q, R) >= 2 * R * (1 - gap_tol):
            best, best_rho = Q, rho
            s = min(s * 1.3, 0.05)
        else:
            s *= 0.5
            if s < 1e-4:
                break
    return best, best_rho


def sono_global(P0, R=1.0, n=None, guard=None, restarts=4, iters=15000,
                rungs=(0.5, 0.6, 0.7, 0.8), trials_per_rung=2,
                continuation=(1.5, 2.0), rng=None, log=None):
    """Minimiseur global de ropelength : multi-départs perturbés (squeezes
    homéomorphes) + réchauffes graduées + polissage de gradient + continuation
    en résolution.

    Retourne (P, rho, info) ; info['basins'] liste les minima de chaque départ
    (le diagnostic de globalité : leur dispersion borne l'écart au minimum).
    `guard` (ex. Alexander) valide chaque candidat."""
    rng = rng or np.random.default_rng(0)
    log = log or (lambda *a, **k: None)
    basins = []
    best, best_rho = None, np.inf
    for r in range(restarts):
        Q = np.asarray(P0, float).copy()
        if r:
            for _ in range(1 + r % 2):
                Q = squeeze(Q, rng.normal(size=3), 0.5 + 0.3 * rng.random())
        Q, _ = sono(Q, R=R, n=n, iters=iters, loosen_start=True)
        Q, rho = polish_gradient(Q, R=R)
        ok = guard(Q) if guard is not None else True
        basins.append((rho, ok))
        log(f"depart {r} : rho = {rho:.2f} {'OK' if ok else 'REJET'}")
        if ok and rho < best_rho:
            best, best_rho = Q, rho
    if best is None:
        raise ValueError("aucun depart valide")
    best, best_rho, _ = anneal(best, R=R, guard=guard, rungs=rungs,
                               trials_per_rung=trials_per_rung,
                               iters=max(4000, iters // 2), n=n, sweeps=1,
                               rng=rng, log=log)
    best, best_rho = polish_gradient(best, R=R)
    for f in continuation:
        Q = resample(best, int(len(best) * f))
        Q, _ = sono(Q, R=R, iters=max(2000, iters // 4), loosen_start=False)
        Q, rho = polish_gradient(Q, R=R)
        ok = guard(Q) if guard is not None else True
        log(f"continuation x{f} : rho = {rho:.2f} {'OK' if ok else 'REJET'}")
        if ok and rho < best_rho * 1.005:
            best, best_rho = Q, min(rho, best_rho)
    info = {"ropelength": ropelength(best, R), "basins": basins,
            "min_gap": min_gap(best, R)}
    return best, info["ropelength"], info


# --------------------------------------------------------------------------- #
#  Courbes de départ usuelles                                                 #
# --------------------------------------------------------------------------- #
def torus_knot(p, q, n=240, R0=2.0, r0=1.0):
    """Nœud torique T(p,q) (3_1 = T(2,3), 5_1 = T(2,5), 7_1 = T(2,7))."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = R0 + r0 * np.cos(q * t)
    return np.stack([rr * np.cos(p * t), rr * np.sin(p * t), r0 * np.sin(q * t)], 1)


def plat_seed(word, n_strands=4, pts_per_seg=50, npts=1400):
    """Amorce compacte par fermeture en PLAT d'une tresse (nœuds à 2 ponts).

    Les brins verticaux subissent les croisements sigma_i du mot (i>0 : brin
    gauche dessus), puis les paires de slots (1,2) et (3,4) sont refermées par
    des chapeaux en haut ET en bas (fermeture en plat, PAS la clôture-trace).
    Bien plus compact que la clôture-trace pour les 2-ponts (validé : 8_3 des
    mots sigma2^a sigma1^b sigma2^c). TOUJOURS garder par Alexander en aval ;
    certains mots donnent 2 composantes (ValueError levée).
    """
    occupant = {k: k for k in range(1, n_strands + 1)}
    paths = {s: [(float(s), 0.0, 0.0)] for s in range(1, n_strands + 1)}
    for layer, g in enumerate(word):
        y0 = float(layer)
        i = abs(g)
        over = g > 0
        for slot in range(1, n_strands + 1):
            s = occupant[slot]
            if slot == i:
                for u in np.linspace(0, 1, pts_per_seg)[1:]:
                    z = (0.35 if over else -0.35) * np.sin(np.pi * u)
                    paths[s].append((i + u, y0 + u, z))
            elif slot == i + 1:
                for u in np.linspace(0, 1, pts_per_seg)[1:]:
                    z = (-0.35 if over else 0.35) * np.sin(np.pi * u)
                    paths[s].append(((i + 1) - u, y0 + u, z))
            else:
                for u in np.linspace(0, 1, pts_per_seg)[1:]:
                    paths[s].append((float(slot), y0 + u, 0.0))
        occupant[i], occupant[i + 1] = occupant[i + 1], occupant[i]
    top = float(len(word))

    def cap(x1, x2, y, sign):
        c, r = (x1 + x2) / 2, (x2 - x1) / 2
        th = np.linspace(0, np.pi, 40)
        return [(c - r * np.cos(t), y + sign * 0.6 * np.sin(t), 0.15 * np.sin(t))
                for t in th]

    conn_bot = {1: 2, 2: 1, 3: 4, 4: 3}
    top_pair = {occupant[1]: occupant[2], occupant[2]: occupant[1],
                occupant[3]: occupant[4], occupant[4]: occupant[3]}
    slot_of = {s: sl for sl, s in occupant.items()}
    curve, s, visited, going_up = [], 1, set(), True
    while s not in visited:
        visited.add(s)
        pth = paths[s] if going_up else paths[s][::-1]
        curve += [tuple(p) for p in (pth[1:] if curve else pth)]
        if going_up:
            s2 = top_pair[s]
            curve += cap(min(slot_of[s], slot_of[s2]), max(slot_of[s], slot_of[s2]),
                         top, +1)[1:]
        else:
            s2 = conn_bot[s]
            curve += cap(min(s, s2), max(s, s2), 0.0, -1)[1:]
        s, going_up = s2, not going_up
    if len(visited) != n_strands:
        raise ValueError(f"fermeture en plat : {len(visited)} brins visités "
                         f"(entrelacs à plusieurs composantes, pas un nœud)")
    return resample(np.array(curve), npts)


def twist_knot_seed(n_half, hand=+1, thread=+1, x_eye=0.10, r=0.35, npts=1200):
    """Amorce compacte de nœud de twist : bande à n_half demi-torsions + agrafe.

    Construction : deux brins antipodaux torsadés de x=-1 à 1, demi-tour à
    droite, et connecteur-agrafe enfilé dans l'œil de la bande à x = x_eye.
    Le type effectif dépend de (n_half, x_eye) --- l'enfilage absorbe les
    torsions à sa gauche --- et se VALIDE par le polynôme d'Alexander :
      (n_half=9, x_eye=0.10) -> 7_2 (Alexander (3,-5,3), les deux chiralités) ;
      l'échelle traverse 0_1, 3_1, 4_1, 5_2, 6_1, 7_2 quand x_eye avance.
    TOUJOURS vérifier le type en aval (garde d'Alexander) avant usage.

    Vertu : démarre dans le bassin compact --- le serrage SONO donne rho=73.7
    (+11.8% du publié 65.92) là où la fermeture de tresse plafonne à 118 (+72%).
    """
    t = np.linspace(-1, 1, 300)
    phi = hand * n_half * np.pi * (t + 1) / 2
    A = np.stack([t, r * np.cos(phi), r * np.sin(phi)], 1)
    B = np.stack([t, -r * np.cos(phi), -r * np.sin(phi)], 1)
    angA = np.arctan2(A[-1, 2], A[-1, 1])
    angB = np.arctan2(B[-1, 2], B[-1, 1])
    if angB < angA:
        angB += 2 * np.pi
    ang = np.linspace(angA, angB, 60)
    U = np.stack([1 + 0.5 * np.sin(np.linspace(0, np.pi, 60)),
                  r * np.cos(ang), r * np.sin(ang)], 1)
    yb, zb = B[0, 1], B[0, 2]
    ya, za = A[0, 1], A[0, 2]
    zd = thread
    pts = [B[0], [-1.6, yb, zb], [-1.6, 0.0, 0.9 * zd],
           [x_eye, 0.0, 0.45 * zd], [x_eye, 0.0, -0.45 * zd],
           [-1.6, 0.0, -0.9 * zd], [-1.6, ya, za], A[0]]
    C = []
    for p_, q_ in zip(pts[:-1], pts[1:]):
        for u in np.linspace(0, 1, 30)[1:]:
            C.append((1 - u) * np.array(p_, float) + u * np.array(q_, float))
    curve = np.vstack([A, U[1:], B[::-1][1:], np.array(C)[1:]])
    return resample(curve, npts)
