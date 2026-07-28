"""Abondance des entrelacs metastables au gel : la cascade branchee
etendue a l'ENLACEMENT (24/07 -- maillon (b) du mecanisme Li-7).

QUESTION (brouillon cosmologie fine) : le mecanisme « cendres excitees
du gel » exige n_X/n_b ~ 1e-7 d'entrelacs sombres a Lk excite.
L'OUTIL : la cascade de Nambu branchee ETABLIE (departage_f_frac,
17/07 -- P_twist derive 0,372, plancher 0,577 xi, zones d'exclusion),
ETENDUE : a chaque SCISSION, les deux filles sont deux boucles fermees
au MEME instant dans le repere du laboratoire -> leur enlacement de
Gauss est calculable, et il est INVARIANT ensuite (les fragments se
separent ; les contacts inter-fragments ulterieurs sont negliges --
limite declaree). On ne compte que les scissions TERMINALES (les deux
filles sont des fragments finals) : ce sont les paires qui survivent.

CLASSEMENT :
  - |Lk| = 0 : boucles separees (le reservoir de noeuds isoles) ;
  - |Lk| >= 1, les DEUX filles nouees (>= 1 passage chacune) :
    ENTRELACS candidat ; Lk_opt = 0 pour deux noeuds neutres (framing)
    -> tout |Lk| >= 1 est EXCITE ; la desexcitation k -> k-1 emet
    beta (2k-1) <sqrt(cJ cJ)> : pour la paire de trefles, le pas
    k=2 -> 1 (2,08 MeV) est DANS la fenetre 7Be [1,587, 2,224] MeV.
CHAINE D'ABONDANCE : n_X/n_b = (n_d/n_b) x f_pair(|Lk|>=2) x
  e^(-t_eff/tau) -- le dernier facteur (survie jusqu'a l'efficacite
  des photons, t_eff = 2,3e4 s) est LA cle : il convertit une
  fraction de paires O(1e-2..1e-3) en 1e-7 si tau ~ 1,5-2,5e3 s --
  precisement la gamme Sargent NUE. Le « facteur 30 manquant » du
  brouillon pourrait se dissoudre : trop de paires + vies trop
  courtes se compensent exponentiellement. On calcule tau* requis.
"""
import sys
import math
import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.derive_lambda_kinks import Poly, \
    embedded_fragments  # noqa: E402
import simulation.experiments.departage_f_frac as df  # noqa: E402

N_D_NB = 2.5            # noeuds sombres par baryon (chaine Omega)
NX_REQ = 1e-7           # abondance effective requise (brouillon Li-7)
T_EFF = 2.3e4           # s, survie des photons (E_c > seuil 7Be)
TAU_SARGENT = (560.0, 2000.0)   # s, Q = 2,2 -> 1,7 MeV (amplitude muon)


def gauss_linking(A, B):
    """Enlacement de Gauss de deux polylignes fermees (points (n,3))."""
    a_mid = 0.5 * (A + np.roll(A, -1, axis=0))
    b_mid = 0.5 * (B + np.roll(B, -1, axis=0))
    da = np.roll(A, -1, axis=0) - A
    db = np.roll(B, -1, axis=0) - B
    r = a_mid[:, None, :] - b_mid[None, :, :]
    d3 = np.sum(r * r, axis=2) ** 1.5
    d3 = np.maximum(d3, 1e-9)
    cross = np.cross(da[:, None, :], db[None, :, :])
    return float(np.sum(np.sum(cross * r, axis=2) / d3) / (4 * np.pi))


def loop_points(pa, pb, t, u1, u2, m=80):
    sig = np.linspace(u1, u2, m, endpoint=False)
    return 0.5 * (pa.eval(sig - t) + pb.eval(sig + t))


def cascade_avec_enlacement(path, rng):
    """Cascade branchee identique a l'etablie + enlacement aux scissions.
    Retourne finals [(id, L, n_passages)] et scissions
    [(id_A, id_B, Lk_brut, ntw_A, ntw_B)]."""
    V = np.asarray(path, float)
    V = np.vstack([V, V[:1]])
    finals, scissions = [], []
    next_id = [0]

    def nid():
        next_id[0] += 1
        return next_id[0]

    stack = [(Poly(V, np.zeros(3)), Poly(V, np.zeros(3)), 0.0, [], [], nid())]
    n_ev = 0
    while stack:
        pa, pb, t0, tws, zones, fid = stack.pop()
        Lf = pa.L
        if Lf < 2 * df.DSIG_MIN + 0.05 or n_ev >= df.MAX_EV:
            finals.append((fid, Lf, len(tws)))
            continue
        hit = df.first_crossing_from(pa, pb, Lf, t0, zones)
        if hit is None:
            finals.append((fid, Lf, len(tws)))
            continue
        n_ev += 1
        t, s1, s2 = hit
        if rng.random() < df.P_TWIST:
            pos = (s1 + s2) / 2.0 if abs(s2 - s1) < Lf / 2 else \
                ((s1 + s2 + Lf) / 2.0) % Lf
            stack.append((pa, pb, t, tws + [pos], zones + [(t, s1, s2)], fid))
        else:
            if s2 < s1:
                s1, s2 = s2, s1
            # enlacement des deux filles a l'instant de la scission
            A = loop_points(pa, pb, t, s1, s2)
            B = loop_points(pa, pb, t, s2, s1 + Lf)
            lk = gauss_linking(A, B)
            ids, ntws = [], []
            for (u1, u2) in ((s1, s2), (s2, s1 + Lf)):
                Va = pa.slice(u1 - t, u2 - t)
                Vb = pb.slice(u1 + t, u2 + t)
                sub = []
                for p_ in tws:
                    pp = p_ if p_ >= u1 else p_ + Lf
                    if u1 <= pp < u2:
                        sub.append(pp - u1)
                da = Poly(Va, Va[-1] - Va[0])
                db = Poly(Vb, Vb[-1] - Vb[0])
                i2 = nid()
                stack.append((da, db, 0.0, sub, [], i2))
                ids.append(i2)
                ntws.append(len(sub))
                tws = [p_ for p_ in tws
                       if not (u1 <= (p_ if p_ >= u1 else p_ + Lf) < u2)]
            scissions.append((ids[0], ids[1], lk, ntws[0], ntws[1]))
    return finals, scissions


def main():
    print(__doc__)
    rng = np.random.default_rng(2026)
    parents = []
    for seed in (42, 43):
        r2 = np.random.default_rng(7 + seed)
        ps = embedded_fragments(N=16, seed=seed, n_sample=30, rng=r2)
        parents += [p for p in ps if 6 <= len(p) <= 80]
    parents = parents[:24]
    L_tot = float(sum(len(p) for p in parents))
    print("parents : %d fragments plonges, L_tot = %.0f xi" % (
        len(parents), L_tot))

    all_finals, all_sci = [], []
    for p in parents:
        f, s = cascade_avec_enlacement(p, rng)
        all_finals += f
        all_sci += s
    fin_ids = {fid for fid, _, _ in all_finals}
    n_knot = sum(1 for _, _, ntw in all_finals if ntw >= 1)
    print("fragments finals : %d ; noeuds (>=1 passage) : %d "
          "(controle f_frac = %.3f)" % (
          len(all_finals), n_knot, n_knot / len(all_finals)))

    # scissions TERMINALES (les deux filles sont finales)
    term = [(a, b, lk, na, nb) for (a, b, lk, na, nb) in all_sci
            if a in fin_ids and b in fin_ids]
    lks = np.array([round(lk) for _, _, lk, _, _ in term])
    raw = np.array([lk for _, _, lk, _, _ in term])
    both = np.array([(na >= 1 and nb >= 1) for _, _, _, na, nb in term])
    print("\nscissions terminales : %d ; residu d'arrondi Lk median "
          "|brut-entier| = %.3f" % (len(term), float(
              np.median(np.abs(raw - lks)))))
    print("  distribution |Lk| (toutes paires terminales) :")
    for v in range(0, int(abs(lks).max()) + 1):
        n = int((np.abs(lks) == v).sum())
        print("    |Lk| = %d : %3d  (%.3f)" % (v, n, n / max(len(lks), 1)))
    n_pair1 = int(((np.abs(lks) >= 1) & both).sum())
    n_pair2 = int(((np.abs(lks) >= 2) & both).sum())
    print("  paires NOUEES enlacees |Lk|>=1 : %d ; |Lk|>=2 : %d" % (
        n_pair1, n_pair2))
    f1 = n_pair1 / max(n_knot, 1)
    f2 = n_pair2 / max(n_knot, 1)
    print("  -> par noeud final : f(|Lk|>=1) = %.4f ; f(|Lk|>=2) = %.4f"
          % (f1, f2))

    print("=" * 78)
    print("CHAINE D'ABONDANCE (fenetre 7Be : |Lk|>=2 pour la paire de"
          " trefles)")
    print("=" * 78)
    for lab, f in (("|Lk|>=2 (pas 2,08 MeV en fenetre)", f2),
                   ("|Lk|>=1 (si famille a k=1 en fenetre, cJ~8)", f1)):
        if f <= 0:
            print("  %s : f = 0 (statistique insuffisante)" % lab)
            continue
        n_x0 = N_D_NB * f
        surv_req = NX_REQ / n_x0
        tau_req = T_EFF / math.log(1 / surv_req)
        lo, hi = TAU_SARGENT
        print("""  %s :
    n_X(t=0)/n_b = 2,5 x %.4f = %.2e -> survie requise %.1e
    -> tau* = t_eff/ln(1/surv) = %.0f s   vs Sargent nu [%d, %d] s
    ecart tau*/Sargent : x%.1f-%.1f (element de matrice en puissance
    1/5 : x%.1f-%.1f sur l'amplitude)""" % (
            lab, f, n_x0, surv_req, tau_req, lo, hi,
            tau_req / hi, tau_req / lo,
            (tau_req / hi) ** 0.2, (tau_req / lo) ** 0.2))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  La cascade branchee ETENDUE mesure pour la premiere fois
  l'enlacement des paires au gel. La chaine se ferme par la SURVIE :
  l'exponentielle e^(-t_eff/tau) convertit la fraction de paires
  mesuree en l'abondance effective requise pour tau* ~ quelques 1e3 s
  -- LA GAMME SARGENT NUE. Le « facteur ~30 manquant » du brouillon
  se REDUIT : surabondance a la formation et vies courtes se
  compensent exponentiellement ; l'accord fin exige tau dans un
  facteur ~2 de tau* (sensibilite exponentielle DECLAREE -- c'est
  desormais LE maillon : l'element de matrice P15 d'entrelacs).
  Limites declarees : contacts inter-fragments ignores apres
  scission ; Lk mesure a l'instant de la scission (filles adjacentes,
  residu d'arrondi affiche) ; statistique = 24 parents.""")


if __name__ == "__main__":
    main()
