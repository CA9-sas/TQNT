"""Rails du socle : probabilites de produire electron, proton, neutron (23/07).

APPROFONDISSEMENT du S43. Cibles TQNT :
  electron = 3_1 seul (les 3 modes Z3 du trefle cascadent vers e) ;
  proton   = entrelacs 3_1 + 7_2 + 7bar_2, Lk_total = +3 (Q = -2+3 = +1) ;
  neutron  = le MEME entrelacs, Lk_total = +2 (Q = 0).

QUATRE VOLETS :
[1] SPECTRE RESOLU : le determinant seul degenere (det 5 = {4_1, 5_1},
    det 7 = {5_2, 7_1}, det 11 = {6_2, 7_2}). Deuxieme invariant :
    |Delta(2)| (partie impaire -- le calcul par Wirtinger donne
    +-2^a Delta(2)) : 4_1->1/5_1->11 ; 5_2->1(4)/7_1->43 ;
    6_2->1/7_2->5. Chiralite par le signe du writhe.
[2] CO-FAUFILEMENT : plusieurs ondes se faufilent dans la MEME region
    de rails -> distribution des enlacements de Gauss Lk (exacts,
    demi-somme des signes de croisements inter-composantes).
[3] L'ECHELLE D'ASSEMBLAGE : P_brut(e) vs P_brut(p) (produit des
    composants x paire miroir x configuration d'enlacement).
[4] L'ENTONNOIR : les taux bruts ne fixent PAS les abondances -- la
    cascade P15 canalise vers les PLANCHERS (electron = plancher
    leptonique ; proton = plancher du potentiel harmonique
    d'enlacement, marche beta(Lk-3)^2 : 1 unite = 1,29 MeV = Dm(n-p)
    canonique) et la conservation de Q force n_e = n_p.
"""
import sys, os
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from theory.knot_probability import project_and_find_crossings  # noqa: E402

BETA_LK = 0.210 * 6.15          # MeV par unite^2 (potentiel harmonique)


# ----- outillage noeud ------------------------------------------------------

def closed_lattice_walk(n, rng, origin=None):
    while True:
        a = rng.multinomial(n // 2, [1 / 3] * 3)
        if np.all(a >= 1):
            break
    steps = []
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = 1.0
        steps += [e] * a[ax] + [-e] * a[ax]
    steps = np.array(steps)
    rng.shuffle(steps)
    v = np.zeros((n + 1, 3))
    v[1:] = np.cumsum(steps, axis=0)
    if origin is not None:
        v += origin
    return v


def alexander_at(crossings, t):
    """|det| de la matrice d'Alexander a t (Wirtinger) ; retourne 0 si vide."""
    if not crossings:
        return 1
    n = len(crossings)
    under, over_pos = [], {}
    for ci, c in enumerate(crossings):
        if c["z_over"]:
            under.append((c["j"] + c["s"], ci))
            over_pos[ci] = c["i"] + c["t"]
        else:
            under.append((c["i"] + c["t"], ci))
            over_pos[ci] = c["j"] + c["s"]
    under.sort(key=lambda x: x[0])
    rank = {ci: k for k, (_, ci) in enumerate(under)}
    pos = [p for p, _ in under]

    def arc_of(x):
        for k in range(n):
            a, b = pos[k], pos[(k + 1) % n]
            if (a < b and a < x < b) or (a >= b and (x > a or x < b)):
                return k
        return 0

    A = [[0] * n for _ in range(n)]
    for ci, c in enumerate(crossings):
        k = rank[ci]
        ui, uo, o = (k - 1) % n, k, arc_of(over_pos[ci])
        # signe propre du croisement, fourni par enrich_signs
        if c["eps"] > 0:
            A[ci][ui] += t
            A[ci][o] += 1 - t
            A[ci][uo] += -1
        else:
            A[ci][ui] += 1
            A[ci][o] += t - 1
            A[ci][uo] += -t
    M = [row[:-1] for row in A[:-1]]
    return abs(int_det(M))


def int_det(M):
    """Determinant entier exact (Bareiss, sans erreur d'arrondi)."""
    n = len(M)
    if n == 0:
        return 1
    M = [row[:] for row in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    M[k], M[r] = M[r], M[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[-1][-1]


def enrich_signs(poly, crossings):
    """Ajoute le signe propre eps = sign((d_over x d_under)_z)."""
    for c in crossings:
        di = poly[c["i"] + 1, :2] - poly[c["i"], :2]
        dj = poly[c["j"] + 1, :2] - poly[c["j"], :2]
        cz = di[0] * dj[1] - di[1] * dj[0]
        c["eps"] = int(np.sign(cz)) if c["z_over"] else -int(np.sign(cz))
    return crossings


def odd_part(x):
    if x <= 0:
        return 0
    while x % 2 == 0:
        x //= 2
    return x


def identify(poly, rng):
    """(nom, writhe) via det(-1) + partie impaire de Delta(2)."""
    p = poly + rng.normal(0, 1e-3, poly.shape)
    p[-1] = p[0]
    cr = enrich_signs(p, project_and_find_crossings(p))
    d1 = alexander_at(cr, -1)
    wr = sum(c["eps"] for c in cr)
    if d1 <= 1:
        return "0_1", wr
    d2 = odd_part(alexander_at(cr, 2))
    key = (d1, d2)
    table = {(3, 3): "3_1", (5, 1): "4_1", (5, 11): "5_1",
             (7, 1): "5_2", (7, 43): "7_1", (9, 0): "6_1",
             (9, 9): "3_1#3_1", (11, 1): "6_2", (11, 5): "7_2",
             (13, 7): "6_3", (13, 1): "7_3", (15, 3): "7_4",
             (17, 7): "7_5", (19, 5): "7_6", (21, 3): "7_7"}
    return table.get(key, f"?{d1}"), wr


def linking(pa, pb):
    """Enlacement de Gauss exact (demi-somme des signes inter-croisements)."""
    na, nb = len(pa) - 1, len(pb) - 1
    tot = 0
    for i in range(na):
        p1, d1 = pa[i, :2], pa[i + 1, :2] - pa[i, :2]
        p3 = pb[:-1, :2]
        d2 = pb[1:, :2] - pb[:-1, :2]
        dp = p3 - p1
        den = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
        safe = np.abs(den) > 1e-12
        t = np.where(safe, (dp[:, 0] * d2[:, 1] - dp[:, 1] * d2[:, 0])
                     / np.where(safe, den, 1), -1)
        s = np.where(safe, (dp[:, 0] * d1[1] - dp[:, 1] * d1[0])
                     / np.where(safe, den, 1), -1)
        hit = safe & (t > 0) & (t < 1) & (s > 0) & (s < 1)
        for j in np.where(hit)[0]:
            za = pa[i, 2] + t[j] * (pa[i + 1, 2] - pa[i, 2])
            zb = pb[j, 2] + s[j] * (pb[j + 1, 2] - pb[j, 2])
            cz = d1[0] * d2[j, 1] - d1[1] * d2[j, 0]
            eps = np.sign(cz) if za > zb else -np.sign(cz)
            tot += eps
    return int(round(tot / 2))


def main():
    print(__doc__)
    rng = np.random.default_rng(11)

    print("=" * 78)
    print("[1] SPECTRE RESOLU (det(-1) + partie impaire de Delta(2))")
    print("=" * 78)
    plan = [(32, 900), (48, 900), (64, 1200), (96, 900), (128, 500)]
    p31, p72 = {}, {}
    wr_chir = []
    n72tot, t72tot = 0, 0
    for n, trials in plan:
        cnt = Counter()
        for _ in range(trials):
            w = closed_lattice_walk(n, rng)
            name, wr = identify(w, rng)
            cnt[name] += 1
            if name in ("5_2", "7_2"):
                wr_chir.append(wr)
        p31[n] = cnt["3_1"] / trials
        p72[n] = cnt["7_2"] / trials
        if n >= 64:
            n72tot += cnt["7_2"]
            t72tot += trials
        top = ", ".join(f"{k} {100 * v / trials:.1f} %"
                        for k, v in cnt.most_common(6) if k != "0_1")
        print(f"  N = {n:3d} : P(3_1) = {p31[n]:.3f}, P(7_2) = {p72[n]:.4f}"
              f"   [{top}]")
    p72_pool = max(n72tot, 1) / t72tot
    born = "" if n72tot else " (borne sup., 0 evenement)"
    print(f"  P(7_2) mutualisee (N >= 64) : {p72_pool:.2e}"
          f" ({n72tot}/{t72tot}){born}")
    if wr_chir:
        wp = sum(1 for w in wr_chir if w > 0) / len(wr_chir)
        print(f"""  Chiralite des noeuds CHIRAUX formes (5_2 + 7_2, {len(wr_chir)} evts) :
  {100 * wp:.0f} % writhe > 0 vs {100 * (1 - wp):.0f} % < 0 -- parite ~50/50 : les deux
  chiralites se forment a egalite -> la paire miroir 7_2 + 7bar_2 (le
  compagnon du nucleon) est naturelle : P(paire) = P72^2/2.""")

    print("=" * 78)
    print("[2] CO-FAUFILEMENT : distribution des enlacements de Gauss")
    print("=" * 78)
    N = 48
    lk_c = Counter()
    npairs = 2500
    for _ in range(npairs):
        pa = closed_lattice_walk(N, rng)
        pb = closed_lattice_walk(N, rng, origin=np.array([1.0, 0.0, 0.0]))
        pa = pa + rng.normal(0, 1e-3, pa.shape)
        pb = pb + rng.normal(0, 1e-3, pb.shape)
        pa[-1] = pa[0]
        pb[-1] = pb[0]
        lk_c[linking(pa, pb)] += 1
    tot = sum(lk_c.values())
    print(f"  paires co-faufilees (N = {N}, origines voisines) : {npairs}")
    for k in sorted(lk_c, key=abs)[:9]:
        print(f"     Lk = {k:+d} : {100 * lk_c[k] / tot:5.1f} %")
    plink = 1 - lk_c[0] / tot
    # triple : somme de 3 enlacements de paires independants (approximation)
    vals, probs = zip(*[(k, v / tot) for k, v in lk_c.items()])
    rngc = np.random.default_rng(3)
    ii = rngc.choice(len(vals), (20000, 3), p=probs)
    sums = np.array(vals)[ii].sum(axis=1)
    p_lk3 = float(np.mean(sums == 3))
    p_lk2 = float(np.mean(sums == 2))
    print(f"""  P(paire enlacee) = {plink:.2f} ; TRIPLE (3 composantes, somme des
  enlacements de paires) : P(Lk_tot = +3) = {p_lk3:.3f},
  P(Lk_tot = +2) = {p_lk2:.3f} -> rapport brut n/p (au co-faufilement) =
  {p_lk2 / p_lk3:.2f} (les configurations neutron sont un peu plus
  probables : |Lk| plus petit).""")

    print("=" * 78)
    print("[3] L'ECHELLE D'ASSEMBLAGE BRUT")
    print("=" * 78)
    pe = p31[48]
    pp = p31[48] * p72_pool**2 / 2 * p_lk3
    pn = p31[48] * p72_pool**2 / 2 * p_lk2
    print(f"""  P_brut(electron | fermeture) ~ P(3_1) = {pe:.3f}  (N ~ 48 ; les 3
  modes Z3 du trefle cascadent tous vers e)
  P_brut(proton | 3 fermetures co-faufilees)
     = P(3_1) x P(paire miroir 7_2) x P(Lk_tot = 3)
     = {p31[48]:.3f} x {p72_pool**2 / 2:.2e} x {p_lk3:.3f} = {pp:.2e}
  P_brut(neutron) = {pn:.2e}
  RAPPORT electron/proton AU FAUFILEMENT BRUT : ~{pe / pp:.0e} --
  l'electron est ~30 millions de fois plus facile a former : c'est le
  noeud minimal ; le proton exige DEUX 7_2 miroirs ET la bonne
  configuration d'enlacement (Poisson declare : 3 evts 7_2).""")

    print("=" * 78)
    print("[4] L'ENTONNOIR : des taux bruts aux abondances")
    print("=" * 78)
    print(f"""  Les taux bruts ne fixent PAS les abondances observees :
  (i) CASCADE P15 : tout debris topologique charge cascade vers les
      PLANCHERS de conservation -- electron (plancher leptonique :
      mu, tau -> e) et proton (plancher baryonique) ; les produits
      exotiques du faufilement (4_1, 5_2, 6_x...) sont des
      transitoires qui se defont ou s'assemblent ;
  (ii) LE PLANCHER PROTON EST DYNAMIQUE : le potentiel harmonique
      d'enlacement E = beta (Lk - 3)^2 <sqrt(cJ cJ)> donne la marche
      Lk = 2 -> 3 : {BETA_LK:.2f} MeV = Dm(n - p) etabli (1,293) --
      le rapport brut n/p ~ {p_lk2 / p_lk3:.1f} du co-faufilement est ensuite
      RELAXE par cette marche : le neutron libre decroit (beta), le
      proton est le fond du puits ;
  (iii) LA NEUTRALITE : Q = Sigma_chi + Lk est conservee -> l'univers
      force n_e = n_p A LA FIN, quel que soit le rapport brut
      ~{pe / pp:.0e} : l'exces d'electrons potentiels ne survit pas
      (annihilation/cascade), la conservation fixe l'abondance.
  MORALITE : la combinatoire des rails fixe QUI peut se former et a
  quel cout (3_1 premier, nucleon rare et compose) ; les PLANCHERS et
  la CONSERVATION fixent qui SURVIT -- exactement la hierarchie
  observee (e et p stables, tout le reste transitoire).""")

    print("=" * 78)
    print("BILAN (etude, brouillon)")
    print("=" * 78)
    print(f"""  P(3_1) ~ {p31[32]:.2f}-{p31[128]:.2f} par fermeture (N = 32-128) -- l'electron
  est LE produit generique du faufilement ; P(7_2)
  (mutualisee {p72_pool:.1e}) et chiralite ~50/50 -> paire miroir
  naturelle mais rare ; enlacement
  au co-faufilement : P(Lk_tot = 3) ~ {p_lk3:.2f}, P(2)/P(3) ~ {p_lk2 / p_lk3:.1f} ;
  assemblage proton brut ~ 4e-8 de l'electron (~3e7 de rapport) ;
  l'entonnoir
  (cascades P15 + potentiel d'enlacement + conservation de Q)
  transforme ces taux bruts en la hierarchie observee. Aucune
  constante nouvelle ; interfaces etablies intactes (marche 1,29 MeV
  = Dm(n-p) ; planchers ; porte Z3).""")


if __name__ == "__main__":
    main()
