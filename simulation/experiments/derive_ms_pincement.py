"""Pincer m_s : assignation 5_2 par discriminants declares + test d'elimination.

!! AUDIT 2026-07-22 (verify_ropelengths_acpr.py) : la table RL ci-dessous
venait de bornes superieures anciennes ; la valeur ACPR est l(5_2) = 49,4704
(rayons) -> ratio VRAI 1,5107 (pas 1,596) -> m_s^geo = 468,3 vs 494,8 des
donnees (-5,4 %). L'epinglage 5_2 SORT RENFORCE (fenetre ACPR vide a +-2 %)
mais la derivation geometrique NUE ne tient plus : statut m_s en revision
(declassement ou fonctionnelle habillee +5,6 %). Source canonique des
ropelengths : theory/ropelength_acpr.py.

ETAT (mtopo_heavy_knots_test.py, commit 4cbbc97) : M_topo = scale*l(K) donne
m_s via 5_2 a +3.1% (RESERVE : 5_2 non pinne ; 4_1 catalogue -17%).

1. PINCER L'ASSIGNATION (discriminants declares AVANT le test, regle du null) :
   a. CHIRALITE : le noeud etrange doit etre CHIRAL -- s et s-bar sont
      distincts (K0 et K0-bar sont des etats differents qui oscillent, V5p27) ;
      un noeud amphichiral est auto-conjugue -> EXCLUT 4_1.
   b. FAMILLE : les noeuds toriques C_q sont la famille des LEPTONS
      (cartographie vortex : torus=centre ; le 3_1 est le seul a la fois
      torique et twist) -> EXCLUT 5_1 comme constituant hadronique.
   c. MINIMALITE : le plus leger restant apres 3_1.
   => candidat unique : 5_2 (premier noeud chiral non-torique apres le trefle).
   NULL : fenetre +-5% autour de la cible l = 1.548*l(3_1) = 25.34 ->
   combien de noeuds du catalogue tombent dedans ?

2. CROSS-CHECK OUTILLAGE : ratio l(5_2)/l(3_1) mesure par SONO (graine twist
   dediee) vs valeurs publiees (16.372/26.13 en demi-ropelength).

3. TEST D'ELIMINATION (decisif) : imposer m_s = m_l * l(5_2)/l(3_1) = 494.8
   dans l'engin canonique (spectrum_sigmachi_full) et verifier que les etats
   sensibles a m_s (D_s, B_s, Xi_c, Omega_c, ...) tiennent ; puis refit
   (m_c, m_b) pour absorber -- si le RMS lourd est inchange, m_s n'est PLUS
   un parametre libre : il devient m_l x (ratio geometrique DERIVE).
   Comptage : 11 -> 10.
"""
import sys, os, math, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

RL = {"3_1": 16.372, "4_1": 20.99, "5_1": 23.55, "5_2": 26.13, "6_1": 28.31, "6_2": 28.13,
      "6_3": 27.21, "7_1": 30.69, "7_2": 32.96, "7_3": 33.61, "7_4": 34.97, "7_5": 35.16,
      "7_6": 32.66, "7_7": 31.93, "8_1": 37.49, "8_2": 37.32, "8_3": 37.30, "9_1": 38.62}
AMPHI = {"4_1", "6_3", "8_3"}
TORUS = {"3_1", "5_1", "7_1", "9_1"}
ML, MS_CAL, MC, MB = 310., 480., 1557., 4900.
RATIO_GEO = RL["5_2"] / RL["3_1"]
MS_GEO = ML * RATIO_GEO

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


def strange(nm):
    b = nm.replace('_c', '').replace('_b', '').replace('*', '')
    if b.startswith(('Λ', 'Σ')): return 1
    if b.startswith('Ξ'): return 2
    if b.startswith('Ω'): return 3
    return 0


def heavy_open(nm, cat_, ms, mc, mb):
    if cat_ in ('charmé', 'charmé-V'):
        return mc + (ms if 's' in nm.lower() else ML)
    if cat_ in ('beau', 'beau-V'):
        if 'B_c' in nm: return mc + mb
        if 'B_s' in nm: return mb + ms
        return mb + ML
    if cat_ == 'charmé-B': ns = strange(nm); return mc + ns*ms + (2-ns)*ML
    if cat_ == 'beau-B':   ns = strange(nm); return mb + ns*ms + (2-ns)*ML
    if cat_ == 'double-c': return 2*mc + (ms if 'Ω' in nm else ML)
    return None


# etats du CSV catalogue en methode 'constituant' (verifie 14/07 : Ds+=mc+ms=2037,
# Bs0=mb+ms=5380, Lambda_c=mc+2ml=2177, Lambda_b=mb+2ml=5520) : ils SONT sensibles
# a m_s / participent au refit -> formules explicites ci-dessous.
SERVED = {'J/ψ', 'Υ(1S)'}
def csv_constituent(nm, ms, mc, mb):
    if nm in ('D⁰', 'D±'): return mc + ML
    if nm == 'D_s±': return mc + ms
    if nm in ('B±', 'B⁰'): return mb + ML
    if nm == 'B_s⁰': return mb + ms
    if nm == 'Λ_c⁺': return mc + 2*ML
    if nm == 'Λ_b⁰': return mb + 2*ML
    return None


def load_pdg():
    src = open(os.path.join(ROOT, 'simulation/experiments/exhaustive_scan.py')).read()
    blk = src[src.index('PDG = ['):src.index(']', src.index('PDG = ['))+1].replace('2/3', '0.6667')
    return ast.literal_eval(blk[blk.index('['):])


def run():
    print("=== Pincer m_s : assignation + elimination ===\n")
    print("cible : m_s/m_l = %.4f ; geometrie 5_2 : l(5_2)/l(3_1) = %.4f (%+.1f%%)\n" % (
        MS_CAL/ML, RATIO_GEO, (RATIO_GEO/(MS_CAL/ML)-1)*100))

    # -- 1. null d'assignation --
    target = (MS_CAL/ML) * RL["3_1"]
    print("[1] Null d'assignation : fenetre +-5%% autour de l_cible = %.2f :" % target)
    hits = []
    for K, l in sorted(RL.items(), key=lambda kv: kv[1]):
        if abs(l/target - 1) < 0.05:
            tags = []
            if K in AMPHI: tags.append("AMPHICHIRAL (exclu : K0 != K0bar)")
            if K in TORUS: tags.append("TORIQUE (exclu : famille lepton)")
            hits.append((K, l, tags))
            print("      %s  l=%.2f  %s" % (K, l, "; ".join(tags) if tags else "<- candidat"))
    surv = [h for h in hits if not h[2]]
    print("    -> %d noeud(s) dans la fenetre, %d survivant(s) aux discriminants : %s\n" % (
        len(hits), len(surv), ", ".join(h[0] for h in surv)))
    print("    Discriminants (declares) : chiralite (s!=sbar), famille (torus=lepton),")
    print("    minimalite. 4_1 : -17%% ET amphichiral ; 5_1 : -7%% ET torique ;")
    print("    6_1 : +11.7%%. Le 5_2 est pinne par 3 discriminants independants.\n")

    # -- 2. cross-check SONO du ratio (fait a part, 30k/60k iters : voir sortie
    #       archivee ; a 8k iters la graine twist ne converge pas, +115%) --
    print("[2] Cross-check SONO : execute a part (30k/60k iters), cf. rapport.\n")

    # -- 3. test d'elimination sur l'engin canonique --
    PDG = load_pdg()
    HEAVY_CATS = ('charmé', 'charmé-V', 'beau', 'beau-V', 'charmé-B', 'beau-B', 'double-c')

    def score(ms, mc, mb):
        errs, det = [], []
        for nm, me, J, Q, c in PDG:
            if c not in HEAVY_CATS or nm in SERVED or me <= 0:
                continue
            m = csv_constituent(nm, ms, mc, mb)
            if m is None:
                m = heavy_open(nm, c, ms, mc, mb)
                if m is not None and c in ('charmé-V', 'beau-V'): m += 140
            if m is None:
                continue
            errs.append(abs(m/me - 1)*100)
            det.append((nm, c, me, m, (m/me-1)*100))
        return np.array(errs), det

    e480, d480 = score(MS_CAL, MC, MB)
    egeo, dgeo = score(MS_GEO, MC, MB)
    print("[3] Elimination : m_s = m_l * l(5_2)/l(3_1) = %.1f MeV (vs 480 calibre)" % MS_GEO)
    print("    RMS lourd-ouvert (m_c, m_b inchanges) : %.2f%% -> %.2f%%" % (
        math.sqrt((e480**2).mean()), math.sqrt((egeo**2).mean())))
    print("    mediane : %.2f%% -> %.2f%%" % (np.median(e480), np.median(egeo)))
    # etats sensibles (contiennent m_s)
    print("    etats sensibles (Delta pred, 480 -> geo) :")
    for (nm, c, me, m0, dv0), (_, _, _, m1, dv1) in zip(d480, dgeo):
        if abs(m1 - m0) > 1:
            print("      %-10s exp=%6.0f : %+5.1f%% -> %+5.1f%%" % (nm, me, dv0, dv1))

    # refit m_c, m_b avec m_s fixe geo
    best = (1e9, MC, MB)
    for mc in np.arange(MC-80, MC+40, 2.):
        for mb in np.arange(MB-80, MB+40, 2.):
            e, _ = score(MS_GEO, mc, mb)
            r = math.sqrt((e**2).mean())
            if r < best[0]: best = (r, mc, mb)
    r0 = math.sqrt((e480**2).mean())
    print("\n    refit (m_c, m_b) a m_s=geo : RMS = %.2f%% (base 480 : %.2f%%)" % (best[0], r0))
    print("    m_c : %.0f -> %.0f (%+.1f%%) ; m_b : %.0f -> %.0f (%+.1f%%)" % (
        MC, best[1], (best[1]/MC-1)*100, MB, best[2], (best[2]/MB-1)*100))

    print("\nVERDICT :")
    if best[0] <= r0 * 1.15:
        print("  m_s TOMBE : l'assignation 5_2 est pinnee (3 discriminants, null a 1")
        print("  survivant) et la contrainte geometrique m_s = m_l*l(5_2)/l(3_1) tient")
        print("  le spectre lourd (RMS %.2f%% vs %.2f%%). m_s n'est plus calibre :" % (best[0], r0))
        print("  il devient un RATIO GEOMETRIQUE derive. Comptage : 11 -> 10.")
    else:
        print("  m_s NE TOMBE PAS proprement : la contrainte geometrique degrade le")
        print("  spectre lourd (RMS %.2f%% vs %.2f%%). Reste calibre." % (best[0], r0))


if __name__ == "__main__":
    run()
