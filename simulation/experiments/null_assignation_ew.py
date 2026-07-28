"""(c) Null d'assignation du quartet electrofaible (17/07).

REGLE V6 (sec:v6-null-assignation) : tout accord par assignation doit
passer le null. Le quartet EW (W=7_1+7_2+8_3, Z=7_2+8_3+9_1,
H=7_1+8_3+10_1, top=8_2+8_3+10_1 ; RMS 0.74 %) ne l'avait pas passe.

NULL : dans la classe de formules m = a0 U_EM (c_J)^3.55 (secteur EW,
f=1), combien de quadruplets de triplets (parmi tous les multisets de 3
noeuds du catalogue) reproduisent les 4 masses PDG au meme RMS <= 0.74 %,
en s'autorisant UNE calibration globale par candidat (RMS centre --
necessaire : le schema U_EM disponible, Abraham N50K, differe du schema
calibre post-glitch des commits, qui ne reproduit pas W=80.7 ici) ?

Si le compte attendu >> 1 : l'accord n'est pas une evidence d'assignation
(liberte combinatoire suffisante). Contrainte structurelle testee a part :
les 4 triplets partagent un noeud commun (le « quartet 8_3 »).
"""
import sys
import itertools
import numpy as np

sys.path.insert(0, ".")
from theory.mass_formula import mass
from theory.knot_catalog import KNOT_NAMES

PDG = {"W": 80.377e3, "Z": 91.1876e3, "H": 125.25e3, "top": 172.69e3}
RMS_TARGET = 0.0074


def run():
    knots = [k for k in KNOT_NAMES if k != '0_1']
    print("catalogue : %d noeuds" % len(knots))
    trips, logs = [], []
    for c in itertools.combinations_with_replacement(knots, 3):
        try:
            m = mass("+".join(c), "ew")
        except Exception:
            continue
        if m > 0:
            trips.append(c)
            logs.append(np.log(m))
    logs = np.array(logs)
    T = len(trips)
    print("triplets valides : %d ; quadruplets possibles ~ %.1e\n" % (T, float(T) ** 4))

    X = {k: logs - np.log(v) for k, v in PDG.items()}
    rng = np.random.default_rng(3)
    N = 4_000_000
    idx = rng.integers(0, T, size=(N, 4))
    M = np.stack([X["W"][idx[:, 0]], X["Z"][idx[:, 1]],
                  X["H"][idx[:, 2]], X["top"][idx[:, 3]]], 1)
    rms = M.std(axis=1)          # RMS centre (calibration globale libre)
    P = float((rms <= RMS_TARGET).mean())
    exp_count = P * float(T) ** 4
    print("[NULL A] calibration globale libre :")
    print("    P(RMS <= 0.74 %%) = %.2e -> quadruplets attendus = %.1e" % (P, exp_count))

    # NULL B : noeud commun impose aux 4 triplets
    counts = []
    for k in knots:
        sel = np.array([k in t for t in trips])
        idxs = np.nonzero(sel)[0]
        if len(idxs) < 4:
            continue
        n2 = min(400_000, len(idxs) ** 2 * 16)
        jj = rng.integers(0, len(idxs), size=(n2, 4))
        m4 = np.stack([X["W"][idxs[jj[:, 0]]], X["Z"][idxs[jj[:, 1]]],
                       X["H"][idxs[jj[:, 2]]], X["top"][idxs[jj[:, 3]]]], 1)
        p = float((m4.std(axis=1) <= RMS_TARGET).mean())
        counts.append((k, p * float(len(idxs)) ** 4))
    tot = sum(c for _, c in counts)
    top3 = sorted(counts, key=lambda x: -x[1])[:3]
    print("\n[NULL B] noeud commun impose (somme sur les %d noeuds) :" % len(counts))
    print("    quadruplets attendus = %.1e (dominants : %s)" % (
        tot, ", ".join("%s %.0e" % (k, c) for k, c in top3)))

    print("\nVERDICT :")
    if exp_count > 10:
        print(" - Le quartet EW NE PASSE PAS le null a calibration libre : ~%.0e" % exp_count)
        print("   quadruplets equivalents attendus dans la classe de formules --")
        print("   l'accord a 0.74 %% n'est PAS une evidence d'assignation ;")
        print("   la contrainte structurelle (noeud commun) ne suffit pas non")
        print("   plus (%.0e attendus)." % tot)
        print(" - A DECLASSER dans le registre du null : le quartet EW =")
        print("   COMPATIBILITE (assignation sous-determinee), pas preuve.")
        print("   Ce qui survit : la STRUCTURE (les 4 contiennent 8_3, doublet")
        print("   de writhe, predictions d'auto-conjugaison) -- pas le chiffre.")
    else:
        print(" - Quadruplets attendus %.1e : l'accord est significatif." % exp_count)
    print(" - Declare : schema U_EM Abraham N50K (le schema calibre post-")
    print("   glitch des commits n'est pas regenerable triplet par triplet).")


if __name__ == "__main__":
    run()
