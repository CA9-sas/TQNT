r"""
VÉRIFICATION DE COHÉRENCE de la catégorisation topologique (407 particules).

Contrôle les invariants logiques de `categorize_pdg_topological.py` :
  1. cœur cohérent avec la catégorie (lepton→3₁, octet→7₂, décuplet→5₂,
     EW→8₃, méson→K+K̄, base→—) ;
  2. signature (symétrie/multipôle/Alexander) cohérente avec le cœur ;
  3. spin-statistique (catégories fermioniques → J demi-entier, mésons → entier ;
     BASE et EW sont mixtes par construction, exclus) ;
  4. charge ENTIÈRE (TQNT : Q=Σχ+Lk ∈ ℤ) — les charges fractionnaires sont des
     étiquettes de quark SM (notées, non-anomalies) ;
  5. conjugaison de charge : chaque particule non-auto-conjuguée a exactement une
     antiparticule (préfixe « anti- », sans collision), de charge opposée, même
     masse et même catégorie ; les auto-conjuguées (Q=0) n'en ont pas ;
  6. pas de doublon de nom.

Sortie : liste des anomalies (doit être vide) + notes documentées.
"""
from __future__ import annotations
from fractions import Fraction
from collections import Counter

from simulation.experiments.categorize_pdg_topological import (
    build_rows, CORE_SIGNATURE, _anti_name, SELF_CONJUGATE, PDG_EXTENSION)

EXPECT_CORE = {
    "LEPTON": "3_1", "BARYON OCTET": "7_2", "BARYON DÉCUPLET": "5_2",
    "SECTEUR ÉLECTROFAIBLE": "8_3", "MÉSON (résiduel K+K̄)": "K+K̄",
    "MÉSON (quarkonium)": "K+K̄", "BASE (c=0, sans nœud)": "—",
    "BARYON EXOTIQUE (pentaquark)": "≥5 nœuds", "MÉSON EXOTIQUE (tétra)": "K+K̄+…",
    "BARYON EXCITÉ": "7_2/5_2",
}
FERMION_CATS = {"LEPTON", "BARYON OCTET", "BARYON DÉCUPLET", "BARYON EXCITÉ",
                "BARYON EXOTIQUE (pentaquark)"}
MESON_CATS = {"MÉSON (résiduel K+K̄)", "MÉSON (quarkonium)", "MÉSON EXOTIQUE (tétra)"}


def _int_charge(Q):
    return Q is None or Fraction(Q).limit_denominator(9).denominator == 1


def _half_integer(J):
    return (Fraction(J).limit_denominator(4) * 2) % 2 == 1


def verify():
    rows = build_rows()
    V, NOTES = [], []
    seen = set()
    for r in rows:
        if r["name"] in seen:
            V.append(("DOUBLON", f"nom répété : {r['name']}"))
        seen.add(r["name"])

    for r in rows:
        nm, J, Q, cat, core = r["name"], r["J"], r["Q"], r["cat"], r["core"]
        base_core = core.replace(" (miroir)", "")
        if cat in EXPECT_CORE and base_core != EXPECT_CORE[cat]:
            V.append(("CŒUR", f"{nm}: {cat} attend {EXPECT_CORE[cat]}, a {base_core}"))
        exp = CORE_SIGNATURE.get(base_core)
        if exp:
            esym, emulti, ealx = exp
            want = esym + " miroir" if r["anti"] else esym
            if r["sym"] != want:
                V.append(("SIGN", f"{nm}: symétrie {r['sym']!r} != {want!r}"))
            if r["multi"] != emulti:
                V.append(("SIGN", f"{nm}: multipôle {r['multi']!r} != {emulti!r}"))
            if r["alx"] != ealx:
                V.append(("SIGN", f"{nm}: Alexander {r['alx']!r} != {ealx!r}"))
        if cat in FERMION_CATS and not _half_integer(J):
            V.append(("SPIN", f"{nm}: fermion {cat} mais J={J} entier"))
        if cat in MESON_CATS and _half_integer(J):
            V.append(("SPIN", f"{nm}: méson {cat} mais J={J} demi-entier"))
        if not _int_charge(Q):
            NOTES.append(f"{nm}: Q={Q} = étiquette quark SM (attendu, documenté)")

    selfset = set(SELF_CONJUGATE) | {n for (n, *_r, sc) in PDG_EXTENSION if sc}
    anti = {r["name"]: r for r in rows if r["anti"]}
    for r in (x for x in rows if not x["anti"]):
        nm, Q = r["name"], r["Q"]
        an = _anti_name(nm)
        if nm in selfset:
            if Q not in (0, None):
                V.append(("SELF-Q", f"{nm}: auto-conjugué mais Q={Q}≠0"))
            if an in anti:
                V.append(("SELF-ANTI", f"{nm}: auto-conjugué mais a une antiparticule"))
        else:
            if an not in anti:
                V.append(("ANTI-MANQ", f"{nm}: pas d'antiparticule (Q={Q})"))
            else:
                a = anti[an]
                if Q is not None and a["Q"] != -Q:
                    V.append(("ANTI-Q", f"{nm}: Q={Q}, anti Q={a['Q']} (≠ -Q)"))
                if a["m"] != r["m"]:
                    V.append(("ANTI-M", f"{nm}: masse anti ≠"))
                if a["cat"] != r["cat"]:
                    V.append(("ANTI-CAT", f"{nm}: catégorie anti ≠"))
    return rows, V, NOTES


def main():
    rows, V, NOTES = verify()
    npart = sum(not r["anti"] for r in rows)
    print(f"=== COHÉRENCE : {len(rows)} lignes ({npart} + {len(rows) - npart} anti) ===\n")
    if not V:
        print("✓ AUCUNE incohérence sur tous les invariants.")
    else:
        for k, n in Counter(k for k, _ in V).most_common():
            print(f"  {n:3d}  {k}")
        for k, msg in V:
            print(f"   [{k}] {msg}")
    if NOTES:
        print(f"\nNOTES documentées (non-anomalies) : {len(NOTES)}")
        for n in NOTES:
            print(f"   • {n}")
    return not V


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
