r"""
CATÉGORISATION TOPOLOGIQUE de 200+ particules PDG « à la manière TQNT ».

Applique systématiquement la méthode développée dans
`docs/brouillon_Ukappa_Utau_geometrie.tex` :
  1. taxonomie de l'AXE DE COMPLEXITÉ (6 catégories : base c=0 sans nœud /
     lepton / baryon octet / baryon décuplet / secteur EW / méson résiduel,
     + sous-cas exotiques et matière noire) ;
  2. cœur de nœud CATÉGORIEL par famille (lepton=3₁ torique, octet=7₂,
     décuplet=5₂, EW=8₃ amphichiral partagé, méson=construction K+K̄) ;
  3. signature de symétrie / multipôle du cœur (torique→dipôle pur Q=0
     robuste ; chiral→quadrupôle O(1) ; amphichiral→auto-conjugué) ;
  4. contrôle d'Alexander des cœurs (nœuds validés : 3₁,5₂,7₂,8₃,6₂,8₁,10₁,8₂).

RÉSERVES HONNÊTES (héritées du programme « courbes fausses ») :
  - la CATÉGORIE est robuste (elle découle du type de particule) ;
  - l'ENTRELACS complet est SOUS-DÉTERMINÉ (médiane ~4 candidats/particule,
    liberté d'assignation) — seul le CŒUR catégoriel est fixé ;
  - la signature MULTIPOLAIRE ne distingue QUE le lepton (dipôle pur) ; tout
    nœud non-torique donne O(1). La partition tient sur chiralité / identité /
    construction, pas sur le quadrupôle.
  - l'antiparticule = nœud MIROIR (chiralité). Les cœurs amphichiraux (8₃)
    sont leur propre antiparticule.

Source de la liste PDG (~122) : `exhaustive_scan.py` (lue par AST, non ré-
saisie). Étendue au-delà de 200 par les antiparticules (miroirs distincts).
Sortie : `docs/TQNT_categorisation_topologique.xlsx`.
"""
from __future__ import annotations
import ast
import os
from fractions import Fraction

HERE = os.path.dirname(__file__)
SCAN = os.path.join(HERE, "exhaustive_scan.py")


# --------------------------------------------------------------------------- #
#  1. Extraction de la liste PDG depuis exhaustive_scan.py (AST, sans exécuter)#
# --------------------------------------------------------------------------- #
def load_pdg():
    with open(SCAN, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PDG":
                    # eval restreint (la liste contient des expressions type 2/3)
                    return eval(compile(ast.Expression(node.value), SCAN, "eval"),
                                {"__builtins__": {}}, {})
    raise RuntimeError("liste PDG introuvable")


# --------------------------------------------------------------------------- #
#  2. Signature de cœur validée par Alexander (repair_curves_braid)           #
# --------------------------------------------------------------------------- #
# cœur -> (symétrie, multipôle ||Q||/(|m|R_g), Alexander)
CORE_SIGNATURE = {
    "3_1":  ("torique C3",  "0.00 (dipôle PUR, robuste)", "(1,-1,1)"),
    "7_2":  ("chiral C1",   "0.93 (quadrupôle O(1))",     "(3,-5,3)"),
    "5_2":  ("chiral C1",   "0.91 (quadrupôle O(1))",     "(2,-3,2)"),
    "8_3":  ("amphichiral", "0.86 (auto-conjugué)",       "(2,-4,5,-4,2)"),
    "K+K̄":  ("paire miroir","O(1) (non discriminant)",    "construction"),
    "—":    ("onde libre",  "N.A. (sans nœud)",           "—"),
    "≥5 nœuds": ("entrelacs","O(1)",                       "multi-composante"),
}


# --------------------------------------------------------------------------- #
#  3. Règles de catégorisation (déterministes : secteur + J -> catégorie)     #
# --------------------------------------------------------------------------- #
BASE_SECTORS = {"neutrino", "boson-jauge"}
MESON_SECTORS = {"méson-PS", "méson-V", "méson-T", "scalaire", "axial",
                 "charmonium", "bottomonium", "charmé", "charmé-V", "beau",
                 "beau-V"}
MESON_EXOTIC = {"exotique-X", "exotique-Y", "exotique-Zc", "exotique-Zb",
                "tetraquark", "tetra-c"}
BARYON_EXOTIC = {"pentaquark"}
# secteurs de RÉSONANCE baryonique (excitations orbitales/radiales, PAS le
# multiplet fondamental) : le J=3/2 d'un N* n'en fait pas un décuplet.
RESONANCE_BARYON = {"N*", "Δ*", "Λ*", "Σ**", "Ξ**",
                    "Λ_b*", "Σ_b*", "Ξ_b*", "Ω_b*"}


def is_half_integer(J):
    return (Fraction(J).limit_denominator(4) * 2) % 2 == 1


def categorize(name, m, J, Q, sector):
    """Retourne (catégorie, cœur, note_role)."""
    if sector in BASE_SECTORS or name == "γ":
        role = "photon (quantum de jauge)" if "γ" in name or sector == "boson-jauge" \
            else "neutrino (onde χ auto-rotative)"
        return "BASE (c=0, sans nœud)", "—", role
    if sector == "lepton":
        return "LEPTON", "3_1", "mode Z₃ du trèfle (e/μ/τ)"
    if sector == "EW":
        return "SECTEUR ÉLECTROFAIBLE", "8_3", "entrelacs contenant 8₃ (amphichiral partagé)"
    # fermions restants = baryons
    if is_half_integer(J):
        if sector in BARYON_EXOTIC:
            return "BARYON EXOTIQUE (pentaquark)", "≥5 nœuds", "entrelacs B=1 multi-nœuds"
        if sector in RESONANCE_BARYON:
            return "BARYON EXCITÉ", "7_2/5_2", f"résonance baryonique J={J} (excitation orbitale/radiale)"
        if J == 0.5:
            return "BARYON OCTET", "7_2", "cœur 7₂ (chiral), J=½"
        if J == 1.5:
            return "BARYON DÉCUPLET", "5_2", "cœur 5₂ (chiral), J=3/2"
        return "BARYON EXCITÉ", "7_2/5_2", f"cœur baryonique, J={J} (excité orbital/radial)"
    # bosons restants = mésons
    if sector in MESON_EXOTIC:
        return "MÉSON EXOTIQUE (tétra)", "K+K̄+…", "entrelacs résiduel multi-nœuds"
    if sector in ("charmonium", "bottomonium"):
        return "MÉSON (quarkonium)", "K+K̄", "K+K̄ diagonal (auto-conjugué, tour radiale)"
    return "MÉSON (résiduel K+K̄)", "K+K̄", "construction knot+miroir"


# particules auto-conjuguées (propre antiparticule) : pas de miroir distinct
SELF_CONJUGATE = {"γ", "π⁰", "η", "η'", "Z⁰", "H",
                  "η_c(1S)", "η_c(2S)", "η_b(1S)", "J/ψ", "ψ(2S)", "ψ(3770)",
                  "ψ(4040)", "Υ(1S)", "Υ(2S)", "Υ(3S)",
                  "χ_c0(1P)", "χ_c1(1P)", "χ_c2(1P)",
                  "χ_b0(1P)", "χ_b1(1P)", "χ_b2(1P)",
                  "ω", "φ", "f₀(500)", "f₀(980)", "f₀(1370)", "f₂(1270)",
                  "X(3872)", "X(3915)", "X(4140)", "X(4274)", "X(6900)",
                  "Y(4260)", "Y(4360)", "Y(4660)"}


# --------------------------------------------------------------------------- #
#  Extension PDG (états établis absents de exhaustive_scan.py)                 #
#  format : (nom, m_MeV, J, Q, secteur, auto_conjugué)                         #
# --------------------------------------------------------------------------- #
PDG_EXTENSION = [
    # ═══ MÉSONS LÉGERS NON ÉTRANGES (excités) ═══
    ("π(1300)⁰", 1300, 0, 0, "méson-PS", True), ("π₂(1670)⁰", 1670, 2, 0, "méson-T", True),
    ("η(1295)", 1294, 0, 0, "méson-PS", True), ("η(1405)", 1409, 0, 0, "méson-PS", True),
    ("η(1475)", 1476, 0, 0, "méson-PS", True), ("η₂(1645)", 1617, 2, 0, "méson-T", True),
    ("ρ(1450)⁰", 1465, 1, 0, "méson-V", True), ("ρ(1700)⁰", 1720, 1, 0, "méson-V", True),
    ("ρ₃(1690)⁰", 1689, 3, 0, "méson-V", True), ("ω(1420)", 1410, 1, 0, "méson-V", True),
    ("ω(1650)", 1670, 1, 0, "méson-V", True), ("ω₃(1670)", 1667, 3, 0, "méson-V", True),
    ("h₁(1170)", 1166, 1, 0, "axial", True), ("h₁(1415)", 1416, 1, 0, "axial", True),
    ("f₁(1285)", 1281, 1, 0, "axial", True), ("f₁(1420)", 1426, 1, 0, "axial", True),
    ("f₀(1500)", 1506, 0, 0, "scalaire", True), ("f₀(1710)", 1704, 0, 0, "scalaire", True),
    ("f₂'(1525)", 1517, 2, 0, "méson-T", True), ("f₂(1950)", 1936, 2, 0, "méson-T", True),
    ("f₄(2050)", 2018, 4, 0, "méson-T", True), ("a₀(1450)⁰", 1474, 0, 0, "scalaire", True),
    ("a₂(1700)⁰", 1698, 2, 0, "méson-T", True), ("b₁(1235)⁰", 1230, 1, 0, "axial", True),
    ("π₁(1600)⁰", 1660, 1, 0, "exotique-X", True),  # hybride exotique
    # ═══ MÉSONS ÉTRANGES (excités) ═══
    ("K(1460)±", 1460, 0, 1, "méson-PS", False), ("K*(1410)±", 1414, 1, 1, "méson-V", False),
    ("K₁(1270)±", 1253, 1, 1, "axial", False), ("K₁(1400)±", 1403, 1, 1, "axial", False),
    ("K*₀(1430)±", 1425, 0, 1, "scalaire", False), ("K₂*(1430)±", 1427, 2, 1, "méson-T", False),
    ("K*(1680)±", 1718, 1, 1, "méson-V", False), ("K₂(1770)±", 1773, 2, 1, "méson-T", False),
    ("K₃*(1780)±", 1776, 3, 1, "méson-V", False), ("K₄*(2045)±", 2045, 4, 1, "méson-T", False),
    ("K*(892)⁰", 895.5, 1, 0, "méson-V", False),
    # ═══ MÉSONS CHARMÉS / BEAUX (excités) ═══
    ("D₁(2420)⁰", 2422, 1, 0, "charmé-V", False), ("D₂*(2460)⁰", 2461, 2, 0, "charmé-V", False),
    ("D₂*(2460)±", 2465, 2, 1, "charmé-V", False), ("D_s0*(2317)±", 2318, 0, 1, "charmé", False),
    ("D_s1(2460)±", 2459, 1, 1, "charmé-V", False), ("D_s1(2536)±", 2535, 1, 1, "charmé-V", False),
    ("D_s2*(2573)±", 2569, 2, 1, "charmé-V", False), ("B₁(5721)⁰", 5726, 1, 0, "beau-V", False),
    ("B₂*(5747)⁰", 5739, 2, 0, "beau-V", False), ("B_s1(5830)⁰", 5829, 1, 0, "beau-V", False),
    ("B_s2*(5840)⁰", 5840, 2, 0, "beau-V", False), ("B_s*⁰", 5415, 1, 0, "beau-V", False),
    # ═══ CHARMONIUM / BOTTOMONIUM (excités) ═══
    ("h_c(1P)", 3525, 1, 0, "charmonium", True), ("ψ(3823)", 3823, 2, 0, "charmonium", True),
    ("ψ(4160)", 4191, 1, 0, "charmonium", True), ("ψ(4415)", 4421, 1, 0, "charmonium", True),
    ("χ_c1(3872)", 3872, 1, 0, "charmonium", True), ("η_c2(1D)", 3820, 2, 0, "charmonium", True),
    ("Υ(4S)", 10579, 1, 0, "bottomonium", True), ("Υ(10860)", 10885, 1, 0, "bottomonium", True),
    ("Υ(11020)", 11000, 1, 0, "bottomonium", True), ("η_b(2S)", 9999, 0, 0, "bottomonium", True),
    ("h_b(1P)", 9899, 1, 0, "bottomonium", True), ("h_b(2P)", 10260, 1, 0, "bottomonium", True),
    ("χ_b0(2P)", 10233, 0, 0, "bottomonium", True), ("χ_b1(2P)", 10255, 1, 0, "bottomonium", True),
    ("χ_b2(2P)", 10269, 2, 0, "bottomonium", True), ("χ_b1(3P)", 10514, 1, 0, "bottomonium", True),
    ("Υ(1D)", 10164, 2, 0, "bottomonium", True),
    # ═══ BARYONS N* / Δ* (résonances) ═══
    ("N(1520)", 1515, 1.5, 1, "N*", False), ("N(1675)", 1675, 2.5, 1, "N*", False),
    ("N(1680)", 1685, 2.5, 1, "N*", False), ("N(1720)", 1720, 1.5, 1, "N*", False),
    ("N(1875)", 1875, 1.5, 1, "N*", False), ("N(1900)", 1920, 1.5, 1, "N*", False),
    ("N(2190)", 2180, 3.5, 1, "N*", False), ("N(2220)", 2250, 4.5, 1, "N*", False),
    ("N(2250)", 2280, 4.5, 1, "N*", False), ("Δ(1900)", 1860, 0.5, 1, "Δ*", False),
    ("Δ(1910)", 1900, 0.5, 1, "Δ*", False), ("Δ(1920)", 1920, 1.5, 1, "Δ*", False),
    ("Δ(1930)", 1950, 2.5, 1, "Δ*", False), ("Δ(1950)", 1930, 3.5, 1, "Δ*", False),
    ("Δ(2420)", 2400, 5.5, 1, "Δ*", False),
    # ═══ HYPÉRONS EXCITÉS Λ* / Σ* / Ξ* ═══
    ("Λ(1405)", 1405, 0.5, 0, "Λ*", False), ("Λ(1520)", 1519, 1.5, 0, "Λ*", False),
    ("Λ(1600)", 1600, 0.5, 0, "Λ*", False), ("Λ(1670)", 1674, 0.5, 0, "Λ*", False),
    ("Λ(1690)", 1690, 1.5, 0, "Λ*", False), ("Λ(1820)", 1820, 2.5, 0, "Λ*", False),
    ("Λ(1830)", 1825, 2.5, 0, "Λ*", False), ("Λ(2100)", 2100, 3.5, 0, "Λ*", False),
    ("Σ(1660)", 1660, 0.5, 1, "Σ**", False), ("Σ(1670)", 1675, 1.5, 1, "Σ**", False),
    ("Σ(1775)", 1775, 2.5, 1, "Σ**", False), ("Σ(1915)", 1915, 2.5, 1, "Σ**", False),
    ("Σ(2030)", 2030, 3.5, 1, "Σ**", False), ("Ξ(1690)", 1690, 0.5, 0, "Ξ**", False),
    ("Ξ(1820)", 1823, 1.5, 0, "Ξ**", False), ("Ξ(2030)", 2025, 2.5, 0, "Ξ**", False),
    # ═══ BARYONS LOURDS (excités) ═══
    ("Λ_c(2595)⁺", 2592, 0.5, 1, "charmé-B", False), ("Λ_c(2625)⁺", 2628, 1.5, 1, "charmé-B", False),
    ("Λ_c(2860)⁺", 2856, 1.5, 1, "charmé-B", False), ("Λ_c(2880)⁺", 2881, 2.5, 1, "charmé-B", False),
    ("Ξ_c'(2578)", 2578, 0.5, 1, "charmé-B", False), ("Ξ_c(2645)", 2646, 1.5, 1, "charmé-B", False),
    ("Ξ_c(2790)", 2792, 0.5, 1, "charmé-B", False), ("Ξ_c(2815)", 2817, 1.5, 1, "charmé-B", False),
    ("Ω_c(2770)*", 2766, 1.5, 0, "charmé-B", False), ("Ω_cc⁺", 3738, 0.5, 1, "double-c", False),
    ("Λ_b(5912)⁰", 5912, 0.5, 0, "beau-B", False), ("Λ_b(5920)⁰", 5920, 1.5, 0, "beau-B", False),
    ("Ξ_b'(5935)⁻", 5935, 0.5, -1, "beau-B", False), ("Ξ_b(5945)⁰", 5949, 1.5, 0, "beau-B", False),
    ("Ξ_b(6100)⁻", 6100, 1.5, -1, "beau-B", False), ("Ξ_bc⁰", 6900, 0.5, 0, "double-c", False),
    # ═══ EXOTIQUES (récents) ═══
    ("P_c(4457)⁺", 4457, 0.5, 1, "pentaquark", False), ("P_cs(4338)", 4338, 0.5, 0, "pentaquark", False),
    ("Z_cs(3985)", 3985, 1, 1, "exotique-Zc", False), ("Z_cs(4000)", 4003, 1, 1, "exotique-Zc", False),
    ("T_cs(2900)⁰", 2900, 0, 0, "tetraquark", False), ("T_cs0(2900)", 2866, 0, 0, "tetraquark", False),
    ("X(3960)", 3960, 0, 0, "tetra-c", True), ("χ_c0(3915)", 3922, 0, 0, "charmonium", True),
    ("T_ψψ(6900)", 6900, 0, 0, "tetra-c", True), ("Z(4430)⁺", 4478, 1, 1, "exotique-Zc", False),
]


def build_rows():
    pdg = [(n, m, J, Q, s, None) for (n, m, J, Q, s) in load_pdg()] + \
          [(n, m, J, Q, s, sc) for (n, m, J, Q, s, sc) in PDG_EXTENSION]
    rows = []
    for name, m, J, Q, sector, self_conj in pdg:
        cat, core, role = categorize(name, m, J, Q, sector)
        # charge fractionnaire = étiquette de quark SM ; en TQNT la charge
        # Q=Σχ+Lk est ENTIÈRE (l'objet réel est l'entrelacs EW à Q entier)
        if Q is not None and Fraction(Q).limit_denominator(9).denominator != 1:
            role += " [Q fractionnaire = étiquette quark SM ; TQNT : entrelacs Q entier]"
        sym, multi, alx = CORE_SIGNATURE.get(core, ("—", "—", "—"))
        rows.append(dict(name=name, m=m, J=J, Q=Q, sector=sector, cat=cat,
                         core=core, sym=sym, multi=multi, alx=alx, role=role,
                         anti=False))
        # auto-conjugué : flag explicite (extension) sinon jeu de noms (base)
        is_self = self_conj if self_conj is not None else (name in SELF_CONJUGATE)
        # antiparticule = miroir (sauf auto-conjugué)
        if not is_self and Q is not None:
            aname = _anti_name(name)
            rows.append(dict(name=aname, m=m, J=J, Q=(-Q if Q else 0),
                             sector=sector, cat=cat, core=core + " (miroir)",
                             sym=sym + " miroir", multi=multi, alx=alx,
                             role="antiparticule = nœud miroir", anti=True))
    return rows


def _anti_name(name):
    """Nom de l'antiparticule. Préfixe « anti- » UNIFORME pour éviter les
    collisions : Σ⁺ et Σ⁻ sont des baryons DISTINCTS (uus/dds), pas une paire
    particule-antiparticule ; nommer l'anti de Σ⁺ « Σ⁻ » écraserait le vrai Σ⁻.
    « anti-Σ⁺ » est sans ambiguïté (charge opposée gérée séparément)."""
    return "anti-" + name


# --------------------------------------------------------------------------- #
#  4. Sortie                                                                   #
# --------------------------------------------------------------------------- #
def main():
    rows = build_rows()
    from collections import Counter
    catcount = Counter(r["cat"] for r in rows)
    print(f"TOTAL catégorisé : {len(rows)} particules "
          f"({sum(not r['anti'] for r in rows)} + {sum(r['anti'] for r in rows)} antiparticules)\n")
    print("Répartition par catégorie TQNT :")
    for cat, n in catcount.most_common():
        print(f"  {n:4d}  {cat}")

    try:
        import openpyxl
    except ImportError:
        print("\n(openpyxl absent : pas d'excel, table imprimée seulement)")
        return rows
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Catégorisation topologique"
    hdr = ["Particule", "m (MeV)", "J", "Q", "secteur PDG",
           "CATÉGORIE TQNT", "cœur (catégoriel)", "symétrie cœur",
           "multipôle ||Q||/(|m|R_g)", "Alexander cœur", "rôle / note", "anti?"]
    ws.append(hdr)
    for r in rows:
        ws.append([r["name"], r["m"], r["J"], r["Q"], r["sector"], r["cat"],
                   r["core"], r["sym"], r["multi"], r["alx"], r["role"],
                   "oui" if r["anti"] else ""])
    # onglet réserves
    ws2 = wb.create_sheet("Réserves")
    for line in [
        "RÉSERVES (méthode = brouillon_Ukappa_Utau_geometrie.tex)",
        "",
        "1. La CATÉGORIE est robuste : elle découle du type de particule (spin, secteur).",
        "2. L'ENTRELACS complet est SOUS-DÉTERMINÉ (médiane ~4 candidats/particule).",
        "   Seul le CŒUR catégoriel est fixé (lepton 3₁ / octet 7₂ / décuplet 5₂ /",
        "   EW 8₃ / méson K+K̄).",
        "3. La signature MULTIPOLAIRE ne distingue QUE le lepton (torique, dipôle pur",
        "   Q=0 exact). Tout nœud non-torique donne O(1) : le ratio ne sépare pas",
        "   baryon/EW/méson. La partition tient sur chiralité / identité / construction.",
        "4. L'antiparticule = nœud MIROIR ; les cœurs amphichiraux (8₃) sont leur",
        "   propre antiparticule.",
        "5. Cœurs validés par Alexander (repair_curves_braid.py) : 3₁,5₂,7₂,8₃,6₂,8₁,10₁,8₂.",
        "6. Limite du contrôle : 7₄ et 9₂ partagent le même Alexander (4,-7,4).",
    ]:
        ws2.append([line])
    out = os.path.join(HERE, "..", "..", "docs", "TQNT_categorisation_topologique.xlsx")
    out = os.path.normpath(out)
    wb.save(out)
    print(f"\nExcel : {out}")
    return rows


if __name__ == "__main__":
    main()
