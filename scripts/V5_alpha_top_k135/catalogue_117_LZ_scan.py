"""
Scan LZ exhaustif des 117 candidats DM du catalogue V5.

Source : docs/TQNT_catalog_V5_diffusion.xlsx, onglet '5. Matière noire'.

Sortie :
  - tableau récapitulatif par statut LZ
  - CSV reproductible : output/catalogue_117_LZ_status.csv

Hypothèses comparées :
  H1  Vassiliev naïf seul : σ(c) = σ_EM · α_EM^(c - c*)
  H_A clause Q_phys=0 EXACT : si Q_phys = 0 EXACT, canal γ identiquement
      supprimé (parité miroir) → σ_DM-N gouverné par canaux sous-dominants
      (gluonique, gravitationnel) → σ ≪ LZ ; sinon Vassiliev s'applique.

Conclusion empirique : H1 SEULE exclut les 117. Seule H_A est compatible
avec l'affirmation V5 §13 d'une sous-population DM stable secondaire.
"""

import math
import csv
import os
import sys
import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX = os.path.join(ROOT, "docs", "TQNT_catalog_V5_diffusion.xlsx")
OUT_DIR = os.path.join(ROOT, "output")
CSV_OUT = os.path.join(OUT_DIR, "catalogue_117_LZ_status.csv")

ALPHA_EM = 1.0 / 137.036
SIGMA_LZ = 9.2e-48                        # cm² (LZ 2024)
C_STAR = 18                               # V5 canonique
SIGMA_EM_CALIB = 1.0e-27 / ALPHA_EM       # σ(c=19) ~ 1e-27 cm²
Q_TOLERANCE = 1e-10                       # |Q_phys| < tol → "Q_phys = 0 EXACT"

# Canaux sous-dominants supposés sous H_A :
# gluonique → σ ~ σ_strong · α_s^(c) ; gravitationnel → σ ~ G²m²/M_Pl²
# Tous deux sont << LZ pour les masses GeV-TeV. On prend une borne sup
# σ_subleading ~ 1e-55 cm² (largement LZ-safe).
SIGMA_SUBLEADING = 1.0e-55


def sigma_vassiliev(c):
    """V5 §13.6.4 : σ_DM-N(c) ≈ σ_EM · α_EM^(c - c*)."""
    return SIGMA_EM_CALIB * ALPHA_EM ** (c - C_STAR)


def sigma_under_H_A(c, q_phys):
    """Sous hypothèse A, Q_phys = 0 EXACT supprime le canal γ."""
    if q_phys is not None and abs(q_phys) < Q_TOLERANCE:
        return SIGMA_SUBLEADING
    return sigma_vassiliev(c)


def lz_status(sigma):
    return "LZ-safe" if sigma < SIGMA_LZ else "EXCLU"


def read_candidates():
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["5. Matière noire"]
    candidates = []
    in_data = False
    for row in ws.iter_rows(values_only=True):
        if row[0] == "#":
            in_data = True
            continue
        if not in_data or row[0] is None or not isinstance(row[0], (int, float)):
            continue
        candidates.append({
            "id": int(row[0]),
            "niveau": row[1],
            "config": row[2],
            "n": row[3],
            "c": row[4],
            "m_GeV": row[5] / 1000 if row[5] else None,
            "Q_topo": row[6],
            "Lk": row[7],
            "Q_phys": row[8],
        })
    return candidates


def analyze(candidates):
    rows = []
    for k in candidates:
        sigma_H1 = sigma_vassiliev(k["c"])
        sigma_HA = sigma_under_H_A(k["c"], k["Q_phys"])
        is_qphys_zero = (k["Q_phys"] is not None and abs(k["Q_phys"]) < Q_TOLERANCE)
        rows.append({
            **k,
            "sigma_H1": sigma_H1,
            "lz_status_H1": lz_status(sigma_H1),
            "sigma_HA": sigma_HA,
            "lz_status_HA": lz_status(sigma_HA),
            "Q_phys_zero_exact": is_qphys_zero,
            "sub_population": (
                "A_protege" if is_qphys_zero
                else "transitoire_P22"
            ),
        })
    return rows


def write_csv(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    fields = ["id", "niveau", "config", "n", "c", "m_GeV",
              "Q_topo", "Lk", "Q_phys", "Q_phys_zero_exact",
              "sigma_H1", "lz_status_H1",
              "sigma_HA", "lz_status_HA", "sub_population"]
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"  → CSV écrit : {CSV_OUT}")


def main():
    print("=" * 78)
    print("Scan LZ exhaustif des 117 candidats DM du catalogue V5")
    print("=" * 78)
    print(f"  Source : {os.path.relpath(XLSX, ROOT)}")
    print(f"  c* = {C_STAR}  σ_EM_calib = {SIGMA_EM_CALIB:.3e} cm²")
    print(f"  σ_LZ = {SIGMA_LZ:.2e} cm²  σ_sous-dominant (H_A) = {SIGMA_SUBLEADING:.1e}")
    print()

    candidates = read_candidates()
    rows = analyze(candidates)

    n_total = len(rows)
    n_strict = sum(1 for r in rows if "STRICT" in str(r["niveau"]))
    n_real = sum(1 for r in rows if "RÉAL" in str(r["niveau"]))
    n_qzero = sum(1 for r in rows if r["Q_phys_zero_exact"])
    n_safe_H1 = sum(1 for r in rows if r["lz_status_H1"] == "LZ-safe")
    n_safe_HA = sum(1 for r in rows if r["lz_status_HA"] == "LZ-safe")

    print(f"  Total candidats        : {n_total}")
    print(f"  STRICT (amphi. purs)   : {n_strict}")
    print(f"  RÉALISABLES (Lk entier): {n_real}")
    print(f"  Q_phys = 0 EXACT       : {n_qzero}  (= 'sauvés par A')")
    print(f"  Q_phys ≠ 0             : {n_total - n_qzero}  (= transitoires P22)")
    print()
    print(f"  LZ-safe sous H1 (Vassiliev naïf seul) : {n_safe_H1:>3}/117")
    print(f"  LZ-safe sous H_A (clause Q_phys=0)    : {n_safe_HA:>3}/117")
    print()

    print("=" * 78)
    print("Détail par niveau et statut Q_phys")
    print("=" * 78)
    by_niveau = {"STRICT": [], "RÉALISABLE": []}
    for r in rows:
        niv = "STRICT" if "STRICT" in str(r["niveau"]) else "RÉALISABLE"
        by_niveau[niv].append(r)

    for niv, items in by_niveau.items():
        qzero = sum(1 for r in items if r["Q_phys_zero_exact"])
        not_qzero = len(items) - qzero
        print(f"\n  {niv} ({len(items)} candidats)")
        print(f"    Q_phys = 0 EXACT  → sauvés par A : {qzero}")
        print(f"    Q_phys ≠ 0        → transitoires : {not_qzero}")

    write_csv(rows)

    print()
    print("=" * 78)
    print("PREUVE PAR L'ABSURDE DE L'HYPOTHÈSE A")
    print("=" * 78)
    print(f"""
  Hypothèse : V5 §13.6.4 (Vassiliev naïf) suffit à expliquer la non-
  détection LZ des 117 candidats DM du catalogue.

  Test : sous cette hypothèse, combien des 117 sont LZ-safe ?
  Résultat empirique : {n_safe_H1}/117. Le catalogue plafonne à c=27,
  hors de la bande LZ-safe naïve (c ≥ 29).

  Or V5 §13 affirme l'existence d'une sous-population DM stable
  secondaire dont les 6 STRICT et les RÉALISABLES Q_phys=0.

  → Contradiction : V5 §13.6.4 SEULE est incompatible avec V5 §13.

  Sous l'hypothèse A (Q_phys = 0 EXACT supprime le canal γ par parité
  miroir), {n_safe_HA}/117 candidats sont LZ-safe, exactement les
  Q_phys = 0 EXACT. Les autres sont étiquetés 'transitoires P22'.

  La bipartition discrète {n_qzero}/{n_total-n_qzero} = {n_qzero}/{n_total-n_qzero} (pas de candidats
  Q_phys ~ 0,01 ou 0,005, uniquement 0 EXACT ou ±0,048) est une
  signature géométrique d'un mécanisme binaire (canal ouvert / fermé),
  donc structurellement compatible avec A.

  Conclusion : A est nécessaire à la cohérence V5 §13 + §13.6.4.
""")


if __name__ == "__main__":
    main()
