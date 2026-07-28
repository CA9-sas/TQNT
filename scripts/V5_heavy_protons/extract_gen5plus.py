"""
Extraction des candidats « protons lourds Gen 5+ » du catalogue V5.

Critères topologiques (calqués sur le proton ordinaire 3_1+7_2+7_2) :
  - n = 3 composantes (baryon confiné, facteur f(3) = 0,77)
  - J spectre contient 0,5 (spin fondamental ½)
  - Q_phys_min compatible avec une charge entière +1 après correction P21
  - c_tot ∈ [18, 30] (Gen 5+ : strictement plus complexe que le proton à c=17)

Le proton (référence Gen 1) :
  config = 3_1+7_2+7_2
  c = 17  J = 0,5  Q_topo Jones = 4,238  Lk_PDG = -3
  m = 938,54 MeV

On extrait tous les entrelacs satisfaisant les contraintes ci-dessus,
ordonne par masse croissante, et exporte en CSV + Markdown.
"""

import openpyxl
from pathlib import Path
import csv
import sys

XLSX = Path("docs/TQNT_catalog_V5_diffusion.xlsx")
OUT_DIR = Path("output")
OUT_CSV = OUT_DIR / "heavy_protons_gen5plus.csv"
OUT_MD = OUT_DIR / "heavy_protons_gen5plus.md"

# Référence : proton Q_phys_min mesuré
Q_PHYS_PROTON = 0.238
Q_PHYS_TOL = 0.02   # tolérance sur Q_phys_min (modulo 1 indistinguable)
C_MIN = 18
C_MAX = 30


def has_spin_half(j_spectre: str) -> bool:
    """Le spectre contient-il 0.5 ?"""
    if not j_spectre:
        return False
    tokens = [t.strip() for t in str(j_spectre).split(",")]
    return any(t == "0.5" or t == "1/2" for t in tokens)


def q_phys_compatible_proton(q_phys_min: float | None) -> bool:
    """Q_phys_min compatible avec une charge entière +1 modulo P21."""
    if q_phys_min is None:
        return False
    try:
        q = float(q_phys_min)
    except (TypeError, ValueError):
        return False
    # Charge cible +1 : Q_phys_min ≈ ±Q_PHYS_PROTON ou ≈ ±(1 - Q_PHYS_PROTON)
    candidates = [Q_PHYS_PROTON, -Q_PHYS_PROTON,
                  1.0 - Q_PHYS_PROTON, -(1.0 - Q_PHYS_PROTON)]
    return any(abs(q - c) < Q_PHYS_TOL for c in candidates)


def main():
    if not XLSX.exists():
        sys.exit(f"Catalogue introuvable : {XLSX}")
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["2. Catalogue unifié"]

    candidates = []
    for row in ws.iter_rows(min_row=3, values_only=True):
        try:
            num = int(row[0]) if row[0] is not None else None
            n = int(row[2]) if row[2] is not None else 0
            c = int(row[3]) if row[3] is not None else 0
        except (TypeError, ValueError):
            continue
        config = str(row[1] or "").strip()
        if n != 3:
            continue
        if not (C_MIN <= c <= C_MAX):
            continue
        if not has_spin_half(row[4]):
            continue
        if not q_phys_compatible_proton(row[7]):
            continue
        m_ots = row[9]
        if m_ots is None:
            continue

        candidates.append({
            "num": num,
            "config": config,
            "n": n,
            "c": c,
            "J": row[4],
            "Q_topo": row[5],
            "Lk_neutralisant": row[6],
            "Q_phys_min": row[7],
            "Lk_PDG": row[8],
            "m_V5_MeV": float(m_ots),
            "tau_TQNT": row[12],
            "type": row[13],
            "particule_PDG": row[14],
            "m_exp_MeV": row[15] if len(row) > 15 else None,
        })

    candidates.sort(key=lambda r: r["m_V5_MeV"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(candidates[0].keys()) if candidates else
                           ["num","config","n","c","J","Q_topo","Lk_neutralisant",
                            "Q_phys_min","Lk_PDG","m_V5_MeV","tau_TQNT","type",
                            "particule_PDG","m_exp_MeV"])
        w.writeheader()
        for r in candidates:
            w.writerow(r)

    # Markdown table
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("# Candidats « protons lourds Gen 5+ » V5 (c ∈ [18, 30])\n\n")
        fh.write(f"Filtre : n=3, J=½, Q_phys ≈ +1 (proton-like), c ∈ [{C_MIN}, {C_MAX}]\n\n")
        fh.write(f"**{len(candidates)} candidats trouvés.** Source : catalogue V5 onglet 2.\n\n")
        fh.write("| # | Config | c | m_V5 (MeV) | Q_phys_min | Lk_PDG | τ_TQNT | Type | PDG ? |\n")
        fh.write("|---:|---|---:|---:|---:|---:|---|---|---|\n")
        for r in candidates:
            pdg = r.get("particule_PDG") or ""
            pdg_str = str(pdg).replace("|", "\\|") if pdg else "—"
            type_str = str(r.get("type") or "")
            tau_str = str(r.get("tau_TQNT") or "?")
            fh.write(f"| {r['num']} | `{r['config']}` | {r['c']} | "
                     f"{r['m_V5_MeV']:.1f} | {r['Q_phys_min']:.3f} | "
                     f"{r['Lk_PDG']} | {tau_str} | {type_str} | {pdg_str} |\n")

    print(f"Candidats Gen 5+ trouvés : {len(candidates)}")
    print(f"  Plage de masse : {candidates[0]['m_V5_MeV']:.1f} → "
          f"{candidates[-1]['m_V5_MeV']:.1f} MeV")
    print(f"  Plage de c     : {min(r['c'] for r in candidates)} → "
          f"{max(r['c'] for r in candidates)}")
    print(f"  Sortie CSV : {OUT_CSV}")
    print(f"  Sortie MD  : {OUT_MD}")

    # Affiche les 10 plus légers et 5 plus lourds en console
    print("\n--- 10 plus légers ---")
    for r in candidates[:10]:
        pdg = r.get("particule_PDG") or "—"
        print(f"  #{r['num']:5d}  {r['config']:30s}  c={r['c']:>2d}  "
              f"m={r['m_V5_MeV']:>9.1f} MeV  PDG={pdg}")
    print("\n--- 5 plus lourds ---")
    for r in candidates[-5:]:
        pdg = r.get("particule_PDG") or "—"
        print(f"  #{r['num']:5d}  {r['config']:30s}  c={r['c']:>2d}  "
              f"m={r['m_V5_MeV']:>9.1f} MeV  PDG={pdg}")


if __name__ == "__main__":
    main()
