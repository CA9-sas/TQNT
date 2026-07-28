"""
Patch 2 — annoter l'onglet '5. Matière noire' du catalogue V5 avec une
colonne 'Statut LZ (clause Q_phys=0)'.

Convention :
  - Q_phys = 0 EXACT       → 'LZ-safe (A)'        — DM stable secondaire
  - |Q_phys| > 0            → 'EXCLU (Vassiliev)'  — transitoire P22

Source : TQNT_detaillee_V5_FR.tex §18.12.2 (clause Q_phys-bipartition).

Effet : ajoute la colonne en position 10 (après Q_phys), conserve toutes
les cellules existantes, ajoute également un mini-récapitulatif en bas.
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX = os.path.join(ROOT, "docs", "TQNT_catalog_V5_diffusion.xlsx")

Q_TOLERANCE = 1e-10

GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
RED_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
HEADER_FONT = Font(bold=True)
HEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
CENTER = Alignment(horizontal="center", vertical="center")


def main():
    wb = openpyxl.load_workbook(XLSX)
    ws = wb["5. Matière noire"]

    # Locate header row
    header_row_idx = None
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if row[0] == "#":
            header_row_idx = i
            break
    if header_row_idx is None:
        print("ERREUR : en-tête '#' introuvable.", file=sys.stderr)
        sys.exit(1)

    new_col = 10
    ws.cell(row=header_row_idx, column=new_col, value="Statut LZ (clause Q_phys=0)")
    ws.cell(row=header_row_idx, column=new_col).font = HEADER_FONT
    ws.cell(row=header_row_idx, column=new_col).fill = HEADER_FILL
    ws.cell(row=header_row_idx, column=new_col).alignment = CENTER

    n_safe = 0
    n_excl = 0
    last_data_row = header_row_idx
    for row_idx in range(header_row_idx + 1, ws.max_row + 1):
        id_cell = ws.cell(row=row_idx, column=1).value
        q_phys_cell = ws.cell(row=row_idx, column=9).value
        if not isinstance(id_cell, (int, float)):
            continue
        if q_phys_cell is None:
            continue
        last_data_row = row_idx
        is_zero = abs(float(q_phys_cell)) < Q_TOLERANCE
        if is_zero:
            label = "LZ-safe (A)"
            fill = GREEN_FILL
            n_safe += 1
        else:
            label = "EXCLU (Vassiliev)"
            fill = RED_FILL
            n_excl += 1
        cell = ws.cell(row=row_idx, column=new_col, value=label)
        cell.fill = fill
        cell.alignment = CENTER

    # Set sensible column width
    from openpyxl.utils import get_column_letter
    ws.column_dimensions[get_column_letter(new_col)].width = 26

    # Recap row 2 lines below last data
    recap_row = last_data_row + 2
    ws.cell(row=recap_row, column=1,
            value=f"Bilan clause Q_phys=0  (TQNT_detaillee_V5_FR.tex §18.12.2)")
    ws.cell(row=recap_row, column=1).font = HEADER_FONT
    ws.cell(row=recap_row + 1, column=1,
            value=f"  LZ-safe (A) — Q_phys = 0 exact   : {n_safe} / 117")
    ws.cell(row=recap_row + 2, column=1,
            value=f"  EXCLU (Vassiliev) — Q_phys ≠ 0   : {n_excl} / 117")
    ws.cell(row=recap_row + 3, column=1,
            value=f"  Référence : scripts/V5_alpha_top_k135/catalogue_117_LZ_scan.py")

    wb.save(XLSX)
    print(f"Catalogue annoté : {n_safe} LZ-safe / {n_excl} EXCLU sur 117")
    print(f"Sauvegardé : {XLSX}")


if __name__ == "__main__":
    main()
