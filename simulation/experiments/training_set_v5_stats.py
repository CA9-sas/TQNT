"""Stats du training set V5 (Excel diffusion) — pour comparer au test aveugle."""
import numpy as np
import openpyxl

wb = openpyxl.load_workbook(
    "docs/TQNT_catalog_V5_diffusion.xlsx", read_only=True
)
ws = wb["3. Particules PDG"]
rows = list(ws.iter_rows(values_only=True))[2:]

v5_pcts, plast_pcts, names = [], [], []
for r in rows:
    if r[0] is None:
        continue
    try:
        d_v5 = float(str(r[5]).replace("+", ""))
        d_plast = float(str(r[8]).replace("+", ""))
        names.append(r[0])
        v5_pcts.append(abs(d_v5))
        plast_pcts.append(abs(d_plast))
    except Exception:
        pass


def stats(a, name):
    print(f"  {name:<14s}  N={len(a):2d}  mean_abs={a.mean():6.2f}%  "
          f"median_abs={np.median(a):5.2f}%  RMS={np.sqrt(np.mean(a**2)):5.2f}%  "
          f"max={a.max():6.2f}%")


v5 = np.array(v5_pcts)
pl = np.array(plast_pcts)
print("=== TRAINING SET V5 (Excel diffusion) ===\n")
print("Toutes particules (incluant photon, outlier) :")
stats(v5, "V5 strict")
stats(pl, "V5 plast.")
print()
mask = v5 < 100  # élimine photon (Δ ≈ 5e5%)
print("Sans le photon (qui est noté m_exp=0 dans le fichier) :")
stats(v5[mask], "V5 strict")
stats(pl[mask], "V5 plast.")
print()
print("Particules >5% écart :")
for nm, d in zip(names, v5):
    if d > 5:
        print(f"    {nm:>14s}  Δ = {d:6.2f}%")
