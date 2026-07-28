r"""
Statistiques agrégées V5 strict vs V5+P23 (plasticité) à partir du catalogue
officiel ``docs/TQNT_catalog_V5_diffusion.xlsx`` (feuille « 3. Particules PDG »).

Contrairement à ``exhaustive_scan_p23.py`` (qui applique aveuglément ξ_BS
additif aux 118 matchs du scan), ce script utilise les colonnes ``m V5``
et ``m plast`` du catalogue formel — colonnes calculées par la formule
multiplicative officielle de P23 :

    m_mesurée(L) = m(L) · [1 + σ(L) · γ(L) · ε(n)]

avec ε(n) = 0 si n = 1 (P23 ne s'applique pas aux nœuds isolés), σ = +1
pour les pseudo-scalaires (J=0), −1 sinon, γ = C_plast/Σ U_κU_τ, et
C_plast = −2,92 MeV calibré sur μ⁻.

Aussi : 8 particules sont réassignées par P23 (la config V5 strict optimale
n'est plus la même une fois la plasticité prise en compte).

Usage :
    python3 -m simulation.experiments.p23_catalog_stats
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

try:
    import openpyxl
except ImportError:
    raise SystemExit("openpyxl requis : pip install openpyxl")

CATALOG = Path(__file__).resolve().parents[2] / "docs" / "TQNT_catalog_V5_diffusion.xlsx"
if not CATALOG.exists():
    raise SystemExit(f"Catalogue introuvable : {CATALOG}")

wb = openpyxl.load_workbook(CATALOG, data_only=True)
ws = wb["3. Particules PDG"]

rows = list(ws.iter_rows(values_only=True))
data = []
for r in rows[2:]:  # skip 2 header rows
    if r[0] is None:
        continue
    name, m_exp, tier, cfg_v5, m_v5, d_v5, cfg_pl, m_pl, d_pl, _ = r
    if m_exp is None or d_v5 is None:
        continue
    try:
        d_v5_f = float(str(d_v5).replace("+", ""))
        d_pl_f = float(str(d_pl).replace("+", "")) if d_pl is not None else d_v5_f
    except (ValueError, TypeError):
        continue
    data.append({"name": name, "m_exp": m_exp, "cfg_v5": cfg_v5, "d_v5": d_v5_f,
                 "cfg_pl": cfg_pl, "d_pl": d_pl_f,
                 "changed": cfg_v5 != cfg_pl})

# Exclure le photon (Δ_V5 ~ 5×10⁵ % parce que m_exp = 0)
clean = [d for d in data if abs(d["d_v5"]) < 100]

d_v5 = np.array([d["d_v5"] for d in clean])
d_pl = np.array([d["d_pl"] for d in clean])
a_v5 = np.abs(d_v5)
a_pl = np.abs(d_pl)

print(f"═══ Statistiques V5 strict vs V5+P23 sur {len(clean)} particules PDG ═══")
print(f"(catalogue : {CATALOG.name}, feuille « 3. Particules PDG », hors photon)\n")

print(f"  {'Métrique':<22} {'V5 strict':>12} {'V5 + P23':>12} {'gain':>10}")
print("  " + "-" * 60)
rms_v5, rms_pl = float(np.sqrt(np.mean(d_v5**2))), float(np.sqrt(np.mean(d_pl**2)))
med_v5, med_pl = float(np.median(a_v5)), float(np.median(a_pl))
mean_v5, mean_pl = float(np.mean(a_v5)), float(np.mean(a_pl))
max_v5, max_pl = float(np.max(a_v5)), float(np.max(a_pl))
for label, vv, vp in [("RMS", rms_v5, rms_pl), ("Médiane |Δm/m|", med_v5, med_pl),
                       ("Moyenne |Δm/m|", mean_v5, mean_pl), ("Max |Δm/m|", max_v5, max_pl)]:
    gain = (vp - vv) / vv * 100
    print(f"  {label:<22} {vv:>11.3f}% {vp:>11.3f}% {gain:>+9.1f}%")

print()
for thr in [0.5, 1.0, 2.0, 5.0]:
    n_v5 = int((a_v5 < thr).sum())
    n_pl = int((a_pl < thr).sum())
    print(f"  Couverture <{thr:>4.1f}%   :   V5 = {n_v5:>3}/{len(clean)}, "
          f"+ P23 = {n_pl:>3}/{len(clean)}  ({n_pl - n_v5:+d})")

changed = [d for d in clean if d["changed"]]
print(f"\n  Réassignées par P23 : {len(changed)}")
for d in changed:
    print(f"    {d['name']:<14} {d['cfg_v5']:<22} ({d['d_v5']:+.2f}%) "
          f"→ {d['cfg_pl']:<22} ({d['d_pl']:+.2f}%)")
