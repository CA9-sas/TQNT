"""Catalogue TQNT-V6 complet : masse au repos + masse haute énergie + durée de vie.

Repart de docs/TQNT_catalog_complete.xlsx (masses repos m_Ω de V5, vérité-terrain)
et AJOUTE pour chaque configuration :
  - m_repos     : masse Ω_ts au repos (confinée), reprise du catalogue V5 ;
  - m_deconf    : masse à HAUTE ÉNERGIE = Σ masses des composantes isolées
                  (l'entrelacs se sépare ; les sous-noeuds deviennent libres) ;
  - E_liaison   : m_repos − m_deconf = énergie de liaison topologique (tension
                  d'enlacement), libérée au déconfinement ;
  - E_seuil     : seuil de déconfinement = E₀·Σρ_J ~ ħc/r_link (rétraction χ) ;
  - tau_TQNT    : durée de vie — stable si plus léger de son secteur (charge, spin),
                  sinon estimation Sargent depuis l'écart de masse au plus proche.

Sortie : docs/TQNT_catalog_V6_complete.xlsx
"""
from __future__ import annotations
import re
import numpy as np
import openpyxl

from theory.chi_field import jones_rho

SRC = "docs/TQNT_catalog_complete.xlsx"
OUT = "docs/TQNT_catalog_V6_complete.xlsx"
E0 = 33.0          # MeV : E₀ tel que E_K=E₀·ρ_J ~ Λ_knot~200 MeV (ħc/r_link)
GF, HBAR, PI = 1.1664e-5, 6.582e-25, np.pi


def components(cfg):
    return [k.strip() for k in re.split(r'[+\s]+', str(cfg)) if re.match(r'^\d+_\d+$', k.strip())]


def main():
    wb = openpyxl.load_workbook(SRC, data_only=True, read_only=True)
    ws = wb["Catalogue TQNT V5"]
    rows = [r for r in ws.iter_rows(min_row=7, values_only=True) if r[1] and r[4]]

    # masses isolées (repos) des 19 briques
    iso = {r[1].strip(): float(r[4]) for r in rows if str(r[2]) == "1"}

    cat = []
    for r in rows:
        cfg = str(r[1]).strip()
        comp = components(cfg)
        if not comp:
            continue
        m_repos = float(r[4])
        m_deconf = sum(iso[k] for k in comp)
        rhoJ = sum(jones_rho(k) for k in comp)
        E_seuil = E0 * rhoJ
        spin = str(r[5]) if r[5] else ""
        Qtopo = r[6]
        try:
            Q = round(float(str(Qtopo).replace("+", ""))) if Qtopo not in (None, "—") else None
        except ValueError:
            Q = None
        cat.append({"cfg": cfg, "n": r[2], "c": r[3], "m_repos": m_repos,
                    "m_deconf": m_deconf, "E_liaison": m_repos - m_deconf,
                    "E_seuil": E_seuil, "spin": spin, "Q": Q,
                    "pdg": r[9], "m_exp": r[10], "tau_exp": r[14]})

    # Durée de vie : LA formule validée Γ = G_F² |M|² Φ_n(Q), Q physique = ΔΩ_ts.
    # Q = écart de masse (V5) au produit de désintégration = config plus lourde
    # juste en dessous, de charge compatible (|ΔQ|≤1, un lepton émis). |M|²=1 (base),
    # Φ_3 = Q⁵/192π³ (3-corps). Découplée de la prédiction de masse (Q physique).
    M2 = 1.0
    for e in cat:
        daughters = [o for o in cat if o["m_repos"] < e["m_repos"] - 0.01
                     and (e["Q"] is None or o["Q"] is None or abs((o["Q"] or 0) - (e["Q"] or 0)) <= 1)]
        if not daughters:
            e["tau"] = "stable"; e["Qdec"] = 0.0; e["daughter"] = "—"
            continue
        dau = max(daughters, key=lambda o: o["m_repos"])   # plus proche en dessous
        Q = (e["m_repos"] - dau["m_repos"]) / 1000.0        # GeV
        e["Qdec"] = Q * 1000; e["daughter"] = dau["cfg"]
        G = GF**2 * M2 * Q**5 / (192*PI**3)                 # Φ_3 (3-corps)
        e["tau"] = HBAR / G if G > 0 else "stable"

    # écriture
    out = openpyxl.Workbook(); o = out.active; o.title = "Catalogue V6"
    o.append(["#", "Config TQNT", "n", "c", "m_repos (MeV)", "m_deconf HE (MeV)",
              "E_liaison (MeV)", "E_seuil déconf (MeV)", "Spin", "Q", "produit désint.",
              "Q_désint (MeV)", "τ = G_F²Φ_3(Q) (s)", "Particule PDG", "m_exp (MeV)", "τ_exp (mesuré)"])
    for i, e in enumerate(cat, 1):
        tau = e["tau"] if isinstance(e["tau"], str) else f"{e['tau']:.2e}"
        o.append([i, e["cfg"], e["n"], e["c"], round(e["m_repos"], 2), round(e["m_deconf"], 2),
                  round(e["E_liaison"], 2), round(e["E_seuil"], 1), e["spin"], e["Q"],
                  e.get("daughter", "—"), round(e.get("Qdec", 0), 2), tau,
                  e["pdg"], e["m_exp"], e["tau_exp"]])
    out.save(OUT)
    nstab = sum(1 for e in cat if e["tau"] == "stable")
    print(f"{len(cat)} configs écrites dans {OUT} ; stables : {nstab}")
    print("Validation (Γ = G_F²·Φ_3(Q), |M|²=1) :")
    byc = {e["cfg"].replace(" ", ""): e for e in cat}
    for nm, key, exp in [("neutron", "3_1+7_2+7_2", "879 s (β, |M|²≈10)"),
                         ("muon", "3_1+6_1", "2,2e-6 s")]:
        e = byc.get(key)
        if e:
            tau = e["tau"] if isinstance(e["tau"], str) else f"{e['tau']:.2e}"
            print(f"  {nm:>8} {key}: produit={e.get('daughter')} Q={e.get('Qdec',0):.2f} MeV → τ={tau}  (exp {exp})")


if __name__ == "__main__":
    main()
