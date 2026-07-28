"""Catalogue TQNT-V6 résolu en (config, Lk) + durée de vie par la formule validée.

Chaque configuration de noeuds porte un MULTIPLET de charge : les états (config, Lk)
diffèrent par le nombre d'enlacement Lk, donc par la charge Q = Σχ + Lk et par la
masse (tension d'enlacement) :

    m(config, Lk) = m_Ω + β·⟨√(c_J c_J)⟩·(Lk − Lk_opt)²        (β = 0,210 MeV)

avec Lk_opt = n (composantes) — l'état fondamental est (config, Lk_opt), de masse m_Ω
(catalogue V5). La désintégration FAIBLE = Lk → Lk±1 vers Lk_opt (émission d'un lepton,
ΔQ = ±1) ; on applique la formule validée :

    Γ = G_F²·|M|²·Φ_3(Q_désint),   Φ_3 = Q⁵/192π³,   Q_désint = écart de tension − m_e

État fondamental (Lk = Lk_opt) : aucun Lk inférieur plus léger → STABLE (le proton sort
stable tout seul). Validé : neutron (Lk=Lk_opt−1) → proton, Q = 0,78 MeV.

Catalogue UNIFIÉ : chaque état (config, Lk) porte, dans la même ligne,
  - AU REPOS (E=0) : spin, charge Q, masse m_repos, durée de vie τ_repos ;
  - À HAUTE ÉNERGIE (E ≫ E_seuil) : masse déconfinée m_HE = Σ masses des composantes
    isolées, et le seuil E_seuil = E₀·Σρ_J ~ ħc/r_link ~ 200 MeV.
Concept clé : la particule est DYNAMIQUE selon l'énergie captée ; la valeur mesurée
se situe dans l'INTERVALLE [m_HE, m_repos] (ΔE), le seuil marquant le déconfinement.

Sortie : docs/TQNT_catalog_V6_unifie.xlsx
"""
from __future__ import annotations
import re
import numpy as np
import openpyxl

from theory.chi_field import jones_rho
from theory.jones import braid_index
from simulation.experiments.reassign_pdg import sigma_chi  # χ = σ·b (validé)
from simulation.experiments.lepton_koide_z3 import lepton_masses, koide_Q, M_E, M_MU, M_TAU, M_P
from simulation.experiments.quark_koide_z3 import SECTORS, koide_params
from simulation.experiments.ew_vev_derivation import derive_vev, EW_KNOTS, VEV_EXP
from simulation.experiments.alpha_em_cs import alpha_em, quantum_dim_half, delta_oscillation, is_rigid_em_level, K as CS_K
from simulation.experiments.composition_kinetics_v6 import P_combination, CR as _CR
from openpyxl.styles import PatternFill, Font
from openpyxl.formatting.rule import ColorScaleRule
import math as _math

# --- Correction du glitch de Frenet sur 7_5 (catalogue source : 5959 MeV, aberrant) ---
# Ω_ts propre (vérifié : nucléon 938,2 ; concorde avec le catalogue à 0,11% hors 7_5)
# pour recalculer les configs contenant 7_5 avec un 7_5 cohérent (~141 MeV).
from collections import Counter as _Counter
from theory.alexander import alexander_determinant as _adet
from theory.omega_ts import frenet_energies as _fren, _N_LOBES as _NLOB
_OK = list(_NLOB.keys())
_CRn = {k: int(k.split('_')[0]) for k in _OK}
_Bn = {k: braid_index(k) for k in _OK}
_DETn = {k: _adet(k) for k in _OK}
_UKUT = {}
for _k in _OK:
    _uk, _ut, _ = _fren(_k)
    _UKUT[_k] = _uk * _ut
_REF75 = _UKUT["3_1"]
_MC75, _ML75 = 0.147883, 0.022450


def _omega_clean(cl, a=0.5283, p=0.7796, q=0.7218, x=-2.456, beta=1.651, rho0=10.622):
    """Ω_ts reconstruit proprement (Frenet cohérent), pour corriger les configs 7_5."""
    n = len(cl); nf = ((n + 1) / 2) ** (-x); sc = 1.0
    for k, c in _Counter(cl).items():
        if c >= 2:
            sc *= (rho0 * _CRn[k] ** (-beta)) ** (c - 1)
    h = sum(((_CRn[k] * _MC75 + _NLOB[k] * _ML75) * (_Bn[k] * _DETn[k] / 6) ** p) ** a
            for k in cl) ** (1 / a)
    return nf * sc * h * (sum(_UKUT[k] for k in cl) / _REF75) ** q

SRC = "docs/TQNT_catalog_complete.xlsx"
OUT = "docs/TQNT_catalog_V6_unifie.xlsx"
BETA = 0.210       # MeV : tension d'enlacement
ME = 0.511         # MeV
GF, HBAR, PI = 1.1664e-5, 6.582e-25, np.pi
VUD, GA = 0.9737, 1.2756
AMPHI = {"4_1", "6_3", "8_3"}


def gamma_beta(Q_MeV):
    """Taux β PROPRE (espace de phase de Fermi, électron massif), pas Q⁵.
    Γ = (G_F² V_ud²/2π³)(1+3g_A²)·∫ p_e E_e (W0−E_e)² dE_e, W0 = Q + m_e.
    → neutron (Q=0,782) ≈ 914 s. Tend vers ~Q⁵ pour Q≫m_e."""
    W0 = Q_MeV + ME
    E = np.linspace(ME, W0, 400)
    p = np.sqrt(np.clip(E**2 - ME**2, 0, None))
    F = np.sum(p * E * (W0 - E)**2) * (E[1] - E[0]) / 1000.0**5   # GeV⁵
    return (GF**2 * VUD**2 / (2*PI**3)) * (1 + 3*GA**2) * F


def comps(cfg):
    return [k for k in re.split(r'[+\s]+', str(cfg)) if re.match(r'^\d+_\d+$', k)]


_CHI = {}
def chi(k):
    if k not in _CHI:
        _CHI[k] = 0 if k in AMPHI else sigma_chi(k)[1]
    return _CHI[k]


def mean_cc(cl):
    """⟨√(c_J(i)·c_J(j))⟩ sur les paires (i<j) ; nucléon → 6,15."""
    cj = [jones_rho(k) for k in cl]
    if len(cj) < 2:
        return cj[0] if cj else 1.0
    pairs = [np.sqrt(cj[i]*cj[j]) for i in range(len(cj)) for j in range(i+1, len(cj))]
    return float(np.mean(pairs))


def main():
    wb = openpyxl.load_workbook(SRC, data_only=True, read_only=True)
    ws = wb["Catalogue TQNT V5"]
    base = []
    iso = {}
    for r in ws.iter_rows(min_row=7, values_only=True):
        if not r[1] or not r[4]:
            continue
        cl = comps(r[1])
        if not cl:
            continue
        if str(r[2]) == "1":
            iso[cl[0]] = float(r[4])          # masse du noeud isolé (composante libre)
        base.append({"cfg": str(r[1]).strip(), "cl": cl, "n": int(r[2]), "c": r[3],
                     "mOmega": float(r[4]), "spin": str(r[5]) if r[5] else "",
                     "pdg": r[9], "tau_exp": r[14]})

    # CORRECTION du glitch de Frenet sur 7_5 (catalogue : 5959 MeV aberrant, 12× les
    # autres 7-nœuds). On recalcule les configs contenant 7_5 avec l'Ω_ts propre.
    n_fix = 0
    for b in base:
        if "7_5" in b["cl"]:
            b["mOmega"] = round(_omega_clean(b["cl"]), 2)
            n_fix += 1
    if "7_5" in iso:
        iso["7_5"] = round(_omega_clean(["7_5"]), 2)
    print(f"  correction 7_5 : {n_fix} configs recalculées (7_5 single 5959→{iso.get('7_5')} MeV)")

    E0 = 33.0   # MeV : E₀ tel que E_K = E₀·Σρ_J ~ ħc/r_link ~ 200 MeV (seuil déconf.)

    # générer le multiplet (config, Lk) : Lk de Lk_opt-2 à Lk_opt+2
    M2 = 1.0
    rows = []
    for b in base:
        n = b["n"]; Lk_opt = n
        Schi = sum(chi(k) for k in b["cl"])
        cc = mean_cc(b["cl"])
        for dLk in range(-2, 3):
            Lk = Lk_opt + dLk
            Q = Schi + Lk
            m = b["mOmega"] + BETA*cc*dLk**2
            # désintégration faible : vers Lk_opt (tension plus basse) par pas de 1
            if dLk == 0:
                tau = "stable"; Qdec = 0.0; prod = "—"
            else:
                step = -1 if dLk > 0 else +1          # vers Lk_opt
                Lk2 = Lk + step
                m2 = b["mOmega"] + BETA*cc*(Lk2-Lk_opt)**2
                Qdec = m - m2 - ME                    # MeV (lepton émis)
                if Qdec <= 0:
                    tau = "stable"; prod = "—"
                else:
                    G = gamma_beta(Qdec)              # espace de phase β propre (Fermi)
                    tau = HBAR / G
                    prod = f"{b['cfg']} (Lk={Lk2}, Q={Schi+Lk2:+d})"
            rows.append({**b, "Lk": Lk, "Q": Q, "m": m, "Qdec": Qdec, "tau": tau, "prod": prod,
                         "cc": cc, "Schi": Schi, "chan": "β" if tau != "stable" else "—"})

    # nombre baryonique : B = 1 si (3 composantes ET un noeud c≥7), sinon 0.
    # (proton 3_1+7_2+7_2 → B=1 ; pion 3_1+3_1+5_2 → B=0 ; muon, 7_2+7_2 → B=0)
    def baryon(b):
        return 1 if b["n"] == 3 and any(int(k.split("_")[0]) >= 7 for k in b["cl"]) else 0
    for e in rows:
        e["B"] = baryon(e)

    # CANAL DE CONFIG (charge constante) : DISSOLUTION = on retire des composantes
    # (le produit Y est un SOUS-multiensemble des noeuds de X — pas de transformation
    # de noeud). Y prend le Lk qui conserve la charge Q ; on conserve aussi B.
    # Q_désint = m(X)−m(Y), Γ = G_F²·Φ_3(Q). On garde la désint. la plus RAPIDE.
    import itertools
    from collections import Counter
    mOmega = {tuple(sorted(b["cl"])): b["mOmega"] for b in base}

    def submulti(cl):
        """sous-multiensembles propres non vides (au moins 1 composante retirée)."""
        seen = set()
        for r in range(1, len(cl)):
            for sub in itertools.combinations(cl, r):
                t = tuple(sorted(sub))
                if t in seen or t not in mOmega:
                    continue
                seen.add(t); yield list(t)

    for e in rows:
        best = None
        for Y in submulti(e["cl"]):
            nY = len(Y); SchiY = sum(chi(k) for k in Y); ccY = mean_cc(Y)
            BY = 1 if nY == 3 and any(int(k.split("_")[0]) >= 7 for k in Y) else 0
            if BY != e["B"]:
                continue                              # conserve le nombre baryonique
            LkY = e["Q"] - SchiY                       # Lk qui donne la même charge Q
            mY = mOmega[tuple(sorted(Y))] + BETA*ccY*(LkY - nY)**2
            if mY < e["m"] - 0.5 and (best is None or mY < best[1]):
                best = (Y, mY)
        if best:
            Qd = e["m"] - best[1]
            tau_cfg = HBAR / (GF**2 * M2 * (Qd/1000.0)**5 / (192*PI**3))
            tb = e["tau"] if isinstance(e["tau"], (int, float)) else float("inf")
            if tau_cfg < tb:
                e["tau"] = tau_cfg; e["Qdec"] = Qd; e["chan"] = "config"
                e["prod"] = " + ".join(best[0])

    # HAUTE ÉNERGIE (E ≫ E_seuil : déconfinement). Par config :
    #   - masse : m_HE = Σ masses des composantes isolées (le link se sépare)
    #   - spin & charge : CONSERVÉS (Noether) → identiques au repos
    #   - durée de vie : l'état déconfiné est TRANSITOIRE, il se re-confine en
    #     τ_HE ~ ħ/E_seuil (échelle d'interaction forte ~10⁻²⁴ s)
    for e in rows:
        e["m_HE"] = sum(iso[k] for k in e["cl"])
        e["E_seuil"] = E0 * sum(jones_rho(k) for k in e["cl"])
        e["tau_HE"] = HBAR / (e["E_seuil"] / 1000.0)        # ħ/E_seuil (re-confinement)
        # RESTRUCTURATION : à h.é. on RÉSOUT les constituants (≠ total conservé).
        # charge résolue = Σχ (constituants) ; ΔQ = Lk (charge portée par l'enlacement)
        e["Qchi_HE"] = e["Schi"]                             # charge des constituants
        e["Lk_charge"] = e["Q"] - e["Schi"]                 # = Lk (enlacement)
        # spin résolu = Σ sᵢ (constituants) ; orbital = Σsᵢ − S (porté par l'enlacement)
        e["sum_si"] = sum((braid_index(k) - 1) / 2 for k in e["cl"])
        try:
            S_tot = float(str(e["spin"]).split(",")[0].strip())
        except (ValueError, AttributeError):
            S_tot = 0.0
        e["orbital"] = e["sum_si"] - S_tot
        e["spin_comp"] = "+".join(f"{(braid_index(k)-1)/2:g}" for k in e["cl"])
        # probabilité de composition P(L) (cinétique V6) ; log10 pour lisibilité/couleur
        try:
            e["P_compo"] = P_combination("+".join(e["cl"]), Lk=e["Lk"], Lk_opt=e["n"])
            e["logP"] = _math.log10(e["P_compo"]) if e["P_compo"] > 0 else None
        except (KeyError, ValueError):
            e["P_compo"] = None; e["logP"] = None

    out = openpyxl.Workbook(); o = out.active; o.title = "Catalogue V6 unifié"
    # En-tête : identité | AU REPOS (E=0) | HAUTE ÉNERGIE (E≫E_seuil) | PDG
    # NB : la particule mesurée est DYNAMIQUE dans l'intervalle [m_HE, m_repos]
    #      selon l'énergie captée ; le seuil E_seuil marque le déconfinement.
    o.append(["#", "Config", "n", "c", "log P_compo", "Spin tot", "Q tot",
              "m_repos (MeV)", "canal", "τ_repos (s)", "B",
              "Σχ const. (h.é.)", "Lk (charge enlac.)", "Σsᵢ const. (h.é.)",
              "orbital (spin enlac.)", "spins compos.", "m_h.é. (MeV)", "τ_h.é. (s)",
              "E_seuil (MeV)", "ΔE (MeV)", "produit désint.", "PDG (fond.)", "τ_exp"])
    for i, e in enumerate(rows, 1):
        tau = e["tau"] if isinstance(e["tau"], str) else f"{e['tau']:.3e}"
        fond = e["Lk"] == e["n"]
        o.append([i, e["cfg"], e["n"], e["c"],
                  round(e["logP"], 2) if e["logP"] is not None else "",
                  e["spin"], e["Q"], round(e["m"], 2), e["chan"], tau, e["B"],
                  e["Qchi_HE"], e["Lk_charge"], f"{e['sum_si']:g}", f"{e['orbital']:+g}",
                  e["spin_comp"], round(e["m_HE"], 2), f"{e['tau_HE']:.2e}",
                  round(e["E_seuil"], 1), round(e["m"] - e["m_HE"], 2),
                  e["prod"], e["pdg"] if fond else "", e["tau_exp"] if fond else ""])
    _colorize_main(o, rows)
    _add_probable_sheet(out, rows)
    _add_summary_sheets(out)
    _add_4knot_sheet(out)
    out.save(OUT)
    nstab = sum(1 for e in rows if e["tau"] == "stable")
    print(f"{len(rows)} états (config,Lk) écrits dans {OUT} ; stables : {nstab}")
    print("Validation (β = Lk±1 ; config = charge constante, dissolution) :")
    by = {}
    for e in rows:
        by[(e["cfg"].replace(" ", ""), e["Lk"])] = e
    cases = [("électron", "3_1", 1), ("muon", "3_1+6_1", 2), ("proton", "3_1+7_2+7_2", 3),
             ("neutron", "3_1+7_2+7_2", 2), ("pion+", "3_1+3_1+5_2", 3)]
    exp = {"électron": "stable", "muon": "2,2e-6", "proton": "stable", "neutron": "879", "pion+": "2,6e-8"}
    for nm, cfg, Lk in cases:
        e = by.get((cfg, Lk))
        if e:
            tau = e["tau"] if isinstance(e["tau"], str) else f"{e['tau']:.2e}"
            print(f"  {nm:>9} {cfg} Lk={Lk}: Q={e['Q']:+d} B={e['B']} canal={e['chan']:>6} "
                  f"Qdéc={e['Qdec']:.1f} → τ={tau}  (exp {exp[nm]})")


def _add_4knot_sheet(out, top_n=400):
    """Combinatoire 4-nœuds, FILTRÉE par probabilité de composition (prédictif).

    On énumère tous les multiensembles 4-nœuds, on calcule P(L) (FIABLE : combinatoire
    + croisements), la charge Q=Σχ+Lk (FIABLE) et la parité de spin (FIABLE), et on garde
    les top_n par probabilité. La masse est INDICATIVE (Ω_ts calibré ~10 %, non vérifiée).
    """
    import itertools
    from theory.omega_ts import omega_ts_axiomatic
    AMPHI = {"4_1", "6_3", "8_3"}
    knots = list(_CR.keys())

    def chi(k):
        return 0 if k in AMPHI else sigma_chi(k)[1]

    # P(L) à Lk optimal (=n, tension nulle) pour tous les 4-multiensembles
    allc = []
    for combo in itertools.combinations_with_replacement(knots, 4):
        cfg = "+".join(combo)
        P = P_combination(cfg, Lk=4, Lk_opt=4)  # p=P_PERC dérivé (percolation)   # Lk=n → E_link=0
        allc.append((combo, P))
    allc.sort(key=lambda x: -x[1])
    keep = allc[:top_n]

    ws = out.create_sheet("4-nœuds probables")
    ws.append([f"Combinatoire 4-nœuds filtrée par P(L) — top {top_n} sur "
               f"{len(allc)} multiensembles. Vert = plus probable. "
               "P, Q, spin FIABLES ; masse INDICATIVE (~10 %, Ω_ts calibré)."])
    ws.append(["Config", "C (crois.)", "log P_compo", "Σχ", "Q (Lk=4)", "parité spin",
               "m indicative (GeV)"])
    for combo, P in keep:
        C = sum(_CR[k] for k in combo)
        Sx = sum(chi(k) for k in combo)
        par = "boson" if (sum(braid_index(k) for k in combo) - 4) % 2 == 0 else "fermion"
        try:
            m = omega_ts_axiomatic("+".join(combo)) / 0.867 / 1000
            mstr = round(m, 1)
        except Exception:
            mstr = "?"
        ws.append(["+".join(combo), C, round(_math.log10(P), 2) if P > 0 else "",
                   f"{Sx:+d}", f"{Sx+4:+d}", par, mstr])
    ws.conditional_formatting.add(
        f"C3:C{len(keep)+2}",
        ColorScaleRule(start_type="min", start_color="F8696B",
                       mid_type="percentile", mid_value=50, mid_color="FFEB84",
                       end_type="max", end_color="63BE7B"))
    ws.freeze_panes = "A3"
    print(f"  feuille '4-nœuds probables' : top {top_n} par P(L) (sur {len(allc)} configs)")


def _colorize_main(o, rows):
    """Code couleur : échelle rouge→vert sur log P_compo (probabilité de composition) ;
    fond vert pour les configs identifiées à une particule PDG (proches de l'expérience)."""
    n = len(rows)
    # échelle de couleur sur la probabilité (col E) : rouge=improbable, vert=probable
    o.conditional_formatting.add(
        f"E2:E{n+1}",
        ColorScaleRule(start_type="percentile", start_value=10, start_color="F8696B",
                       mid_type="percentile", mid_value=50, mid_color="FFEB84",
                       end_type="percentile", end_value=90, end_color="63BE7B"))
    green = PatternFill("solid", fgColor="C6EFCE")
    bold = Font(bold=True)
    npdg = 0
    for i, e in enumerate(rows, 2):
        pdg = e.get("pdg", "")
        if e["Lk"] == e["n"] and pdg and not str(pdg).startswith("("):
            for col in (2, 22):  # Config, PDG
                o.cell(row=i, column=col).fill = green
            o.cell(row=i, column=2).font = bold
            npdg += 1
    o.freeze_panes = "A2"
    o.cell(row=1, column=2).comment = None
    print(f"  couleurs : {npdg} configs PDG en vert ; échelle prob. sur log P_compo")


def _add_probable_sheet(out, rows):
    """Feuille des particules identifiées (PDG), triées par probabilité de composition."""
    ws = out.create_sheet("Particules probables")
    ws.append(["Particules PDG identifiées, triées par PROBABILITÉ de composition P(L) "
               "— vert = plus probable / mieux formée (n bas, C bas, Lk optimal)"])
    ws.append(["PDG", "Config", "n", "c", "log P_compo", "m_repos (MeV)", "Spin", "Q",
               "canal", "τ_repos (s)"])
    sel = [e for e in rows if e["Lk"] == e["n"] and e.get("pdg")
           and not str(e["pdg"]).startswith("(") and e["logP"] is not None]
    sel.sort(key=lambda e: -e["logP"])
    for e in sel:
        tau = e["tau"] if isinstance(e["tau"], str) else f"{e['tau']:.2e}"
        ws.append([e["pdg"], e["cfg"], e["n"], e["c"], round(e["logP"], 2),
                   round(e["m"], 2), e["spin"], e["Q"], e["chan"], tau])
    if sel:
        ws.conditional_formatting.add(
            f"E3:E{len(sel)+2}",
            ColorScaleRule(start_type="min", start_color="F8696B",
                           mid_type="percentile", mid_value=50, mid_color="FFEB84",
                           end_type="max", end_color="63BE7B"))
    ws.freeze_panes = "A3"
    print(f"  feuille 'Particules probables' : {len(sel)} particules PDG triées par P(L)")


def _add_summary_sheets(out):
    """Feuilles de synthèse V6 : leptons (Z₃ générations), quarks (Z₃ couleur),
    secteur EW + α. Reflète les acquis de la session (dérivations Koide/vev/α).

    NB : les LEPTONS ne sont PAS dans le catalogue d'entrelacs ci-dessus — ce sont
    des nœuds 3₁ SIMPLES (ponctuels) en trois générations Z₃ (cf. falsification du
    muon composite par sa nature ponctuelle >10 TeV). Leurs masses suivent Koide.
    """
    import math
    # 1) Leptons = 3₁ en générations Z₃ (ponctuels) ; masse = m_p/3·Koide(2/9)
    ws = out.create_sheet("Leptons (Z₃ trèfle)")
    ws.append(["Lepton = 3₁ (génération g_k), PONCTUEL — charge −1, spin ½ ; m_k = (m_p/3)(1+√2 cos(2/9+2πk/3))²"])
    ws.append(["génération", "nœud", "k (mode Z₃)", "m prédite (MeV)", "m exp (MeV)", "écart %", "charge", "spin"])
    mk = lepton_masses(M_P / 3, 2 / 9)              # k=0→τ, 1→e, 2→μ
    for k, nm, exp in [(1, "e (g1)", M_E), (2, "μ (g2)", M_MU), (0, "τ (g3)", M_TAU)]:
        ws.append([nm, "3_1", k, round(mk[k], 3), exp, round((mk[k] / exp - 1) * 100, 2), -1, "1/2"])
    ws.append([])
    ws.append([f"Koide Q = {koide_Q([M_E, M_MU, M_TAU]):.5f} (= 2/3, dérivé Z₃+√2) ; phase δ = 2/cr² = 2/9 ; échelle m₀ = m_p/3"])

    # 2) Quarks = Z₃ déformé par la couleur ; nombre baryonique = trialité
    ws = out.create_sheet("Quarks (Z₃ couleur)")
    ws.append(["Quarks : même forme Z₃ que les leptons mais DÉFORMÉE par la couleur (r>√2, δ<2/9)"])
    ws.append(["secteur", "m₀ (MeV)", "Koide Q", "r", "r/√2", "δ", "δ·(9/2)"])
    for nm, ms in SECTORS.items():
        m0, Q, r, d = koide_params(ms)
        ws.append([nm, round(m0, 1), round(Q, 4), round(r, 3), round(r / math.sqrt(2), 3),
                   round(d, 4), round(d / (2 / 9), 2)])
    ws.append([])
    ws.append(["Couleur SU(3) = Z₃ du trèfle (3 lobes = 3 couleurs)."])
    ws.append(["Nombre baryonique B = trialité SU(3) = (n_q − n_q̄)/3 = charge Z₃ de la couleur."])
    ws.append(["=> B est topologique (trialité couleur), PAS le type de nœud ; π(méson)≡n(baryon) par le nœud, distincts par la trialité."])
    ws.append(["Échelles : leptons m₀=m_p/3 (topo) ; down ≈2·m_p/3 (hadronique) ; up = top≈vev/√2 (EW, non topo)."])

    # 3) Secteur électrofaible + α_EM
    ws = out.create_sheet("EW + alpha (k=135)")
    ws.append(["Secteur électrofaible : pas de condensat ; W/Z/H = entrelacs lourds ; vev DÉRIVÉ"])
    ws.append(["boson", "config (entrelacs)", "masse TQNT (GeV)"])
    for b, (m, cfg) in EW_KNOTS.items():
        ws.append([b, cfg, m])
    ws.append([])
    mW, mZ = EW_KNOTS["W"][0], EW_KNOTS["Z"][0]
    vev, s2W, g = derive_vev(mW, mZ)
    ws.append(["Dérivation du vev :", "", ""])
    ws.append(["sin²θ_W = 1−(m_W/m_Z)²", round(s2W, 4), "(mesuré 0,2312)"])
    ws.append(["g = √(4πα)/sinθ_W", round(g, 4), ""])
    ws.append(["vev = 2 m_W / g (GeV)", round(vev, 1), f"exp {VEV_EXP} → {round((vev/VEV_EXP-1)*100,1)}%"])
    ws.append([])
    ws.append(["α_EM = 1/(k+2) (Chern-Simons, k+h∨=k+2, h∨=2) :", "", ""])
    ws.append([f"k = {CS_K}", f"α = 1/{CS_K+2} = {alpha_em():.8f}", "exp 1/137,036 (+0,026%)"])
    ws.append(["d_½(k=135) = 2cos(π/137)", round(quantum_dim_half(), 6), f"→ δ_oscill = {delta_oscillation():.4f}"])
    ws.append([f"k+2={CS_K+2} premier, k impair ∉{{10,16,28}} → invariant modulaire diagonal UNIQUE (EM rigide) : {is_rigid_em_level()}"])
    ws.append(["Hiérarchie dissoute : pas de scalaire de Higgs fondamental (H=entrelacs) ⇒ pas de divergence quadratique."])


if __name__ == "__main__":
    main()
