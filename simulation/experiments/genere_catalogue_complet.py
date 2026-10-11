"""Génère le catalogue complet TQNT : docs/TQNT_catalog_complete.xlsx
(28/07/2026 — état canonique V6 du 28/07).

12 onglets : Lisez-moi ; Règles fondamentales ; Constantes & échelles ;
Secteurs S ; Catalogue cœur (37) ; Spectre étendu (226) ; Bosons & EW ;
Neutrinos ; Nucléon & charge ; Matière noire ; Prédictions datées ;
Comptage & statuts.

Sources : docs/catalogue_sigma_chi.csv (canonique 07/07),
output/socle_224_resultats.csv (moteur aligné), CLAUDE.md/V6 (règles,
constantes, assignations EW/DM, préregistration). Tous les statuts
épistémiques sont reportés (dérivé / théorème / conditionnel /
identifié / calibré / compatibilité) — règle du test nul rappelée.
"""
import csv
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BLEU = "1A4A7A"
BLEU_CLAIR = "DFE9F4"
GRIS = "F4F6F8"
OR = "B8860B"

H_FONT = Font(bold=True, color="FFFFFF", size=11)
H_FILL = PatternFill("solid", fgColor=BLEU)
S_FONT = Font(bold=True, color=BLEU, size=12)
WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Border(bottom=Side(style="thin", color="CCCCCC"))


def sheet(wb, name, widths):
    ws = wb.create_sheet(name)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    return ws


def header(ws, row, cols):
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=row, column=j, value=c)
        cell.font = H_FONT
        cell.fill = H_FILL
        cell.alignment = WRAP
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def rows(ws, start, data, wrap_cols=None):
    r = start
    for row in data:
        for j, v in enumerate(row, 1):
            cell = ws.cell(row=r, column=j, value=v)
            cell.border = THIN
            if wrap_cols is None or j in wrap_cols:
                cell.alignment = WRAP
        r += 1
    return r


def titre(ws, text, row=1):
    c = ws.cell(row=row, column=1, value=text)
    c.font = S_FONT
    return row + 2


def main():
    wb = Workbook()
    wb.remove(wb.active)

    # ================= 1. LISEZ-MOI =================
    ws = sheet(wb, "Lisez-moi", [110])
    lignes = [
        "TQNT — CATALOGUE COMPLET DES PARTICULES (état canonique V6, 28 juillet 2026)",
        "",
        "Théorie Quantique des Nœuds Topologiques, 6e édition. Un champ unique χ (connexion de type",
        "Chern-Simons SU(2), niveau k=135) sur la trame ondulante de l'espace-temps. Les nœuds de ce",
        "champ SONT les particules ; masse, spin, charge et durée de vie émergent d'invariants topologiques.",
        "",
        "ONGLETS : Règles fondamentales | Constantes & échelles | Secteurs S | Catalogue cœur (37 états",
        "canoniques) | Spectre étendu (226 états) | Nœuds premiers (pool) | ENTRELACS COMPLETS (1539,",
        "jusqu'à 30 croisements) | 117 candidats DM | Bosons & EW | Neutrinos | Nucléon & charge |",
        "Matière noire | Prédictions datées | Comptage & statuts.",
        "",
        "STATUTS ÉPISTÉMIQUES (toujours reportés) : dérivé (chaîne complète) ; théorème (démontré sous",
        "les primitives) ; conditionnel (dérivé sous identifications déclarées) ; identifié (valeur reconnue",
        "dans une structure, non dérivée) ; calibré (ajusté une fois, global) ; compatibilité (accord",
        "numérique sous assignation — ne vaut PAS preuve, cf. règle du test nul).",
        "",
        "RÈGLE DU TEST NUL : tout accord obtenu par choix d'assignation doit battre le hasard des",
        "assignations équivalentes avant d'être présenté comme preuve. Hiérarchie du méta-test nul (B.6) :",
        "4 résultats au-delà de tout facteur d'essais (hyperfin V-P, échelle de recouplage, Koide,",
        "oscillations) ; une grappe de conditionnels portés par leurs mécanismes ; le reste = contexte.",
        "",
        "SOURCES : catalogue_sigma_chi.csv (canonique 07/07) ; socle_224_resultats.csv (moteur aligné",
        "21-22/07) ; monographie TQNT-V6 (FR 328 p / EN 323 p) et ses ~640 scripts.",
        "Contact : lilian.cariou@ca9.fr — Licence CC-BY-4.0.",
    ]
    for i, l in enumerate(lignes, 1):
        c = ws.cell(row=i, column=1, value=l)
        if i == 1:
            c.font = S_FONT
        c.alignment = WRAP

    # ================= 2. RÈGLES FONDAMENTALES =================
    ws = sheet(wb, "Règles fondamentales", [30, 60, 42])
    r = titre(ws, "Les règles TQNT — définitions et théorèmes")
    header(ws, r, ["Règle", "Énoncé", "Statut / origine"])
    r = rows(ws, r + 1, [
        ("Masse (définition, P21)", "m = √σ_χ × S — une seule échelle √σ_χ = 418,47 MeV, un facteur "
         "sans dimension S fixé par la classe topologique (secteur). La fonctionnelle maîtresse "
         "M² = (Σmᵢ)² + 2πσ_χ(L + n/2 + 1,38·n_r) est la forme close de S pour le secteur "
         "composite-excité (pas une seconde définition).", "définition canonique ; sectorialité "
         "mécanisée (temps de rails→base, sens→génération)"),
        ("Charge", "Q = Σχ + Lk (χ = auto-enlacement signé du nœud, Lk = enlacement d'entrelacs). "
         "La charge naît de l'enroulement dans les rails ; un objet jamais entré (photon, neutrino) "
         "a Q = 0 structurellement.", "dérivé"),
        ("Spin", "Spin = arité de tresse (b−1) ; ruban 4π → spin-statistique dérivée, Dirac forcé. "
         "Le taux de rotation ω est structurellement caché (théorème de lecture) : toutes les lois "
         "utilisent J.", "dérivé / théorème"),
        ("Antimatière", "Antiparticule = nœud MIROIR : V_K̄(t) = V_K(t⁻¹). Amphichiral = sa propre "
         "antiparticule.", "dérivé"),
        ("Durée de vie", "Fermi/Sargent : Γ = N·m⁵/(384π³·E_relax⁴), g_topo = 1 exactement (pas de "
         "constante de couplage faible) ; Q = Δm pilote (corr. −0,99).", "dérivé (structure) ; "
         "constantes MS en entrée"),
        ("Les forces — 2 sources", "Tube χ : forte = sa tension (σ_χ) ; EM = sa limite de jauge "
         "(α = 1/(k+2)) ; faible = sa reconnexion (P15, contact ΔLk=±1). Trame : gravitation = "
         "réponse aux déficits (Ω² = 1 + κ_g|Ψ|²).", "structurel"),
        ("Gravitation", "G = ħc/M_Pl² (l'amplitude de la trame — équation UNIQUE de G). Le nucléon "
         "LIT la rigidité : m_base = M_Pl·√((12/7)α¹⁸) → G à −0,05 % du CODATA.", "théorème de "
         "lecture ; ne jamais présenter la nucléonique comme l'équation de G"),
        ("α et k", "α = 1/(k+2) = 1/137, k = 135 fixé par le trèfle : k+2 = round((8π/3)·ℓ(3₁)) = 137.",
         "conditionnel (verrou entier)"),
        ("VEV électrofaible", "v = √(2π)·α⁸·M_Pl·(1+α/4π) = 246,227 GeV (+28 ppm).", "conditionnel "
         "(croisement-ouvert sur l'ancre)"),
        ("Familles", "N_c = N_gen = 3 = le Z₃ du trèfle (le même). Saturation Z₃ dérivée : JAMAIS de "
         "4e famille, AUCUN stérile.", "dérivé"),
        ("Les 3 régimes de la trame", "Lumière = onde jamais entrée dans un rail (S=0, spectatrice, "
         "c exact) ; nœud = onde entrée et verrouillée (déficit = masse) ; neutrino = onde "
         "auto-rotative OUVERTE qui effleure sans entrer (m = ħc/2R).", "consolidé"),
        ("Cosmologie", "Friedmann mécanisé (H² = 8πGρ/3), Ω_tot = 1 EN THÉORÈME, w = −1 par "
         "construction ; l'indiscernabilité de ΛCDM+RG = théorème de la trame.", "théorème"),
        ("Interdits (Règle 0)", "Ne JAMAIS raisonner en quarks/saveur/CKM/hypercharge comme "
         "fondamentaux : concepts SM dérivés. Le proton n'est PAS « fait de quarks » : entrelacs "
         "3₁+7₂+7̄₂. Les phénomènes réels (oscillations, transitions) = reconnexions P15.", "règle de méthode"),
    ], wrap_cols={1, 2, 3})

    # ================= 3. CONSTANTES & ÉCHELLES =================
    ws = sheet(wb, "Constantes & échelles", [26, 22, 18, 60])
    r = titre(ws, "Constantes et échelles canoniques (V6, 28/07/2026)")
    header(ws, r, ["Grandeur", "Valeur", "Statut", "Origine / note"])
    r = rows(ws, r + 1, [
        ("√σ_χ (tension de corde)", "418,47 MeV", "calibré", "recalibrée 18/07 (leptons+octet, accord 7e-5) ; en dérivent Regge 2πσ_χ, Λ_χ, E_Z3, σ_Cornell"),
        ("m_ℓ (nœud trèfle)", "310,30 MeV (phys)", "conditionnel", "(3/4)√σ_χ/(1+πα/2) — gonflement magnétique de la charge ; transmutation (pendant de Λ_QCD)"),
        ("m_s (nœud 5₂)", "494,8 MeV", "calibré + indice", "épinglage 5₂ renforcé ; ratio ACPR −5,4 %, habillage +5,6 % non dérivé (déclassé 22/07)"),
        ("m_c / m_b (nœuds lourds)", "1557 / 4952 MeV", "calibrés", "pas des quarks (Règle 0)"),
        ("f_π (échelle chirale)", "~92 MeV", "calibré", "secteur Goldstone (GMOR)"),
        ("β_⊗ (tension d'enlacement)", "0,210 MeV", "calibré (cadre)", "mécanisme identifié : piégeage de fluxoïde E=Φ_χ²/2L_ind (+6,6 %)"),
        ("Ratio radial", "1,38", "calibré", "excités radiaux (n_r)"),
        ("k (niveau Chern-Simons)", "135", "conditionnel", "verrou du trèfle round((8π/3)ℓ(3₁))=137 ; α=1/(k+2)"),
        ("α_EM", "1/137", "conditionnel", "= 1/(k+2)"),
        ("G", "6,6710e-11 (prédit)", "théorème de lecture", "G = ħc/M_Pl² ; −0,05 % CODATA ; convention : toute application utilise G_prédit"),
        ("v (VEV)", "246,227 GeV", "conditionnel", "√(2π)α⁸M_Pl(1+α/4π), +28 ppm"),
        ("M_Pl (rigidité)", "1,22089e19 GeV", "primitive", "l'horloge de la trame ; non dérivable (théorème de fermeture des échelles)"),
        ("Λ_χ", "209,2 MeV", "dérivé", "= ½√σ_χ"),
        ("E_Z3 (quantum de génération)", "139,49 MeV", "dérivé", "= ⅔Λ_χ → m_π à −0,06 %"),
        ("πσ_χ (demi-tour du ruban)", "0,5501 GeV²", "dérivé", "hyperfin V−P universel π→B_s à 0,7 %"),
        ("2πσ_χ (pente de Regge)", "1,1002 GeV²", "dérivé", "1 tour de torsion périphérique ; α′=0,9088 GeV⁻²"),
        ("D (tube du plateau)", "0,0214 fm", "dérivé", "L = m/σ_χ (loi longueur-masse)"),
        ("T_c (gel)", "62 ± 3 MeV", "dérivé", "serrage topologique (double reconnexion P15)"),
        ("t_c (marche de G)", "~1,6e-4 s", "dérivé", "naissance des premiers déficits"),
        ("Hints", "A=√2 ; δ=2/9", "portés/décomposés", "A porté par le doublement de sens (conditionnel) ; δ = π/12 − ε (électron quasi nodal)"),
    ], wrap_cols={1, 4})

    # ================= 4. SECTEURS S =================
    ws = sheet(wb, "Secteurs S", [22, 62, 20, 40])
    r = titre(ws, "La sectorialité de S (mécanisée : temps de rails → base, sens → génération)")
    header(ws, r, ["Secteur", "Loi de masse", "Précision", "Mécanisme / note"])
    r = rows(ws, r + 1, [
        ("BASE (γ, ν)", "S = 0 (onde χ libre, aucun nœud)", "exact", "théorème de rails : jamais entrée = jamais magnétisée = sans masse de nouage"),
        ("LEPTONS (e, μ, τ)", "√m_k = μ(1+√2·cos(2πk/3+δ)), μ=(3/4)√σ_χ, δ=π/12−ε", "≤0,005 % ; m_τ 61 ppm", "modes Z₃ du circuit trèfle ; A=√2 porté (2 sens) ; électron quasi nodal (protection) ; m_e = Green retardée ±3 %"),
        ("BARYONS légers", "Regge : S = √(2π(J−α₀)) + n_Z3·E_Z3 (+τ_⊥, ΔLk)", "0,75 % ; base octet 938,2 (0,0 %)", "jonction Y, tenseur Λ_χ/8, clivage d'isospin par fluxoïde"),
        ("GOLDSTONE (π, K, η)", "GMOR : m² = B(mᵢ+mⱼ)", "m_π −0,06 %", "paire K+K̄ — annulation totale du sens (le plus annulé est le plus léger)"),
        ("LOURDS (c, b)", "constituant m=Σmᵢ + torsion 2πσ_χ(L+n/2+1,38n_r) ; quarkonia = Cornell (compression)", "57 états ~0,3 % ; 59 sans ancre 0,58 %", "échelle de recouplage en formes closes (base=trialité, ×3=Z₃, ½=jonction, d−1=√2−1)"),
        ("ÉLECTROFAIBLE (W,Z,H,t)", "entrelacs non confinés : masse par le VEV", "<1,3 % (RMS 0,74 %)", "⚠ null d'assignation ÉCHOUÉ : accords = compatibilités ; la structure porte (8₃ commun)"),
        ("MATIÈRE NOIRE", "M_topo = √σ_χ·ℓ(K) (loi géométrique)", "1,7-2,2 GeV (bord c=23)", "nœuds premiers isolés c≥23 (dérivé) ; neutralisés par framing"),
        ("NEUTRINOS", "m = ħc/2R (courbure d'auto-rotation)", "R ajusté sur Σm_ν≈66 meV", "onde ouverte ; intervalle (0, m_ℓ=314 MeV] ; Koide IMPOSSIBLE (Q_ν≤0,585 — absence prédite et vérifiée)"),
    ], wrap_cols={1, 2, 3, 4})

    # ================= 5. CATALOGUE CŒUR (37) =================
    ws = sheet(wb, "Catalogue cœur (37)", [16, 14, 34, 8, 8, 14, 14, 12, 26])
    r = titre(ws, "Catalogue canonique σ_χ (source : catalogue_sigma_chi.csv, 07/07 — précédence maximale)")
    header(ws, r, ["Particule", "Catégorie", "Configuration (spin/charge)", "J", "Q",
                   "m exp (MeV)", "m TQNT (MeV)", "Écart %", "Méthode"])
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    data = []
    with open("docs/catalogue_sigma_chi.csv") as f:
        for row in csv.DictReader(f):
            data.append((row["particule"], row["categorie"], row["config_spin_charge"],
                         num(row["spin_J"]), row["charge_Q"], num(row["m_exp_MeV"]),
                         num(row["m_theo_MeV"]), num(row["delta_m_pct"]), row["methode"]))
    r = rows(ws, r + 1, data, wrap_cols={3, 9})

    # ================= 6. SPECTRE ÉTENDU (226) =================
    ws = sheet(wb, "Spectre étendu (226)", [18, 14, 8, 14, 26, 10, 14])
    r = titre(ws, "Table étendue (moteur aligné 21-22/07 ; médiane 1,43 % ; 31/32 spins lus)")
    header(ws, r, ["État", "m exp (GeV)", "J", "m TQNT (GeV)", "Méthode", "J lu", "Fraction trame"])
    data = []
    with open("output/socle_224_resultats.csv") as f:
        for row in csv.DictReader(f):
            data.append((row["etat"],
                         float(row["m_exp_GeV"]) if row["m_exp_GeV"] else None,
                         row["J"],
                         float(row["m_socle_GeV"]) if row["m_socle_GeV"] else None,
                         row["methode"], row["J_lu"], row["fraction_socle"]))
    r = rows(ws, r + 1, data, wrap_cols={5})

    # ================= 7. BOSONS & EW =================
    ws = sheet(wb, "Bosons & EW", [14, 26, 16, 16, 12, 50])
    r = titre(ws, "Bosons et secteur électrofaible (assignations canoniques post-glitch 7₅)")
    header(ws, r, ["Particule", "Contenu topologique", "m TQNT (GeV)", "m exp (GeV)", "Écart %", "Note"])
    r = rows(ws, r + 1, [
        ("γ (photon)", "onde χ libre (aucun nœud)", "0", "0", "exact", "S=0 : théorème de rails — jamais entrée, jamais magnétisée, spectatrice"),
        ("gluon", "vortex du tube χ (pas une particule libre)", "—", "—", "—", "programme exploratoire (Ising/parafermion Z₃) ; confiné par construction"),
        ("W", "7₁ + 7₂ + 8₃", "80,68", "80,38", "+0,39", "plancher du quartet EW ; « 2 familles » dérivé (niveau 2 fermé par construction)"),
        ("Z", "7₂ + 8₃ + 9₁", "90,56", "91,19", "−0,69", "doublet miroir dégénéré (option i) — toute séparation Z/H = signature CP-impaire"),
        ("H", "7₁ + 8₃ + 10₁", "125,26", "125,25", "+0,05", "idem doublet miroir"),
        ("top", "8₂ + 8₃ + 10₁", "170,41", "172,57", "−1,25", "réassigné sans le 7₅ aberrant"),
        ("⚠ AVERTISSEMENT", "les quatre contiennent 8₃ (amphichiral)", "", "", "", "NULL D'ASSIGNATION ÉCHOUÉ (17/07) : ~8e5 quadruplets équivalents — les accords = COMPATIBILITÉS ; ce qui porte = la STRUCTURE (8₃ commun, auto-conjugaison)"),
    ], wrap_cols={2, 6})

    # ================= 8. NEUTRINOS =================
    ws = sheet(wb, "Neutrinos", [24, 18, 18, 18, 50])
    r = titre(ws, "Neutrinos : ondes auto-rotatives ouvertes (m = ħc/2R) — Dirac, 0νββ = 0 exact")
    header(ws, r, ["État", "m (meV)", "R (μm)", "λ_C (μm)", "Note"])
    r = rows(ws, r + 1, [
        ("ν₁ (boomerang, NO)", "≈5", "≈19,7", "≈248", "R ajusté sur Σm_ν ≈ 66 meV (raffinement, pas dérivation)"),
        ("ν₂", "≈10 (≥8,68 osc.)", "≈9,9", "≈124", "plancher oscillations √Δm²₂₁"),
        ("ν₃", "≈51 (≥50,3 osc.)", "≈1,94", "≈24,4", "plancher √Δm²₃₁ ; sous DESI Σ<72 meV"),
        ("", "", "", "", ""),
        ("Intervalle d'existence", "0 < m ≤ 314 MeV", "R ≥ 0,314 fm", "", "borne haute = SEUIL DE NOUAGE (m_ℓ, dérivé) ; le ν observé vit ~6e9 SOUS le seuil"),
        ("Q = 0 (structurel)", "", "", "", "jamais entré dans un rail → pas d'enroulement → pas de charge"),
        ("Faible seulement", "", "", "", "jamais magnétisé → ni forte ni EM ; seule la reconnexion P15"),
        ("Koide : IMPOSSIBLE", "Q_ν ≤ 0,585", "", "", "absence STRUCTURELLE prédite (pas de nœud → pas de circuit) et VÉRIFIÉE par les oscillations"),
        ("SN1987A", "compatible", "", "", "étalement 1,1e-4 s ≪ bouffée 13 s ; l'avance de 3 h = émission (shock breakout), superluminique TRIPLEMENT exclu"),
        ("Falsifiables", "Σm_ν ≈ 66 meV", "", "", "DESI < 60 meV réfuterait ; tout signal 0νββ réfuterait ; tout stérile réfuterait"),
    ], wrap_cols={1, 2, 5})

    # ================= 9. NUCLÉON & CHARGE =================
    ws = sheet(wb, "Nucléon & charge", [26, 30, 60])
    r = titre(ws, "Le nucléon = entrelacs 3₁ + 7₂ + 7̄₂ (cœur fermionique + compagnons miroir)")
    header(ws, r, ["Grandeur", "Valeur", "Règle / origine"])
    r = rows(ws, r + 1, [
        ("Composition", "3₁ (cJ=3,3, χ=−2) + 7₂ (χ=+3) + 7̄₂ (χ=−3)", "Σχ = −2 préservé par la paire miroir ; les compagnons ont des chiralités opposées"),
        ("Charge", "Q = Σχ + Lk = −2 + Lk", "proton : Lk=+3 → Q=+1 ; neutron : Lk=+2 → Q=0"),
        ("m(proton)", "938,2 MeV (0,0 %)", "base de Regge octet (J=½), √σ_χ·S"),
        ("Δm(n−p)", "1,293 MeV (0,1 %)", "DÉRIVÉ : tension d'enlacement β(Lk−Lk_opt)²·√(cJcJ), Lk_opt=+3"),
        ("Spin ½", "arité de tresse", "1 fermion 3₁ + 2 bosons 7₂ : ½⊗1⊗1 → octet ½ / décuplet 3/2"),
        ("Moment magnétique", "porté par le cœur trèfle", "d_tube = λ̄_C(p) à 1 % ; les compagnons miroir se compensent (théorème de la paire)"),
        ("μ_p − μ_n", "4,71 μ_N (~1 %)", "différentiel d'enlacement (r_link/λ̄_C) — les absolus non dérivés"),
        ("r_p", "0,841 fm", "rayon d'exclusion a = 4λ̄_C"),
        ("m_base → G", "m_base = M_Pl√((12/7)α¹⁸)", "la lecture de la rigidité par 18 croisements → G −0,05 %"),
        ("Stabilité", "τ_p ~ α⁻¹⁸ (fini, ~8e33 ans)", "18 croisements partagés avec la gravité ; neutron libre 879 s / lié stable"),
    ], wrap_cols={1, 2, 3})

    # ================= 10. MATIÈRE NOIRE =================
    ws = sheet(wb, "Matière noire", [26, 20, 20, 60])
    r = titre(ws, "Matière noire : nœuds premiers ISOLÉS, c ≥ 23 (seuil dérivé — découplage + survie)")
    header(ws, r, ["Objet / classe", "Masse", "Section efficace", "Note"])
    r = rows(ws, r + 1, [
        ("Réservoir dominant", "1,7-2,2 GeV (bord c=23-25)", "σ_eq ~ 1e-48 cm² [2,5e-49, 4,6e-48]", "T(2,q≥23), T(3,q≥12) ; neutralisés par framing (Q=0) ; polarisabilité d'anneau conducteur α_pol=R³"),
        ("Détection directe", "ABSENCE PRÉDITE", "~7 dex sous DS-50", "LZ/XENONnT (>5 GeV) : nul prédit ✓ ; DS-20k/XLZD (1,7-2,2 GeV) : absence prédite — toute détection réfuterait"),
        ("6 états STRICTS (amphichiraux)", "1,07-1,64 GeV (ACPR)", "raies γ fines", "4₁+8₃ 1,07 ; 6₃+6₃ 1,10 ; 6₃+8₃ 1,22 ; 8₃+8₃ 1,35 ; 4₁+4₁+8₃ 1,47 ; 6₃+6₃+6₃ 1,64 — ×8-13 sous les limites Fermi 2015"),
        ("Extensions EW-neutres", "1,75 / 1,89 GeV", "", "4₁+8₃+8₃ et 6₃+8₃+8₃ (quartet 8₃ en double)"),
        ("111 RÉALISABLES", "diverses", "", "Q_phys=0 via Lk entier — transitoires/contextuels (P22)"),
        ("Ω_DM/Ω_b", "≈0,6 [0,2, 1,3] à T_c dérivé", "", "chaîne entièrement dérivée (relique symétrique / miette CP) — ordre de Planck (5,45) ; comptage mesuré à l'ancien T_c : 3,9±0,4"),
        ("Stabilité", "τ > 1e347 s", "", "barrière Chern-Simons k=135"),
    ], wrap_cols={1, 2, 3, 4})

    # ================= 11. PRÉDICTIONS DATÉES =================
    ws = sheet(wb, "Prédictions datées", [34, 30, 46])
    r = titre(ws, "Prédictions datées (pré-enregistrement du 27/07/2026 — opposables)")
    header(ws, r, ["Prédiction", "Valeur TQNT", "Réfutation si"])
    r = rows(ws, r + 1, [
        ("Ω_cc — CONFIRMÉE", "3715-3788 MeV (LHCb 06/26 : ~3727 ✓)", "antécédent déclaré (dérivation aveugle en pratique)"),
        ("Ξ_cc* (J=3/2)", "3659 MeV", "mesure ~3690 (réseau/QM)"),
        ("Ω_b*", "6067 MeV", "hors ± quelques MeV"),
        ("Ω_b(1P, j=0) — le 5e", "6297-6304 MeV", "absence ou hors fenêtre"),
        ("B₀*/B₁′ (bottom j_K=1/2)", "centroïde 5665 MeV (sép. 6,6)", "5591/5517 ; ~5700-5750 (autres modèles)"),
        ("Quintet Σ_b(1P)", "6076 ; 6092/6102 ; 6111/6128 MeV", "niveaux j0/j2 absents dans Λ_bπ"),
        ("H_cc⁺⁺ (0⁺)", "4497±10 MeV, lié, cτ~30 μm", "aucun signal à vertex déplacé"),
        ("H_cc excité (1⁺)", "4572±10 (seuil Λ_cΛ_c)", "TOUT PIC > 4573 MeV RÉFUTE"),
        ("H léger (uuddss)", "ABSENT (2481, +250)", "toute détection"),
        ("Ω_ccc / Ω_bbb", "4782 MeV strict / hors cadre", "écart net / état lié standard"),
        ("Z/H", "dégénérescence exacte", "toute séparation (= CP-impaire EW)"),
        ("Jamais de 4e famille / stérile", "saturation Z₃", "toute confirmation"),
        ("0νββ", "taux = 0 exactement (Dirac)", "tout signal"),
        ("Σm_ν", "≈66 meV", "DESI < 60 meV"),
        ("6 raies γ", "1,07-1,64 GeV", "absence aux sensibilités ×10 ; continuum étalé = tension"),
        ("Ω_tot", "= 1 exact (théorème)", "toute courbure confirmée"),
        ("w", "= −1 constant", "w(z) confirmé (DESI doit régresser)"),
        ("H₀ / S₈", "doivent régresser", "physique nouvelle confirmée"),
        ("Métrologie de G", "convergence vers 6,6710e-11", "consolidation à 6,674 (réfute à 5e-4)"),
        ("Anomalies CMB", "non-réplication croisée", "toute réplication croisée"),
    ], wrap_cols={1, 2, 3})

    # ================= 12. COMPTAGE & STATUTS =================
    ws = sheet(wb, "Comptage & statuts", [30, 80])
    r = titre(ws, "Le comptage des paramètres et la hiérarchie du méta-test nul")
    header(ws, r, ["Rubrique", "Contenu"])
    r = rows(ws, r + 1, [
        ("Calibrés TQNT : 10 (→9)", "masse (9) : √σ_χ, ratio radial 1,38, τ_⊥, hyperfin lourd, m_ℓ, m_s, m_c, m_b, f_π ; cadre (1) : β_⊗. Si le verrou k/m_ℓ est retenu : NEUF. Contre 19+ pour le Modèle Standard ; ZÉRO par particule."),
        ("Dérivés (échantillon)", "v (+28 ppm), G (−0,05 %), α=1/(k+2), Regge 2πσ_χ, Λ_χ, E_Z3, πσ_χ, T_c, c_min DM=23, exposant P22, structure de Fermi (g_topo=1), N_c=N_gen=3, Ω_tot=1, w=−1, m_e (Green retardée ±3 %)"),
        ("Hints", "A=√2 (porté par le doublement de sens) ; δ=2/9 (décomposé π/12−ε, électron quasi nodal)"),
        ("Noyau dur (méta-test nul)", "4 résultats au-delà de TOUT facteur d'essais : hyperfin V−P (p≤3,8e-11), échelle de recouplage (1e-10), Koide (2e-4), oscillations (2,7e-4) — noyau ×1905 tentatives : 3,9e-25"),
        ("Conditionnels", "v, G, n_s, Fermi, E=αm_ℓ — le poids est dans les MÉCANISMES, pas les seuls nombres"),
        ("Cohérences (pas porteurs)", "k=137 seul (p~0,64), m_e ±3 %, Ω_cc seule, A_s, η_B — ~250 observables au total = CONTEXTE, pas l'argument"),
        ("Hérités (non expliqués)", "Li-7 (mécanisme des cendres réfuté au mur d'efficacité), muon puzzle (gerbes), g−2 du muon (voie P23 archivée)"),
        ("Falsifications passées", "F1 (inflation pure, fermée par Starobinsky) ; F2 (réchauffage, fermée 17/07) ; F3 (raies Class E, levée 15/07 aux masses canoniques) — AUCUNE falsification ouverte"),
    ], wrap_cols={1, 2})

    # ================= 13. NŒUDS PREMIERS (POOL) =================
    from theory.ropelength_acpr import ROPELENGTH_ACPR
    SQS = 418.47
    AMPHI = {"4_1", "6_3", "8_3"}
    TORIQUE = {"3_1", "5_1", "7_1", "9_1"}
    ROLES = {
        "3_1": "cœur leptonique (e/μ/τ = modes Z₃) ; cœur du nucléon",
        "4_1": "amphichiral — DM stricte (4₁+8₃...)",
        "5_1": "torique C5 — composante d'habillage g.3",
        "5_2": "nœud étrange (m_s) ; cœur décuplet",
        "6_1": "chaîne twist (contrôle)", "6_2": "méson diagonal (ρ⁰)",
        "6_3": "amphichiral — DM stricte", "7_1": "torique C7 — EW (W, H)",
        "7_2": "compagnon du nucléon (paire miroir 7₂+7̄₂) ; EW (W, Z)",
        "7_3": "—", "7_4": "—", "7_5": "⚠ courbe historique glitchée (écarté des EW)",
        "7_6": "—", "7_7": "—", "8_1": "méson diagonal (ω)",
        "8_2": "cœur top (EW)", "8_3": "amphichiral — COMMUN aux 4 EW ; DM",
        "9_1": "torique C9 — EW (Z)",
    }
    # état libre (§sec:v6-noeuds-libres, 28/07) : dénouage faible P15,
    # tau = u * hbar * 384 pi^3 v^4 / Q^5 (structure de Fermi dérivée,
    # g_topo=1, N=1 borne sup. ; contrôle muon -0,4 %)
    import math as _math
    HBAR_GS = 6.582119569e-25                        # GeV s
    DENOM_P15 = 384 * _math.pi**3 * 246.22**4        # GeV^4
    M_L = 310.0   # m_ℓ CALIBRÉE (canonique ; = ropelength_acpr.ML)
    U_DENOUAGE = {"3_1": 1, "4_1": 1, "5_1": 2, "5_2": 1, "6_1": 1,
                  "6_2": 1, "6_3": 1, "7_1": 3, "7_2": 1, "7_3": 2,
                  "7_4": 2, "7_5": 2, "7_6": 1, "7_7": 1, "8_1": 1,
                  "8_2": 2, "8_3": 2, "9_1": 4}

    def m_libre(K):
        return M_L * ROPELENGTH_ACPR[K] / ROPELENGTH_ACPR["3_1"]

    def tau_libre(K):
        """durée de vie de l'état libre (dénouage P15, u étapes) [s]."""
        if K not in ROPELENGTH_ACPR or K not in U_DENOUAGE:
            return None
        return U_DENOUAGE[K] * HBAR_GS * DENOM_P15 / (m_libre(K) / 1000.0) ** 5

    ws = sheet(wb, "Nœuds premiers (pool)", [10, 8, 12, 16, 16, 12, 6, 22, 60])
    r = titre(ws, "Le pool des nœuds premiers (ropelengths ACPR 22/07 ; M_topo = m_ℓ·ℓ/ℓ(3₁) — loi géométrique canonique) "
              "+ ÉTAT LIBRE prédit (§sec:v6-noeuds-libres)")
    header(ws, r, ["Nœud", "c", "ℓ (ACPR)", "M_topo = m libre (MeV)", "Type", "Amphichiral", "u",
                   "État libre : τ (dénouage P15)", "Rôle document principalique V6"])
    data = []
    for K in sorted(ROPELENGTH_ACPR, key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1]))):
        ell = ROPELENGTH_ACPR[K]
        c = int(K.split("_")[0])
        typ = "torique" if K in TORIQUE else ("amphichiral" if K in AMPHI else "chiral")
        if K == "3_1":
            t_lbl = "— (état libre = leptons e/μ/τ : 0,511/105,7/1777)"
        else:
            t = tau_libre(K)
            t_lbl = "%.1e s" % t + (" (échelle — loi torique ouverte)" if K in TORIQUE else "")
        data.append((K, c, ell, round(m_libre(K), 1),
                     typ, "oui" if K in AMPHI else "non", U_DENOUAGE.get(K), t_lbl, ROLES.get(K, "—")))
    r = rows(ws, r + 1, data, wrap_cols={8, 9})
    ws.cell(row=r + 1, column=1, value="NB : M_topo = m_ℓ·ℓ(K)/ℓ(3₁), m_ℓ = 310 (calibrée ; loi du plateau léger, "
            "RÉFUTÉE pour charme/beauté — plafond ~1,4 GeV) ; réservoir DM = c ≥ 23 (hors pool). ÉTAT LIBRE "
            "(§sec:v6-noeuds-libres) : τ = u·384π³v⁴ħ/Q⁵ (dénouage faible P15, u = nb de dénouage, N=1 borne sup. ; "
            "contrôle : muon à −0,4 % du PDG sous la même formule) ; production par PAIRES seulement (théorème de "
            "l'entrée) → libres uniquement si un champ critique sépare la paire (P22 ξ≫1) ; 3₁ exclu (son état libre "
            "EST le lepton chargé) ; toriques = échelle seule (loi d'état libre ouverte).").alignment = WRAP

    # ================= 14. ENTRELACS COMPLETS (1539) =================
    from openpyxl import load_workbook
    wb5 = load_workbook("docs/TQNT_catalog_V5_diffusion.xlsx", read_only=True)
    ws5 = wb5["2. Catalogue unifié"]
    def m_geo(config):
        tot = 0.0
        for part in str(config).split("+"):
            k = part.strip()
            if k not in ROPELENGTH_ACPR:
                return None
            tot += m_libre(k)   # m_ℓ = 310 (harmonisé — l'ex-313,85 Koide non consolidé)
        return round(tot, 1)

    # τ PDG des états assignés (la TQNT dérive la STRUCTURE Fermi/Sargent,
    # le nombre par état est le PDG ; résonances : τ = ħ/Γ)
    TAU_PDG = {
        "e⁻": "stable (exact — plancher leptonique)",
        "μ⁻": "2,197e-6 s", "τ⁻": "2,90e-13 s",
        "π±": "2,603e-8 s", "π⁰": "8,4e-17 s",
        "K±": "1,238e-8 s", "K⁰": "K_S 9,0e-11 / K_L 5,1e-8 s",
        "η": "5,0e-19 s (Γ=1,3 keV)", "η'": "3,4e-21 s (Γ=0,19 MeV)",
        "ρ": "4,4e-24 s (Γ=149 MeV)", "ω": "7,6e-23 s (Γ=8,7 MeV)",
        "φ": "1,5e-22 s (Γ=4,2 MeV)",
        "p": "~2,6e41 s (~8e33 ans, τ_p = τ_n·α⁻¹⁸)",
        "n": "878,4 s (libre ; lié avec p : stable)",
        "Λ": "2,63e-10 s", "Σ⁺": "8,0e-11 s", "Σ⁻": "1,48e-10 s",
        "Ξ⁰": "2,90e-10 s", "Ξ⁻": "1,64e-10 s",
        "Δ⁺⁺": "5,6e-24 s (Γ≈117 MeV)", "Ω⁻": "8,2e-11 s",
        "Λ_c⁺": "2,03e-13 s", "Λ_b⁰": "1,46e-12 s",
        "D⁰": "4,10e-13 s", "D±": "1,03e-12 s", "D_s±": "5,0e-13 s",
        "J/ψ": "7,1e-21 s (Γ=93 keV)", "Υ": "1,2e-20 s (Γ=54 keV)",
        "B±": "1,64e-12 s", "B⁰": "1,52e-12 s", "B_s⁰": "1,52e-12 s",
        "W": "3,2e-25 s (Γ=2,09 GeV)", "Z": "2,6e-25 s (Γ=2,50 GeV)",
        "H": "2,1e-22 s (Γ≈3,2 MeV)", "t": "4,6e-25 s (Γ=1,42 GeV)",
        "W±": "3,2e-25 s (Γ=2,09 GeV)", "Z⁰": "2,6e-25 s (Γ=2,50 GeV)",
        "Υ(1S)": "1,2e-20 s (Γ=54 keV)", "Υ(2S)": "2,1e-20 s (Γ=32 keV)",
        "Υ(3S)": "3,2e-20 s (Γ=20 keV)",
        "η_c(1S)": "2,1e-23 s (Γ=32 MeV)", "η_b(1S)": "~7e-23 s (Γ~10 MeV)",
        "χ_c0(1P)": "6,2e-23 s (Γ=10,7 MeV)", "χ_c1(1P)": "7,8e-22 s (Γ=0,84 MeV)",
        "χ_c2(1P)": "3,3e-22 s (Γ=1,97 MeV)", "χ_b0(1P)": "~5e-22 s (Γ~1,3 MeV est.)",
        "K*(892)": "1,3e-23 s (Γ=51 MeV)", "K*(1430)": "6,0e-24 s (Γ≈109 MeV)",
        "D*(2010)±": "7,9e-21 s (Γ=83,4 keV)", "D_s*": "≳3,5e-19 s (Γ<1,9 keV)",
        "B_c±": "5,1e-13 s",
        "Σ_c(2455)": "3,5e-22 s (Γ=1,9 MeV)", "Σ_c(2520)": "4,5e-23 s (Γ=14,7 MeV)",
        "Ξ*⁰": "7,2e-23 s (Γ=9,1 MeV)",
        "Ξ_b⁰": "1,48e-12 s", "Ξ_b(6227)": "~3,5e-23 s (Γ≈19 MeV)",
        "Σ_b(6097)": "~2,3e-23 s (Γ≈29 MeV)", "Σ_b*±": "~6,3e-23 s (Γ≈10 MeV)",
        "Ξ_cc⁺⁺": "2,56e-13 s", "Ω_cc⁰": "non mesurée (attendu ~1e-13 s)",
        "Ω_c⁰": "2,68e-13 s", "Ω_b(6316)": "étroite (Γ non résolue)",
        "Δ(1600)": "~2,6e-24 s (Γ≈250 MeV)", "Δ(1905)": "~2,0e-24 s (Γ≈330 MeV)",
        "N(1710)": "~4,7e-24 s (Γ≈140 MeV)",
        "a₀(980)": "~1e-23 s (Γ 50-100 MeV)", "a₁(1260)": "~1,6e-24 s (Γ 250-600 MeV)",
        "a₂(1320)": "6,2e-24 s (Γ=107 MeV)", "b₁(1235)": "4,6e-24 s (Γ=142 MeV)",
        "f₀(500)": "~1,2e-24 s (Γ 400-700 MeV)", "f₀(1370)": "~2e-24 s (Γ 200-500 MeV)",
        "X(3915)": "~3,3e-23 s (Γ≈20 MeV)", "X(4140)": "~3e-23 s (Γ~20 MeV)",
        "X(4274)": "~1,2e-23 s (Γ≈56 MeV)", "X(6900)": "~8e-24 s (Γ≈80 MeV)",
        "Y(4260)": "~1,2e-23 s (Γ≈55 MeV)", "Y(4360)": "~9e-24 s (Γ≈74 MeV)",
        "Y(4660)": "~9e-24 s (Γ≈72 MeV)",
        "Z_c(3900)": "~2,4e-23 s (Γ≈28 MeV)", "Z_c(4430)": "~3,7e-24 s (Γ≈180 MeV)",
        "Z_b(10610)": "~3,6e-23 s (Γ≈18 MeV)", "Z_b(10650)": "~5,7e-23 s (Γ≈11,5 MeV)",
        "P_c(4312)": "~6,6e-23 s (Γ≈10 MeV)", "P_c(4337)": "~2,3e-23 s (Γ≈29 MeV)",
        "P_c(4450)": "~3,3e-23 s (Γ≈20 MeV)", "P_cs(4459)": "~3,8e-23 s (Γ≈17 MeV)",
        "T_ccs(2941)": "~1e-23 s (Γ large)",
    }
    # correspondance noms V5 -> catalogue cœur V6 (loi de secteur)
    V5_TO_CORE = {"e⁻": "e-", "μ⁻": "mu-", "τ⁻": "tau-", "π±": "pi+", "π⁰": "pi0",
                  "K±": "K+", "K⁰": "K0", "η": "eta", "η'": "eta'", "ρ": "rho",
                  "ω": "omega", "φ": "phi", "p": "p", "n": "n", "Λ": "Lambda",
                  "Σ⁺": "Sigma+", "Σ⁻": "Sigma-", "Ξ⁰": "Xi0", "Ξ⁻": "Xi-",
                  "Δ⁺⁺": "Delta", "Ω⁻": "Omega-", "Λ_c⁺": "Lambda_c",
                  "Λ_b⁰": "Lambda_b", "D⁰": "D0", "D±": "D+", "D_s±": "Ds+",
                  "J/ψ": "J/psi", "Υ": "Ups", "Υ(1S)": "Ups", "B±": "B+", "B⁰": "B0", "B_s⁰": "Bs0",
                  "W": "W", "W±": "W", "Z": "Z", "Z⁰": "Z", "H": "H", "t": "top"}
    # états excités/exotiques assignés -> nom du spectre 226 (moteur aligné,
    # m_socle) ; quarkonia/B_c EXCLUS (compression : hors fonctionnelle par
    # principe — leurs valeurs CSV sont des placeholders de base)
    EXC_TO_226 = {
        "D*(2010)±": "D*(2010)+", "D_s*": "Ds*(2112)",
        "Σ_c(2455)": "Sc(2455)", "Σ_c(2520)": "Sc(2520)",
        "Ξ*⁰": "Xi(1530)3/2+", "Δ(1600)": "Dl(1600)3/2+",
        "Δ(1905)": "Dl(1905)5/2+", "N(1710)": "N(1710)1/2+",
        "Ξ_b⁰": "Xb(5792)", "Ξ_b(6227)": "Xb(6227)LHCb",
        "Σ_b(6097)": "Sb(6097)LHCb", "Σ_b*±": "Sb*(5832)",
        "Ξ_cc⁺⁺": "Xicc(3621)LHCb", "Ω_c⁰": "Oc(2695)",
        "a₂(1320)": "a2(1320)", "K*(1430)": "K2*(1430)",
        "Z_c(3900)": "Zc(3900)=DD*", "Z_b(10610)": "Zb(10610)=BB*",
        "Z_b(10650)": "Zb(10650)=B*B*", "P_c(4312)": "Pc(4312)=ScD",
        "P_c(4450)": "Pc(4457)=ScD*", "P_cs(4459)": "Pcs(4459)=XicD*",
        "X(4140)": "X(4140)", "X(4274)": "X(4274)", "X(6900)": "X(6900)LHC",
    }
    socle_m = {}
    with open("output/socle_224_resultats.csv") as f:
        for row in csv.DictReader(f):
            v = num(row["m_socle_GeV"])
            if v:
                socle_m[row["etat"]] = round(v * 1000.0, 1)
    core_m = {}
    with open("docs/catalogue_sigma_chi.csv") as f:
        for row in csv.DictReader(f):
            core_m[row["particule"]] = num(row["m_theo_MeV"])

    def tau_v6(pdg_lbl, tau5, n, c):
        lbl = str(pdg_lbl or "")
        t5 = str(tau5 or "")
        if "MN" in lbl or "matière noire" in lbl or "10³⁴⁷" in t5:
            return "DM : stable (τ > 1e347 s, barrière CS k=135)"
        if lbl.strip() in ("e⁻", "p"):
            return "stable (e : plancher leptonique ; p : fini ~8e33 ans)"
        if lbl.strip() and lbl.strip() not in ("—", "⚪ libre"):
            return "instable : Fermi/Sargent (Q=Δm, g_topo=1) — τ = PDG (structure dérivée)"
        if (n or 0) >= 2:
            return "instable : cascade P15 (~1e-18 s — théorème du désert)"
        if (c or 0) >= 23:
            return "réservoir DM : stable (τ > 1e347 s)"
        if (c or 0) >= 18:
            return "exclu/évacué (c_min DM dérivé = 23)"
        return "cœur réalisable (contextuel, P22)"

    ws = sheet(wb, "Entrelacs complets (1539)", [8, 26, 6, 6, 10, 12, 12, 10, 10, 22, 18, 34, 22, 16, 12, 30])
    r = titre(ws, "ÉNUMÉRATION EXHAUSTIVE : 1539 entrelacs jusqu'à 30 croisements — topologie canonique, "
              "masses et stabilités V6")
    ws.cell(row=r - 1, column=1, value="⚠ ΣM_topo = somme des masses topologiques BRUTES des composants "
            "(échelle des constituants, m_ℓ=310) — PAS la masse prédite : « σ_χ prévaut » — la config porte le spin "
            "et la charge, la MASSE suit la loi de son SECTEUR. « m V6 » = loi de secteur pour les états "
            "assignés (via le catalogue cœur) ; nœud n=1 non assigné = masse d'ÉTAT LIBRE prédite "
            "(§sec:v6-noeuds-libres) ; entrelacs sombre = estimation ΣM_topo (déclarée). Assignations affichées = "
            "HISTORIQUES V5 (le document principal V6 diffère : n = 3₁+7₂+7̄₂, π = 3₁+3̄₁, μ/τ = modes Z₃ du trèfle — cf. Catalogue "
            "cœur). « τ V6 » = nombres par classe : assignés → τ PDG (la TQNT dérive la STRUCTURE Fermi/Sargent, "
            "g_topo=1, pas un nombre par état) ; nœud n=1 libre → dénouage P15 τ = u·384π³v⁴ħ/Q⁵ (contrôle muon "
            "−0,4 %) ; DM → barrière CS ; DÉSERT (cascade P15) → volontairement SANS nombres fins (~1e-18 s "
            "générique). États excités/molécules assignés : m V6 = moteur 226 aligné ; quarkonia et B_c = secteur "
            "COMPRESSION : masse hors fonctionnelle PAR PRINCIPE (pas de m V6 ; τ = PDG). Les τ Ω_ts V5 sont "
            "ARCHIVÉS (barème réfuté).").alignment = WRAP
    header(ws, r, ["#", "Config", "n", "c", "J spectre", "Q_topo", "Lk neutr.", "Q_phys", "Lk PDG",
                   "ΣM_topo composants (MeV) — pas la masse prédite", "m V6 (MeV, loi de secteur / état libre)",
                   "τ/stabilité V6 (classe)", "τ V6 (s)", "Particule (assign. V5 — historique)", "m_exp (MeV)", "Note"])
    data = []
    for row in ws5.iter_rows(min_row=3, values_only=True):
        if row[0] is None:
            continue
        numero, cfg, n, c, J, qt, lkn, qp, lkpdg, mots, _, _, tau5, _, pdg, mexp = row[:16]
        lbl = str(pdg or "").strip()
        key = V5_TO_CORE.get(lbl)
        mv6 = core_m.get(key) if key else None
        if mv6 is None and lbl in EXC_TO_226:
            mv6 = socle_m.get(EXC_TO_226[lbl])   # moteur 226 (excités/molécules)
        if mv6 is None and lbl == "Ω_cc⁰":
            mv6 = "3715-3788 (prédit ; LHCb 2026 ~3727 ✓)"
        cls = tau_v6(pdg, tau5, n, c)
        tau_num = None
        if "cascade P15" in cls:
            pass                                   # désert : sans nombres (demande 28/07)
        elif lbl in TAU_PDG and ("Fermi/Sargent" in cls or "stable (e" in cls):
            tau_num = TAU_PDG[lbl]
        elif "réalisable" in cls:                  # nœud n=1 : état libre prédit
            K = str(cfg).strip()
            if K == "3_1":                         # état libre = les leptons (modes Z₃)
                tau_num = "— (état libre = leptons e/μ/τ)"
            else:
                t = tau_libre(K)
                if t is not None:
                    suffix = " (échelle — loi torique ouverte)" if K in TORIQUE else " (dénouage P15)"
                    tau_num = ("%.1e s" % t) + suffix
                else:
                    tau_num = "dénouage P15 (ℓ hors table ACPR — cible)"
                if mv6 is None and K in ROPELENGTH_ACPR:
                    mv6 = round(m_libre(K), 1)
        elif "1e347" in cls or "DM" in cls:        # stricts + réservoir : barrière CS
            tau_num = "> 1e347 s (barrière CS k=135)"
            if mv6 is None:
                mv6 = m_geo(cfg)
        elif "exclu" in cls:                       # bande 18-22 : évacuée
            tau_num = "population nulle (évacuée au gel)"
            if mv6 is None:
                mv6 = m_geo(cfg)
        data.append((numero, cfg, n, c, J, qt, lkn, qp, lkpdg, m_geo(cfg), mv6,
                     cls, tau_num, pdg, mexp, row[18]))
    r = rows(ws, r + 1, data, wrap_cols={2, 12, 13, 16})
    print("  entrelacs exportés :", len(data),
          "; m V6 renseignées :", sum(1 for d in data if d[10] is not None),
          "; τ V6 renseignées :", sum(1 for d in data if d[12] is not None))

    # ================= 15. 117 CANDIDATS DM =================
    ws = sheet(wb, "117 candidats DM", [8, 14, 26, 6, 6, 20, 10, 8, 10, 16, 18])
    r = titre(ws, "Les 117 candidats matière noire (6 STRICTS amphichiraux + 111 RÉALISABLES Lk-neutralisés) — "
              "masses géométriques ACPR ; statuts LZ")
    header(ws, r, ["id", "Niveau", "Config", "n", "c", "ΣM_topo (MeV) — prédiction sombre (estimation)",
                   "Q_topo", "Lk", "Q_phys", "Statut LZ (H1)", "Sous-population"])
    import csv as _csv
    data = []
    with open("output/catalogue_117_LZ_status.csv") as f:
        for row in _csv.DictReader(f):
            data.append((int(row["id"]), row["niveau"], row["config"], int(row["n"]), int(row["c"]),
                         m_geo(row["config"]), row["Q_topo"], row["Lk"], row["Q_phys"],
                         row["lz_status_H1"], row["sub_population"]))
    r = rows(ws, r + 1, data, wrap_cols={3})
    ws.cell(row=r + 1, column=1, value="NB : les 6 STRICTS canoniques (ACPR 22/07) : 4₁+8₃ 1,07 GeV ; 6₃+6₃ 1,10 ; "
            "6₃+8₃ 1,22 ; 8₃+8₃ 1,35 ; 4₁+4₁+8₃ 1,47 ; 6₃+6₃+6₃ 1,64 → 6 raies γ fines.").alignment = WRAP

    out = "docs/TQNT_catalog_complete.xlsx"
    wb.save(out)
    print("écrit :", out)
    for ws in wb.worksheets:
        print("  onglet %-24s : %d lignes" % (ws.title, ws.max_row))


if __name__ == "__main__":
    main()
