#!/usr/bin/env python3
r"""
Met à jour automatiquement les cross-citations entre les papiers TQNT
dans :
  - les fichiers .tex de chaque papier (\bibitem et footnote titre)
  - les fichiers .json metadata Zenodo (related_identifiers)
  - les fichiers de cover letter (mentions DOI)

Une fois les nouveaux DOIs obtenus (via zenodo_upload.py / zenodo_batch.sh),
modifier le dictionnaire DOIS ci-dessous puis exécuter ce script.

Usage :
  # 1) Éditer ce fichier pour mettre les nouveaux DOIs réels
  # 2) Lancer :
  python3 -m simulation.experiments.update_cross_doi

  # Mode --check pour valider sans modifier (affiche ce qui changerait)
  python3 -m simulation.experiments.update_cross_doi --check
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJ_ROOT / "docs"


# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION : DOIs des papiers
#
# REMPLACER les "PENDING" par les vrais DOIs obtenus de Zenodo
# après publication (format complet : 10.5281/zenodo.XXXXXXX)
# ──────────────────────────────────────────────────────────────────────────
DOIS = {
    # Concept DOI = résout vers la dernière version disponible (recommandé
    # pour citations stables). Version DOI = pointe vers une version
    # spécifique (utile pour la traçabilité d'une publication particulière).
    # Les documents .tex utilisent le concept DOI pour rester valides
    # quand une nouvelle version est déposée.
    "V5":     "10.5281/zenodo.19131538",     # Monographe TQNT — concept DOI
                                              # (version V5 spécifique :
                                              #  10.5281/zenodo.20539411)
    "paper1": "10.5281/zenodo.20552673",     # CP/Jones (publié)
    "paper2": "10.5281/zenodo.20552914",     # AJ → Schrödinger (publié)
    "paper3": "10.5281/zenodo.20559267",     # DM Pearls (publié v1.1 le 2026-06-06)
    "paper4": "10.5281/zenodo.20559279",     # Solar topologique (publié v1.1 le 2026-06-06)
    "paper5": "10.5281/zenodo.20559287",     # Confinement topologique (publié 2026-06-06)
    "paper6": "10.5281/zenodo.20559293",     # Qubits chiraux (publié 2026-06-06)
}

# Version DOIs (utile pour citations historiques figées sur une version) :
#   V5  spécifique : 10.5281/zenodo.20539411 (instance V5 unique)
# Pour citer "la dernière TQNT", utiliser le concept DOI ci-dessus.
# Pour citer "TQNT V5 telle qu'au 4 juin 2026", utiliser le version DOI.

# Titres pour les références bibliographiques
TITLES_EN = {
    "V5":     "TQNT-V5: Quantum Theory of Topological Knots",
    "paper1": "A CP-violating phase from the Jones polynomial at $q = e^{2\\pi i/137}$",
    "paper2": "From the AJ conjecture to emergent Schrödinger dynamics",
    "paper3": "Dark matter pearls: compact objects from topological Fermi pressure",
    "paper4": "Topological photovoltaic conversion",
    "paper5": "Topological electron confinement for electrical storage",
    "paper6": "Nodal topological quantum computer",
}

TITLES_FR = {
    "V5":     "TQNT-V5 : Théorie Quantique des Nœuds Topologiques",
    "paper1": "Une phase de violation CP à partir du polynôme de Jones en $q = e^{2\\pi i/137}$",
    "paper2": "De la conjecture AJ à la dynamique de Schrödinger émergente",
    "paper3": "Perles de matière noire : objets compacts à partir d'une pression de Fermi topologique",
    "paper4": "Conversion photovoltaïque par résonance topologique",
    "paper5": "Stockage électrique par confinement topologique d'électrons",
    "paper6": "Ordinateur quantique topologique nodal",
}

# Mapping .tex file → paper ID (pour éviter qu'un papier se cite lui-même)
TEX_TO_PAPER = {
    "cp_jones_paper.tex":                          "paper1",
    "cp_jones_paper_FR.tex":                       "paper1",
    "aj_schrodinger_paper.tex":                    "paper2",
    "aj_schrodinger_paper_FR.tex":                 "paper2",
    "dm_pearls_paper.tex":                         "paper3",
    "dm_pearls_paper_FR.tex":                      "paper3",
    "topological_solar_paper_FR.tex":              "paper4",
    "topological_battery_paper_FR.tex":            "paper5",
    "topological_quantum_computer_paper_FR.tex":   "paper6",
    "TQNT_V5_companion_arxiv.tex":                 None,  # companion, peut citer tous
}

# Mapping .json file → paper ID
JSON_TO_PAPER = {
    "zenodo_metadata.json":         "V5",
    "zenodo_metadata_paper1.json":  "paper1",
    "zenodo_metadata_paper2.json":  "paper2",
    "zenodo_metadata_paper3.json":  "paper3",
    "zenodo_metadata_paper4.json":  "paper4",
    "zenodo_metadata_paper5.json":  "paper5",
    "zenodo_metadata_paper6.json":  "paper6",
}


# ──────────────────────────────────────────────────────────────────────────
# Mises à jour
# ──────────────────────────────────────────────────────────────────────────
def update_tex_dois(tex_path: Path, check: bool = False) -> int:
    """Remplace les anciens DOIs par les nouveaux dans un fichier .tex.

    Cherche les motifs :
      - doi:10.5281/zenodo.XXXXXXX
      - zenodo.XXXXXXX
      - https://doi.org/10.5281/zenodo.XXXXXXX

    Retourne le nombre de remplacements effectués.
    """
    if not tex_path.exists():
        return 0

    content = tex_path.read_text(encoding="utf-8")
    original = content
    changes = []

    # Pour chaque DOI connu, vérifier si une version "PENDING" est mentionnée
    for paper_id, doi in DOIS.items():
        if doi == "PENDING":
            continue
        # Pas de remplacement sur cette pass : c'est le DOI canonique.
        # Mais on signale les références "à venir", "à mettre à jour", etc.
        for marker in ("[DOI Zenodo à venir]", "[DOI à venir]",
                       "[à venir]", "zenodo.XXXXXXX"):
            if marker in content:
                pass  # détecté mais pas remplacé sans DOI cible spécifique

    # Mise à jour des sub-DOIs PENDING -> DOI réel
    # On cherche les patterns explicites contenant le sub-ID
    for paper_id, doi in DOIS.items():
        if doi == "PENDING":
            continue
        # Cherche un placeholder spécifique au paper
        placeholder_patterns = [
            f"DOI à venir pour {paper_id}",
            f"forthcoming for {paper_id}",
            f"[{paper_id}:DOI]",
        ]
        for pat in placeholder_patterns:
            if pat in content:
                content = content.replace(pat, doi)
                changes.append(f"  {pat!r} → {doi}")

    if content != original:
        if not check:
            tex_path.write_text(content, encoding="utf-8")
        return len(changes)
    return 0


def update_json_related_ids(json_path: Path, check: bool = False) -> int:
    """Ajoute/met à jour les `related_identifiers` pour pointer vers
    tous les autres papiers TQNT (avec relation `cites` ou `hasPart`)."""
    if not json_path.exists():
        return 0

    json_file = json_path.name
    self_id = JSON_TO_PAPER.get(json_file)
    if self_id is None:
        return 0

    with open(json_path) as f:
        data = json.load(f)

    original_rels = data.get("related_identifiers", [])
    existing_dois = set()
    for ri in original_rels:
        existing_dois.add(ri.get("identifier"))

    new_rels = list(original_rels)
    additions = []

    # Décider qui cite qui
    if self_id == "V5":
        # V5 a pour parties (hasPart) tous les papiers
        relation = "hasPart"
        targets = ["paper1", "paper2", "paper3", "paper4", "paper5", "paper6"]
    else:
        # Chaque paper cite V5 (isSupplementTo, déjà fait) + les autres papers (cites)
        relation = "cites"
        targets = [pid for pid in
                   ("paper1", "paper2", "paper3", "paper4", "paper5", "paper6")
                   if pid != self_id]

    for target in targets:
        target_doi = DOIS.get(target)
        if not target_doi or target_doi == "PENDING":
            continue
        if target_doi in existing_dois:
            continue
        new_entry = {
            "identifier": target_doi,
            "relation": relation,
            "resource_type": "publication-preprint" if target.startswith("paper")
                             else "publication-report",
            "scheme": "doi",
        }
        new_rels.append(new_entry)
        additions.append(f"  + {relation} → {target_doi}  ({target})")

    if additions:
        data["related_identifiers"] = new_rels
        # Préserver la note interne si présente
        if not check:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return len(additions), additions

    return 0, []


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Mode validation : afficher les modifications sans appliquer.",
    )
    args = parser.parse_args()

    # Vérifier qu'il reste des DOIs PENDING
    pending = [pid for pid, doi in DOIS.items() if doi == "PENDING"]
    if pending:
        print("=" * 60)
        print("⚠️  DOIs non encore définis :")
        for pid in pending:
            print(f"  - {pid}")
        print()
        print("  ÉDITER ce fichier (simulation/experiments/update_cross_doi.py)")
        print("  pour remplacer 'PENDING' par les vrais DOIs Zenodo obtenus.")
        print("=" * 60)
        if not args.check:
            print("\nSeules les mises à jour partielles seront appliquées.")
        print()

    if args.check:
        print("=== MODE --check : aucun fichier ne sera modifié ===\n")

    # 1) Mise à jour des .tex
    print("→ Mise à jour des fichiers .tex")
    total_tex_changes = 0
    for tex_name in TEX_TO_PAPER:
        tex_path = DOCS_DIR / tex_name
        n = update_tex_dois(tex_path, check=args.check)
        if n > 0:
            print(f"  ✓ {tex_name}  ({n} remplacement(s))")
            total_tex_changes += n
    if total_tex_changes == 0:
        print("  (aucun remplacement nécessaire)")
    print()

    # 2) Mise à jour des .json metadata
    print("→ Mise à jour des Related identifiers JSON Zenodo")
    total_json_additions = 0
    for json_name in JSON_TO_PAPER:
        json_path = DOCS_DIR / json_name
        result = update_json_related_ids(json_path, check=args.check)
        if isinstance(result, tuple):
            n, additions = result
        else:
            n = result
            additions = []
        if n > 0:
            print(f"  ✓ {json_name}  ({n} ajout(s))")
            for a in additions:
                print(a)
            total_json_additions += n
    if total_json_additions == 0:
        print("  (aucun ajout nécessaire)")
    print()

    # Synthèse
    print("=" * 60)
    print(f"  Total modifications : {total_tex_changes + total_json_additions}")
    print(f"    - {total_tex_changes} dans les fichiers .tex")
    print(f"    - {total_json_additions} dans les fichiers .json")
    if args.check:
        print(f"  Mode --check : aucun fichier modifié")
    else:
        print(f"  Fichiers mis à jour avec succès")
    print("=" * 60)

    if total_tex_changes > 0 and not args.check:
        print("\n→ Étape suivante : recompiler les PDFs concernés :")
        print("  cd docs && for f in *.tex; do pdflatex -interaction=nonstopmode $f; done")
        print("  (puis recompiler une 2e fois pour les références croisées)")

    if total_json_additions > 0 and not args.check:
        print("\n→ Sur Zenodo : éditer les dépôts existants (V5, P1, P2)")
        print("  pour y ajouter manuellement les nouveaux Related identifiers")
        print("  via le bouton 'Edit' sur la page de chaque dépôt.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
