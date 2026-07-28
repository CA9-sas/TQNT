#!/usr/bin/env python3
r"""
Upload automatisé d'un dépôt Zenodo à partir d'un fichier
``zenodo_metadata_paperX.json``.

Convertit les métadonnées au format API Zenodo, upload les fichiers
attachés, publie le dépôt et imprime le DOI obtenu.

Usage typique :

  # 1) Obtenir un token Zenodo : https://zenodo.org/account/settings/applications/
  #    (scopes nécessaires : deposit:write + deposit:actions)
  export ZENODO_TOKEN="votre_token_personnel"

  # 2) Tester d'abord sur le sandbox (DOI fictif, dépôt effacé périodiquement)
  python3 -m simulation.experiments.zenodo_upload \
      docs/zenodo_metadata_paper3.json \
      --file docs/dm_pearls_paper.pdf \
      --file docs/dm_pearls_paper.tex \
      --sandbox

  # 3) Publication réelle sur zenodo.org (DOI permanent)
  python3 -m simulation.experiments.zenodo_upload \
      docs/zenodo_metadata_paper3.json \
      --file docs/dm_pearls_paper.pdf \
      --file docs/dm_pearls_paper.tex

  # 4) Dry-run (validation sans upload)
  python3 -m simulation.experiments.zenodo_upload \
      docs/zenodo_metadata_paper3.json \
      --file docs/dm_pearls_paper.pdf \
      --dry-run

  # 5) Créer un draft seulement (pas de publication finale)
  python3 -m simulation.experiments.zenodo_upload \
      docs/zenodo_metadata_paper3.json \
      --file docs/dm_pearls_paper.pdf \
      --draft

Le script affiche :
  - L'ID de dépôt Zenodo
  - L'URL du draft (pour édition manuelle si besoin)
  - Le DOI permanent une fois publié
  - L'URL publique du dépôt

Sécurité :
  Le token n'est jamais affiché ni sauvegardé. Ne pas commiter
  de fichier .env contenant ZENODO_TOKEN.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Erreur : 'requests' requis. Installer avec : pip install requests")


# ──────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────
ZENODO_PROD = "https://zenodo.org"
ZENODO_SANDBOX = "https://sandbox.zenodo.org"

# Conversion notre format → format API Zenodo
LICENSE_MAP = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc0-1.0": "CC0-1.0",
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
}


# ──────────────────────────────────────────────────────────────────────────
# Conversion JSON gabarit → format API Zenodo
# ──────────────────────────────────────────────────────────────────────────
def to_zenodo_api_format(raw: dict) -> dict:
    """Transforme notre JSON métadonnée vers le format attendu par
    l'API REST Zenodo.

    Différences principales :
      - retire les champs internes (préfixe `_`)
      - convertit la licence en MAJUSCULES (CC-BY-4.0 vs cc-by-4.0)
      - ajoute le champ `term` aux subjects (label libre)
      - retire les champs non supportés (title_en, _*_note)
    """
    out = {}

    # Champs simples copiés tels quels
    simple_fields = (
        "title", "description", "version", "publication_date",
        "language", "upload_type", "publication_type",
        "access_right", "notes", "method",
        "imprint_publisher",
    )
    for k in simple_fields:
        if k in raw and not k.startswith("_"):
            out[k] = raw[k]

    # Listes simples
    if "keywords" in raw:
        out["keywords"] = list(raw["keywords"])
    if "references" in raw:
        out["references"] = list(raw["references"])

    # Creators (passer tel quel, mais retirer ORCID si vide)
    if "creators" in raw:
        out["creators"] = []
        for c in raw["creators"]:
            entry = {"name": c["name"]}
            if c.get("affiliation"):
                entry["affiliation"] = c["affiliation"]
            if c.get("orcid"):
                entry["orcid"] = c["orcid"]
            out["creators"].append(entry)

    # License : convertir au format Zenodo (SPDX uppercase)
    if "license" in raw:
        lic = raw["license"]
        out["license"] = LICENSE_MAP.get(lic.lower(), lic)

    # Subjects : ajouter le champ `term` requis par l'API
    if "subjects" in raw:
        out["subjects"] = []
        for s in raw["subjects"]:
            entry = dict(s)
            if "term" not in entry and "identifier" in entry:
                entry["term"] = entry["identifier"]
            out["subjects"].append(entry)

    # Communities (filtrer les vides)
    if "communities" in raw and raw["communities"]:
        out["communities"] = [
            {"identifier": c["identifier"] if isinstance(c, dict) else c}
            for c in raw["communities"]
        ]

    # Related identifiers (passer tel quel, retirer les clés inconnues)
    if "related_identifiers" in raw:
        out["related_identifiers"] = []
        for ri in raw["related_identifiers"]:
            entry = {
                "identifier": ri["identifier"],
                "relation": ri["relation"],
            }
            if "resource_type" in ri:
                entry["resource_type"] = ri["resource_type"]
            if "scheme" in ri:
                entry["scheme"] = ri["scheme"]
            out["related_identifiers"].append(entry)

    return out


# ──────────────────────────────────────────────────────────────────────────
# Opérations API
# ──────────────────────────────────────────────────────────────────────────
def api_create_deposition(base_url: str, token: str) -> dict:
    """Crée un dépôt vide et retourne sa réponse JSON."""
    r = requests.post(
        f"{base_url}/api/deposit/depositions",
        params={"access_token": token},
        json={},
    )
    r.raise_for_status()
    return r.json()


def api_upload_file(bucket_url: str, file_path: Path, token: str) -> dict:
    """Upload un fichier vers le bucket du dépôt."""
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    with open(file_path, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{file_path.name}",
            data=f,
            params={"access_token": token},
        )
    r.raise_for_status()
    return r.json()


def api_set_metadata(base_url: str, deposition_id: int,
                     metadata: dict, token: str) -> dict:
    """Met à jour les métadonnées du dépôt."""
    r = requests.put(
        f"{base_url}/api/deposit/depositions/{deposition_id}",
        params={"access_token": token},
        json={"metadata": metadata},
    )
    r.raise_for_status()
    return r.json()


def api_publish(base_url: str, deposition_id: int, token: str) -> dict:
    """Publie le dépôt (action irréversible, attribue le DOI)."""
    r = requests.post(
        f"{base_url}/api/deposit/depositions/{deposition_id}/actions/publish",
        params={"access_token": token},
    )
    r.raise_for_status()
    return r.json()


def api_new_version(base_url: str, existing_id: int, token: str) -> dict:
    """Crée une nouvelle version d'un dépôt publié.

    Retourne le dépôt draft de la nouvelle version (avec son propre ID
    et son propre DOI réservé). La nouvelle version hérite des fichiers
    et métadonnées de la précédente — les fichiers à remplacer doivent
    être supprimés explicitement avant l'upload des nouveaux."""
    r = requests.post(
        f"{base_url}/api/deposit/depositions/{existing_id}/actions/newversion",
        params={"access_token": token},
    )
    r.raise_for_status()
    # La réponse contient links.latest_draft → URL du nouveau dépôt
    latest_draft_url = r.json()["links"]["latest_draft"]
    new_id = int(latest_draft_url.rstrip("/").rsplit("/", 1)[-1])
    # Fetch the new draft pour avoir bucket_url etc.
    r2 = requests.get(
        f"{base_url}/api/deposit/depositions/{new_id}",
        params={"access_token": token},
    )
    r2.raise_for_status()
    return r2.json()


def api_delete_all_files(base_url: str, deposition_id: int, token: str) -> int:
    """Supprime tous les fichiers d'un dépôt draft. Retourne le nombre
    de fichiers supprimés. Nécessaire avant new-version pour remplacer
    les PDFs par des versions corrigées."""
    r = requests.get(
        f"{base_url}/api/deposit/depositions/{deposition_id}/files",
        params={"access_token": token},
    )
    r.raise_for_status()
    files = r.json()
    for f in files:
        rd = requests.delete(
            f"{base_url}/api/deposit/depositions/{deposition_id}/files/{f['id']}",
            params={"access_token": token},
        )
        rd.raise_for_status()
    return len(files)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Upload automatisé d'un dépôt Zenodo depuis un fichier "
            "zenodo_metadata_paperX.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "metadata_json",
        type=Path,
        help="Chemin vers le fichier zenodo_metadata_paperX.json",
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        action="append",
        required=False,
        help="Fichier à uploader (PDF, TeX, etc.). Répéter pour plusieurs fichiers.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Utiliser sandbox.zenodo.org (test, DOI fictif).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valider la conversion JSON sans rien envoyer à Zenodo.",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Créer un draft sans publier (modifiable manuellement après).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token Zenodo (par défaut : variable d'environnement ZENODO_TOKEN).",
    )
    parser.add_argument(
        "--new-version",
        type=int,
        default=None,
        metavar="EXISTING_ID",
        help=(
            "Publier comme nouvelle version d'un dépôt existant. "
            "EXISTING_ID est l'ID du dépôt original "
            "(ex: 20559267 pour DOI 10.5281/zenodo.20559267). "
            "Les anciens fichiers sont supprimés, les nouveaux fichiers "
            "et métadonnées sont uploadés. Bumper le champ `version` "
            "dans le JSON avant pour différencier."
        ),
    )
    args = parser.parse_args()

    # 1) Charger et valider le JSON métadonnée
    if not args.metadata_json.exists():
        sys.exit(f"Fichier introuvable : {args.metadata_json}")
    with open(args.metadata_json) as f:
        raw = json.load(f)

    # 2) Convertir au format API
    metadata = to_zenodo_api_format(raw)

    print(f"=== Métadonnées converties ===")
    print(f"  Titre        : {metadata['title'][:80]}...")
    print(f"  Auteur       : {metadata['creators'][0]['name']}")
    print(f"  ORCID        : {metadata['creators'][0].get('orcid', '—')}")
    print(f"  Type         : {metadata.get('publication_type', '?')}")
    print(f"  Version      : {metadata.get('version', '?')}")
    print(f"  Date         : {metadata.get('publication_date', '?')}")
    print(f"  Licence      : {metadata.get('license', '?')}")
    print(f"  Keywords     : {len(metadata.get('keywords', []))}")
    print(f"  Subjects     : {len(metadata.get('subjects', []))}")
    print(f"  Related ids  : {len(metadata.get('related_identifiers', []))}")
    print(f"  References   : {len(metadata.get('references', []))}")
    print(f"  Description  : {len(metadata.get('description', ''))} caractères")

    files = args.file or []
    print(f"\n=== Fichiers à uploader ({len(files)}) ===")
    for f in files:
        if not f.exists():
            sys.exit(f"  ✗ Fichier introuvable : {f}")
        size = f.stat().st_size / 1024
        print(f"  ✓ {f}  ({size:.1f} KB)")

    if args.dry_run:
        print("\n=== Mode --dry-run : aucun envoi à Zenodo ===")
        print("Métadonnées validées avec succès.")
        return 0

    # 3) Obtenir le token
    token = args.token or os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit(
            "\n✗ Aucun token fourni. Définir ZENODO_TOKEN dans l'environnement\n"
            "  ou utiliser --token. Obtenir un token sur :\n"
            "  https://zenodo.org/account/settings/applications/"
        )

    # 4) Choix prod / sandbox
    base_url = ZENODO_SANDBOX if args.sandbox else ZENODO_PROD
    print(f"\n=== Cible : {base_url} ===")
    if args.sandbox:
        print("  (mode sandbox : DOI fictif, dépôt effacé périodiquement)")

    # 5) Créer le dépôt (nouveau OU nouvelle version d'un existant)
    if args.new_version is not None:
        print(f"\n→ Nouvelle version du dépôt #{args.new_version}...")
        try:
            dep = api_new_version(base_url, args.new_version, token)
        except requests.HTTPError as e:
            sys.exit(
                f"✗ Échec création new-version : {e.response.status_code} "
                f"{e.response.text}"
            )
        deposition_id = dep["id"]
        bucket_url = dep["links"]["bucket"]
        edit_url = dep["links"]["html"]
        print(f"  ✓ Nouveau draft créé, ID = {deposition_id}")
        print(f"  Edit URL : {edit_url}")

        # Supprimer les anciens fichiers (sinon ils restent dans la nouvelle version)
        print(f"\n→ Suppression des fichiers de la version précédente...")
        try:
            n_deleted = api_delete_all_files(base_url, deposition_id, token)
            print(f"  ✓ {n_deleted} fichier(s) supprimé(s)")
        except requests.HTTPError as e:
            sys.exit(
                f"  ✗ Échec suppression : {e.response.status_code} "
                f"{e.response.text}"
            )
    else:
        print("\n→ Création du dépôt vide...")
        try:
            dep = api_create_deposition(base_url, token)
        except requests.HTTPError as e:
            sys.exit(f"✗ Échec création : {e.response.status_code} {e.response.text}")
        deposition_id = dep["id"]
        bucket_url = dep["links"]["bucket"]
        edit_url = dep["links"]["html"]
        print(f"  ✓ Dépôt créé, ID = {deposition_id}")
        print(f"  Edit URL : {edit_url}")

    # 6) Upload des fichiers
    if files:
        print(f"\n→ Upload de {len(files)} fichier(s)...")
        for f in files:
            try:
                api_upload_file(bucket_url, f, token)
                print(f"  ✓ {f.name}")
            except requests.HTTPError as e:
                sys.exit(f"  ✗ Échec upload {f.name} : {e.response.text}")

    # 7) Pousser les métadonnées
    print("\n→ Envoi des métadonnées...")
    try:
        api_set_metadata(base_url, deposition_id, metadata, token)
        print(f"  ✓ Métadonnées enregistrées")
    except requests.HTTPError as e:
        sys.exit(
            f"  ✗ Échec métadonnées : {e.response.status_code}\n"
            f"    Réponse Zenodo : {e.response.text}\n"
            f"    Le dépôt {deposition_id} a été créé mais n'est pas publié.\n"
            f"    Vous pouvez l'éditer manuellement à : {edit_url}"
        )

    # 8) Publier (ou laisser en draft)
    if args.draft:
        print(f"\n=== Mode --draft : dépôt non publié ===")
        print(f"  Édition manuelle : {edit_url}")
        print(f"  Pour publier ultérieurement, cliquer 'Publish' sur la page.")
        return 0

    print("\n→ Publication finale (irréversible)...")
    try:
        pub = api_publish(base_url, deposition_id, token)
    except requests.HTTPError as e:
        sys.exit(
            f"  ✗ Échec publication : {e.response.status_code}\n"
            f"    Réponse Zenodo : {e.response.text}\n"
            f"    Le draft existe à : {edit_url}\n"
            f"    Vous pouvez l'éditer puis publier manuellement."
        )

    doi = pub.get("doi", "—")
    record_url = pub["links"].get("record_html", "—")
    print("\n" + "═" * 60)
    print(" 🎉 PUBLICATION RÉUSSIE")
    print("═" * 60)
    print(f"  DOI permanent : {doi}")
    print(f"  URL publique  : {record_url}")
    print(f"  ID dépôt      : {deposition_id}")
    if args.sandbox:
        print("\n  ⚠️  DOI SANDBOX uniquement (non citable hors test).")
    print("═" * 60)
    print(f"\n  → Mettre à jour les .tex avec :")
    print(f"    doi:{doi}")
    print(f"  → Ajouter le DOI dans Related identifiers des autres dépôts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
